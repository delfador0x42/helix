//! Append-only data log: primary storage. Never modified in place.
//! Format: LogHeader (8B) + sequential Entry/Delete records.

use std::io::{Read, Seek, SeekFrom, Write};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

pub const LOG_MAGIC: [u8; 4] = *b"AMRL";
pub const LOG_VERSION: u32 = 1;
const LOG_HEADER_SIZE: u64 = 8;
const ENTRY_HEADER_SIZE: usize = 12;
const DELETE_RECORD_SIZE: usize = 8;

pub struct LogEntry {
    pub offset: u32,
    pub topic: String,
    pub body: String,
    pub timestamp_min: i32,
}

/// Create data.log with header if absent.
pub fn ensure_log(dir: &Path) -> Result<PathBuf, String> {
    let path = dir.join("data.log");
    if path.exists() { return Ok(path); }
    let mut f = File::create(&path).map_err(|e| format!("create data.log: {e}"))?;
    f.write_all(&LOG_MAGIC).map_err(|e| e.to_string())?;
    f.write_all(&LOG_VERSION.to_le_bytes()).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())?;
    Ok(path)
}

/// Append one entry. Returns log offset.
pub fn append_entry(log_path: &Path, topic: &str, body: &str, ts_min: i32) -> Result<u32, String> {
    let mut f = OpenOptions::new().append(true).open(log_path)
        .map_err(|e| format!("open data.log: {e}"))?;
    let offset = f.seek(SeekFrom::End(0)).map_err(|e| e.to_string())? as u32;
    write_entry(&mut f, topic, body, ts_min)?;
    f.sync_data().map_err(|e| e.to_string())?;
    Ok(offset)
}

/// Append entry to already-open handle (no fsync). For batch writes.
pub fn append_entry_to(f: &mut File, topic: &str, body: &str, ts_min: i32) -> Result<u32, String> {
    let offset = f.seek(SeekFrom::End(0)).map_err(|e| e.to_string())? as u32;
    write_entry(f, topic, body, ts_min)?;
    Ok(offset)
}

fn write_entry(f: &mut impl Write, topic: &str, body: &str, ts_min: i32) -> Result<(), String> {
    let tb = topic.as_bytes();
    let bb = body.as_bytes();
    let hdr = entry_header(tb.len() as u8, bb.len() as u32, ts_min);
    f.write_all(&hdr).map_err(|e| e.to_string())?;
    f.write_all(tb).map_err(|e| e.to_string())?;
    f.write_all(bb).map_err(|e| e.to_string())?;
    Ok(())
}

/// Append a delete tombstone.
pub fn append_delete(log_path: &Path, target_offset: u32) -> Result<(), String> {
    let mut f = OpenOptions::new().append(true).open(log_path)
        .map_err(|e| format!("open data.log: {e}"))?;
    let mut rec = [0u8; DELETE_RECORD_SIZE];
    rec[0] = 0x02;
    rec[4..8].copy_from_slice(&target_offset.to_le_bytes());
    f.write_all(&rec).map_err(|e| e.to_string())?;
    f.sync_data().map_err(|e| e.to_string())?;
    Ok(())
}

pub fn read_entry_from(f: &mut File, offset: u32) -> Result<LogEntry, String> {
    f.seek(SeekFrom::Start(offset as u64)).map_err(|e| e.to_string())?;
    let mut hdr = [0u8; ENTRY_HEADER_SIZE];
    f.read_exact(&mut hdr).map_err(|e| format!("read header: {e}"))?;
    if hdr[0] != 0x01 { return Err("not an entry record".into()); }
    let tl = hdr[1] as usize;
    let bl = u32::from_le_bytes([hdr[2], hdr[3], hdr[4], hdr[5]]) as usize;
    let ts = i32::from_le_bytes([hdr[6], hdr[7], hdr[8], hdr[9]]);
    let mut buf = vec![0u8; tl + bl];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(LogEntry {
        offset,
        topic: String::from_utf8_lossy(&buf[..tl]).into(),
        body: String::from_utf8_lossy(&buf[tl..]).into(),
        timestamp_min: ts,
    })
}

/// Iterate all live entries (single-pass, filters tombstones).
pub fn iter_live(log_path: &Path) -> Result<Vec<LogEntry>, String> {
    let data = fs::read(log_path).map_err(|e| format!("read data.log: {e}"))?;
    if data.len() < LOG_HEADER_SIZE as usize { return Err("data.log too small".into()); }
    if data[..4] != LOG_MAGIC { return Err("bad data.log magic".into()); }

    let mut entries = Vec::new();
    let mut deleted = crate::fxhash::FxHashSet::default();
    let mut pos = LOG_HEADER_SIZE as usize;

    while pos < data.len() {
        match data[pos] {
            0x01 => {
                if pos + ENTRY_HEADER_SIZE > data.len() { break; }
                let offset = pos as u32;
                let tl = data[pos + 1] as usize;
                let bl = u32::from_le_bytes([data[pos+2], data[pos+3], data[pos+4], data[pos+5]]) as usize;
                let ts = i32::from_le_bytes([data[pos+6], data[pos+7], data[pos+8], data[pos+9]]);
                let rec_end = pos + ENTRY_HEADER_SIZE + tl + bl;
                if rec_end > data.len() { break; }
                let topic = String::from_utf8_lossy(&data[pos+ENTRY_HEADER_SIZE..pos+ENTRY_HEADER_SIZE+tl]).into();
                let body = String::from_utf8_lossy(&data[pos+ENTRY_HEADER_SIZE+tl..rec_end]).into();
                entries.push(LogEntry { offset, topic, body, timestamp_min: ts });
                pos = rec_end;
            }
            0x02 => {
                if pos + DELETE_RECORD_SIZE > data.len() { break; }
                let target = u32::from_le_bytes([data[pos+4], data[pos+5], data[pos+6], data[pos+7]]);
                deleted.insert(target);
                pos += DELETE_RECORD_SIZE;
            }
            _ => break,
        }
    }
    if !deleted.is_empty() { entries.retain(|e| !deleted.contains(&e.offset)); }
    Ok(entries)
}

/// Rewrite data.log without deleted entries.
pub fn compact_log(dir: &Path) -> Result<String, String> {
    let log_path = dir.join("data.log");
    let entries = iter_live(&log_path)?;
    let before = fs::metadata(&log_path).map(|m| m.len()).unwrap_or(0);
    let tmp = dir.join("data.log.tmp");
    {
        let mut f = File::create(&tmp).map_err(|e| e.to_string())?;
        f.write_all(&LOG_MAGIC).map_err(|e| e.to_string())?;
        f.write_all(&LOG_VERSION.to_le_bytes()).map_err(|e| e.to_string())?;
        for e in &entries {
            write_entry(&mut f, &e.topic, &e.body, e.timestamp_min)?;
        }
        f.sync_all().map_err(|e| e.to_string())?;
    }
    fs::rename(&tmp, &log_path).map_err(|e| e.to_string())?;
    let after = fs::metadata(&log_path).map(|m| m.len()).unwrap_or(0);
    Ok(format!("compacted: {} entries, {} → {} bytes", entries.len(), before, after))
}

fn entry_header(topic_len: u8, body_len: u32, ts_min: i32) -> [u8; ENTRY_HEADER_SIZE] {
    let mut h = [0u8; ENTRY_HEADER_SIZE];
    h[0] = 0x01;
    h[1] = topic_len;
    h[2..6].copy_from_slice(&body_len.to_le_bytes());
    h[6..10].copy_from_slice(&ts_min.to_le_bytes());
    h
}

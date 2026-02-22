//! Read helix's data.log binary format (read-only).
//! Format: 8B header (AMRL + version) + sequential entry/delete records.

use std::path::Path;

const LOG_MAGIC: [u8; 4] = *b"AMRL";
const ENTRY_HDR: usize = 12; // type(1) + topic_len(1) + body_len(4) + ts_min(4) + pad(2)
const DELETE_REC: usize = 8;  // type(1) + pad(3) + target_offset(4)

pub struct Entry {
    pub offset: u32,
    pub topic: String,
    pub body: String,
    pub timestamp_min: i32,
}

impl Entry {
    /// Extract content lines (skip metadata lines starting with '[').
    pub fn content(&self) -> String {
        self.body.lines()
            .filter(|l| !(l.starts_with('[') && l.ends_with(']')))
            .collect::<Vec<_>>().join("\n")
    }

    /// Extract tags from [tags: ...] metadata line.
    pub fn tags(&self) -> Vec<String> {
        for line in self.body.lines() {
            if let Some(rest) = line.strip_prefix("[tags: ") {
                if let Some(inner) = rest.strip_suffix(']') {
                    return inner.split(',').map(|s| s.trim().to_lowercase()).collect();
                }
            }
        }
        Vec::new()
    }
}

/// Read all live entries from data.log (filters out deleted).
pub fn read_entries(path: &Path) -> Result<Vec<Entry>, String> {
    let data = std::fs::read(path).map_err(|e| format!("read data.log: {e}"))?;
    if data.len() < 8 { return Err("data.log too small".into()); }
    if data[..4] != LOG_MAGIC { return Err("bad data.log magic".into()); }

    let mut entries = Vec::new();
    let mut deleted = std::collections::HashSet::new();
    let mut pos = 8; // skip header

    while pos < data.len() {
        match data[pos] {
            0x01 => {
                if pos + ENTRY_HDR > data.len() { break; }
                let tl = data[pos + 1] as usize;
                let bl = u32::from_le_bytes([data[pos+2], data[pos+3], data[pos+4], data[pos+5]]) as usize;
                let ts = i32::from_le_bytes([data[pos+6], data[pos+7], data[pos+8], data[pos+9]]);
                let rec_end = pos + ENTRY_HDR + tl + bl;
                if rec_end > data.len() { break; }
                let topic = String::from_utf8_lossy(&data[pos+ENTRY_HDR..pos+ENTRY_HDR+tl]).into();
                let body = String::from_utf8_lossy(&data[pos+ENTRY_HDR+tl..rec_end]).into();
                entries.push(Entry { offset: pos as u32, topic, body, timestamp_min: ts });
                pos = rec_end;
            }
            0x02 => {
                if pos + DELETE_REC > data.len() { break; }
                let target = u32::from_le_bytes([data[pos+4], data[pos+5], data[pos+6], data[pos+7]]);
                deleted.insert(target);
                pos += DELETE_REC;
            }
            _ => break,
        }
    }
    if !deleted.is_empty() { entries.retain(|e| !deleted.contains(&e.offset)); }
    Ok(entries)
}

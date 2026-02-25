//! MCP server — stdio JSON-RPC loop, server state, streaming helpers.

use crate::json::Value;
use std::io::{self, BufRead, Write as _};
use std::path::Path;
use std::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};

// ══════════ Server State ══════════

static SESSION_LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());
static INDEX: RwLock<Option<Vec<u8>>> = RwLock::new(None);
static INDEX_DIRTY: AtomicBool = AtomicBool::new(false);
static DIRTY_AT: Mutex<Option<std::time::Instant>> = Mutex::new(None);
static REBUILD_ACTIVE: AtomicBool = AtomicBool::new(false);

pub(crate) fn log_session(msg: String) {
    if let Ok(mut log) = SESSION_LOG.lock() {
        if log.len() >= 200 { log.drain(..50); }
        log.push(msg);
    }
}

pub(crate) fn store_index(data: Vec<u8>) {
    if let Ok(mut g) = INDEX.write() { *g = Some(data); }
}

pub(crate) fn with_index<F, R>(f: F) -> Option<R>
where F: FnOnce(&[u8]) -> R {
    INDEX.read().ok().and_then(|g| g.as_ref().map(|d| f(d)))
}

pub(crate) fn with_index_slice<F, R>(f: F) -> Result<R, String>
where F: FnOnce(Option<&[u8]>) -> R {
    let guard = INDEX.read().map_err(|e| e.to_string())?;
    Ok(f(guard.as_ref().map(|d| d.as_slice())))
}

pub(crate) fn with_session_log<F, R>(f: F) -> R
where F: FnOnce(&[String]) -> R {
    let log = SESSION_LOG.lock().unwrap_or_else(|e| e.into_inner());
    f(&log)
}

pub(crate) fn after_write(_dir: &Path) {
    INDEX_DIRTY.store(true, Ordering::Release);
    if let Ok(mut g) = DIRTY_AT.lock() { if g.is_none() { *g = Some(std::time::Instant::now()); } }
}

/// Async index rebuild — reads never block. REBUILD_ACTIVE prevents concurrent rebuilds.
pub(crate) fn ensure_index_fresh(dir: &Path) {
    if !INDEX_DIRTY.load(Ordering::Acquire) {
        let marker = std::path::Path::new("/tmp/helix-external-write");
        if marker.exists() {
            std::fs::remove_file(marker).ok();
            INDEX_DIRTY.store(true, Ordering::Release);
            if let Ok(mut g) = DIRTY_AT.lock() { *g = Some(std::time::Instant::now()); }
        }
    }
    if !INDEX_DIRTY.load(Ordering::Acquire) { return; }
    let should = DIRTY_AT.lock().ok().map_or(false, |g| {
        matches!(*g, Some(t) if t.elapsed() >= std::time::Duration::from_millis(50))
    });
    if !should { return; }
    if REBUILD_ACTIVE.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).is_err() {
        return;
    }
    INDEX_DIRTY.store(false, Ordering::Release);
    if let Ok(mut g) = DIRTY_AT.lock() { *g = None; }
    let dir = dir.to_path_buf();
    std::thread::spawn(move || {
        match crate::index::rebuild(&dir, true) {
            Ok((_, bytes)) => store_index(bytes),
            Err(_) => { if let Ok(d) = std::fs::read(dir.join("index.bin")) { store_index(d); } }
        }
        REBUILD_ACTIVE.store(false, Ordering::Release);
    });
}

// ══════════ Server Loop ══════════

const INIT_RESULT: &str = r#"{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"helix","version":"1.2.0"}}"#;

pub fn run(dir: &Path) -> Result<(), String> {
    crate::config::ensure_dir(dir)?;
    crate::datalog::ensure_log(dir)?;
    match crate::index::rebuild(dir, true) {
        Ok((_, bytes)) => store_index(bytes),
        Err(_) => { if let Ok(d) = std::fs::read(dir.join("index.bin")) { store_index(d); } }
    }
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut line_buf = String::with_capacity(4096);
    let mut reader = io::BufReader::new(stdin.lock());
    loop {
        line_buf.clear();
        match reader.read_line(&mut line_buf) {
            Ok(0) => break, Ok(_) => {}, Err(e) => return Err(e.to_string()),
        }
        let line = line_buf.trim();
        if line.is_empty() || line.len() > 10_000_000 { continue; }
        let msg = match crate::json::parse(line) { Ok(v) => v, Err(_) => continue };
        let method = msg.get("method").and_then(|v| v.as_str()).unwrap_or("");
        let id = msg.get("id");
        match method {
            "initialize" => {
                let id_json = id_to_json(id);
                let mut out = stdout.lock();
                let _ = write!(out, r#"{{"jsonrpc":"2.0","id":{id_json},"result":{INIT_RESULT}}}"#);
                let _ = writeln!(out); let _ = out.flush();
            }
            "notifications/initialized" | "initialized" => {}
            "tools/list" => {
                let id_json = id_to_json(id);
                let tools_json = crate::tools::tool_list_json();
                let mut out = stdout.lock();
                let _ = write!(out, r#"{{"jsonrpc":"2.0","id":{id_json},"result":{tools_json}}}"#);
                let _ = writeln!(out); let _ = out.flush();
            }
            "tools/call" => {
                let p = msg.get("params");
                let name = p.and_then(|p| p.get("name")).and_then(|v| v.as_str()).unwrap_or("");
                let id_json = id_to_json(id);
                if name == "_reload" {
                    let report = crate::reload::reload_verify(dir);
                    let mut out = stdout.lock();
                    let _ = write_rpc_ok(&mut out, &id_json, &report);
                    let _ = out.flush(); drop(out);
                    crate::reload::do_reexec();
                    continue;
                }
                let args = p.and_then(|p| p.get("arguments"));
                let mut out = stdout.lock();
                let ok = match crate::dispatch::dispatch(name, args, dir) {
                    Ok(ref text) => write_rpc_ok(&mut out, &id_json, text),
                    Err(ref e) => write_rpc_err(&mut out, &id_json, e),
                };
                if let Err(e) = ok { eprintln!("helix: write error: {e}"); break; }
                let _ = out.flush();
            }
            "ping" => {
                let id_json = id_to_json(id);
                let mut out = stdout.lock();
                let _ = write!(out, r#"{{"jsonrpc":"2.0","id":{id_json},"result":{{}}}}"#);
                let _ = writeln!(out); let _ = out.flush();
            }
            _ if id.is_some() => {
                let id_json = id_to_json(id);
                let mut out = stdout.lock();
                let _ = write!(out, r#"{{"jsonrpc":"2.0","id":{id_json},"error":{{"code":-32601,"message":"method not found"}}}}"#);
                let _ = writeln!(out); let _ = out.flush();
            }
            _ => {}
        }
    }
    Ok(())
}

// ══════════ IdBuf + Streaming ══════════

struct IdBuf { bytes: [u8; 24], len: u8 }
impl std::ops::Deref for IdBuf {
    type Target = str;
    fn deref(&self) -> &str { unsafe { std::str::from_utf8_unchecked(&self.bytes[..self.len as usize]) } }
}
impl std::fmt::Display for IdBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str(self) }
}

fn id_to_json(id: Option<&Value>) -> IdBuf {
    match id {
        Some(Value::Num(n)) if n.fract() == 0.0 => {
            let mut buf = IdBuf { bytes: [0u8; 24], len: 0 };
            let v = *n as i64;
            if v == 0 { buf.bytes[0] = b'0'; buf.len = 1; return buf; }
            let (neg, mut uv) = if v < 0 { (true, (-(v as i128)) as u64) } else { (false, v as u64) };
            let mut i = 24;
            while uv > 0 { i -= 1; buf.bytes[i] = b'0' + (uv % 10) as u8; uv /= 10; }
            if neg { i -= 1; buf.bytes[i] = b'-'; }
            buf.bytes.copy_within(i..24, 0);
            buf.len = (24 - i) as u8;
            buf
        }
        Some(v) => {
            let s = v.to_string();
            let mut buf = IdBuf { bytes: [0u8; 24], len: s.len().min(24) as u8 };
            buf.bytes[..buf.len as usize].copy_from_slice(&s.as_bytes()[..buf.len as usize]);
            buf
        }
        None => { let mut buf = IdBuf { bytes: [0u8; 24], len: 4 }; buf.bytes[..4].copy_from_slice(b"null"); buf }
    }
}

fn write_rpc_ok(w: &mut impl io::Write, id: &str, text: &str) -> io::Result<()> {
    w.write_all(b"{\"jsonrpc\":\"2.0\",\"id\":")?;
    w.write_all(id.as_bytes())?;
    w.write_all(b",\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"")?;
    write_json_escaped(w, text)?;
    w.write_all(b"\"}]}}\n")
}

fn write_rpc_err(w: &mut impl io::Write, id: &str, msg: &str) -> io::Result<()> {
    w.write_all(b"{\"jsonrpc\":\"2.0\",\"id\":")?;
    w.write_all(id.as_bytes())?;
    w.write_all(b",\"error\":{\"code\":-32603,\"message\":\"")?;
    write_json_escaped(w, msg)?;
    w.write_all(b"\"}}\n")
}

fn write_json_escaped(w: &mut impl io::Write, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    let mut last = 0;
    for (i, &b) in bytes.iter().enumerate() {
        let esc: &[u8] = match b {
            b'"' => b"\\\"", b'\\' => b"\\\\", b'\n' => b"\\n", b'\r' => b"\\r", b'\t' => b"\\t",
            c if c < 0x20 => {
                if last < i { w.write_all(&bytes[last..i])?; }
                write!(w, "\\u{:04x}", c)?;
                last = i + 1; continue;
            }
            _ => continue,
        };
        if last < i { w.write_all(&bytes[last..i])?; }
        w.write_all(esc)?;
        last = i + 1;
    }
    if last < bytes.len() { w.write_all(&bytes[last..])?; }
    Ok(())
}

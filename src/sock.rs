//! Unix domain socket for hook queries against the in-memory index.
//! MCP server spawns a listener thread; hook processes connect for zero-I/O queries.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

/// Socket path: <dir>/hook.sock
pub fn sock_path(dir: &Path) -> PathBuf { dir.join("hook.sock") }

/// Start the socket listener thread. Returns guard for RAII cleanup.
pub fn start_listener(dir: &Path) -> Option<SockGuard> {
    let path = sock_path(dir);
    let _ = std::fs::remove_file(&path);
    let listener = match UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => { eprintln!("helix: sock bind failed: {e}"); return None; }
    };
    listener.set_nonblocking(false).ok();
    let dir2 = dir.to_path_buf();
    let path2 = path.clone();
    let handle = std::thread::spawn(move || {
        for stream in listener.incoming() {
            match stream {
                Ok(s) => { handle_conn(s, &dir2); }
                Err(e) => {
                    if !path2.exists() { break; }
                    if e.kind() != std::io::ErrorKind::WouldBlock {
                        eprintln!("helix: sock accept: {e}");
                    }
                }
            }
        }
    });
    Some(SockGuard { path, _handle: handle })
}

pub struct SockGuard { path: PathBuf, _handle: std::thread::JoinHandle<()> }

impl Drop for SockGuard {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.path); }
}

/// Handle a single hook query. 100ms timeout, 512-byte BufReader.
fn handle_conn(stream: UnixStream, _dir: &Path) {
    stream.set_read_timeout(Some(std::time::Duration::from_millis(100))).ok();
    stream.set_write_timeout(Some(std::time::Duration::from_millis(100))).ok();

    let mut reader = BufReader::with_capacity(512, &stream);
    let mut line = String::with_capacity(256);
    if reader.read_line(&mut line).is_err() { return; }
    let line = line.trim();
    if line.is_empty() { return; }

    let op = crate::hook::extract_json_str(line, "op").unwrap_or("");
    let result = match op {
        "search" => {
            let req = match crate::json::parse(line) { Ok(v) => v, Err(_) => return };
            handle_search(&req)
        }
        "topics" => handle_topics(),
        "ambient" => handle_ambient_fast(line),
        "hook_ambient" => handle_hook_relay(line),
        _ => String::new(),
    };

    let mut writer = stream;
    let _ = writer.write_all(result.as_bytes());
    let _ = writer.write_all(b"\n");
    let _ = writer.flush();
}

/// Search the in-memory index.
/// Request: {"op":"search","query":"cache","limit":5}
fn handle_search(req: &crate::json::Value) -> String {
    let query = req.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let limit = req.get("limit").and_then(|v| v.as_f64()).unwrap_or(5.0) as usize;
    crate::mcp::with_index(|data| {
        let hits = crate::index::search_index(
            data, query, &crate::index::FilterPred::none(), limit, true,
        ).unwrap_or_default();
        let mut out = String::with_capacity(hits.len() * 100);
        for (i, h) in hits.iter().enumerate() {
            if i > 0 { out.push('\n'); }
            out.push_str(&h.snippet);
        }
        out
    }).unwrap_or_default()
}

/// Return topic table from in-memory index.
/// Request: {"op":"topics"}
fn handle_topics() -> String {
    crate::mcp::with_index(|data| {
        let mut topics = crate::index::topic_table(data).unwrap_or_default();
        topics.sort_unstable_by(|a, b| a.1.cmp(&b.1));
        let mut out = String::with_capacity(topics.len() * 24);
        for (i, (_, name, count)) in topics.iter().enumerate() {
            if i > 0 { out.push_str(", "); }
            out.push_str(name);
            out.push_str(" (");
            crate::text::itoa_push(&mut out, *count as u32);
            out.push(')');
        }
        out
    }).unwrap_or_default()
}

/// Combined ambient hook query — fast string extraction, no full JSON parse.
/// Request: {"op":"ambient","stem":"cache","path":"/full/path","syms":["removed1"]}
fn handle_ambient_fast(line: &str) -> String {
    let stem = match crate::hook::extract_json_str(line, "stem") {
        Some(s) if !s.is_empty() => s,
        _ => return String::new(),
    };
    let file_path = crate::hook::extract_json_str(line, "\"path\"").unwrap_or("");
    let syms = extract_syms_array(line);
    crate::mcp::with_index(|data| {
        crate::hook::query_ambient(data, stem, file_path, &syms, None)
    }).unwrap_or_default()
}

/// Extract string array from "syms":["a","b","c"] without full JSON parse.
fn extract_syms_array(line: &str) -> Vec<&str> {
    let needle = "\"syms\":[";
    let pos = match line.find(needle) { Some(p) => p + needle.len(), None => return Vec::new() };
    let rest = &line[pos..];
    let end = match rest.find(']') { Some(e) => e, None => return Vec::new() };
    rest[..end].split('"')
        .enumerate().filter(|(i, _)| i % 2 == 1)
        .map(|(_, s)| s).filter(|s| !s.is_empty()).collect()
}

/// Handle hook relay: full Claude Code hook stdin with spliced op field.
/// Ambient: {"op":"hook_ambient","tool_name":"Read","tool_input":{"file_path":"..."}}
/// Subagent: {"op":"hook_ambient","type":"subagent-start"}
fn handle_hook_relay(line: &str) -> String {
    let htype = crate::hook::extract_json_str(line, "type").unwrap_or("");
    if htype == "subagent-start" {
        let topics = handle_topics();
        if topics.is_empty() {
            return crate::hook::hook_output(
                "HELIX KNOWLEDGE STORE: You have access to helix MCP tools. \
                 Search before starting work.");
        }
        let mut msg = String::with_capacity(128 + topics.len());
        msg.push_str(
            "HELIX KNOWLEDGE STORE: You have access to helix MCP tools. \
             BEFORE starting work, call mcp__helix__search with keywords \
             relevant to your task. Topics: ");
        msg.push_str(&topics);
        return crate::hook::hook_output(&msg);
    }

    // Ambient: extract tool, file, stem
    let tool = crate::hook::extract_json_str(line, "tool_name").unwrap_or("");
    let is_edit = tool == "Edit";
    match tool {
        "Read" | "Edit" | "Write" | "Glob" | "Grep" | "NotebookEdit" => {}
        _ => return String::new(),
    }
    let path = crate::hook::extract_json_str(line, "file_path")
        .or_else(|| crate::hook::extract_json_str(line, "\"path\""))
        .unwrap_or("");
    if path.is_empty() { return String::new(); }
    let stem = std::path::Path::new(path)
        .file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if stem.len() < 3 { return String::new(); }

    let syms = if is_edit {
        match crate::json::parse(line) {
            Ok(val) => crate::hook::extract_removed_syms(&val, stem),
            Err(_) => vec![],
        }
    } else { vec![] };
    let sym_refs: Vec<&str> = syms.iter().map(|s| s.as_str()).collect();

    let ctx = crate::mcp::with_index(|data| {
        crate::hook::query_ambient(data, stem, path, &sym_refs, None)
    }).unwrap_or_default();
    if ctx.is_empty() { return String::new(); }
    crate::hook::hook_output(&ctx)
}

/// Client: query the running MCP server's socket. Returns None if unavailable.
pub fn query(dir: &Path, request: &str) -> Option<String> {
    let path = sock_path(dir);
    let mut stream = UnixStream::connect(&path).ok()?;
    stream.set_read_timeout(Some(std::time::Duration::from_millis(50))).ok();
    stream.set_write_timeout(Some(std::time::Duration::from_millis(50))).ok();
    stream.write_all(request.as_bytes()).ok()?;
    stream.write_all(b"\n").ok()?;
    stream.flush().ok()?;
    let mut reader = BufReader::with_capacity(1024, stream);
    let mut response = String::with_capacity(512);
    reader.read_line(&mut response).ok()?;
    let trimmed = response.trim();
    if trimmed.is_empty() { None } else { Some(trimmed.to_string()) }
}

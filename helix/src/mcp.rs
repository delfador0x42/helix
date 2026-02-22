//! MCP server: stdio JSON-RPC, in-memory index, debounced rebuilds, 7 tools.
//! Merges mcp.rs (428) + dispatch.rs (504) + tools.rs (231) = 1163 → ~450 lines.

use crate::json::Value;
use std::io::{self, BufRead, Write as _};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};

// ══════════ Server State ══════════

static SESSION_LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());
static INDEX: RwLock<Option<Vec<u8>>> = RwLock::new(None);
static INDEX_DIRTY: AtomicBool = AtomicBool::new(false);
static DIRTY_AT: Mutex<Option<std::time::Instant>> = Mutex::new(None);
static REBUILD_ACTIVE: AtomicBool = AtomicBool::new(false);

pub(crate) fn log_session(msg: String) {
    if let Ok(mut log) = SESSION_LOG.lock() { log.push(msg); }
}

pub(crate) fn store_index(data: Vec<u8>) {
    if let Ok(mut g) = INDEX.write() { *g = Some(data); }
}

pub(crate) fn with_index<F, R>(f: F) -> Option<R>
where F: FnOnce(&[u8]) -> R {
    INDEX.read().ok().and_then(|g| g.as_ref().map(|d| f(d)))
}

pub(crate) fn after_write(_dir: &Path) {
    INDEX_DIRTY.store(true, Ordering::Release);
    if let Ok(mut g) = DIRTY_AT.lock() { if g.is_none() { *g = Some(std::time::Instant::now()); } }
}

/// Async index rebuild — reads never block. Old index serves reads while rebuild
/// runs in a background thread. REBUILD_ACTIVE prevents concurrent rebuilds.
fn ensure_index_fresh(dir: &Path) {
    // Detect external writes (hook auto-capture writes to data.log directly)
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
    // CAS: only one rebuild thread at a time
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
                let tools_json = tool_list_json();
                let mut out = stdout.lock();
                let _ = write!(out, r#"{{"jsonrpc":"2.0","id":{id_json},"result":{tools_json}}}"#);
                let _ = writeln!(out); let _ = out.flush();
            }
            "tools/call" => {
                let p = msg.get("params");
                let name = p.and_then(|p| p.get("name")).and_then(|v| v.as_str()).unwrap_or("");
                let id_json = id_to_json(id);
                if name == "_reload" {
                    let mut out = stdout.lock();
                    let _ = write_rpc_ok(&mut out, &id_json, "reloading helix...");
                    let _ = out.flush(); drop(out);
                    do_reload();
                    continue;
                }
                let args = p.and_then(|p| p.get("arguments"));
                let mut out = stdout.lock();
                let ok = match dispatch(name, args, dir) {
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

// ══════════ Dispatch ══════════

pub fn dispatch(name: &str, args: Option<&Value>, dir: &Path) -> Result<String, String> {
    match name {
        "store" | "batch" | "edit" => {}
        _ => ensure_index_fresh(dir),
    }
    match name {
        "store" => {
            let topic = arg(args, "topic"); let text = arg(args, "text");
            let tags = opt(arg(args, "tags")); let force = abool(args, "force");
            let terse = abool(args, "terse"); let source = opt(arg(args, "source"));
            let conf = arg(args, "confidence").parse::<f64>().ok().filter(|c| *c >= 0.0 && *c <= 1.0);
            let links = opt(arg(args, "links"));
            let result = crate::write::store(dir, topic, text, tags, force, source, conf, links)?;
            after_write(dir);
            log_session(format!("[{}] {}", topic, result.lines().next().unwrap_or("stored")));
            if terse { Ok(result.lines().next().unwrap_or(&result).to_string()) } else { Ok(result) }
        }
        "batch" => dispatch_batch(args, dir),
        "search" => dispatch_search(args, dir),
        "brief" => {
            let query = arg(args, "query");
            if query.is_empty() {
                crate::cache::with_corpus(dir, |cached| {
                    let compact = abool(args, "compact");
                    let mut topics: std::collections::BTreeMap<&str, (usize, i32)> = std::collections::BTreeMap::new();
                    for e in cached {
                        let (c, latest) = topics.entry(&e.topic).or_insert((0, 0));
                        *c += 1; if e.timestamp_min > *latest { *latest = e.timestamp_min; }
                    }
                    let mut sorted: Vec<_> = topics.into_iter().collect();
                    sorted.sort_by(|a, b| b.1.1.cmp(&a.1.1));
                    let limit = if compact { 5 } else { sorted.len() };
                    let mut out = format!("{} topics, {} entries\n\n", sorted.len(), cached.len());
                    use std::fmt::Write;
                    for (topic, (count, latest)) in sorted.iter().take(limit) {
                        let _ = writeln!(out, "  {topic} ({count}, last: {})", crate::time::minutes_to_date_str(*latest));
                    }
                    if sorted.len() > limit { let _ = writeln!(out, "  ... +{} more", sorted.len() - limit); }
                    out
                })
            } else {
                let detail = arg(args, "detail"); let detail = if detail.is_empty() { "summary" } else { detail };
                let since = arg(args, "since").parse::<u64>().ok();
                let focus = opt(arg(args, "focus"));
                crate::brief::run(dir, query, detail, since, focus)
            }
        }
        "read" => {
            let topic = arg(args, "topic");
            let log_path = crate::config::log_path(dir);
            let entries = crate::write::topic_entries(&log_path, topic)?;
            if entries.is_empty() { return Err(format!("topic '{topic}' not found")); }
            let idx_str = arg(args, "index");
            if !idx_str.is_empty() {
                let idx: usize = idx_str.parse().map_err(|_| format!("invalid index: '{idx_str}'"))?;
                let e = entries.get(idx).ok_or_else(|| format!("index {idx} out of range"))?;
                return Ok(format!("[{idx}] {}\n{}", crate::time::minutes_to_date_str(e.timestamp_min), e.body));
            }
            let mut out = String::new();
            use std::fmt::Write;
            for (i, e) in entries.iter().enumerate() {
                let _ = writeln!(out, "[{i}] {}", crate::time::minutes_to_date_str(e.timestamp_min));
                for line in e.body.lines() { let _ = writeln!(out, "  {line}"); }
                let _ = writeln!(out);
            }
            let _ = writeln!(out, "{} entries in {topic}", entries.len());
            Ok(out)
        }
        "edit" => dispatch_edit(args, dir),
        "topics" => dispatch_topics(args, dir),
        _ => Err(format!("unknown tool: {name}")),
    }
}

fn dispatch_batch(args: Option<&Value>, dir: &Path) -> Result<String, String> {
    let verbose = abool(args, "verbose");
    let items = args.and_then(|a| a.get("entries"))
        .and_then(|v| match v { Value::Arr(a) => Some(a), _ => None })
        .ok_or("entries must be an array")?;
    if items.len() > 30 { return Err(format!("batch too large ({}, max 30)", items.len())); }
    let _lock = crate::lock::FileLock::acquire(dir)?;
    crate::config::ensure_dir(dir)?;
    let log_path = crate::datalog::ensure_log(dir)?;
    let mut f = std::fs::OpenOptions::new().append(true).open(&log_path)
        .map_err(|e| format!("open data.log: {e}"))?;
    let (mut ok, mut out) = (0, Vec::new());
    let mut batch_tokens: Vec<(String, crate::fxhash::FxHashSet<String>)> = Vec::new();
    'batch: for (i, item) in items.iter().enumerate() {
        let topic = item.get("topic").and_then(|v| v.as_str()).unwrap_or("");
        let text = item.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let tags = item.get("tags").and_then(|v| v.as_str());
        let source = item.get("source").and_then(|v| v.as_str());
        if topic.is_empty() || text.is_empty() {
            out.push(format!("  [{}] skipped: missing topic or text", i + 1)); continue;
        }
        let tokens: crate::fxhash::FxHashSet<String> = crate::text::tokenize(text)
            .into_iter().filter(|t| t.len() >= 3).collect();
        if tokens.len() >= 6 {
            for (pt, ptok) in &batch_tokens {
                if *pt != topic { continue; }
                let isect = tokens.iter().filter(|t| ptok.contains(*t)).count();
                let union = tokens.len() + ptok.len() - isect;
                if union > 0 && isect as f64 / union as f64 > 0.70 {
                    out.push(format!("  [{}] skipped: similar to earlier entry", i + 1));
                    continue 'batch;
                }
            }
            batch_tokens.push((topic.to_string(), tokens));
        }
        match crate::write::batch_entry_to(&mut f, topic, text, tags, source) {
            Ok(msg) => {
                ok += 1;
                let first = msg.lines().next().unwrap_or(&msg);
                out.push(format!("  [{}] {}", i + 1, first));
                log_session(format!("[{}] {}", topic, first));
            }
            Err(e) => out.push(format!("  [{}] err: {}", i + 1, e.lines().next().unwrap_or(&e))),
        }
    }
    if ok > 0 { let _ = f.sync_all(); }
    drop(f); drop(_lock);
    if ok > 0 { after_write(dir); }
    if verbose { Ok(format!("batch: {ok}/{} stored\n{}", items.len(), out.join("\n"))) }
    else { Ok(format!("batch: {ok}/{} stored", items.len())) }
}

fn dispatch_search(args: Option<&Value>, dir: &Path) -> Result<String, String> {
    let raw_query = arg(args, "query"); let detail = arg(args, "detail");
    // Parse field prefixes (tag:X, topic:X, source:X) from query string
    let parsed = crate::text::parse_query_filters(raw_query);
    let query = if parsed.query.is_empty() { raw_query } else { &parsed.query };
    let mut filter = build_filter(args);
    // Query prefixes fill in filters not already set by explicit params
    if filter.tag.is_none() { filter.tag = parsed.tag; }
    if filter.topic.is_none() { filter.topic = parsed.topic; }
    if filter.source.is_none() { filter.source = parsed.source; }
    let limit = arg(args, "limit").parse::<usize>().ok();
    match detail {
        "count" => crate::search::count(dir, query, &filter),
        "topics" => crate::search::run_topics(dir, query, &filter),
        "grouped" => {
            let guard = INDEX.read().map_err(|e| e.to_string())?;
            let idx = guard.as_ref().map(|d| d.as_slice());
            let r = crate::search::run_grouped(dir, query, limit, &filter, idx);
            drop(guard); r
        }
        _ => {
            let guard = INDEX.read().map_err(|e| e.to_string())?;
            let idx = guard.as_ref().map(|d| d.as_slice());
            let r = match detail {
                "full" => crate::search::run(dir, query, limit, &filter, idx),
                "brief" => crate::search::run_brief(dir, query, limit, &filter, idx),
                _ => crate::search::run_medium(dir, query, limit, &filter, idx),
            };
            drop(guard); r
        }
    }
}

fn dispatch_edit(args: Option<&Value>, dir: &Path) -> Result<String, String> {
    let action = arg(args, "action"); let topic = arg(args, "topic");
    let result = match action {
        "append" => {
            let text = arg(args, "text");
            let idx = arg(args, "index").parse::<usize>().ok();
            let needle = opt(arg(args, "match_str")); let tag = opt(arg(args, "tag"));
            if idx.is_some() || needle.is_some() || tag.is_some() {
                crate::write::append_to(dir, topic, text, idx, needle, tag)?
            } else { crate::write::append(dir, topic, text)? }
        }
        "revise" => {
            let text = arg(args, "text");
            let idx = arg(args, "index").parse::<usize>().ok();
            crate::write::revise(dir, topic, text, idx, opt(arg(args, "match_str")))?
        }
        "delete" => {
            let all = abool(args, "all");
            let idx = arg(args, "index").parse::<usize>().ok();
            crate::write::delete(dir, topic, all, idx, opt(arg(args, "match_str")))?
        }
        "tag" => {
            let idx = arg(args, "index").parse::<usize>().ok();
            crate::write::tag(dir, topic, opt(arg(args, "tags")), opt(arg(args, "remove")),
                idx, opt(arg(args, "match_str")))?
        }
        "rename" => {
            let new_name = arg(args, "new_name");
            let _lock = crate::lock::FileLock::acquire(dir)?;
            let log_path = crate::config::log_path(dir);
            let entries = crate::write::topic_entries(&log_path, topic)?;
            if entries.is_empty() { return Err(format!("{topic} not found")); }
            let sanitized = crate::config::sanitize_topic(new_name);
            for e in &entries {
                crate::datalog::append_entry(&log_path, &sanitized, &e.body, e.timestamp_min)?;
                crate::datalog::append_delete(&log_path, e.offset)?;
            }
            format!("renamed {topic} → {sanitized} ({} entries)", entries.len())
        }
        "merge" => {
            let into = arg(args, "into");
            let _lock = crate::lock::FileLock::acquire(dir)?;
            let log_path = crate::config::log_path(dir);
            let entries = crate::write::topic_entries(&log_path, topic)?;
            if entries.is_empty() { return Err(format!("{topic} not found")); }
            let sanitized = crate::config::sanitize_topic(into);
            for e in &entries {
                crate::datalog::append_entry(&log_path, &sanitized, &e.body, e.timestamp_min)?;
                crate::datalog::append_delete(&log_path, e.offset)?;
            }
            format!("merged {} entries from {topic} into {sanitized}", entries.len())
        }
        _ => return Err(format!("edit action: append|revise|delete|tag|rename|merge (got '{action}')")),
    };
    after_write(dir);
    Ok(result)
}

fn dispatch_topics(args: Option<&Value>, dir: &Path) -> Result<String, String> {
    use std::fmt::Write;
    let action = arg(args, "action");
    match action {
        "recent" => {
            let days = arg(args, "days").parse::<u64>().unwrap_or(7);
            let hours = arg(args, "hours").parse::<u64>().ok();
            crate::cache::with_corpus(dir, |cached| {
                let now = crate::time::LocalTime::now().to_days();
                let cutoff = if let Some(h) = hours { now - (h as i64 / 24).max(1) } else { now - days as i64 };
                let mut entries: Vec<_> = cached.iter().filter(|e| e.day() >= cutoff).collect();
                entries.sort_by(|a, b| b.timestamp_min.cmp(&a.timestamp_min));
                let mut out = String::new();
                for e in &entries {
                    let _ = writeln!(out, "  [{}] {} {}", e.topic,
                        crate::time::minutes_to_date_str(e.timestamp_min), crate::text::truncate(log_preview(&e.body), 80));
                }
                if entries.is_empty() { format!("no entries in last {days} days\n") }
                else { let _ = writeln!(out, "{} entries", entries.len()); out }
            })
        }
        "entries" => {
            let topic = arg(args, "topic");
            let log_path = crate::config::log_path(dir);
            let all = crate::write::topic_entries(&log_path, topic)?;
            if all.is_empty() { return Err(format!("topic '{topic}' not found")); }
            let idx_str = arg(args, "index");
            if !idx_str.is_empty() {
                let idx: usize = idx_str.parse().map_err(|_| format!("invalid index: '{idx_str}'"))?;
                let e = all.get(idx).ok_or_else(|| format!("index {idx} out of range"))?;
                return Ok(format!("[{idx}] {}\n{}", crate::time::minutes_to_date_str(e.timestamp_min), e.body));
            }
            let needle = opt(arg(args, "match_str"));
            let mut out = String::new();
            for (i, e) in all.iter().enumerate() {
                if let Some(n) = needle { if !e.body.to_lowercase().contains(&n.to_lowercase()) { continue; } }
                let _ = writeln!(out, "  [{i}] {} {}", crate::time::minutes_to_date_str(e.timestamp_min),
                    crate::text::truncate(log_preview(&e.body), 80));
            }
            let _ = writeln!(out, "{} entries in {topic}", all.len());
            Ok(out)
        }
        "stats" => {
            let detail = arg(args, "detail");
            match detail {
                "index" => {
                    let guard = INDEX.read().map_err(|e| e.to_string())?;
                    match guard.as_ref() { Some(d) => crate::index::index_info(d), None => Ok("no index loaded".into()) }
                }
                "tags" => crate::cache::with_corpus(dir, |cached| {
                    let mut tc: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
                    for e in cached { for t in e.tags() { *tc.entry(t.clone()).or_default() += 1; } }
                    let mut sorted: Vec<_> = tc.iter().collect();
                    sorted.sort_by(|a, b| b.1.cmp(a.1));
                    let mut out = String::new();
                    for (tag, count) in &sorted { let _ = writeln!(out, "  #{tag}: {count}"); }
                    if sorted.is_empty() { "no tags\n".into() }
                    else { let _ = writeln!(out, "{} unique tags", sorted.len()); out }
                }),
                _ => crate::cache::with_corpus(dir, |cached| {
                    let mut topics: crate::fxhash::FxHashSet<&str> = crate::fxhash::FxHashSet::default();
                    let (mut oldest, mut newest) = (i32::MAX, i32::MIN);
                    for e in cached {
                        topics.insert(&e.topic);
                        if e.timestamp_min < oldest { oldest = e.timestamp_min; }
                        if e.timestamp_min > newest { newest = e.timestamp_min; }
                    }
                    let mut out = format!("{} topics, {} entries\n", topics.len(), cached.len());
                    if !cached.is_empty() {
                        let _ = writeln!(out, "  oldest: {}", crate::time::minutes_to_date_str(oldest));
                        let _ = writeln!(out, "  newest: {}", crate::time::minutes_to_date_str(newest));
                    }
                    out
                }),
            }
        }
        "xref" => {
            let topic = arg(args, "topic");
            let guard = INDEX.read().map_err(|e| e.to_string())?;
            let data = guard.as_ref().ok_or("no index loaded")?;
            let tid = crate::index::resolve_topic(data, topic).ok_or_else(|| format!("'{topic}' not in index"))?;
            let edges = crate::index::xref_edges(data)?;
            let mut out = String::new(); let mut found = 0;
            for (src, dst, count) in &edges {
                if *dst == tid {
                    let _ = writeln!(out, "  {} → {topic} ({count})", crate::index::topic_name(data, *src).unwrap_or_default());
                    found += 1;
                }
                if *src == tid {
                    let _ = writeln!(out, "  {topic} → {} ({count})", crate::index::topic_name(data, *dst).unwrap_or_default());
                    found += 1;
                }
            }
            if found == 0 { Ok(format!("no cross-references for {topic}\n")) }
            else { let _ = writeln!(out, "{found} cross-references"); Ok(out) }
        }
        "graph" => {
            let focus = opt(arg(args, "focus"));
            let guard = INDEX.read().map_err(|e| e.to_string())?;
            let data = guard.as_ref().ok_or("no index loaded")?;
            let edges = crate::index::xref_edges(data)?;
            let mut out = String::new(); let mut count = 0;
            for (src, dst, n) in &edges {
                let s = crate::index::topic_name(data, *src).unwrap_or_default();
                let d = crate::index::topic_name(data, *dst).unwrap_or_default();
                if let Some(f) = focus { if !s.contains(f) && !d.contains(f) { continue; } }
                let _ = writeln!(out, "  {s} → {d} ({n})"); count += 1;
            }
            if count == 0 { Ok("no edges\n".into()) } else { let _ = writeln!(out, "{count} edges"); Ok(out) }
        }
        "stale" => crate::cache::with_corpus(dir, |cached| {
            let mut out = String::new(); let mut found = 0;
            for e in cached {
                if let Some(src) = e.source() {
                    let path = src.split(':').next().unwrap_or(src);
                    if let Some(resolved) = crate::config::resolve_source(path) {
                        let entry_t = std::time::UNIX_EPOCH + std::time::Duration::from_secs((e.timestamp_min as u64) * 60);
                        if let Ok(mt) = std::fs::metadata(&resolved).and_then(|m| m.modified()) {
                            if mt > entry_t {
                                let _ = writeln!(out, "  [{}] {} → stale ({})", e.topic,
                                    crate::text::truncate(log_preview(&e.body), 60), src);
                                found += 1;
                            }
                        }
                    }
                }
            }
            if found == 0 { "all sourced entries are fresh\n".into() }
            else { let _ = writeln!(out, "{found} stale entries"); out }
        }),
        "prune" => {
            let days = arg(args, "days").parse::<u64>().unwrap_or(30);
            crate::cache::with_corpus(dir, |cached| {
                let now = crate::time::LocalTime::now().to_days();
                let mut latest: std::collections::BTreeMap<&str, i64> = std::collections::BTreeMap::new();
                for e in cached { let v = latest.entry(&e.topic).or_insert(i64::MIN); if e.day() > *v { *v = e.day(); } }
                let mut stale: Vec<_> = latest.into_iter().filter(|(_, d)| now - d > days as i64).collect();
                stale.sort_by_key(|(_, d)| -d);
                let mut out = String::new();
                for (topic, d) in &stale { let _ = writeln!(out, "  {topic}: {} days since last entry", now - d); }
                if stale.is_empty() { format!("no stale topics (>{days} days)\n") }
                else { let _ = writeln!(out, "{} stale topics (>{days} days)", stale.len()); out }
            })
        }
        "compact" => {
            let result = crate::datalog::compact_log(dir)?;
            after_write(dir); Ok(result)
        }
        "export" => crate::cache::with_corpus(dir, |cached| {
            let mut out = String::from("{\n"); let mut cur = String::new(); let mut first_t = true; let mut in_arr = false;
            for e in cached {
                if e.topic.as_str() != cur {
                    if in_arr { out.push_str("\n  ]"); in_arr = false; }
                    if !first_t { out.push_str(",\n"); }
                    out.push_str("  \""); crate::json::escape_into(&e.topic, &mut out); out.push_str("\": [");
                    cur = e.topic.to_string(); first_t = false; in_arr = true;
                } else { out.push(','); }
                use std::fmt::Write;
                let _ = write!(out, "\n    {{\"ts\":{},\"body\":\"", e.timestamp_min);
                crate::json::escape_into(&e.body, &mut out); out.push_str("\"}");
            }
            if in_arr { out.push_str("\n  ]"); }
            out.push_str("\n}\n"); out
        }),
        "import" => {
            let json = arg(args, "json");
            let data = crate::json::parse(json)?;
            let _lock = crate::lock::FileLock::acquire(dir)?;
            crate::config::ensure_dir(dir)?;
            let log_path = crate::datalog::ensure_log(dir)?;
            let mut f = std::fs::OpenOptions::new().append(true).open(&log_path)
                .map_err(|e| format!("open data.log: {e}"))?;
            let mut count = 0;
            if let Value::Obj(pairs) = data {
                for (topic, entries) in &pairs {
                    if let Value::Arr(items) = entries {
                        for item in items {
                            let body = item.get("body").and_then(|v| v.as_str()).unwrap_or("");
                            let ts = item.get("ts").and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
                            if !body.is_empty() { crate::datalog::append_entry_to(&mut f, topic, body, ts)?; count += 1; }
                        }
                    }
                }
            }
            let _ = f.sync_all(); drop(f); drop(_lock);
            after_write(dir); crate::cache::invalidate();
            Ok(format!("imported {count} entries"))
        }
        "reindex" => {
            let (result, bytes) = crate::index::rebuild(dir, true)?;
            store_index(bytes); Ok(result)
        }
        "session" => {
            let log = SESSION_LOG.lock().map_err(|e| e.to_string())?;
            if log.is_empty() { return Ok("no stores this session".into()); }
            let mut out = format!("{} stores this session:\n", log.len());
            for entry in log.iter() { out.push_str("  "); out.push_str(entry); out.push('\n'); }
            Ok(out)
        }
        _ => {
            // Default: list topics
            crate::cache::with_corpus(dir, |cached| {
                let mut topics: std::collections::BTreeMap<&str, (usize, i32)> = std::collections::BTreeMap::new();
                for e in cached {
                    let (c, latest) = topics.entry(&e.topic).or_insert((0, 0));
                    *c += 1; if e.timestamp_min > *latest { *latest = e.timestamp_min; }
                }
                let mut sorted: Vec<_> = topics.into_iter().collect();
                sorted.sort_by(|a, b| b.1.1.cmp(&a.1.1));
                let mut out = String::new();
                for (topic, (count, latest)) in &sorted {
                    let _ = writeln!(out, "  {topic} ({count}, {})", crate::time::minutes_to_date_str(*latest));
                }
                let _ = writeln!(out, "{} topics, {} entries", sorted.len(), cached.len());
                out
            })
        }
    }
}

// ══════════ Helpers ══════════

fn log_preview(body: &str) -> &str {
    body.lines().find(|l| { let t = l.trim(); !t.is_empty() && !crate::text::is_metadata_line(t) })
        .map(|l| l.trim()).unwrap_or("")
}

fn arg<'a>(args: Option<&'a Value>, key: &str) -> &'a str {
    args.and_then(|a| a.get(key)).and_then(|v| v.as_str()).unwrap_or("")
}
fn opt(s: &str) -> Option<&str> { if s.is_empty() { None } else { Some(s) } }
fn abool(args: Option<&Value>, key: &str) -> bool { let s = arg(args, key); s == "true" || s == "1" }

fn build_filter(args: Option<&Value>) -> crate::index::Filter {
    let after_raw = arg(args, "after"); let before_raw = arg(args, "before");
    let after = if after_raw.is_empty() {
        crate::time::relative_to_date(arg(args, "days").parse().ok(), arg(args, "hours").parse().ok()).unwrap_or_default()
    } else { crate::time::resolve_date_shortcut(after_raw) };
    let before = crate::time::resolve_date_shortcut(before_raw);
    let tag = arg(args, "tag"); let topic = arg(args, "topic");
    crate::index::Filter {
        after: if after.is_empty() { None } else { crate::time::parse_date_days(&after) },
        before: if before.is_empty() { None } else { crate::time::parse_date_days(&before) },
        tag: if tag.is_empty() { None } else { Some(tag.to_string()) },
        topic: if topic.is_empty() { None } else { Some(topic.to_string()) },
        source: None,
        mode: if arg(args, "mode") == "or" { crate::index::SearchMode::Or } else { crate::index::SearchMode::And },
    }
}

// ══════════ Tool Schemas ══════════

static TOOL_CACHE: Mutex<Option<Arc<str>>> = Mutex::new(None);

fn tool_list_json() -> Arc<str> {
    if let Ok(g) = TOOL_CACHE.lock() { if let Some(ref c) = *g { return Arc::clone(c); } }
    let result = Value::Obj(vec![("tools".into(), tool_list())]);
    let json: Arc<str> = result.to_string().into();
    if let Ok(mut g) = TOOL_CACHE.lock() { *g = Some(Arc::clone(&json)); }
    json
}

fn tool(name: &str, desc: &str, req: &[&str], props: &[(&str, &str, &str)]) -> Value {
    Value::Obj(vec![
        ("name".into(), Value::Str(name.into())),
        ("description".into(), Value::Str(desc.into())),
        ("inputSchema".into(), Value::Obj(vec![
            ("type".into(), Value::Str("object".into())),
            ("properties".into(), Value::Obj(props.iter().map(|(n, t, d)|
                ((*n).into(), Value::Obj(vec![
                    ("type".into(), Value::Str((*t).into())),
                    ("description".into(), Value::Str((*d).into())),
                ]))
            ).collect())),
            ("required".into(), Value::Arr(req.iter().map(|r| Value::Str((*r).into())).collect())),
        ])),
    ])
}

fn batch_tool() -> Value {
    let entry_schema = Value::Obj(vec![
        ("type".into(), Value::Str("object".into())),
        ("properties".into(), Value::Obj(vec![
            ("topic".into(), Value::Obj(vec![("type".into(), Value::Str("string".into())), ("description".into(), Value::Str("Topic name".into()))])),
            ("text".into(), Value::Obj(vec![("type".into(), Value::Str("string".into())), ("description".into(), Value::Str("Entry content".into()))])),
            ("tags".into(), Value::Obj(vec![("type".into(), Value::Str("string".into())), ("description".into(), Value::Str("Comma-separated tags".into()))])),
            ("source".into(), Value::Obj(vec![("type".into(), Value::Str("string".into())), ("description".into(), Value::Str("Source file: path/to/file:line".into()))])),
        ])),
        ("required".into(), Value::Arr(vec![Value::Str("topic".into()), Value::Str("text".into())])),
    ]);
    Value::Obj(vec![
        ("name".into(), Value::Str("batch".into())),
        ("description".into(), Value::Str("Store multiple entries in one call. Each entry: {topic, text, tags?}. Faster than sequential store calls.".into())),
        ("inputSchema".into(), Value::Obj(vec![
            ("type".into(), Value::Str("object".into())),
            ("properties".into(), Value::Obj(vec![
                ("entries".into(), Value::Obj(vec![
                    ("type".into(), Value::Str("array".into())),
                    ("items".into(), entry_schema),
                    ("description".into(), Value::Str("Array of entries to store".into())),
                ])),
                ("verbose".into(), Value::Obj(vec![("type".into(), Value::Str("string".into())),
                    ("description".into(), Value::Str("Set to 'true' for per-entry details (default: terse count only)".into()))])),
            ])),
            ("required".into(), Value::Arr(vec![Value::Str("entries".into())])),
        ])),
    ])
}

const FILTER_PROPS: &[(&str, &str, &str)] = &[
    ("limit", "string", "Max results to return (default: unlimited)"),
    ("after", "string", "Only entries on/after date (YYYY-MM-DD or 'today'/'yesterday'/'this-week')"),
    ("before", "string", "Only entries on/before date (YYYY-MM-DD or 'today'/'yesterday')"),
    ("days", "string", "Number of days (shortcut for after=N-days-ago)"),
    ("hours", "string", "Number of hours (overrides days)"),
    ("tag", "string", "Only entries with this tag"),
    ("topic", "string", "Limit search to a single topic"),
    ("mode", "string", "Search mode: 'and' (default, all terms must match) or 'or' (any term matches)"),
];

fn tool_list() -> Value {
    let search_props: Vec<(&str, &str, &str)> = [
        ("query", "string", "Search query"),
        ("detail", "string", "Result detail level: 'full' (complete entry), 'medium' (default, 2 lines), 'brief' (topic+first line), 'count' (match count only), 'topics' (hits per topic), 'grouped' (results by topic)"),
    ].into_iter().chain(FILTER_PROPS.iter().copied()).collect();

    Value::Arr(vec![
        tool("store", "Store a timestamped knowledge entry under a topic. Warns on duplicate content.",
            &["topic", "text"],
            &[("topic", "string", "Topic name"), ("text", "string", "Entry content"),
              ("tags", "string", "Comma-separated tags (e.g. 'bug,p0,iris')"),
              ("force", "string", "Set to 'true' to bypass duplicate detection"),
              ("source", "string", "Source file reference: 'path/to/file:line'. Enables staleness detection."),
              ("terse", "string", "Set to 'true' for minimal response (just first line)"),
              ("confidence", "string", "Confidence level 0.0-1.0 (default: 1.0). Affects search ranking."),
              ("links", "string", "Space-separated references: 'topic:index topic:index'. Creates narrative links.")]),
        batch_tool(),
        tool("search", "Search all knowledge files (case-insensitive). Splits CamelCase/snake_case. Falls back to OR when AND finds nothing. Use detail param: 'full' (complete entry), 'medium' (default, 2 lines), 'brief' (topic+first line), 'count' (match count only), 'topics' (hits per topic).",
            &[], &search_props),
        tool("brief", "One-shot compressed briefing for a topic or pattern. Primary way to load a mental model. Default output is a ~15-line summary; use detail='scan' for category one-liners, detail='full' for complete entries. Use since=N for entries from last N hours only. Supports glob patterns like 'iris-*' for multi-topic views. Without query: session start briefing (activity-weighted topics + velocity).",
            &[],
            &[("query", "string", "Topic, keyword, or glob pattern (e.g. 'iris-*', 'engine', 'amaranthine-codebase')"),
              ("detail", "string", "Output tier: 'summary' (default, ~15 lines), 'scan' (category one-liners), 'full' (complete entries)"),
              ("since", "string", "Only entries from last N hours (e.g. '24' for last day, '48' for 2 days)"),
              ("focus", "string", "Comma-separated category names to show (e.g. 'gotchas,invariants'). Only matching categories appear in output."),
              ("compact", "string", "Set to 'true' for compact meta-briefing (top 5 topics only)")]),
        tool("read", "Read the full contents of a specific topic file.", &["topic"],
            &[("topic", "string", "Topic name"), ("index", "string", "Fetch a single entry by index (0-based)")]),
        tool("edit", "Modify entries. action: append (add text to entry), revise (overwrite entry text), delete (remove entries), tag (add/remove tags), rename (rename topic), merge (merge topics).",
            &["action", "topic"],
            &[("action", "string", "Operation: append|revise|delete|tag|rename|merge"),
              ("topic", "string", "Topic name (or source topic for rename/merge)"),
              ("text", "string", "Text content (for append/revise)"),
              ("index", "string", "Entry index number"), ("match_str", "string", "Substring to find entry"),
              ("tag", "string", "Append to most recent entry with this tag (append only)"),
              ("tags", "string", "Comma-separated tags to add (tag action)"),
              ("remove", "string", "Comma-separated tags to remove (tag action)"),
              ("all", "string", "Set to 'true' to delete entire topic (delete action)"),
              ("new_name", "string", "New topic name (rename action)"),
              ("into", "string", "Target topic to merge INTO (merge action)")]),
        tool("topics", "Browse & maintain knowledge base. Default: list all topics. Use action param for other operations.",
            &[],
            &[("action", "string", "Operation: list(default)|recent|entries|stats|xref|graph|stale|prune|compact|export|import|reindex|session"),
              ("topic", "string", "Topic name (for entries/xref)"), ("days", "string", "Number of days (default: 7 for recent, 30 for prune)"),
              ("hours", "string", "Number of hours (overrides days for recent)"),
              ("detail", "string", "Output: default|'tags'|'index' (for stats action)"),
              ("index", "string", "Entry index (for entries action)"), ("match_str", "string", "Filter entries matching substring"),
              ("focus", "string", "Glob pattern to filter topics (graph action)"),
              ("json", "string", "JSON string to import (import action)"),
              ("refresh", "string", "Set to 'true' to show stale entries + current source (stale action)")]),
        tool("_reload", "Re-exec the server binary to pick up code changes.", &[], &[]),
    ])
}

/// Copy release binary over installed binary, codesign, re-exec.
fn do_reload() {
    use std::os::unix::process::CommandExt;
    let exe = match std::env::current_exe() { Ok(p) => p, Err(_) => return };
    let home = std::env::var("HOME").unwrap_or_default();
    let src = std::path::PathBuf::from(&home).join("wudan/dojo/crash3/llm_double_helix/target/release/helix");
    if src.exists() {
        let tmp = exe.with_extension("tmp");
        if std::fs::copy(&src, &tmp).is_ok() {
            if std::fs::rename(&tmp, &exe).is_ok() {
                let _ = std::process::Command::new("codesign")
                    .args(["-s", "-", "-f"]).arg(&exe).output();
            } else { let _ = std::fs::remove_file(&tmp); }
        }
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let _err = std::process::Command::new(&exe).args(&args).exec();
    eprintln!("helix reload failed: {_err}");
}

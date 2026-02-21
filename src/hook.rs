//! Claude Code hook handlers — 9 hooks for knowledge-augmented development.
//!
//! Hook types (by event):
//!   session       SessionStart         — meta-briefing on session start/resume/compact
//!   prompt        UserPromptSubmit     — automatic KB search on every user question
//!   ambient       PreToolUse           — 5-layer file-aware context injection
//!   post-build    PostToolUse(Bash)    — build result storage prompt
//!   error-context PostToolUseFailure   — search KB for error patterns on failures
//!   pre-compact   PreCompact           — re-inject KB summary before context loss
//!   stop          Stop                 — debounced findings reminder
//!   subagent      SubagentStart        — inject topic list into subagent context
//!   approve-mcp   PermissionRequest    — auto-approve helix MCP tool calls
//!
//! Performance: all hooks use mmap(2) for zero-copy index access, direct string
//! formatting (no Value tree allocations), and fast JSON extraction without parsing.

use std::io::Read;
use std::path::Path;

// ══════════ Dispatch ══════════

pub fn run(hook_type: &str, dir: &Path) -> Result<String, String> {
    if hook_type == "approve-mcp" { return Ok(APPROVE_MCP_RESPONSE.into()); }
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).ok();
    let input = input.trim();
    match hook_type {
        "session" => session(input, dir),
        "prompt" => prompt_submit(input, dir),
        "ambient" => ambient(input, dir),
        "post-build" => post_build(input),
        "error-context" => error_context(input, dir),
        "pre-compact" => pre_compact(dir),
        "stop" => stop(input),
        "subagent" => subagent_start(dir),
        _ => Err(format!("unknown hook type: {hook_type}")),
    }
}

// ══════════ Infrastructure ══════════

/// Memory-map index.bin for zero-copy queries. Direct mmap(2) syscall.
/// Mapping lives until process exit (no munmap needed for short-lived hook processes).
fn mmap_index(dir: &Path) -> Option<&'static [u8]> {
    let f = std::fs::File::open(dir.join("index.bin")).ok()?;
    let len = f.metadata().ok()?.len() as usize;
    if len < std::mem::size_of::<crate::format::Header>() { return None; }
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    }
    let ptr = unsafe { mmap(std::ptr::null_mut(), len, 1 /* PROT_READ */, 2 /* MAP_PRIVATE */, f.as_raw_fd(), 0) };
    drop(f); // close fd — mapping persists
    if ptr.is_null() || ptr as usize == usize::MAX { return None; }
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Build hook JSON output — zero Value allocations, direct string formatting.
pub fn hook_output(context: &str) -> String {
    let mut out = String::with_capacity(64 + context.len());
    out.push_str(r#"{"hookSpecificOutput":{"additionalContext":""#);
    crate::json::escape_into(context, &mut out);
    out.push_str(r#""}}"#);
    out
}

/// Index search: AND mode (all terms required).
fn idx_search(data: &[u8], query: &str, limit: usize) -> Vec<crate::index::SearchHit> {
    crate::index::search_index(data, query, &crate::index::FilterPred::none(), limit, true)
        .unwrap_or_default()
}

/// Index search: OR mode (any term matches).
fn idx_search_or(data: &[u8], query: &str, limit: usize) -> Vec<crate::index::SearchHit> {
    crate::index::search_index(data, query, &crate::index::FilterPred::none(), limit, false)
        .unwrap_or_default()
}

/// Find entry IDs with [source:] metadata matching filename.
fn source_entries_for_file(data: &[u8], filename: &str) -> Vec<u32> {
    crate::index::sourced_entries(data).unwrap_or_default()
        .into_iter().filter(|(_, _, path, _)| path.contains(filename))
        .map(|(eid, _, _, _)| eid).collect()
}

// ══════════ Hook: SessionStart ══════════

/// Inject KB meta-briefing on session start/resume/compact/clear.
/// Different behavior based on source: startup gets full briefing,
/// compact gets a reminder that KB is available (context was just lost).
fn session(input: &str, dir: &Path) -> Result<String, String> {
    let source = extract_json_str(input, "source").unwrap_or("startup");
    let data = match mmap_index(dir) {
        Some(d) => d,
        None => return Ok(hook_output(
            "HELIX KNOWLEDGE STORE: Available via mcp__helix__* tools. Search before starting work.")),
    };
    let topics = crate::index::topic_table(data).unwrap_or_default();
    if topics.is_empty() {
        return Ok(hook_output("HELIX KNOWLEDGE STORE: Empty. Store findings as you work."));
    }
    let total_entries: u16 = topics.iter().map(|(_, _, c)| c).sum();
    let mut msg = String::with_capacity(512);

    match source {
        "compact" => msg.push_str("HELIX KB (post-compaction): "),
        "resume" => msg.push_str("HELIX KB (session resumed): "),
        _ => msg.push_str("HELIX KNOWLEDGE STORE: "),
    }
    push_u32(&mut msg, total_entries as u32);
    msg.push_str(" entries across ");
    push_u32(&mut msg, topics.len() as u32);
    msg.push_str(" topics.\nTopics: ");
    let mut sorted = topics;
    sorted.sort_by(|a, b| b.2.cmp(&a.2));
    for (i, (_, name, count)) in sorted.iter().take(15).enumerate() {
        if i > 0 { msg.push_str(", "); }
        msg.push_str(name);
        msg.push_str(" ("); push_u32(&mut msg, *count as u32); msg.push(')');
    }
    msg.push_str("\nBEFORE starting work, call mcp__helix__search with keywords relevant to your task.");
    Ok(hook_output(&msg))
}

// ══════════ Hook: UserPromptSubmit ══════════

/// Automatic KB search triggered by user questions. Extracts keywords from the
/// prompt, searches the index, and injects matching entries as context.
/// Conservative: skips short prompts (<10 chars), long pastes (>500 chars),
/// and prompts with too few search terms (<2).
fn prompt_submit(input: &str, dir: &Path) -> Result<String, String> {
    if input.is_empty() { return Ok(String::new()); }
    let prompt_text = extract_json_str(input, "prompt").unwrap_or("");
    if prompt_text.len() < 10 || prompt_text.len() > 500 { return Ok(String::new()); }

    let data = match mmap_index(dir) { Some(d) => d, None => return Ok(String::new()) };
    let terms = crate::text::query_terms(prompt_text);
    if terms.len() < 2 { return Ok(String::new()); }

    let query = terms.iter().take(6).cloned().collect::<Vec<_>>().join(" ");
    let hits = idx_search_or(data, &query, 3);
    if hits.is_empty() { return Ok(String::new()); }

    let mut out = String::with_capacity(256);
    out.push_str("helix knowledge (relevant to your question):\n");
    for h in &hits {
        out.push_str("  ");
        out.push_str(&h.snippet);
        out.push('\n');
    }
    Ok(hook_output(&out))
}

// ══════════ Hook: PreToolUse (Ambient) ══════════

/// 5-layer smart ambient context injection on file access.
/// Fast-path byte scanning for tool_name/file_path (no full JSON parse).
fn ambient(input: &str, dir: &Path) -> Result<String, String> {
    if input.is_empty() { return Ok(String::new()); }
    let tool = extract_json_str(input, "tool_name").unwrap_or("");
    let is_edit = tool == "Edit";
    match tool {
        "Read" | "Edit" | "Write" | "Glob" | "Grep" | "NotebookEdit" => {}
        _ => return Ok(String::new()),
    }
    let file_path = extract_json_str(input, "file_path")
        .or_else(|| extract_json_str(input, "\"path\""))
        .unwrap_or("");
    if file_path.is_empty() { return Ok(String::new()); }
    let stem = std::path::Path::new(file_path)
        .file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if stem.len() < 3 { return Ok(String::new()); }

    let syms = if is_edit {
        match crate::json::parse(input) {
            Ok(val) => extract_removed_syms(&val, stem),
            Err(_) => vec![],
        }
    } else { vec![] };

    let data = match mmap_index(dir) { Some(d) => d, None => return Ok(String::new()) };
    let sym_refs: Vec<&str> = syms.iter().map(|s| s.as_str()).collect();
    let out = query_ambient(data, stem, file_path, &sym_refs);
    if out.is_empty() { return Ok(String::new()); }
    Ok(hook_output(&out))
}

// ══════════ Hook: PostToolUse (Build) ══════════

/// After build commands, remind to store results. Async — non-blocking.
fn post_build(input: &str) -> Result<String, String> {
    let is_build = (input.contains("xcodebuild") && input.contains("build"))
        || input.contains("cargo build") || input.contains("swift build")
        || input.contains("swiftc ");
    if !is_build { return Ok(String::new()); }
    Ok(POST_BUILD_RESPONSE.into())
}

// ══════════ Hook: PostToolUseFailure (Error Context) ══════════

/// When a tool fails, search KB for similar error patterns.
fn error_context(input: &str, dir: &Path) -> Result<String, String> {
    if input.is_empty() { return Ok(String::new()); }
    let error = extract_json_str(input, "error").unwrap_or("");
    if error.len() < 15 { return Ok(String::new()); }

    let data = match mmap_index(dir) { Some(d) => d, None => return Ok(String::new()) };
    let terms = crate::text::query_terms(error);
    if terms.len() < 2 { return Ok(String::new()); }

    let query = terms.iter().take(8).cloned().collect::<Vec<_>>().join(" ");
    let hits = idx_search_or(data, &query, 3);
    if hits.is_empty() { return Ok(String::new()); }

    let mut out = String::with_capacity(256);
    out.push_str("helix: relevant knowledge for this error:\n");
    for h in &hits { out.push_str("  "); out.push_str(&h.snippet); out.push('\n'); }
    Ok(hook_output(&out))
}

// ══════════ Hook: PreCompact ══════════

/// Before context compaction, re-inject KB summary so awareness survives.
fn pre_compact(dir: &Path) -> Result<String, String> {
    let data = match mmap_index(dir) { Some(d) => d, None => return Ok(String::new()) };
    let topics = crate::index::topic_table(data).unwrap_or_default();
    if topics.is_empty() { return Ok(String::new()); }

    let mut msg = String::with_capacity(512);
    msg.push_str("CONTEXT PRESERVED — HELIX KB: ");
    push_u32(&mut msg, topics.len() as u32);
    msg.push_str(" topics available. After compaction, search helix for knowledge.\nTopics: ");
    let mut sorted = topics;
    sorted.sort_by(|a, b| b.2.cmp(&a.2));
    for (i, (_, name, count)) in sorted.iter().take(15).enumerate() {
        if i > 0 { msg.push_str(", "); }
        msg.push_str(name); msg.push_str(" ("); push_u32(&mut msg, *count as u32); msg.push(')');
    }
    Ok(hook_output(&msg))
}

// ══════════ Hook: Stop ══════════

/// Debounced (120s) reminder to store findings. Checks stop_hook_active
/// to prevent infinite loops (per Claude Code hook protocol).
fn stop(input: &str) -> Result<String, String> {
    if extract_json_str(input, "stop_hook_active") == Some("true") {
        return Ok(String::new());
    }
    let stamp = "/tmp/helix-hook-stop.last";
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    if let Ok(content) = std::fs::read_to_string(stamp) {
        if let Ok(last) = content.trim().parse::<u64>() {
            if now.saturating_sub(last) < 120 { return Ok(String::new()); }
        }
    }
    std::fs::write(stamp, now.to_string()).ok();
    Ok(hook_output("STOPPING: Store any non-obvious findings in helix before ending."))
}

// ══════════ Hook: SubagentStart ══════════

/// Inject dynamic topic list from index. Prefers mmap over socket.
fn subagent_start(dir: &Path) -> Result<String, String> {
    let fallback = "HELIX KNOWLEDGE STORE: You have access to helix MCP tools. \
         Search before starting work.";
    let topic_list = mmap_index(dir)
        .and_then(|data| {
            let topics = crate::index::topic_table(data).ok()?;
            let mut list: Vec<String> = topics.iter()
                .map(|(_, name, count)| format!("{name} ({count})")).collect();
            list.sort();
            Some(list.join(", "))
        })
        .or_else(|| crate::sock::query(dir, r#"{"op":"topics"}"#));
    let msg = match topic_list {
        Some(list) if !list.is_empty() => format!(
            "HELIX KNOWLEDGE STORE: You have access to helix MCP tools. \
             BEFORE starting work, call mcp__helix__search with keywords \
             relevant to your task. Topics: {list}"),
        _ => fallback.into(),
    };
    Ok(hook_output(&msg))
}

// ══════════ Fast JSON Extraction ══════════

/// Fast JSON string extraction: find "key":"value" without full parse.
/// Stack-allocated needle — zero heap allocation.
pub fn extract_json_str<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let kb = key.as_bytes();
    let quoted = kb.first() == Some(&b'"');
    let mut needle_buf = [0u8; 80];
    let nlen = if quoted {
        if kb.len() + 2 > needle_buf.len() { return None; }
        needle_buf[..kb.len()].copy_from_slice(kb);
        needle_buf[kb.len()] = b':';
        needle_buf[kb.len() + 1] = b'"';
        kb.len() + 2
    } else {
        if kb.len() + 4 > needle_buf.len() { return None; }
        needle_buf[0] = b'"';
        needle_buf[1..1 + kb.len()].copy_from_slice(kb);
        needle_buf[1 + kb.len()] = b'"';
        needle_buf[2 + kb.len()] = b':';
        needle_buf[3 + kb.len()] = b'"';
        kb.len() + 4
    };
    let needle = unsafe { std::str::from_utf8_unchecked(&needle_buf[..nlen]) };
    let pos = json.find(needle)?;
    let rest = &json[pos + nlen..];
    let bytes = rest.as_bytes();
    let mut end = 0;
    while end < bytes.len() {
        if bytes[end] == b'"' && (end == 0 || bytes[end - 1] != b'\\') {
            return Some(&rest[..end]);
        }
        end += 1;
    }
    None
}

// ══════════ Smart Ambient Context (5 layers) ══════════

/// Extract symbols removed by an Edit (refactor impact detection).
pub fn extract_removed_syms(input: &crate::json::Value, stem: &str) -> Vec<String> {
    let ti = input.get("tool_input");
    let old = ti.and_then(|t| t.get("old_string")).and_then(|v| v.as_str()).unwrap_or("");
    let new_str = ti.and_then(|t| t.get("new_string")).and_then(|v| v.as_str()).unwrap_or("");
    if old.len() < 8 { return vec![]; }
    let extract = |s: &str| -> crate::fxhash::FxHashSet<String> {
        s.split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| w.len() >= 4 && w.bytes().any(|b| b.is_ascii_alphabetic()))
            .map(|w| w.to_lowercase()).collect()
    };
    let old_tokens: crate::fxhash::FxHashSet<String> = extract(old)
        .into_iter().filter(|t| t != stem).collect();
    let new_tokens = extract(new_str);
    let mut removed: Vec<String> = old_tokens.into_iter()
        .filter(|t| !new_tokens.contains(t)).collect();
    removed.sort();
    removed.truncate(3);
    removed
}

/// 5-layer smart ambient context with deduplication:
///   1. Source-path matches — entries with [source:] metadata for this file
///   2. Symbol-based OR search — fn/struct/enum names from file (cached)
///   3. Global BM25 — stem keyword search
///   4. Structural coupling — "structural <stem>" query
///   5. Refactor impact — removed symbols (Edit only)
///
/// Adaptive: Layer 2 skipped when Layer 1 returns 5+ hits.
/// Cow<str>: Layer 1 borrows from mmap (zero alloc), other layers own.
pub fn query_ambient(data: &[u8], stem: &str, file_path: &str, syms: &[&str]) -> String {
    let filename = std::path::Path::new(file_path)
        .file_name().and_then(|f| f.to_str()).unwrap_or(stem);
    let mut seen = crate::fxhash::FxHashSet::default();
    let mut pool: Vec<std::borrow::Cow<str>> = Vec::with_capacity(32);

    // Layer 1: Source-path matches
    let l1_start = pool.len();
    let source_ids = source_entries_for_file(data, filename);
    for &eid in &source_ids {
        seen.insert(eid);
        if let Ok(snip) = crate::index::entry_snippet_ref(data, eid) {
            if !snip.is_empty() { pool.push(std::borrow::Cow::Borrowed(snip)); }
        }
    }
    let l1 = pool.len() - l1_start;

    // Layer 2: Symbol-based OR search (skip if L1 ≥ 5)
    let l2_start = pool.len();
    if source_ids.len() < 5 {
        let file_symbols = cached_file_symbols(file_path);
        if !file_symbols.is_empty() {
            let query = build_symbol_query(&file_symbols, stem);
            if !query.is_empty() {
                for h in idx_search_or(data, &query, 8) {
                    if seen.insert(h.entry_id) {
                        pool.push(std::borrow::Cow::Owned(h.snippet));
                        if pool.len() - l2_start >= 5 { break; }
                    }
                }
            }
        }
    }
    let l2 = pool.len() - l2_start;

    // Layer 3: Global BM25 (stem keyword)
    let l3_start = pool.len();
    for h in idx_search(data, stem, 5) {
        if seen.insert(h.entry_id) {
            pool.push(std::borrow::Cow::Owned(h.snippet));
            if pool.len() - l3_start >= 3 { break; }
        }
    }
    let l3 = pool.len() - l3_start;

    // Layer 4: Structural coupling (stack-allocated query when possible)
    let l4_start = pool.len();
    let mut sq_buf = [0u8; 128];
    let sq_prefix = b"structural ";
    let sq_len = sq_prefix.len() + stem.len();
    let structural = if sq_len <= sq_buf.len() {
        sq_buf[..sq_prefix.len()].copy_from_slice(sq_prefix);
        sq_buf[sq_prefix.len()..sq_len].copy_from_slice(stem.as_bytes());
        let sq = unsafe { std::str::from_utf8_unchecked(&sq_buf[..sq_len]) };
        idx_search(data, sq, 3)
    } else {
        idx_search(data, &format!("structural {stem}"), 3)
    };
    for h in structural {
        if seen.insert(h.entry_id) { pool.push(std::borrow::Cow::Owned(h.snippet)); }
    }
    let l4 = pool.len() - l4_start;

    // Layer 5: Refactor impact (Edit only)
    let l5_start = pool.len();
    for sym in syms {
        for hit in idx_search(data, sym, 3) {
            if seen.insert(hit.entry_id) { pool.push(std::borrow::Cow::Owned(hit.snippet)); }
        }
    }
    let l5 = pool.len() - l5_start;

    if pool.is_empty() { return String::new(); }

    // Single output pass
    let est = pool.iter().map(|s| s.len() + 4).sum::<usize>() + 5 * 40;
    let mut out = String::with_capacity(est);
    let counts = [l1, l2, l3, l4, l5];
    let mut pool_idx = 0;
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 { continue; }
        if !out.is_empty() { out.push_str("---\n"); }
        match i {
            0 => { out.push_str("source-linked ("); out.push_str(filename); out.push_str("):\n"); }
            1 => out.push_str("symbol context:\n"),
            2 => { out.push_str("related ("); out.push_str(stem); out.push_str("):\n"); }
            3 => out.push_str("structural coupling:\n"),
            4 => {
                out.push_str("REFACTOR IMPACT (symbols modified: ");
                for (j, sym) in syms.iter().enumerate() {
                    if j > 0 { out.push_str(", "); }
                    out.push_str(sym);
                }
                out.push_str("):\n");
            }
            _ => {}
        }
        for _ in 0..count {
            out.push_str("  "); out.push_str(&pool[pool_idx]); out.push('\n');
            pool_idx += 1;
        }
    }
    out
}

// ══════════ Symbol Extraction (for Layer 2) ══════════

fn extract_file_symbols(path: &str) -> Vec<String> {
    let content = match std::fs::read_to_string(path) { Ok(c) => c, Err(_) => return vec![] };
    static KEYWORDS: &[&str] = &[
        "fn ", "struct ", "enum ", "trait ",
        "func ", "class ", "protocol ", "extension ",
    ];
    let mut symbols = Vec::with_capacity(16);
    for line in content.lines().take(500) {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || trimmed.starts_with("///")
            || trimmed.starts_with('#') || trimmed.starts_with("/*") { continue; }
        for kw in KEYWORDS {
            if let Some(pos) = trimmed.find(kw) {
                let rest = &trimmed[pos + kw.len()..];
                let rest = if *kw == "fn " || *kw == "func " { rest }
                else {
                    rest.trim_start_matches(|c: char| c == '<' || c == '\'')
                        .split(|c: char| c == '>' || c == ' ')
                        .next().unwrap_or(rest)
                };
                let name: String = rest.chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                if name.len() >= 3 && name.as_bytes()[0].is_ascii_alphabetic() {
                    symbols.push(name);
                }
            }
        }
    }
    symbols.sort(); symbols.dedup(); symbols.truncate(20);
    symbols
}

const SYM_CACHE_PATH: &str = "/tmp/helix-sym-cache";

fn cached_file_symbols(path: &str) -> Vec<String> {
    let mtime = match std::fs::metadata(path) {
        Ok(m) => m.modified().ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs()).unwrap_or(0),
        Err(_) => return vec![],
    };
    if let Ok(cache) = std::fs::read_to_string(SYM_CACHE_PATH) {
        let mut lines = cache.lines();
        if let (Some(cp), Some(cm)) = (lines.next(), lines.next()) {
            if cp == path { if let Ok(mt) = cm.parse::<u64>() {
                if mt == mtime { return lines.map(|l| l.to_string()).collect(); }
            }}
        }
    }
    let syms = extract_file_symbols(path);
    let mut buf = String::with_capacity(path.len() + 32 + syms.len() * 20);
    buf.push_str(path); buf.push('\n'); push_u64(&mut buf, mtime);
    for sym in &syms { buf.push('\n'); buf.push_str(sym); }
    std::fs::write(SYM_CACHE_PATH, buf.as_bytes()).ok();
    syms
}

fn build_symbol_query(symbols: &[String], stem: &str) -> String {
    let stem_lower = stem.to_lowercase();
    let mut terms = Vec::with_capacity(symbols.len());
    for sym in symbols {
        for tok in crate::text::tokenize(sym) {
            if tok.len() >= 3 && tok != stem_lower { terms.push(tok); }
        }
    }
    terms.sort(); terms.dedup(); terms.truncate(15);
    terms.join(" ")
}

// ══════════ Hook Management ══════════

/// Generate hooks JSON config for all 9 hooks.
pub fn hooks_config(binary: &str) -> String {
    let mut b = String::with_capacity(binary.len() + 10);
    crate::json::escape_into(binary, &mut b);
    format!(concat!(
        "{{",
        "\"SessionStart\":[{{\"matcher\":\"\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook session\",\"timeout\":5}}]}}],",
        "\"UserPromptSubmit\":[{{\"matcher\":\"\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook prompt\",\"timeout\":5}}]}}],",
        "\"PreToolUse\":[{{\"matcher\":\"Read|Edit|Write|Glob|Grep|NotebookEdit\",",
            "\"hooks\":[{{\"type\":\"command\",\"command\":\"{b} hook ambient\",\"timeout\":5}}]}}],",
        "\"PostToolUse\":[{{\"matcher\":\"Bash\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook post-build\",\"async\":true,\"timeout\":5}}]}}],",
        "\"PostToolUseFailure\":[{{\"matcher\":\"Bash\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook error-context\",\"timeout\":5}}]}}],",
        "\"PreCompact\":[{{\"matcher\":\"\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook pre-compact\",\"timeout\":5}}]}}],",
        "\"Stop\":[{{\"matcher\":\"\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook stop\",\"timeout\":5}}]}}],",
        "\"SubagentStart\":[{{\"matcher\":\"\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook subagent\",\"timeout\":5}}]}}],",
        "\"PermissionRequest\":[{{\"matcher\":\"mcp__helix__.*\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook approve-mcp\"}}]}}]",
        "}}"),
        b = b)
}

/// Install helix hooks into ~/.claude/settings.json.
pub fn install_hooks() -> Result<String, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let settings_dir = format!("{home}/.claude");
    let settings_path = format!("{settings_dir}/settings.json");
    let binary = std::env::current_exe().map_err(|e| format!("exe path: {e}"))?
        .to_string_lossy().to_string();

    std::fs::create_dir_all(&settings_dir).map_err(|e| e.to_string())?;
    let mut settings = if std::path::Path::new(&settings_path).exists() {
        let content = std::fs::read_to_string(&settings_path).map_err(|e| e.to_string())?;
        crate::json::parse(&content)?
    } else {
        crate::json::Value::Obj(vec![])
    };

    let hooks_json = hooks_config(&binary);
    let hooks_value = crate::json::parse(&hooks_json)?;

    match &mut settings {
        crate::json::Value::Obj(entries) => {
            if let Some(pos) = entries.iter().position(|(k, _)| k == "hooks") {
                entries[pos].1 = hooks_value;
            } else {
                entries.push(("hooks".to_string(), hooks_value));
            }
        }
        _ => return Err("settings.json root is not an object".into()),
    }

    std::fs::write(&settings_path, format!("{settings}")).map_err(|e| e.to_string())?;
    Ok(format!("Installed 9 helix hooks to {settings_path}\n\
        Hooks: session, prompt, ambient, post-build, error-context, pre-compact, stop, subagent, approve-mcp\n\
        Restart Claude Code or open /hooks to activate."))
}

/// Remove helix hooks from ~/.claude/settings.json.
pub fn uninstall_hooks() -> Result<String, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let settings_path = format!("{home}/.claude/settings.json");
    if !std::path::Path::new(&settings_path).exists() { return Ok("No settings.json found".into()); }

    let content = std::fs::read_to_string(&settings_path).map_err(|e| e.to_string())?;
    let mut settings = crate::json::parse(&content)?;
    match &mut settings {
        crate::json::Value::Obj(entries) => { entries.retain(|(k, _)| k != "hooks"); }
        _ => {}
    }
    std::fs::write(&settings_path, format!("{settings}")).map_err(|e| e.to_string())?;
    Ok(format!("Removed hooks from {settings_path}"))
}

/// Show current hooks configuration.
pub fn hooks_status() -> Result<String, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let settings_path = format!("{home}/.claude/settings.json");
    if !std::path::Path::new(&settings_path).exists() { return Ok("No settings.json found".into()); }
    let content = std::fs::read_to_string(&settings_path).map_err(|e| e.to_string())?;
    let settings = crate::json::parse(&content)?;
    let hooks = match settings.get("hooks") {
        Some(h) => h, None => return Ok("No hooks configured".into()),
    };
    let events = match hooks {
        crate::json::Value::Obj(pairs) => pairs,
        _ => return Ok("Invalid hooks section".into()),
    };
    let mut out = String::with_capacity(256);
    out.push_str(&events.len().to_string());
    out.push_str(" hook events configured:\n");
    for (event, _) in events { out.push_str("  "); out.push_str(event); out.push('\n'); }
    Ok(out)
}

// ══════════ Helpers ══════════

fn push_u32(buf: &mut String, n: u32) { crate::text::itoa_push(buf, n); }

fn push_u64(buf: &mut String, n: u64) {
    if n == 0 { buf.push('0'); return; }
    let mut digits = [0u8; 20];
    let mut i = 0;
    let mut v = n;
    while v > 0 { digits[i] = b'0' + (v % 10) as u8; v /= 10; i += 1; }
    while i > 0 { i -= 1; buf.push(digits[i] as char); }
}

// ══════════ Constants ══════════

const APPROVE_MCP_RESPONSE: &str =
    r#"{"hookSpecificOutput":{"hookEventName":"PermissionRequest","decision":{"behavior":"allow"}}}"#;

const POST_BUILD_RESPONSE: &str = r#"{"systemMessage":"BUILD COMPLETED. If the build failed with a non-obvious error, store the root cause in helix (topic: build-gotchas). If it succeeded after fixing an issue, store what fixed it."}"#;

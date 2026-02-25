//! Claude Code hook handlers — 9 hooks for knowledge-augmented development.
//!
//! Hook types: session, prompt, ambient, post-build, error-context,
//! pre-compact, stop, subagent, approve-mcp.

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
        "post-build" => crate::hook_build::post_build(input, dir),
        "error-context" => error_context(input, dir),
        "pre-compact" => pre_compact(dir),
        "stop" => stop(input),
        "subagent" => subagent_start(dir),
        _ => Err(format!("unknown hook type: {hook_type}")),
    }
}

pub use crate::hook_mgmt::{install_hooks, uninstall_hooks, hooks_status};

// ══════════ Infrastructure ══════════

pub(crate) fn mmap_index(dir: &Path) -> Option<&'static [u8]> {
    let f = std::fs::File::open(dir.join("index.bin")).ok()?;
    let len = f.metadata().ok()?.len() as usize;
    if len < std::mem::size_of::<crate::format::Header>() { return None; }
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
    }
    let ptr = unsafe { mmap(std::ptr::null_mut(), len, 1, 2, f.as_raw_fd(), 0) };
    drop(f);
    if ptr.is_null() || ptr as usize == usize::MAX { return None; }
    Some(unsafe { std::slice::from_raw_parts(ptr, len) })
}

pub fn hook_output(context: &str) -> String {
    let mut out = String::with_capacity(64 + context.len());
    out.push_str(r#"{"hookSpecificOutput":{"additionalContext":""#);
    crate::json::escape_into(context, &mut out);
    out.push_str(r#""}}"#);
    out
}

pub(crate) fn auto_store(dir: &Path, topic: &str, text: &str, tags: &str) {
    let _ = crate::write::store(dir, topic, text, Some(tags), true, None, Some(0.3), None);
    std::fs::write("/tmp/helix-external-write", b"1").ok();
}

pub(crate) fn idx_search_or(data: &[u8], query: &str, limit: usize) -> Vec<crate::index::SearchHit> {
    crate::index::search_index(data, query, &crate::index::FilterPred::none(), limit, false)
        .unwrap_or_default()
}

pub(crate) fn data_log_mtime(dir: &Path) -> u64 {
    std::fs::metadata(crate::config::log_path(dir)).ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn idx_search(data: &[u8], query: &str, limit: usize) -> Vec<crate::index::SearchHit> {
    crate::index::search_index(data, query, &crate::index::FilterPred::none(), limit, true)
        .unwrap_or_default()
}

pub use crate::json::extract_json_str;

fn push_topic_list(data: &[u8], mut topics: Vec<(u16, String, u16)>, out: &mut String, limit: usize) {
    let rmap: crate::fxhash::FxHashMap<u16, i32> = crate::index::topic_recency(data).into_iter().collect();
    topics.sort_by(|a, b| rmap.get(&b.0).unwrap_or(&0).cmp(rmap.get(&a.0).unwrap_or(&0)));
    for (i, (_, name, count)) in topics.iter().take(limit).enumerate() {
        if i > 0 { out.push_str(", "); }
        out.push_str(name); out.push_str(" ("); crate::text::itoa_push(out, *count as u32); out.push(')');
    }
}

// ══════════ Hook: SessionStart ══════════

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
    crate::text::itoa_push(&mut msg, total_entries as u32);
    msg.push_str(" entries across ");
    crate::text::itoa_push(&mut msg, topics.len() as u32);
    msg.push_str(" topics.\nTopics: ");
    push_topic_list(data, topics, &mut msg, 15);
    msg.push_str("\nBEFORE starting work, call mcp__helix__search with keywords relevant to your task.");
    msg.push_str("\nDISCIPLINE: Externalize reasoning through helix store calls every 2-3 steps.");
    Ok(hook_output(&msg))
}

// ══════════ Hook: UserPromptSubmit ══════════

fn prompt_submit(input: &str, dir: &Path) -> Result<String, String> {
    if input.is_empty() { return Ok(String::new()); }
    let prompt_text = extract_json_str(input, "prompt").unwrap_or("");
    if prompt_text.len() < 10 || prompt_text.len() > 500 { return Ok(String::new()); }
    let lower = prompt_text.as_bytes();
    let has_question = prompt_text.contains('?')
        || (lower.len() > 4 && matches!(&lower[..4], b"how " | b"How " | b"what" | b"What"
            | b"why " | b"Why " | b"wher" | b"Wher"));
    let data = match mmap_index(dir) { Some(d) => d, None => return Ok(String::new()) };
    let terms = crate::text::query_terms(prompt_text);
    if terms.len() < 2 { return Ok(String::new()); }
    if !has_question && terms.len() < 3 { return Ok(String::new()); }
    let query = terms.iter().take(6).cloned().collect::<Vec<_>>().join(" ");
    let hits = idx_search(data, &query, 3);
    let hits = if hits.is_empty() && has_question { idx_search_or(data, &query, 3) } else { hits };
    if hits.is_empty() { return Ok(String::new()); }
    let mut out = String::with_capacity(256);
    out.push_str("helix knowledge (relevant to your question):\n");
    for h in &hits { out.push_str("  "); out.push_str(&h.snippet); out.push('\n'); }
    Ok(hook_output(&out))
}

// ══════════ Hook: PreToolUse (Ambient) ══════════

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
            Ok(val) => crate::ambient::extract_removed_syms(&val, stem),
            Err(_) => vec![],
        }
    } else { vec![] };
    let data = match mmap_index(dir) { Some(d) => d, None => return Ok(String::new()) };
    let sym_refs: Vec<&str> = syms.iter().map(|s| s.as_str()).collect();
    let mut session = crate::session::Session::load_or_new();
    session.record_tool(tool);
    session.tick_tool(data_log_mtime(dir));
    session.track_file(is_edit || tool == "Write" || tool == "NotebookEdit");
    let mut out = crate::ambient::query_ambient(data, stem, file_path, &sym_refs, Some(&mut session));
    if session.tool_calls_since_store >= 10 {
        if !out.is_empty() { out.push_str("---\n"); }
        out.push_str("DISCIPLINE: ");
        crate::text::itoa_push(&mut out, session.tool_calls_since_store);
        out.push_str(" tool calls since last helix store. Externalize your reasoning \
            — store intermediate findings, not just conclusions.");
    }
    session.save();
    if out.is_empty() { return Ok(String::new()); }
    Ok(hook_output(&out))
}

// ══════════ Hook: PostToolUseFailure (Error Context) ══════════

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

fn pre_compact(dir: &Path) -> Result<String, String> {
    let data = match mmap_index(dir) { Some(d) => d, None => return Ok(String::new()) };
    let topics = crate::index::topic_table(data).unwrap_or_default();
    if topics.is_empty() { return Ok(String::new()); }
    let mut msg = String::with_capacity(512);
    msg.push_str("CONTEXT PRESERVED — HELIX KB: ");
    crate::text::itoa_push(&mut msg, topics.len() as u32);
    msg.push_str(" topics available. After compaction, search helix for knowledge.\nTopics: ");
    push_topic_list(data, topics, &mut msg, 15);
    Ok(hook_output(&msg))
}

// ══════════ Hook: Stop ══════════

fn stop(input: &str) -> Result<String, String> {
    if extract_json_str(input, "stop_hook_active") == Some("true") { return Ok(String::new()); }
    let stamp = "/tmp/helix-hook-stop.last";
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    if let Ok(content) = std::fs::read_to_string(stamp) {
        if let Ok(last) = content.trim().parse::<u64>() {
            if now.saturating_sub(last) < 120 { return Ok(String::new()); }
        }
    }
    std::fs::write(stamp, now.to_string()).ok();
    let dir = crate::config::resolve_dir(None);
    let recently_stored = std::fs::metadata(crate::config::log_path(&dir)).ok()
        .and_then(|m| m.modified().ok())
        .and_then(|mt| mt.elapsed().ok())
        .map(|e| e.as_secs() < 1800).unwrap_or(false);
    if recently_stored {
        Ok(hook_output("STOPPING: You've been storing findings. Anything else to capture?"))
    } else {
        Ok(hook_output("STOPPING: No findings stored recently. Capture non-obvious learnings before ending."))
    }
}

// ══════════ Hook: SubagentStart ══════════

fn subagent_start(dir: &Path) -> Result<String, String> {
    let fallback = "HELIX KNOWLEDGE STORE: You have access to helix MCP tools. \
         Search before starting work.";
    let topic_list = mmap_index(dir)
        .and_then(|data| {
            let topics = crate::index::topic_table(data).ok()?;
            let mut sorted: Vec<_> = topics.iter().collect();
            sorted.sort_by(|a, b| a.1.cmp(&b.1));
            let mut list = String::with_capacity(sorted.len() * 24);
            for (i, (_, name, count)) in sorted.iter().enumerate() {
                if i > 0 { list.push_str(", "); }
                list.push_str(name); list.push_str(" (");
                crate::text::itoa_push(&mut list, *count as u32); list.push(')');
            }
            Some(list)
        })
        .or_else(|| crate::sock::query(dir, r#"{"op":"topics"}"#));
    let session = crate::session::Session::load();
    let injected_count = session.as_ref().map(|s| s.injected.len()).unwrap_or(0);
    let focus = session.as_ref().map(|s| &s.focus_topics);
    let msg = match topic_list {
        Some(list) if !list.is_empty() => {
            let mut m = String::with_capacity(256 + list.len());
            m.push_str("HELIX KNOWLEDGE STORE: You have access to helix MCP tools. ");
            if injected_count > 10 { m.push_str("Parent session already has context loaded. "); }
            if let Some(ft) = focus {
                if !ft.is_empty() {
                    m.push_str("Focus topics: ");
                    for (i, t) in ft.iter().take(5).enumerate() {
                        if i > 0 { m.push_str(", "); }
                        m.push_str(t);
                    }
                    m.push_str(". ");
                }
            }
            m.push_str("BEFORE starting work, call mcp__helix__search with keywords \
                         relevant to your task. Topics: ");
            m.push_str(&list);
            m
        }
        _ => fallback.into(),
    };
    Ok(hook_output(&msg))
}

// ══════════ Constants ══════════

const APPROVE_MCP_RESPONSE: &str =
    r#"{"hookSpecificOutput":{"hookEventName":"PermissionRequest","decision":{"behavior":"allow"}}}"#;

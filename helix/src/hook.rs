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
        "post-build" => post_build(input, dir),
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

/// Auto-store low-confidence entry from hook. Errors silently ignored.
fn auto_store(dir: &Path, topic: &str, text: &str, tags: &str) {
    let _ = crate::write::store(dir, topic, text, Some(tags), true, None, Some(0.3), None);
    std::fs::write("/tmp/helix-external-write", b"1").ok();
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
    // Sort by recency (most recently updated first), not entry count
    let recency = crate::index::topic_recency(data);
    sorted.sort_by(|a, b| {
        let ra = recency.iter().find(|&&(id, _)| id == a.0).map(|&(_, d)| d).unwrap_or(0);
        let rb = recency.iter().find(|&&(id, _)| id == b.0).map(|&(_, d)| d).unwrap_or(0);
        rb.cmp(&ra)
    });
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

    // Signal gate: detect knowledge questions vs commands/greetings.
    // Only match question words at the START of the prompt or with '?'.
    let lower = prompt_text.as_bytes();
    let has_question = prompt_text.contains('?')
        || (lower.len() > 4 && matches!(&lower[..4], b"how " | b"How " | b"what" | b"What"
            | b"why " | b"Why " | b"wher" | b"Wher"));
    let data = match mmap_index(dir) { Some(d) => d, None => return Ok(String::new()) };
    let terms = crate::text::query_terms(prompt_text);
    if terms.len() < 2 { return Ok(String::new()); }
    // Non-questions need 3+ substantive terms to be worth searching
    if !has_question && terms.len() < 3 { return Ok(String::new()); }

    let query = terms.iter().take(6).cloned().collect::<Vec<_>>().join(" ");
    // Use AND mode first (all terms must match) — much more selective than OR.
    // Only fall back to OR for questions with 2 terms (where AND is too strict).
    let hits = idx_search(data, &query, 3);
    let hits = if hits.is_empty() && has_question {
        idx_search_or(data, &query, 3)
    } else { hits };
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

    // Session accumulator: load, track, pass to query_ambient, save
    let mut session = crate::session::Session::load_or_new();
    session.record_tool(tool);
    session.track_file(is_edit || tool == "Write" || tool == "NotebookEdit");
    let out = query_ambient(data, stem, file_path, &sym_refs, Some(&mut session));
    session.save();
    if out.is_empty() { return Ok(String::new()); }
    Ok(hook_output(&out))
}

// ══════════ Hook: PostToolUse (Build) ══════════

/// After Bash commands: auto-capture build errors, benchmark timings, perf data.
fn post_build(input: &str, dir: &Path) -> Result<String, String> {
    let is_build = (input.contains("xcodebuild") && input.contains("build"))
        || input.contains("cargo build") || input.contains("swift build")
        || input.contains("swiftc ");
    // Also check for benchmark/timing output even on non-build commands
    let response = extract_json_str(input, "tool_response").unwrap_or(input);
    capture_benchmark(response, dir);
    if !is_build { return Ok(String::new()); }

    let has_error = input.contains("error:") || input.contains("error[E")
        || input.contains("BUILD FAILED") || input.contains("** FAILED **");

    if has_error {
        // Extract tool_response to avoid JSON wrapper noise in error lines
        let response = extract_json_str(input, "tool_response").unwrap_or(input);
        let mut errors: Vec<String> = Vec::with_capacity(5);
        // Split on JSON-escaped newlines (\n = 2 chars) and real newlines
        for part in response.split("\\n").flat_map(|s| s.split('\n')) {
            if errors.len() >= 5 { break; }
            let t = part.trim();
            if (t.contains(": error:") || t.contains("error[E"))
                && !errors.iter().any(|e| e == t)
            {
                let line = crate::text::truncate(t, 200);
                errors.push(line.to_string());
            }
        }
        if errors.is_empty() {
            for part in response.split("\\n").flat_map(|s| s.split('\n')) {
                if errors.len() >= 3 { break; }
                let t = part.trim();
                if t.contains("BUILD FAILED") || t.contains("** FAILED **") || t.contains("aborting") {
                    errors.push(crate::text::truncate(t, 200).to_string());
                }
            }
        }

        // Auto-store errors at low confidence
        if !errors.is_empty() {
            let text = errors.join("\n");
            auto_store(dir, "build-errors", &text, "auto, build-error");
            std::fs::write("/tmp/helix-build-errors", text.as_bytes()).ok();
        }

        // Search KB for matching error patterns
        let mut kb_out = String::new();
        if let Some(data) = mmap_index(dir) {
            let mut total = 0;
            for err in &errors {
                if total >= 3 { break; }
                let terms = crate::text::query_terms(err);
                if terms.len() < 2 { continue; }
                let q = terms.iter().take(6).cloned().collect::<Vec<_>>().join(" ");
                for h in idx_search_or(data, &q, 2) {
                    if total >= 3 { break; }
                    kb_out.push_str("  ");
                    kb_out.push_str(&h.snippet);
                    kb_out.push('\n');
                    total += 1;
                }
            }
        }

        // Build output for Claude's context
        let mut ctx = String::with_capacity(256);
        ctx.push_str("BUILD FAILED (");
        crate::text::itoa_push(&mut ctx, errors.len() as u32);
        ctx.push_str(" errors):\n");
        for e in &errors { ctx.push_str("  "); ctx.push_str(e); ctx.push('\n'); }
        if !kb_out.is_empty() { ctx.push_str("helix matches:\n"); ctx.push_str(&kb_out); }
        ctx.push_str("Store non-obvious root causes: mcp__helix__store(topic:'build-gotchas', ...)");
        // Record failed build in session
        let mut session = crate::session::Session::load_or_new();
        session.record_build(false);
        session.save();
        Ok(hook_output(&ctx))
    } else {
        // Build succeeded — check for fix-pair resolution
        let marker = std::path::Path::new("/tmp/helix-build-errors");
        if marker.exists() {
            if let Ok(prior) = std::fs::read_to_string(marker) {
                let first = prior.lines().next().unwrap_or("unknown error");
                let mut text = String::with_capacity(12 + first.len().min(120));
                text.push_str("RESOLVED: ");
                text.push_str(crate::text::truncate(first, 120));
                auto_store(dir, "build-errors", &text, "auto, build-fix");
            }
            std::fs::remove_file(marker).ok();
        }
        // Record successful build in session
        let mut session = crate::session::Session::load_or_new();
        session.record_build(true);
        session.save();
        Ok(String::new())
    }
}

// ══════════ Benchmark Auto-Capture ══════════

/// Detect and store benchmark/timing data from Bash output.
/// Looks for labeled timing patterns like "name: 123.45ms" or "name: 1.23µs"
/// and throughput patterns like "123.4 tok/s" or "350 GB/s".
fn capture_benchmark(response: &str, dir: &Path) {
    let mut timings: Vec<String> = Vec::new();
    // Scan for timing patterns in each line (split on real newlines and JSON-escaped newlines)
    for part in response.split("\\n").flat_map(|s| s.split('\n')) {
        let t = part.trim();
        if t.is_empty() || t.len() < 5 { continue; }
        // Skip build output lines, errors, and pure log lines
        if t.contains(": error:") || t.contains("warning:") || t.starts_with("Compiling")
            || t.starts_with("Linking") || t.starts_with("Finished") { continue; }
        // Match timing patterns: digits followed by ms/µs/us/ns
        let has_timing = has_timing_pattern(t);
        // Match throughput patterns: digits followed by tok/s, GB/s, MB/s
        let has_throughput = t.contains("tok/s") || t.contains("GB/s") || t.contains("MB/s")
            || t.contains("tokens/s") || t.contains("it/s");
        if has_timing || has_throughput {
            timings.push(crate::text::truncate(t, 150).to_string());
            if timings.len() >= 20 { break; }
        }
    }
    if timings.len() < 2 { return; } // Need at least 2 timing lines to be a benchmark
    let text = timings.join("\n");
    auto_store(dir, "perf-data", &text, "auto, benchmark, performance");
}

/// Check if a string contains a timing pattern like "123.45ms" or "1.23µs".
fn has_timing_pattern(s: &str) -> bool {
    let bytes = s.as_bytes();
    let len = bytes.len();
    for i in 0..len {
        if !bytes[i].is_ascii_digit() { continue; }
        // Find the end of the number (digits and dots)
        let mut j = i + 1;
        while j < len && (bytes[j].is_ascii_digit() || bytes[j] == b'.') { j += 1; }
        if j >= len || j == i + 1 { continue; }
        // Check suffix: ms, µs, us, ns, s (but not just any 's')
        let rest = &s[j..];
        if rest.starts_with("ms") || rest.starts_with("µs") || rest.starts_with("us")
            || rest.starts_with("ns") {
            return true;
        }
        // Bare "s" only if preceded by a decimal point (e.g. "3.15s" but not "32s")
        if rest.starts_with('s') && !rest.get(1..2).map(|c| c.as_bytes()[0].is_ascii_alphanumeric()).unwrap_or(false) {
            if s[i..j].contains('.') { return true; }
        }
    }
    false
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
    let recency = crate::index::topic_recency(data);
    sorted.sort_by(|a, b| {
        let ra = recency.iter().find(|&&(id, _)| id == a.0).map(|&(_, d)| d).unwrap_or(0);
        let rb = recency.iter().find(|&&(id, _)| id == b.0).map(|&(_, d)| d).unwrap_or(0);
        rb.cmp(&ra)
    });
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
    // Context-aware: check if user has been storing findings recently
    let dir = crate::config::resolve_dir(None);
    let recently_stored = std::fs::metadata(dir.join("data.log")).ok()
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
    // Check session: if parent already injected context, tell subagent
    let session = crate::session::Session::load();
    let injected_count = session.as_ref().map(|s| s.injected.len()).unwrap_or(0);
    let focus = session.as_ref().map(|s| &s.focus_topics);

    let msg = match topic_list {
        Some(list) if !list.is_empty() => {
            let mut m = String::with_capacity(256 + list.len());
            m.push_str("HELIX KNOWLEDGE STORE: You have access to helix MCP tools. ");
            if injected_count > 10 {
                m.push_str("Parent session already has context loaded. ");
            }
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

/// 6-layer smart ambient context with deduplication:
///   1. Source-path matches — entries with [source:] metadata for this file
///   2. Symbol-based OR search — fn/struct/enum names from file (cached)
///   3. Global BM25 — stem keyword search
///   4. Structural coupling — "structural <stem>" query
///   5. Refactor impact — removed symbols (Edit only, KB search)
///   6. Code blast radius — removed symbols (Edit only, codegraph usages)
///
/// Adaptive: Layer 2 skipped when Layer 1 returns 5+ hits.
/// Cow<str>: Layer 1 borrows from mmap (zero alloc), other layers own.
pub fn query_ambient(data: &[u8], stem: &str, file_path: &str, syms: &[&str], session: Option<&mut crate::session::Session>) -> String {
    let filename = std::path::Path::new(file_path)
        .file_name().and_then(|f| f.to_str()).unwrap_or(stem);
    // Snapshot injected set for dedup (avoids borrow conflict with mutable session)
    let injected_snapshot = session.as_ref().map(|s| s.injected.clone());
    let mut seen = crate::fxhash::FxHashSet::default();
    // Pre-populate seen with previously injected entries
    if let Some(ref snap) = injected_snapshot {
        for &eid in snap.iter() { seen.insert(eid); }
    }
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

    // Layer 5: Refactor impact (Edit only, KB search)
    let l5_start = pool.len();
    for sym in syms {
        for hit in idx_search(data, sym, 3) {
            if seen.insert(hit.entry_id) { pool.push(std::borrow::Cow::Owned(hit.snippet)); }
        }
    }
    let l5 = pool.len() - l5_start;

    // Layer 6: Code blast radius (Edit only, codegraph)
    // When symbols are removed, find code files that reference them.
    let mut code_blast: Vec<String> = Vec::new();
    if !syms.is_empty() {
        if let Some(root) = find_project_root(file_path) {
            let files = crate::codegraph::walk_source_files(&root);
            for sym in syms {
                let usages = crate::codegraph::find_usages(&root, &files, sym, "", 0);
                if usages.is_empty() { continue; }
                let mut line = String::with_capacity(64);
                line.push_str(sym);
                line.push_str(" → ");
                crate::text::itoa_push(&mut line, usages.len() as u32);
                line.push_str(" refs: ");
                let mut shown_files = crate::fxhash::FxHashSet::default();
                let mut count = 0;
                for u in &usages {
                    if count >= 3 { break; }
                    if shown_files.insert(u.file.as_str()) {
                        if count > 0 { line.push_str(", "); }
                        line.push_str(&u.file);
                        line.push(':');
                        crate::text::itoa_push(&mut line, u.line);
                        count += 1;
                    }
                }
                if usages.len() > 3 {
                    line.push_str(" +");
                    crate::text::itoa_push(&mut line, (usages.len() - 3) as u32);
                    line.push_str(" more");
                }
                code_blast.push(line);
            }
        }
    }
    let l6 = code_blast.len();

    // Directory fallback: when file stem is unknown to KB, try parent dir name.
    // Catches new files in known directories (e.g. new probe in Scanners/).
    if pool.is_empty() || (l1 == 0 && l3 == 0) {
        let dir_name = std::path::Path::new(file_path).parent()
            .and_then(|p| p.file_name()).and_then(|f| f.to_str()).unwrap_or("");
        if dir_name.len() >= 3 && dir_name != stem {
            for h in idx_search(data, dir_name, 3) {
                if seen.insert(h.entry_id) {
                    pool.push(std::borrow::Cow::Owned(h.snippet));
                }
            }
        }
    }
    if pool.is_empty() { return String::new(); }

    // Session bookkeeping: mark injected entries + auto-infer focus topics
    if let Some(sess) = session {
        // Mark all entries we're about to inject
        for &eid in &seen {
            if injected_snapshot.as_ref().map(|s| !s.contains(&eid)).unwrap_or(true) {
                sess.mark_injected(eid);
            }
        }
        // Auto-infer focus topics: count topic_ids, add topics with 3+ hits
        let mut topic_hits: crate::fxhash::FxHashMap<u16, u16> = crate::fxhash::FxHashMap::default();
        for &eid in &seen {
            if let Ok(tid) = crate::index::entry_topic_id(data, eid) {
                *topic_hits.entry(tid).or_insert(0) += 1;
            }
        }
        for (&tid, &count) in &topic_hits {
            if count >= 3 {
                if let Ok(name) = crate::index::topic_name(data, tid) {
                    sess.add_focus_topic(&name);
                }
            }
        }
    }

    // Single output pass
    let est = pool.iter().map(|s| s.len() + 4).sum::<usize>() + 6 * 40
        + code_blast.iter().map(|s| s.len() + 4).sum::<usize>();
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
    // Layer 6: Code blast radius (separate from KB pool)
    if l6 > 0 {
        if !out.is_empty() { out.push_str("---\n"); }
        out.push_str("CODE BLAST RADIUS (callers/references that may break):\n");
        for line in &code_blast {
            out.push_str("  "); out.push_str(line); out.push('\n');
        }
    }
    // Directory fallback entries (after L1-L5)
    if pool_idx < pool.len() {
        let dir_name = std::path::Path::new(file_path).parent()
            .and_then(|p| p.file_name()).and_then(|f| f.to_str()).unwrap_or("directory");
        if !out.is_empty() { out.push_str("---\n"); }
        out.push_str("directory context ("); out.push_str(dir_name); out.push_str("):\n");
        while pool_idx < pool.len() {
            out.push_str("  "); out.push_str(&pool[pool_idx]); out.push('\n');
            pool_idx += 1;
        }
    }
    out
}

// ══════════ Project Root Detection (for Layer 6) ══════════

/// Walk up from a file path to find the project root (Cargo.toml, Package.swift, .git, etc.)
fn find_project_root(file_path: &str) -> Option<std::path::PathBuf> {
    let markers = ["Cargo.toml", "Package.swift", ".git", "Makefile", "build.rs"];
    let mut dir = std::path::Path::new(file_path).parent()?;
    for _ in 0..8 {
        for m in &markers {
            if dir.join(m).exists() { return Some(dir.to_path_buf()); }
        }
        dir = dir.parent()?;
    }
    None
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



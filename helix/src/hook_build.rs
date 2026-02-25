//! Post-build hook — auto-capture build errors, benchmark timings, perf data.

use std::path::Path;

/// After Bash commands: auto-capture build errors, benchmark timings, perf data.
pub(crate) fn post_build(input: &str, dir: &Path) -> Result<String, String> {
    let is_build = (input.contains("xcodebuild") && input.contains("build"))
        || input.contains("cargo build") || input.contains("swift build")
        || input.contains("swiftc ");
    let response = crate::hook::extract_json_str(input, "tool_response").unwrap_or(input);
    capture_benchmark(response, dir);
    if !is_build {
        let mut session = crate::session::Session::load_or_new();
        session.tick_tool(crate::hook::data_log_mtime(dir));
        session.save();
        return Ok(String::new());
    }

    let has_error = input.contains("error:") || input.contains("error[E")
        || input.contains("BUILD FAILED") || input.contains("** FAILED **");

    if has_error {
        let response = crate::hook::extract_json_str(input, "tool_response").unwrap_or(input);
        let mut errors: Vec<String> = Vec::with_capacity(5);
        for part in response.split("\\n").flat_map(|s| s.split('\n')) {
            if errors.len() >= 5 { break; }
            let t = part.trim();
            if (t.contains(": error:") || t.contains("error[E"))
                && !errors.iter().any(|e| e == t)
            {
                errors.push(crate::text::truncate(t, 200).to_string());
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
        if !errors.is_empty() {
            let text = errors.join("\n");
            crate::hook::auto_store(dir, "build-errors", &text, "auto, build-error");
            std::fs::write("/tmp/helix-build-errors", text.as_bytes()).ok();
        }
        let mut kb_out = String::new();
        if let Some(data) = crate::hook::mmap_index(dir) {
            let mut total = 0;
            for err in &errors {
                if total >= 3 { break; }
                let terms = crate::text::query_terms(err);
                if terms.len() < 2 { continue; }
                let q = terms.iter().take(6).cloned().collect::<Vec<_>>().join(" ");
                for h in crate::hook::idx_search_or(data, &q, 2) {
                    if total >= 3 { break; }
                    kb_out.push_str("  "); kb_out.push_str(&h.snippet); kb_out.push('\n');
                    total += 1;
                }
            }
        }
        let mut ctx = String::with_capacity(256);
        ctx.push_str("BUILD FAILED (");
        crate::text::itoa_push(&mut ctx, errors.len() as u32);
        ctx.push_str(" errors):\n");
        for e in &errors { ctx.push_str("  "); ctx.push_str(e); ctx.push('\n'); }
        if !kb_out.is_empty() { ctx.push_str("helix matches:\n"); ctx.push_str(&kb_out); }
        ctx.push_str("Store non-obvious root causes: mcp__helix__store(topic:'build-gotchas', ...)");
        let mut session = crate::session::Session::load_or_new();
        session.record_build(false);
        session.tick_tool(crate::hook::data_log_mtime(dir));
        session.save();
        Ok(crate::hook::hook_output(&ctx))
    } else {
        let marker = std::path::Path::new("/tmp/helix-build-errors");
        if marker.exists() {
            if let Ok(prior) = std::fs::read_to_string(marker) {
                let first = prior.lines().next().unwrap_or("unknown error");
                let mut text = String::with_capacity(12 + first.len().min(120));
                text.push_str("RESOLVED: ");
                text.push_str(crate::text::truncate(first, 120));
                crate::hook::auto_store(dir, "build-errors", &text, "auto, build-fix");
            }
            std::fs::remove_file(marker).ok();
        }
        let mut session = crate::session::Session::load_or_new();
        session.record_build(true);
        session.tick_tool(crate::hook::data_log_mtime(dir));
        session.save();
        Ok(String::new())
    }
}

/// Detect and store benchmark/timing data from Bash output.
fn capture_benchmark(response: &str, dir: &Path) {
    let mut timings: Vec<String> = Vec::new();
    for part in response.split("\\n").flat_map(|s| s.split('\n')) {
        let t = part.trim();
        if t.is_empty() || t.len() < 5 { continue; }
        if t.contains(": error:") || t.contains("warning:") || t.starts_with("Compiling")
            || t.starts_with("Linking") || t.starts_with("Finished") { continue; }
        let has_timing = has_timing_pattern(t);
        let has_throughput = t.contains("tok/s") || t.contains("GB/s") || t.contains("MB/s")
            || t.contains("tokens/s") || t.contains("it/s");
        if has_timing || has_throughput {
            timings.push(crate::text::truncate(t, 150).to_string());
            if timings.len() >= 20 { break; }
        }
    }
    if timings.len() < 2 { return; }
    let text = timings.join("\n");
    crate::hook::auto_store(dir, "perf-data", &text, "auto, benchmark, performance");
}

/// Check if a string contains a timing pattern like "123.45ms" or "1.23us".
fn has_timing_pattern(s: &str) -> bool {
    let bytes = s.as_bytes();
    let len = bytes.len();
    for i in 0..len {
        if !bytes[i].is_ascii_digit() { continue; }
        let mut j = i + 1;
        while j < len && (bytes[j].is_ascii_digit() || bytes[j] == b'.') { j += 1; }
        if j >= len || j == i + 1 { continue; }
        let rest = &s[j..];
        if rest.starts_with("ms") || rest.starts_with("µs") || rest.starts_with("us")
            || rest.starts_with("ns") {
            return true;
        }
        if rest.starts_with('s') && !rest.get(1..2).map(|c| c.as_bytes()[0].is_ascii_alphanumeric()).unwrap_or(false) {
            if s[i..j].contains('.') { return true; }
        }
    }
    false
}

//! Tool dispatch — routes MCP tool calls to handlers.

use crate::json::Value;
use crate::topics::{arg, opt, abool, build_filter};
use std::fmt::Write;
use std::path::Path;

pub(crate) fn dispatch(name: &str, args: Option<&Value>, dir: &Path) -> Result<String, String> {
    match name {
        "store" | "batch" | "edit" => {}
        _ => crate::mcp::ensure_index_fresh(dir),
    }
    match name {
        "store" => {
            let raw_topic = arg(args, "topic"); let text = arg(args, "text");
            let owned_topic;
            let topic = if raw_topic.is_empty() {
                owned_topic = crate::session::Session::load()
                    .and_then(|s| s.focus_topics.first().cloned())
                    .unwrap_or_else(|| "scratch".into());
                owned_topic.as_str()
            } else { raw_topic };
            let tags = opt(arg(args, "tags")); let force = abool(args, "force");
            let terse = abool(args, "terse"); let source = opt(arg(args, "source"));
            let conf = arg(args, "confidence").parse::<f64>().ok().filter(|c| *c >= 0.0 && *c <= 1.0);
            let links = opt(arg(args, "links"));
            let result = crate::write::store(dir, topic, text, tags, force, source, conf, links)?;
            crate::mcp::after_write(dir);
            crate::mcp::log_session(format!("[{}] {}", topic, result.lines().next().unwrap_or("stored")));
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
            for (i, e) in entries.iter().enumerate() {
                let _ = writeln!(out, "[{i}] {}", crate::time::minutes_to_date_str(e.timestamp_min));
                for line in e.body.lines() { let _ = writeln!(out, "  {line}"); }
                let _ = writeln!(out);
            }
            let _ = writeln!(out, "{} entries in {topic}", entries.len());
            Ok(out)
        }
        "edit" => dispatch_edit(args, dir),
        "topics" => crate::topics::dispatch(args, dir),
        "trace" => dispatch_trace(args),
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
                crate::mcp::log_session(format!("[{}] {}", topic, first));
            }
            Err(e) => out.push(format!("  [{}] err: {}", i + 1, e.lines().next().unwrap_or(&e))),
        }
    }
    if ok > 0 { let _ = f.sync_all(); }
    drop(f); drop(_lock);
    if ok > 0 { crate::mcp::after_write(dir); }
    if verbose { Ok(format!("batch: {ok}/{} stored\n{}", items.len(), out.join("\n"))) }
    else {
        let mut terse = format!("batch: {ok}/{} stored", items.len());
        let indices: Vec<&str> = out.iter()
            .filter(|s| !s.contains("skipped") && !s.contains("err:"))
            .filter_map(|s| s.split(']').next().and_then(|p| p.split('[').nth(1)))
            .collect();
        if !indices.is_empty() {
            terse.push_str(" ["); terse.push_str(&indices.join(", ")); terse.push(']');
        }
        Ok(terse)
    }
}

fn dispatch_search(args: Option<&Value>, dir: &Path) -> Result<String, String> {
    let raw_query = arg(args, "query"); let detail = arg(args, "detail");
    let queries_raw = arg(args, "queries");
    if !queries_raw.is_empty() {
        return dispatch_batch_search(queries_raw, detail, args, dir);
    }
    let parsed = crate::text::parse_query_filters(raw_query);
    let query = if parsed.query.is_empty() { raw_query } else { &parsed.query };
    let mut filter = build_filter(args);
    if filter.tag.is_none() { filter.tag = parsed.tag; }
    if filter.topic.is_none() { filter.topic = parsed.topic; }
    if filter.source.is_none() { filter.source = parsed.source; }
    let limit = arg(args, "limit").parse::<usize>().ok();
    match detail {
        "count" => crate::search::count(dir, query, &filter),
        "topics" => crate::search::run_topics(dir, query, &filter),
        "grouped" => crate::mcp::with_index_slice(|idx| {
            crate::search::run_grouped(dir, query, limit, &filter, idx)
        })?,
        _ => {
            let expand = abool(args, "expand");
            let max_lines = if expand { usize::MAX } else {
                arg(args, "lines").parse().unwrap_or(2)
            };
            let full_body = expand || detail == "full";
            crate::mcp::with_index_slice(|idx| match detail {
                "full" => crate::search::run(dir, query, limit, &filter, idx),
                "brief" => crate::search::run_brief(dir, query, limit, &filter, idx),
                _ => crate::search::run_medium(dir, query, limit, &filter, idx, max_lines, full_body),
            })?
        }
    }
}

fn dispatch_batch_search(queries_raw: &str, detail: &str, args: Option<&Value>, dir: &Path)
    -> Result<String, String>
{
    let owned: Vec<String> = match crate::json::parse(queries_raw) {
        Ok(crate::json::Value::Arr(arr)) => arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string())).collect(),
        _ => queries_raw.split(',').map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()).collect(),
    };
    if owned.is_empty() { return Err("queries array is empty".into()); }
    let queries: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
    let filter = build_filter(args);
    let limit = arg(args, "limit").parse::<usize>().ok();
    let full_body = detail == "full";
    let (results, fallback) = crate::mcp::with_index_slice(|idx| {
        crate::score::batch_search_scored(dir, &queries, &filter, limit, idx, full_body)
    })??;
    let mut out = String::new();
    if fallback { let _ = writeln!(out, "(some queries used OR fallback)"); }
    let _ = writeln!(out, "batch search: {} queries, {} results\n", queries.len(), results.len());
    for r in &results {
        let header = r.lines.first().map(|s| s.as_str()).unwrap_or("??");
        let _ = write!(out, "  [{}] {} ({:.1})\n", r.name, header.trim_start_matches("## "), r.score);
        let mut n = 0;
        for line in r.lines.iter().skip(1) {
            if crate::text::is_metadata_line(line) || line.trim().is_empty() { continue; }
            let _ = writeln!(out, "    {}", crate::text::truncate(line.trim(), 100));
            n += 1;
            if !full_body && n >= 2 { break; }
        }
    }
    Ok(out)
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
    crate::mcp::after_write(dir);
    Ok(result)
}

fn dispatch_trace(args: Option<&Value>) -> Result<String, String> {
    let action = arg(args, "action");
    let path_str = arg(args, "path");
    if path_str.is_empty() { return Err("path required (project directory or file)".into()); }
    let path = std::path::PathBuf::from(path_str);
    if !path.exists() { return Err(format!("path not found: {path_str}")); }
    match action {
        "symbols" => {
            if path.is_file() {
                let root = path.parent().unwrap_or(&path);
                Ok(crate::codegraph::file_symbols(root, &path))
            } else {
                let sym_limit = arg(args, "limit").parse::<usize>().unwrap_or(usize::MAX);
                let sym_filter = opt(arg(args, "filter"));
                let files = crate::codegraph::walk_source_files(&path);
                let mut out = String::with_capacity(1024);
                let mut total = 0;
                for file in &files {
                    if let Some(pat) = sym_filter {
                        let fname = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if !glob_match(pat, fname) { continue; }
                    }
                    let content = match std::fs::read_to_string(file) { Ok(c) => c, Err(_) => continue };
                    let rel = file.strip_prefix(&path).unwrap_or(file).to_string_lossy();
                    let syms = crate::codegraph::extract_symbols(&content, &rel);
                    if !syms.is_empty() {
                        let _ = writeln!(out, "{rel} ({} symbols):", syms.len());
                        for s in &syms {
                            let _ = writeln!(out, "  {} {} :{}  {}", s.kind, s.name, s.line, s.signature);
                            total += 1;
                            if total >= sym_limit { break; }
                        }
                    }
                    if total >= sym_limit { break; }
                }
                let _ = writeln!(out, "\n{total} symbols in {} files", files.len());
                Ok(out)
            }
        }
        "blast" => {
            if path.is_file() {
                let root = path.parent().unwrap_or(&path);
                Ok(crate::codegraph::blast_radius(root, &path))
            } else {
                Err("blast requires a file path, not a directory".into())
            }
        }
        "analyze" => {
            if !path.is_dir() { return Err("analyze requires a project directory".into()); }
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("project");
            let analysis = crate::codegraph::analyze_project(&path);
            let entries = crate::codegraph::format_analysis(&analysis, name);
            let dir = crate::config::resolve_dir(None);
            let topic = format!("code-{}", crate::config::sanitize_topic(name));
            let mut out = String::with_capacity(512);
            let mut stored = 0;
            for (label, text) in &entries {
                match crate::write::store(&dir, &topic, text, Some(&format!("auto, code-analysis, {label}")),
                    true, None, Some(0.8), None) {
                    Ok(_) => { stored += 1; }
                    Err(e) => { out.push_str("warn: "); out.push_str(&e); out.push('\n'); }
                }
            }
            std::fs::write("/tmp/helix-external-write", b"1").ok();
            out.push_str("Analyzed ");
            out.push_str(name);
            out.push_str(": ");
            crate::text::itoa_push(&mut out, analysis.files.len() as u32);
            out.push_str(" files, ");
            let total_syms: usize = analysis.files.iter().map(|f| f.symbols.len()).sum();
            crate::text::itoa_push(&mut out, total_syms as u32);
            out.push_str(" symbols, ");
            let total_pats: usize = analysis.files.iter().map(|f| f.patterns.len()).sum();
            crate::text::itoa_push(&mut out, total_pats as u32);
            out.push_str(" patterns, ");
            crate::text::itoa_push(&mut out, analysis.coupling.len() as u32);
            out.push_str(" coupling pairs. Stored ");
            crate::text::itoa_push(&mut out, stored as u32);
            out.push_str(" entries in topic '");
            out.push_str(&topic);
            out.push_str("'.\n\n");
            for (_, text) in &entries {
                out.push_str(text);
                out.push('\n');
            }
            Ok(out)
        }
        _ => {
            let symbol = arg(args, "symbol");
            if symbol.is_empty() { return Err("symbol required (function/struct/enum name to trace)".into()); }
            let root = if path.is_file() { path.parent().unwrap_or(&path).to_path_buf() } else { path };
            Ok(crate::codegraph::trace_symbol(&root, symbol))
        }
    }
}

/// Simple glob: supports *.ext and prefix* patterns.
fn glob_match(pattern: &str, name: &str) -> bool {
    if let Some(ext) = pattern.strip_prefix("*.") {
        name.ends_with(ext) && name.len() > ext.len() && name.as_bytes()[name.len() - ext.len() - 1] == b'.'
    } else if let Some(prefix) = pattern.strip_suffix('*') {
        name.starts_with(prefix)
    } else {
        name == pattern
    }
}

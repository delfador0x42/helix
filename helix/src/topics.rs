//! Topics tool dispatch — browse, maintain, import/export knowledge base.

use crate::json::Value;
use crate::mcp::{after_write, store_index};
use std::fmt::Write;
use std::path::Path;

pub(crate) fn dispatch(args: Option<&Value>, dir: &Path) -> Result<String, String> {
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
        "stats" => dispatch_stats(args, dir),
        "xref" => dispatch_xref(args, dir),
        "graph" => dispatch_graph(args, dir),
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
        "compact" => { let result = crate::datalog::compact_log(dir)?; after_write(dir); Ok(result) }
        "export" => crate::topics_io::dispatch_export(dir),
        "import" => crate::topics_io::dispatch_import(args, dir),
        "reindex" => { let (result, bytes) = crate::index::rebuild(dir, true)?; store_index(bytes); Ok(result) }
        "session" => dispatch_session_log(),
        "checkpoint" => dispatch_checkpoint(args, dir),
        "resume" => match crate::session::Checkpoint::load(dir) {
            Some(cp) => Ok(cp.format_resume()), None => Ok("no checkpoint found".into()),
        },
        "clear_checkpoint" => { crate::session::Checkpoint::clear(dir); Ok("checkpoint cleared".into()) }
        _ => dispatch_list(args, dir),
    }
}

fn dispatch_stats(args: Option<&Value>, dir: &Path) -> Result<String, String> {
    let detail = arg(args, "detail");
    match detail {
        "index" => {
            let guard = crate::mcp::with_index(crate::index::index_info);
            match guard { Some(Ok(s)) => Ok(s), Some(Err(e)) => Err(e), None => Ok("no index loaded".into()) }
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

fn dispatch_xref(args: Option<&Value>, _dir: &Path) -> Result<String, String> {
    let topic = arg(args, "topic");
    crate::mcp::with_index(|data| {
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
    }).unwrap_or_else(|| Err("no index loaded".into()))
}

fn dispatch_graph(args: Option<&Value>, _dir: &Path) -> Result<String, String> {
    let focus = opt(arg(args, "focus"));
    crate::mcp::with_index(|data| {
        let edges = crate::index::xref_edges(data)?;
        let mut out = String::new(); let mut count = 0;
        for (src, dst, n) in &edges {
            let s = crate::index::topic_name(data, *src).unwrap_or_default();
            let d = crate::index::topic_name(data, *dst).unwrap_or_default();
            if let Some(f) = focus { if !s.contains(f) && !d.contains(f) { continue; } }
            let _ = writeln!(out, "  {s} → {d} ({n})"); count += 1;
        }
        if count == 0 { Ok("no edges\n".into()) } else { let _ = writeln!(out, "{count} edges"); Ok(out) }
    }).unwrap_or_else(|| Err("no index loaded".into()))
}

fn dispatch_session_log() -> Result<String, String> {
    Ok(crate::mcp::with_session_log(|log| {
        if log.is_empty() { return "no stores this session".into(); }
        let mut out = format!("{} stores this session:\n", log.len());
        for entry in log { out.push_str("  "); out.push_str(entry); out.push('\n'); }
        out
    }))
}

fn dispatch_checkpoint(args: Option<&Value>, dir: &Path) -> Result<String, String> {
    let task = arg(args, "task");
    if task.is_empty() { return Err("task required for checkpoint".into()); }
    let mut cp = crate::session::Checkpoint::new(task);
    let done = arg(args, "done");
    if !done.is_empty() { cp.done = done.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(); }
    let next = arg(args, "next");
    if !next.is_empty() { cp.next = next.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(); }
    let hyp = arg(args, "hypotheses");
    if !hyp.is_empty() { cp.hypotheses = hyp.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(); }
    let blocked = arg(args, "blocked");
    if !blocked.is_empty() { cp.blocked = blocked.to_string(); }
    let files = arg(args, "files");
    if !files.is_empty() { cp.files = files.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(); }
    cp.save(dir)?;
    Ok(format!("checkpoint saved: {task}"))
}

fn dispatch_list(args: Option<&Value>, dir: &Path) -> Result<String, String> {
    let prefix = opt(arg(args, "topic"));
    crate::cache::with_corpus(dir, |cached| {
        let mut topics: std::collections::BTreeMap<&str, (usize, i32)> = std::collections::BTreeMap::new();
        for e in cached {
            if let Some(p) = prefix {
                if !crate::config::topic_matches_query(e.topic.as_str(), p) { continue; }
            }
            let (c, latest) = topics.entry(&e.topic).or_insert((0, 0));
            *c += 1; if e.timestamp_min > *latest { *latest = e.timestamp_min; }
        }
        if topics.is_empty() {
            return if let Some(p) = prefix { format!("no topics under '{p}'\n") }
            else { "no topics\n".into() };
        }
        format_topic_tree(&topics)
    })
}

fn format_topic_tree(topics: &std::collections::BTreeMap<&str, (usize, i32)>) -> String {
    let total_entries: usize = topics.values().map(|(c, _)| c).sum();
    let has_hierarchy = topics.keys().any(|k| k.contains('/'));
    if !has_hierarchy {
        let mut sorted: Vec<_> = topics.iter().collect();
        sorted.sort_by(|a, b| b.1.1.cmp(&a.1.1));
        let mut out = String::new();
        for (topic, (count, latest)) in &sorted {
            let _ = writeln!(out, "  {topic} ({count}, {})", crate::time::minutes_to_date_str(*latest));
        }
        let _ = writeln!(out, "{} topics, {} entries", sorted.len(), total_entries);
        return out;
    }
    let mut prefixes: std::collections::BTreeMap<String, (usize, i32, usize)> = std::collections::BTreeMap::new();
    for (&topic, &(count, latest)) in topics {
        let parts: Vec<&str> = topic.split('/').collect();
        for depth in 1..parts.len() {
            let prefix = parts[..depth].join("/");
            let e = prefixes.entry(prefix).or_insert((0, 0, 0));
            e.0 += count; if latest > e.1 { e.1 = latest; } e.2 += 1;
        }
    }
    let mut sorted: Vec<_> = topics.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut out = String::new();
    let mut last_top = String::new();
    for (&topic, &(count, latest)) in &sorted {
        let parts: Vec<&str> = topic.split('/').collect();
        let top = parts[0];
        if top != last_top {
            if !last_top.is_empty() { out.push('\n'); }
            if let Some(&(sub_entries, sub_latest, sub_topics)) = prefixes.get(top) {
                let _ = writeln!(out, "{}/ ({} entries, {} subtopics, {})",
                    top, sub_entries, sub_topics, crate::time::minutes_to_date_str(sub_latest));
            }
            last_top = top.to_string();
        }
        let depth = parts.len() - 1;
        if depth == 0 {
            let _ = writeln!(out, "  {topic} ({count}, {})", crate::time::minutes_to_date_str(latest));
        } else {
            let indent = "  ".repeat(depth);
            let leaf = *parts.last().unwrap();
            let _ = writeln!(out, "  {indent}{leaf} ({count}, {})", crate::time::minutes_to_date_str(latest));
        }
    }
    let _ = writeln!(out, "\n{} topics, {} entries", sorted.len(), total_entries);
    out
}

// ══════════ Helpers ══════════

fn log_preview(body: &str) -> &str {
    body.lines().find(|l| { let t = l.trim(); !t.is_empty() && !crate::text::is_metadata_line(t) })
        .map(|l| l.trim()).unwrap_or("")
}

pub(crate) fn arg<'a>(args: Option<&'a Value>, key: &str) -> &'a str {
    args.and_then(|a| a.get(key)).and_then(|v| v.as_str()).unwrap_or("")
}
pub(crate) fn opt(s: &str) -> Option<&str> { if s.is_empty() { None } else { Some(s) } }
pub(crate) fn abool(args: Option<&Value>, key: &str) -> bool { let s = arg(args, key); s == "true" || s == "1" }

pub(crate) fn build_filter(args: Option<&Value>) -> crate::index::Filter {
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

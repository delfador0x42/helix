//! Store, batch, edit, delete, append. Merges store.rs + delete.rs.

use crate::time::LocalTime;
use std::path::Path;

/// Full store with all metadata options.
pub fn store(
    dir: &Path, topic: &str, text: &str, tags: Option<&str>,
    force: bool, source: Option<&str>, confidence: Option<f64>, links: Option<&str>,
) -> Result<String, String> {
    crate::config::ensure_dir(dir)?;
    let _lock = crate::lock::FileLock::acquire(dir)?;
    let log_path = crate::datalog::ensure_log(dir)?;

    let cleaned_tags = tags.map(normalize_tags).or_else(|| auto_detect_tags(text));
    let body = build_body(text, cleaned_tags.as_deref(), source, confidence, links);
    let ts_min = LocalTime::now().to_minutes() as i32;

    let dupe_warn = if !force { check_dupe(dir, topic, text) } else { None };

    let offset = crate::datalog::append_entry(&log_path, topic, &body, ts_min)?;
    crate::cache::append_to_cache(dir, topic, &body, ts_min, offset);

    let tag_echo = cleaned_tags.as_deref().filter(|t| !t.is_empty())
        .map(|t| format!(" [tags: {t}]")).unwrap_or_default();
    let conf_echo = confidence.filter(|c| *c < 1.0)
        .map(|c| format!(" (~{:.0}%)", c * 100.0)).unwrap_or_default();
    let mut msg = format!("stored in {topic}{tag_echo}{conf_echo}");
    if let Some(ref dw) = dupe_warn { msg.push_str(&format!("\n  dupe warning: {dw}")); }
    Ok(msg)
}

/// Lean batch write — no lock, no dupe check.
pub fn batch_entry(
    dir: &Path, topic: &str, text: &str, tags: Option<&str>, source: Option<&str>,
) -> Result<String, String> {
    crate::config::ensure_dir(dir)?;
    let log_path = crate::datalog::ensure_log(dir)?;
    let cleaned_tags = tags.map(normalize_tags);
    let body = build_body(text, cleaned_tags.as_deref(), source, None, None);
    let ts_min = LocalTime::now().to_minutes() as i32;
    crate::datalog::append_entry(&log_path, topic, &body, ts_min)?;
    Ok(format!("stored in {topic}"))
}

/// Batch write to pre-opened handle — no lock, no fsync.
pub fn batch_entry_to(
    f: &mut std::fs::File, topic: &str, text: &str, tags: Option<&str>, source: Option<&str>,
) -> Result<String, String> {
    let cleaned_tags = tags.map(normalize_tags);
    let body = build_body(text, cleaned_tags.as_deref(), source, None, None);
    let ts_min = LocalTime::now().to_minutes() as i32;
    crate::datalog::append_entry_to(f, topic, &body, ts_min)?;
    Ok(format!("stored in {topic}"))
}

/// Append text to last entry in topic (no new timestamp).
pub fn append(dir: &Path, topic: &str, text: &str) -> Result<String, String> {
    let _lock = crate::lock::FileLock::acquire(dir)?;
    let log_path = crate::config::log_path(dir);
    let entries = crate::datalog::iter_live(&log_path)?;
    let last = entries.iter().rev().find(|e| e.topic == topic)
        .ok_or_else(|| format!("{topic} not found — use 'store' first"))?;
    let new_body = format!("{}\n{text}", last.body.trim_end());
    crate::datalog::append_entry(&log_path, topic, &new_body, last.timestamp_min)?;
    crate::datalog::append_delete(&log_path, last.offset)?;
    Ok(format!("appended to last entry in {topic}"))
}

/// Append to specific entry by index or match string.
pub fn append_to(dir: &Path, topic: &str, text: &str,
                 index: Option<usize>, match_str: Option<&str>, tag: Option<&str>,
) -> Result<String, String> {
    let _lock = crate::lock::FileLock::acquire(dir)?;
    let log_path = crate::config::log_path(dir);
    let entries = topic_entries(&log_path, topic)?;
    if entries.is_empty() { return Err(format!("{topic} not found")); }

    let target = if let Some(idx) = index {
        entries.get(idx).ok_or_else(|| format!("index {idx} out of range"))?
    } else if let Some(needle) = match_str {
        let lower = needle.to_lowercase();
        entries.iter().find(|e| e.body.to_lowercase().contains(&lower))
            .ok_or_else(|| format!("no entry matching \"{needle}\""))?
    } else if let Some(t) = tag {
        entries.iter().rev().find(|e| e.body.contains(&format!("[tags: ")) && e.body.to_lowercase().contains(&t.to_lowercase()))
            .ok_or_else(|| format!("no entry with tag \"{t}\""))?
    } else {
        entries.last().unwrap()
    };

    let new_body = format!("{}\n{text}", target.body.trim_end());
    crate::datalog::append_entry(&log_path, topic, &new_body, target.timestamp_min)?;
    crate::datalog::append_delete(&log_path, target.offset)?;
    Ok(format!("appended to entry in {topic}"))
}

/// Revise (overwrite) an entry's text.
pub fn revise(dir: &Path, topic: &str, text: &str,
              index: Option<usize>, match_str: Option<&str>) -> Result<String, String> {
    let _lock = crate::lock::FileLock::acquire(dir)?;
    let log_path = crate::config::log_path(dir);
    let entries = topic_entries(&log_path, topic)?;
    if entries.is_empty() { return Err(format!("{topic} not found")); }

    let target = if let Some(idx) = index {
        entries.get(idx).ok_or_else(|| format!("index {idx} out of range"))?
    } else if let Some(needle) = match_str {
        let lower = needle.to_lowercase();
        entries.iter().find(|e| e.body.to_lowercase().contains(&lower))
            .ok_or_else(|| format!("no entry matching \"{needle}\""))?
    } else {
        return Err("specify index or match_str".into());
    };

    let new_body = format!("[modified: {}]\n{text}", crate::time::minutes_to_date_str(LocalTime::now().to_minutes() as i32));
    crate::datalog::append_entry(&log_path, topic, &new_body, target.timestamp_min)?;
    crate::datalog::append_delete(&log_path, target.offset)?;
    Ok(format!("revised entry in {topic}"))
}

/// Delete entries. Supports: all, by index, by match, last.
pub fn delete(dir: &Path, topic: &str, all: bool, index: Option<usize>,
              match_str: Option<&str>) -> Result<String, String> {
    let _lock = crate::lock::FileLock::acquire(dir)?;
    let log_path = crate::config::log_path(dir);
    let entries = topic_entries(&log_path, topic)?;
    if entries.is_empty() { return Err(format!("topic '{topic}' not found")); }

    if all {
        for e in &entries { crate::datalog::append_delete(&log_path, e.offset)?; }
        return Ok(format!("deleted {topic} ({} entries)", entries.len()));
    }
    if let Some(idx) = index {
        if idx >= entries.len() { return Err(format!("index {idx} out of range")); }
        crate::datalog::append_delete(&log_path, entries[idx].offset)?;
        return Ok(format!("removed entry [{idx}] from {topic}"));
    }
    if let Some(needle) = match_str {
        let lower = needle.to_lowercase();
        let entry = entries.iter().find(|e| e.body.to_lowercase().contains(&lower))
            .ok_or_else(|| format!("no entry matching \"{needle}\""))?;
        crate::datalog::append_delete(&log_path, entry.offset)?;
        return Ok(format!("removed entry matching \"{needle}\" from {topic}"));
    }
    // Default: delete last
    let last = entries.last().unwrap();
    crate::datalog::append_delete(&log_path, last.offset)?;
    Ok(format!("removed last entry from {topic}"))
}

/// Manage tags on an entry.
pub fn tag(dir: &Path, topic: &str, add_tags: Option<&str>, remove_tags: Option<&str>,
           index: Option<usize>, match_str: Option<&str>) -> Result<String, String> {
    let _lock = crate::lock::FileLock::acquire(dir)?;
    let log_path = crate::config::log_path(dir);
    let entries = topic_entries(&log_path, topic)?;
    if entries.is_empty() { return Err(format!("{topic} not found")); }

    let target = if let Some(idx) = index {
        entries.get(idx).ok_or_else(|| format!("index {idx} out of range"))?
    } else if let Some(needle) = match_str {
        let lower = needle.to_lowercase();
        entries.iter().find(|e| e.body.to_lowercase().contains(&lower))
            .ok_or_else(|| format!("no entry matching \"{needle}\""))?
    } else {
        entries.last().unwrap()
    };

    let meta = crate::text::extract_all_metadata(&target.body);
    let mut tags: Vec<String> = meta.tags;
    if let Some(add) = add_tags {
        for t in add.split(',') {
            let t = t.trim().to_lowercase();
            if !t.is_empty() && !tags.contains(&t) { tags.push(t); }
        }
    }
    if let Some(rm) = remove_tags {
        let rm_set: Vec<String> = rm.split(',').map(|t| t.trim().to_lowercase()).collect();
        tags.retain(|t| !rm_set.contains(t));
    }
    tags.sort();

    // Rebuild body with updated tags
    let content_lines: Vec<&str> = target.body.lines()
        .filter(|l| !l.starts_with("[tags: ")).collect();
    let mut new_body = String::new();
    if !tags.is_empty() { new_body.push_str(&format!("[tags: {}]\n", tags.join(", "))); }
    for line in content_lines { new_body.push_str(line); new_body.push('\n'); }
    let new_body = new_body.trim_end().to_string();

    crate::datalog::append_entry(&log_path, topic, &new_body, target.timestamp_min)?;
    crate::datalog::append_delete(&log_path, target.offset)?;
    Ok(format!("updated tags on {topic}: {}", tags.join(", ")))
}

/// Get all live entries for a topic.
pub fn topic_entries(log_path: &Path, topic: &str) -> Result<Vec<crate::datalog::LogEntry>, String> {
    let all = crate::datalog::iter_live(log_path)?;
    Ok(all.into_iter().filter(|e| e.topic == topic).collect())
}

// ══════════ Helpers ══════════

fn build_body(text: &str, tags: Option<&str>, source: Option<&str>,
              confidence: Option<f64>, links: Option<&str>) -> String {
    let mut body = String::new();
    if let Some(t) = tags { if !t.is_empty() { body.push_str(&format!("[tags: {t}]\n")); } }
    if let Some(src) = source { body.push_str(&format!("[source: {src}]\n")); }
    if let Some(c) = confidence { if c < 1.0 { body.push_str(&format!("[confidence: {c}]\n")); } }
    if let Some(l) = links { if !l.is_empty() { body.push_str(&format!("[links: {l}]\n")); } }
    body.push_str(text);
    body
}

fn normalize_tags(raw: &str) -> String {
    let mut tags: Vec<String> = raw.split(',')
        .map(|t| singularize(t.trim()).to_lowercase())
        .filter(|t| !t.is_empty()).collect();
    tags.sort();
    tags.dedup();
    tags.join(", ")
}

fn singularize(s: &str) -> String {
    let s = s.trim();
    if s.len() <= 3 { return s.to_string(); }
    if s.ends_with("ies") && s.len() > 4 { return format!("{}y", &s[..s.len() - 3]); }
    if s.ends_with("sses") { return s[..s.len() - 2].to_string(); }
    if s.ends_with('s') && !s.ends_with("ss") && !s.ends_with("us") && !s.ends_with("is") {
        return s[..s.len() - 1].to_string();
    }
    s.to_string()
}

fn auto_detect_tags(text: &str) -> Option<String> {
    let first = text.lines().find(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_lowercase()).unwrap_or_default();
    const PREFIXES: &[(&str, &str)] = &[
        ("gotcha:", "gotcha"), ("bug:", "gotcha"), ("invariant:", "invariant"),
        ("security:", "invariant"), ("decision:", "decision"), ("design:", "decision"),
        ("data flow:", "data-flow"), ("flow:", "data-flow"),
        ("perf:", "performance"), ("benchmark:", "performance"),
        ("gap:", "gap"), ("missing:", "gap"), ("todo:", "gap"),
        ("how-to:", "how-to"), ("impl:", "how-to"), ("fix:", "how-to"),
        ("module:", "module-map"), ("overview:", "architecture"),
        ("coupling:", "coupling"), ("pattern:", "pattern"),
    ];
    let mut tags = Vec::new();
    for &(prefix, tag) in PREFIXES {
        if first.starts_with(prefix) && !tags.contains(&tag) { tags.push(tag); }
    }
    if tags.is_empty() { None } else { Some(tags.join(", ")) }
}

fn check_dupe(dir: &Path, topic: &str, new_text: &str) -> Option<String> {
    crate::cache::with_corpus(dir, |cached| {
        let new_tokens: crate::fxhash::FxHashSet<String> = crate::text::tokenize(new_text)
            .into_iter().filter(|t| t.len() >= 3).collect();
        if new_tokens.len() < 6 { return None; }
        for e in cached.iter().filter(|e| *e.topic == *topic) {
            let isect = new_tokens.iter().filter(|t| e.tf_map.contains_key(*t)).count();
            let union = new_tokens.len() + e.tf_map.len() - isect;
            if union > 0 && isect as f64 / union as f64 > 0.70 {
                let preview = e.preview();
                return Some(crate::text::truncate(preview, 100).to_string());
            }
        }
        None
    }).ok().flatten()
}

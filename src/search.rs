//! Search output formatting. Scoring lives in index.rs.
//! 6 output modes: full, brief, medium, grouped, topics, count.

use std::fmt::Write;
use std::path::Path;
use crate::text::{query_terms, truncate, extract_tags};
use crate::index::Filter;

pub fn run(dir: &Path, query: &str, limit: Option<usize>, filter: &Filter,
           index_data: Option<&[u8]>) -> Result<String, String> {
    let terms = query_terms(query);
    if terms.is_empty() { return Err("provide a query".into()); }
    let (results, fallback) = crate::index::search_scored(dir, &terms, filter, limit, index_data, true)?;
    let total = results.len();
    let show = limit.map(|l| total.min(l)).unwrap_or(total);
    let mut out = String::new();
    if fallback { let _ = writeln!(out, "(no exact match — showing {} OR results)", results.len()); }
    let mut last_name = String::new();
    for r in results.iter().take(show) {
        if r.name != last_name {
            let _ = writeln!(out, "\n--- {} ---", r.name);
            last_name = r.name.clone();
        }
        for line in &r.lines {
            if !terms.is_empty() && terms.iter().any(|t| contains_ci(line, t)) {
                let _ = writeln!(out, "> {line}");
            } else { let _ = writeln!(out, "{line}"); }
        }
        let _ = writeln!(out);
    }
    if total == 0 { let _ = writeln!(out, "no matches for '{query}'"); }
    else if show < total { let _ = writeln!(out, "(showing {show} of {total} matches)"); }
    else { let _ = writeln!(out, "{total} matching section(s)"); }
    Ok(out)
}

pub fn run_brief(dir: &Path, query: &str, limit: Option<usize>, filter: &Filter,
                 index_data: Option<&[u8]>) -> Result<String, String> {
    let terms = query_terms(query);
    if terms.is_empty() { return Err("provide a query".into()); }
    let (results, fallback) = crate::index::search_scored(dir, &terms, filter, limit, index_data, false)?;
    let mut out = String::new();
    if fallback { let _ = writeln!(out, "(no exact match — showing OR results)"); }
    for r in &results {
        let tags = extract_tags(&r.lines);
        let tag_suffix = tags.map(|t| format!(" {t}")).unwrap_or_default();
        let content = r.lines.iter().skip(1)
            .find(|l| !crate::text::is_metadata_line(l) && !l.trim().is_empty())
            .map(|l| truncate(l.trim().trim_start_matches("- "), 80))
            .unwrap_or("");
        let _ = writeln!(out, "  [{}] {content}{tag_suffix} ({:.1})", r.name, r.score);
    }
    if results.is_empty() { let _ = writeln!(out, "no matches for '{query}'"); }
    else { let _ = writeln!(out, "{} match(es)", results.len()); }
    Ok(out)
}

pub fn run_medium(dir: &Path, query: &str, limit: Option<usize>, filter: &Filter,
                  index_data: Option<&[u8]>) -> Result<String, String> {
    let terms = query_terms(query);
    if terms.is_empty() { return Err("provide a query".into()); }
    let (results, fallback) = crate::index::search_scored(dir, &terms, filter, limit, index_data, false)?;
    let mut out = String::new();
    if fallback { let _ = writeln!(out, "(no exact match — showing OR results)"); }
    for r in &results {
        let header = r.lines.first().map(|s| s.as_str()).unwrap_or("??");
        let tags = extract_tags(&r.lines);
        let tag_str = tags.map(|t| format!(" {t}")).unwrap_or_default();
        let _ = writeln!(out, "  [{}] {}{} ({:.1})", r.name, header.trim_start_matches("## "), tag_str, r.score);
        let mut n = 0;
        for line in r.lines.iter().skip(1) {
            if crate::text::is_metadata_line(line) || line.trim().is_empty() { continue; }
            let _ = writeln!(out, "    {}", truncate(line.trim(), 100));
            n += 1;
            if n >= 2 { break; }
        }
    }
    if results.is_empty() { let _ = writeln!(out, "no matches for '{query}'"); }
    else { let _ = writeln!(out, "{} match(es)", results.len()); }
    Ok(out)
}

pub fn run_topics(dir: &Path, query: &str, filter: &Filter) -> Result<String, String> {
    let terms = query_terms(query);
    if terms.is_empty() { return Err("provide a query".into()); }
    let (hits, fallback) = crate::index::topic_matches(dir, &terms, filter)?;
    let mut out = String::new();
    if hits.is_empty() { let _ = writeln!(out, "no matches for '{query}'"); }
    else {
        if fallback { let _ = writeln!(out, "(no exact match — showing OR results)"); }
        let total: usize = hits.iter().map(|(_, n)| n).sum();
        for (topic, n) in &hits { let _ = writeln!(out, "  {topic}: {n}"); }
        let _ = writeln!(out, "{total} match(es) across {} topic(s)", hits.len());
    }
    Ok(out)
}

pub fn count(dir: &Path, query: &str, filter: &Filter) -> Result<String, String> {
    let terms = query_terms(query);
    if terms.is_empty() { return Err("provide a query".into()); }
    let (total, topics, fallback) = crate::index::count_matches(dir, &terms, filter)?;
    if total > 0 {
        let prefix = if fallback { "(OR) " } else { "" };
        Ok(format!("{prefix}{total} matches across {topics} topics for '{query}'"))
    } else { Ok(format!("0 matches for '{query}'")) }
}

pub fn run_grouped(dir: &Path, query: &str, limit_per_topic: Option<usize>, filter: &Filter,
                   index_data: Option<&[u8]>) -> Result<String, String> {
    let terms = query_terms(query);
    if terms.is_empty() { return Err("query required".into()); }
    let (results, fallback) = crate::index::search_scored(dir, &terms, filter, None, index_data, true)?;
    if results.is_empty() { return Ok(format!("no matches for '{query}'\n")); }
    let cap = limit_per_topic.unwrap_or(5);
    let mut groups: std::collections::BTreeMap<String, Vec<&crate::index::ScoredResult>> = std::collections::BTreeMap::new();
    for r in &results { groups.entry(r.name.clone()).or_default().push(r); }
    let mut order: Vec<(String, f64)> = groups.iter()
        .map(|(n, e)| (n.clone(), e.first().map(|e| e.score).unwrap_or(0.0))).collect();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let total: usize = groups.values().map(|v| v.len()).sum();
    let mut out = String::new();
    if fallback { let _ = writeln!(out, "(no exact match — showing OR results)"); }
    let _ = writeln!(out, "'{}' across {} topics ({} matches):\n", query, groups.len(), total);
    for (name, _) in &order {
        let entries = &groups[name];
        let _ = writeln!(out, "[{}] {} matches", name, entries.len());
        for r in entries.iter().take(cap) {
            let header = r.lines.first().map(|s| s.as_str()).unwrap_or("??");
            let _ = write!(out, "  {} — ", header.trim_start_matches("## "));
            if let Some(line) = r.lines.iter().skip(1)
                .find(|l| !crate::text::is_metadata_line(l) && !l.trim().is_empty()) {
                let _ = writeln!(out, "{}", truncate(line.trim(), 90));
            } else { let _ = writeln!(out); }
        }
        if entries.len() > cap { let _ = writeln!(out, "  ...and {} more", entries.len() - cap); }
        let _ = writeln!(out);
    }
    Ok(out)
}

#[inline]
fn contains_ci(haystack: &str, needle: &str) -> bool {
    let nb = needle.as_bytes();
    if nb.len() > haystack.len() { return false; }
    haystack.as_bytes().windows(nb.len())
        .any(|w| w.iter().zip(nb).all(|(h, n)| h.to_ascii_lowercase() == *n))
}

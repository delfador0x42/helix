//! One-shot briefing: reconstruct + compress + format.
//! Merges reconstruct.rs + compress.rs + briefing.rs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use crate::fxhash::FxHashSet;

// ══════════ Data Types ══════════

struct RawEntry {
    topic: String, body: String, timestamp_min: i32, days_old: i64,
    tags: Vec<String>, relevance: f64,
}

pub(crate) struct Compressed {
    pub(crate) topic: String, pub(crate) body: String, pub(crate) date: String,
    pub(crate) days_old: i64, pub(crate) tags: Vec<String>, pub(crate) relevance: f64,
    pub(crate) source: Option<String>, pub(crate) chain: Option<String>,
    pub(crate) also_in: Vec<String>,
}

#[derive(Clone, Copy)]
pub enum Detail { Summary, Scan, Full }
impl Detail {
    pub fn from_str(s: &str) -> Self {
        match s { "scan" => Self::Scan, "full" => Self::Full, _ => Self::Summary }
    }
}

// ══════════ Orchestrator ══════════

pub fn run(dir: &Path, query: &str, detail: &str, since_hours: Option<u64>,
           focus: Option<&str>) -> Result<String, String> {
    let q = query.to_lowercase();
    let is_glob = q.contains('*');
    let q_sanitized = if is_glob { q.clone() } else { crate::config::sanitize_topic(query) };
    let q_terms = crate::text::query_terms(query);
    let now_days = crate::time::LocalTime::now().to_days();
    let max_days = since_hours.map(|h| if h <= 12 { 0i64 } else { (h as i64 - 1) / 24 });
    let focus_cats: Option<Vec<String>> = focus.map(|f|
        f.split(',').map(|c| c.trim().to_uppercase()).filter(|c| !c.is_empty()).collect()
    );

    crate::cache::with_corpus(dir, |cached| {
        let mut primary_set: BTreeSet<&str> = BTreeSet::new();
        for e in cached {
            let topic = e.topic.as_str();
            if is_glob { if glob_match(&q, topic) { primary_set.insert(topic); } }
            else if crate::config::topic_matches_query(topic, &q_sanitized) { primary_set.insert(topic); }
        }

        let mut entries: Vec<RawEntry> = Vec::new();
        let mut matched: FxHashSet<u32> = FxHashSet::default();

        for e in cached {
            let is_primary = primary_set.contains(e.topic.as_str());
            let is_related = !q_terms.is_empty() && q_terms.iter().any(|t| e.tf_map.contains_key(t));
            if !is_primary && !is_related { continue; }
            let days_old = e.days_old(now_days);
            if let Some(max) = max_days { if days_old > max { continue; } }
            matched.insert(e.offset);
            let mut relevance = if is_primary {
                // Hierarchy proximity: exact match = 10, child = 9, grandchild = 8, ...
                match crate::config::hierarchy_distance(e.topic.as_str(), &q_sanitized) {
                    Some(d) => (10.0 - d as f64).max(5.0),
                    None => 10.0, // glob match or other primary
                }
            } else { 0.0 };
            for t in &q_terms { relevance += *e.tf_map.get(t).unwrap_or(&0) as f64; }
            if !e.has_tag("invariant") && !e.has_tag("architecture") {
                relevance *= 1.0 + 1.0 / (1.0 + days_old as f64 / 7.0);
            }
            relevance *= e.confidence();
            entries.push(RawEntry {
                topic: e.topic.to_string(), body: e.body.clone(),
                timestamp_min: e.timestamp_min, days_old,
                tags: e.tags().to_vec(), relevance,
            });
        }

        if entries.is_empty() {
            return if since_hours.is_some() {
                format!("No entries for '{}' in the last {}h.\n", query, since_hours.unwrap())
            } else { format!("No entries found for '{query}'.\n") };
        }

        let primary: Vec<String> = primary_set.iter().map(|s| s.to_string()).collect();
        let raw_count = entries.len();
        let compressed = compress(entries);
        let d = Detail::from_str(detail);
        crate::brief_fmt::format_output(&compressed, query, raw_count, &primary, d, since_hours, focus_cats.as_deref())
    })
}

// ══════════ Compression ══════════

fn compress(entries: Vec<RawEntry>) -> Vec<Compressed> {
    let mut out: Vec<Compressed> = entries.into_iter().map(|e| {
        let source = extract_source(&e.body);
        let date = crate::time::minutes_to_date_str(e.timestamp_min);
        Compressed {
            topic: e.topic, body: e.body, date, days_old: e.days_old,
            tags: e.tags, relevance: e.relevance, source,
            chain: None, also_in: Vec::new(),
        }
    }).collect();
    dedup(&mut out);
    let tokens: Vec<FxHashSet<String>> = out.iter().map(|e|
        first_content(&e.body).split_whitespace()
            .filter(|w| w.len() >= 3).map(|w| w.to_lowercase()).collect()
    ).collect();
    supersede(&mut out, &tokens);
    temporal_chains(&mut out, &tokens);
    out.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));
    out
}

pub(crate) fn first_content(body: &str) -> &str {
    body.lines().find(|l| { let t = l.trim(); !t.is_empty() && !crate::text::is_metadata_line(t) }).unwrap_or("")
}

fn extract_source(body: &str) -> Option<String> {
    for line in body.lines() {
        if let Some(inner) = line.strip_prefix("[source: ").and_then(|s| s.strip_suffix(']')) {
            return Some(inner.trim().to_string());
        }
    }
    None
}

fn dedup(entries: &mut Vec<Compressed>) {
    let keys: Vec<String> = entries.iter().map(|e| first_content(&e.body).to_lowercase()).collect();
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, key) in keys.iter().enumerate() {
        if key.len() >= 10 { groups.entry(key.clone()).or_default().push(i); }
    }
    let mut remove = Vec::new();
    for indices in groups.values() {
        if indices.len() < 2 { continue; }
        let topics: Vec<&str> = indices.iter().map(|&i| entries[i].topic.as_str()).collect();
        if topics.windows(2).all(|w| w[0] == w[1]) { continue; }
        let best = *indices.iter().max_by(|a, b|
            entries[**a].relevance.partial_cmp(&entries[**b].relevance).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
        let others: Vec<(usize, String)> = indices.iter()
            .filter(|&&i| i != best && entries[i].topic != entries[best].topic)
            .map(|&i| (i, entries[i].topic.clone())).collect();
        for (idx, topic) in others {
            entries[best].also_in.push(topic);
            remove.push(idx);
        }
    }
    remove.sort_unstable(); remove.dedup();
    for &idx in remove.iter().rev() { entries.remove(idx); }
}

fn supersede(entries: &mut [Compressed], tokens: &[FxHashSet<String>]) {
    let mut by_topic: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, e) in entries.iter().enumerate() { by_topic.entry(e.topic.as_str()).or_default().push(i); }
    let mut dimmed: BTreeMap<usize, usize> = BTreeMap::new();
    for (_, indices) in &by_topic {
        for (a, &i) in indices.iter().enumerate() {
            if tokens[i].len() < 3 || dimmed.contains_key(&i) { continue; }
            for &j in &indices[a+1..] {
                if tokens[j].len() < 3 || dimmed.contains_key(&j) { continue; }
                let isect = tokens[i].iter().filter(|t| tokens[j].contains(t.as_str())).count();
                let union = tokens[i].len() + tokens[j].len() - isect;
                if union == 0 || isect * 100 / union < 60 { continue; }
                if (entries[i].days_old - entries[j].days_old).abs() < 2 { continue; }
                if entries[i].days_old > entries[j].days_old { dimmed.insert(i, j); }
                else { dimmed.insert(j, i); }
            }
        }
    }
    for (&old, &newer) in &dimmed {
        entries[old].relevance *= 0.5;
        let preview = crate::text::truncate(first_content(&entries[newer].body), 50);
        entries[old].chain = Some(format!("superseded by: {preview}"));
    }
}

/// Temporal chains: dominant term grouping only (Pass 1).
fn temporal_chains(entries: &mut Vec<Compressed>, _tokens: &[FxHashSet<String>]) {
    let mut groups: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for (i, e) in entries.iter().enumerate() {
        if let Some(term) = dominant_term(first_content(&e.body)) {
            groups.entry((e.topic.clone(), term)).or_default().push(i);
        }
    }
    let mut remove = Vec::new();
    for ((_, term), indices) in &groups {
        if indices.len() < 2 { continue; }
        let mut sorted: Vec<usize> = indices.clone();
        sorted.sort_by(|a, b| entries[*b].days_old.cmp(&entries[*a].days_old));
        let newest = *sorted.last().unwrap();
        let mut chain = String::with_capacity(sorted.len() * 30);
        chain.push_str(term); chain.push_str(": ");
        for (si, &i) in sorted.iter().enumerate() {
            if si > 0 { chain.push_str(" → "); }
            let fc = first_content(&entries[i].body);
            let without = fc.replace(term.as_str(), "");
            let words: Vec<&str> = without.split_whitespace().take(5).collect();
            let step = words.join(" ");
            if step.is_empty() { chain.push_str(&entries[i].date[5..]); }
            else { chain.push_str(&step); chain.push_str(" ("); chain.push_str(&entries[i].date[5..]); chain.push(')'); }
        }
        entries[newest].chain = Some(chain);
        entries[newest].relevance += sorted.len() as f64;
        for &idx in sorted.iter().take(sorted.len() - 1) { remove.push(idx); }
    }
    remove.sort_unstable(); remove.dedup();
    for &idx in remove.iter().rev() { entries.remove(idx); }
}

fn dominant_term(line: &str) -> Option<String> {
    line.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() >= 3 && w.chars().next().map_or(false, |c| c.is_uppercase()))
        .max_by_key(|w| w.len()).map(|w| w.to_string())
}

fn glob_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 { return text.contains(pattern); }
    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() { continue; }
        if i == 0 { if !text.starts_with(part) { return false; } pos = part.len(); }
        else if i == parts.len() - 1 { if !text[pos..].ends_with(part) { return false; } }
        else { match text[pos..].find(part) { Some(idx) => pos += idx + part.len(), None => return false } }
    }
    true
}

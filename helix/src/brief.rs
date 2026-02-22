//! One-shot briefing: reconstruct + compress + format.
//! Merges reconstruct.rs + compress.rs + briefing.rs.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::Path;
use crate::fxhash::FxHashSet;

// ══════════ Data Types ══════════

struct RawEntry {
    topic: String, body: String, timestamp_min: i32, days_old: i64,
    tags: Vec<String>, relevance: f64, confidence: f64, link_in: u16,
}

struct Compressed {
    topic: String, body: String, date: String, days_old: i64,
    tags: Vec<String>, relevance: f64, source: Option<String>,
    chain: Option<String>, also_in: Vec<String>, confidence: f64, link_in: u16,
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
            else if topic.contains(q_sanitized.as_str()) { primary_set.insert(topic); }
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
            let mut relevance = if is_primary { 10.0 } else { 0.0 };
            for t in &q_terms { relevance += *e.tf_map.get(t).unwrap_or(&0) as f64; }
            if !e.has_tag("invariant") && !e.has_tag("architecture") {
                relevance *= 1.0 + 1.0 / (1.0 + days_old as f64 / 7.0);
            }
            relevance *= e.confidence();
            entries.push(RawEntry {
                topic: e.topic.to_string(), body: e.body.clone(),
                timestamp_min: e.timestamp_min, days_old,
                tags: e.tags().to_vec(), relevance, confidence: e.confidence(), link_in: 0,
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
        format_output(&compressed, query, raw_count, &primary, d, since_hours, focus_cats.as_deref())
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
            chain: None, also_in: Vec::new(), confidence: e.confidence, link_in: e.link_in,
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

fn first_content(body: &str) -> &str {
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
        let steps: Vec<String> = sorted.iter().map(|&i| {
            let fc = first_content(&entries[i].body);
            let without = fc.replace(term.as_str(), "");
            let words: Vec<&str> = without.split_whitespace().take(5).collect();
            let step = words.join(" ");
            if step.is_empty() { entries[i].date[5..].to_string() }
            else { format!("{} ({})", step, &entries[i].date[5..]) }
        }).collect();
        let newest = *sorted.last().unwrap();
        entries[newest].chain = Some(format!("{}: {}", term, steps.join(" → ")));
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

// ══════════ Categories + Classification ══════════

const CATEGORIES: &[(&str, &[&str])] = &[
    ("ARCHITECTURE", &["architecture", "module-map", "overview"]),
    ("DATA FLOW", &["pipeline", "data-flow"]),
    ("INVARIANTS", &["invariant", "constraint"]),
    ("GOTCHAS", &["gotcha"]),
    ("DECISIONS", &["decision"]),
    ("HOW-TO", &["how-to", "workflow"]),
    ("PERFORMANCE", &["performance", "zero-alloc"]),
    ("GAPS", &["gap", "missing"]),
];

struct Classification {
    categories: Vec<(&'static str, Vec<usize>)>,
    untagged: Vec<usize>,
}

fn classify(entries: &[Compressed]) -> Classification {
    let fc_lower: Vec<String> = entries.iter().map(|e| first_content(&e.body).to_lowercase()).collect();
    let mut assigned = vec![false; entries.len()];
    let mut categories: Vec<(&'static str, Vec<usize>)> = Vec::new();
    for &(cat, patterns) in CATEGORIES {
        let mut group = Vec::new();
        for (i, e) in entries.iter().enumerate() {
            if assigned[i] { continue; }
            let tag_match = e.tags.iter().any(|t| patterns.contains(&t.as_str()));
            let kw_match = patterns.iter().any(|p| fc_lower[i].contains(p));
            if tag_match || kw_match { group.push(i); assigned[i] = true; }
        }
        if !group.is_empty() { categories.push((cat, group)); }
    }
    let untagged: Vec<usize> = (0..entries.len()).filter(|i| !assigned[*i]).collect();
    Classification { categories, untagged }
}

// ══════════ Formatting ══════════

fn format_output(entries: &[Compressed], query: &str, raw_count: usize,
                 primary: &[String], detail: Detail, since: Option<u64>,
                 focus: Option<&[String]>) -> String {
    let cls = classify(entries);
    let n_topics = entries.iter().map(|e| e.topic.as_str()).collect::<BTreeSet<_>>().len();
    let mut out = String::new();
    let since_note = since.map(|h| format!(" (since {}h)", h)).unwrap_or_default();
    let _ = writeln!(out, "=== {}{} === {} → {} compressed, {} topics\n",
        query.to_uppercase(), since_note, raw_count, entries.len(), n_topics);

    // Topics line
    let mut info: BTreeMap<&str, (usize, i64)> = BTreeMap::new();
    for e in entries {
        let (c, d) = info.entry(&e.topic).or_insert((0, i64::MAX));
        *c += 1; if e.days_old < *d { *d = e.days_old; }
    }
    let _ = write!(out, "TOPICS:");
    for t in primary {
        if let Some((c, d)) = info.get(t.as_str()) {
            let fresh = match *d { 0 => ", today", 1 => ", 1d", 2..=7 => ", week", _ => "" };
            let _ = write!(out, " {} ({}{})", t, c, fresh);
        }
    }
    let _ = writeln!(out, "\n");

    match detail {
        Detail::Summary => {
            // Category distribution
            let _ = write!(out, "CATEGORIES:");
            for (cat, indices) in &cls.categories { let _ = write!(out, " {} {}", cat, indices.len()); }
            if !cls.untagged.is_empty() { let _ = write!(out, " | UNTAGGED {}", cls.untagged.len()); }
            let _ = writeln!(out, "\n");
            // Hot top 5
            let mut hot: Vec<usize> = (0..entries.len()).collect();
            hot.sort_by(|&a, &b| entries[b].relevance.partial_cmp(&entries[a].relevance).unwrap_or(std::cmp::Ordering::Equal));
            let _ = writeln!(out, "HOT:");
            for &i in hot.iter().take(5) { format_oneliner(&mut out, &entries[i]); }
        }
        Detail::Scan => {
            for (cat, indices) in &cls.categories {
                if !cat_matches(cat, focus) { continue; }
                let _ = writeln!(out, "--- {} ({}) ---", cat, indices.len());
                for &i in indices.iter().take(3) { format_oneliner(&mut out, &entries[i]); }
                if indices.len() > 3 { let _ = writeln!(out, "  ... +{} more", indices.len() - 3); }
                let _ = writeln!(out);
            }
            if !cls.untagged.is_empty() && cat_matches("UNTAGGED", focus) {
                let _ = writeln!(out, "--- UNTAGGED ({}) ---", cls.untagged.len());
                for &i in cls.untagged.iter().take(3) { format_oneliner(&mut out, &entries[i]); }
                let _ = writeln!(out);
            }
        }
        Detail::Full => {
            for (cat, indices) in &cls.categories {
                if !cat_matches(cat, focus) { continue; }
                let _ = writeln!(out, "--- {} ({}) ---", cat, indices.len());
                let body_limit = match *cat {
                    "DATA FLOW" | "INVARIANTS" => 10,
                    "DECISIONS" | "ARCHITECTURE" => 8,
                    _ => 5,
                };
                for &i in indices.iter().take(5) { format_entry(&mut out, &entries[i], body_limit); }
                for &i in indices.iter().skip(5).take(10) { format_oneliner(&mut out, &entries[i]); }
                if indices.len() > 15 { let _ = writeln!(out, "  ... +{} more\n", indices.len() - 15); }
            }
            if !cls.untagged.is_empty() && cat_matches("UNTAGGED", focus) {
                let _ = writeln!(out, "--- UNTAGGED ({}) ---", cls.untagged.len());
                for &i in cls.untagged.iter().take(5) { format_oneliner(&mut out, &entries[i]); }
                let _ = writeln!(out);
            }
        }
    }

    let pct = if raw_count > 0 { 100 - (entries.len() * 100 / raw_count) } else { 0 };
    let _ = writeln!(out, "\nSTATS: {} compressed ({}% reduction)", entries.len(), pct);
    out
}

fn format_entry(out: &mut String, e: &Compressed, max_lines: usize) {
    let src = e.source.as_deref().map(|s| format!(" → {s}")).unwrap_or_default();
    let fresh = freshness(e.days_old);
    let _ = writeln!(out, "[{}] {}{}{}", e.topic, e.date, fresh, src);
    if let Some(ref chain) = e.chain {
        let _ = writeln!(out, "  {}", crate::text::truncate(chain, 120));
    }
    let lines: Vec<&str> = e.body.lines().filter(|l| !crate::text::is_metadata_line(l)).collect();
    for l in lines.iter().take(max_lines) { let _ = writeln!(out, "  {}", l.trim()); }
    if lines.len() > max_lines { let _ = writeln!(out, "  ...({} more lines)", lines.len() - max_lines); }
    let _ = writeln!(out);
}

fn format_oneliner(out: &mut String, e: &Compressed) {
    let fc = crate::text::truncate(first_content(&e.body), 80);
    let src = e.source.as_deref().map(|s| format!(" → {s}")).unwrap_or_default();
    let chain = match &e.chain {
        Some(c) if c.starts_with("superseded") => " [SUPERSEDED]".to_string(),
        Some(c) => format!(" ({})", crate::text::truncate(c, 60)),
        None => String::new(),
    };
    let _ = writeln!(out, "  [{}] {}{}{}{}", e.topic, fc, src, chain, freshness(e.days_old));
}

fn freshness(days: i64) -> &'static str {
    match days { 0 => " [TODAY]", 1 => " [1d]", 2..=7 => " [week]", _ => "" }
}

fn cat_matches(cat: &str, focus: Option<&[String]>) -> bool {
    match focus {
        None => true,
        Some(cats) => {
            let up = cat.to_uppercase();
            cats.iter().any(|f| up.contains(f.as_str()) || f.contains(&up))
        }
    }
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

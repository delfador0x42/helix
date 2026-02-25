//! Unified search scoring: index-first with cache fallback. BM25 + diversity.

use std::path::Path;
use crate::fxhash::{FxHashMap, FxHashSet};

pub struct ScoredResult {
    pub name: String,
    pub lines: Vec<String>,
    pub score: f64,
}

#[derive(Clone, Copy, PartialEq)]
pub enum SearchMode { And, Or }

pub struct Filter {
    pub after: Option<i64>,
    pub before: Option<i64>,
    pub tag: Option<String>,
    pub topic: Option<String>,
    pub source: Option<String>,
    pub mode: SearchMode,
}

impl Filter {
    pub fn none() -> Self { Self { after: None, before: None, tag: None, topic: None, source: None, mode: SearchMode::And } }
}

/// Unified search: index-first with cache fallback. AND→OR auto-fallback.
pub fn search_scored(dir: &Path, terms: &[String], filter: &Filter, limit: Option<usize>,
                     index_data: Option<&[u8]>, full_body: bool)
    -> Result<(Vec<ScoredResult>, bool), String>
{
    if terms.is_empty() {
        return score_on_cache(dir, terms, filter, limit);
    }
    let fallback_data;
    let data = match index_data {
        Some(d) => Some(d),
        None => { fallback_data = std::fs::read(dir.join("index.bin")).ok(); fallback_data.as_deref() }
    };
    if let Some(data) = data {
        let tag_ok = filter.tag.as_ref().map_or(true, |t| crate::index_read::resolve_tag(data, t).is_some());
        if tag_ok {
            if let Ok(result) = score_via_index(dir, data, terms, filter, limit, full_body) {
                return Ok(result);
            }
        }
    }
    score_on_cache(dir, terms, filter, limit)
}

fn build_filter_pred(data: &[u8], filter: &Filter) -> crate::index::FilterPred {
    let topic_filter = match filter.topic.as_ref() {
        None => crate::index::TopicFilter::Any,
        Some(name) => {
            if let Some(id) = crate::index_read::resolve_topic(data, name) {
                crate::index::TopicFilter::Exact(id)
            } else {
                let ids: Vec<u16> = crate::index_read::topic_table(data).unwrap_or_default().iter()
                    .filter(|(_, n, _)| crate::config::topic_matches_query(n, name))
                    .map(|(id, _, _)| *id)
                    .collect();
                if ids.is_empty() { crate::index::TopicFilter::Any }
                else if ids.len() == 1 { crate::index::TopicFilter::Exact(ids[0]) }
                else { crate::index::TopicFilter::Prefix(ids) }
            }
        }
    };
    crate::index::FilterPred {
        topic_filter,
        after_days: filter.after.map(|d| d.max(0) as u16).unwrap_or(0),
        before_days: filter.before.map(|d| d.min(u16::MAX as i64) as u16).unwrap_or(u16::MAX),
        tag_mask: filter.tag.as_ref().and_then(|t| crate::index_read::resolve_tag(data, t)).map(|b| 1u32 << b).unwrap_or(0),
        source_needle: filter.source.as_ref().map(|s| s.as_bytes().to_vec()),
    }
}

fn score_via_index(dir: &Path, data: &[u8], terms: &[String], filter: &Filter,
                   limit: Option<usize>, full_body: bool) -> Result<(Vec<ScoredResult>, bool), String>
{
    let pred = build_filter_pred(data, filter);
    let cap = limit.unwrap_or(20);
    let query = terms.join(" ");
    let hits = crate::index::search_index(data, &query, &pred, cap, true)?;
    if hits.is_empty() && filter.mode == SearchMode::And && terms.len() >= 2 {
        let or_hits = crate::index::search_index(data, &query, &pred, cap, false)?;
        if !or_hits.is_empty() { return hydrate_hits(dir, data, terms, &or_hits, true, full_body); }
        return Ok((Vec::new(), false));
    }
    hydrate_hits(dir, data, terms, &hits, false, full_body)
}

fn hydrate_hits(dir: &Path, data: &[u8], terms: &[String], hits: &[crate::index::SearchHit],
                fallback: bool, full_body: bool) -> Result<(Vec<ScoredResult>, bool), String>
{
    if hits.is_empty() { return Ok((Vec::new(), false)); }
    let mut name_cache: FxHashMap<u16, String> = FxHashMap::default();
    let mut log_file = if full_body {
        Some(std::fs::File::open(crate::config::log_path(dir)).map_err(|e| format!("open data.log: {e}"))?)
    } else { None };
    let mut results = Vec::with_capacity(hits.len());
    for hit in hits {
        use std::collections::hash_map::Entry;
        let topic_ref = match name_cache.entry(hit.topic_id) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => match crate::index_read::topic_name(data, hit.topic_id) {
                Ok(n) => e.insert(n), Err(_) => continue,
            },
        };
        let mut score = hit.score;
        if terms.iter().any(|t| topic_ref.contains(t.as_str())) { score *= 1.5; }
        if full_body {
            let entry = crate::datalog::read_entry_from(log_file.as_mut().unwrap(), hit.log_offset)
                .unwrap_or(crate::datalog::LogEntry {
                    offset: hit.log_offset, topic: topic_ref.clone(),
                    body: String::new(), timestamp_min: hit.date_minutes,
                });
            for line in entry.body.lines() {
                if line.starts_with("[tags: ") {
                    let tag_hits = terms.iter().filter(|t| line.contains(t.as_str())).count();
                    if tag_hits > 0 { score *= 1.0 + 0.3 * tag_hits as f64; }
                    break;
                }
            }
            let date = crate::time::minutes_to_date_str(entry.timestamp_min);
            let mut lines = vec![format!("## {date}")];
            for line in entry.body.lines() { lines.push(line.to_string()); }
            results.push(ScoredResult { name: topic_ref.clone(), lines, score });
        } else {
            let tag_line = crate::index_read::reconstruct_tags(data, hit.entry_id).ok().flatten();
            if let Some(ref tl) = tag_line {
                let tag_hits = terms.iter().filter(|t| tl.contains(t.as_str())).count();
                if tag_hits > 0 { score *= 1.0 + 0.3 * tag_hits as f64; }
            }
            let date = crate::time::minutes_to_date_str(hit.date_minutes);
            let mut lines = vec![format!("## {date}")];
            if let Some(tl) = tag_line { lines.push(tl); }
            let prefix = format!("[{}] {} ", topic_ref, date);
            let content = hit.snippet.strip_prefix(&prefix).unwrap_or(&hit.snippet);
            if !content.is_empty() { lines.push(content.to_string()); }
            results.push(ScoredResult { name: topic_ref.clone(), lines, score });
        }
    }
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok((results, fallback))
}

// ── Cache Scoring (fallback) ──

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

#[inline]
pub fn matches_tokens(tf_map: &FxHashMap<String, usize>, terms: &[String], mode: SearchMode) -> bool {
    if terms.is_empty() { return true; }
    match mode {
        SearchMode::And => terms.iter().all(|t| tf_map.contains_key(t)),
        SearchMode::Or => terms.iter().any(|t| tf_map.contains_key(t)),
    }
}

fn passes_filter(e: &crate::cache::CachedEntry, f: &Filter) -> bool {
    if f.after.is_some() || f.before.is_some() {
        let days = e.day();
        if let Some(after) = f.after { if days < after { return false; } }
        if let Some(before) = f.before { if days > before { return false; } }
    }
    if let Some(ref tag) = f.tag { if !e.has_tag(tag) { return false; } }
    true
}

fn score_on_cache(dir: &Path, terms: &[String], filter: &Filter, limit: Option<usize>)
    -> Result<(Vec<ScoredResult>, bool), String>
{
    crate::cache::with_corpus(dir, |cached| {
        let filtered: Vec<&crate::cache::CachedEntry> = cached.iter()
            .filter(|e| {
                if let Some(ref t) = filter.topic {
                    if !crate::config::topic_matches_query(e.topic.as_str(), t) { return false; }
                }
                passes_filter(e, filter)
            }).collect();
        let n = filtered.len() as f64;
        let avgdl = if filtered.is_empty() { 1.0 } else {
            filtered.iter().map(|e| e.word_count).sum::<usize>() as f64 / n
        };
        let mut dfs = vec![0usize; terms.len()];
        for e in &filtered {
            for (i, t) in terms.iter().enumerate() { if e.tf_map.contains_key(t) { dfs[i] += 1; } }
        }
        let cap = limit.unwrap_or(filtered.len());
        let score_mode = |mode: SearchMode| -> Vec<ScoredResult> {
            let mut scored: Vec<(f64, usize)> = filtered.iter().enumerate()
                .filter(|(_, e)| matches_tokens(&e.tf_map, terms, mode))
                .filter_map(|(idx, e)| {
                    let len_norm = 1.0 - BM25_B + BM25_B * e.word_count as f64 / avgdl.max(1.0);
                    let mut score = 0.0;
                    for (i, term) in terms.iter().enumerate() {
                        let tf = *e.tf_map.get(term).unwrap_or(&0) as f64;
                        if tf == 0.0 { continue; }
                        let idf = ((n - dfs[i] as f64 + 0.5) / (dfs[i] as f64 + 0.5) + 1.0).ln();
                        score += idf * (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * len_norm);
                    }
                    if score == 0.0 { return None; }
                    if terms.iter().any(|t| e.topic.contains(t.as_str())) { score *= 1.5; }
                    let tag_hits = terms.iter().filter(|t| e.tags().iter().any(|tag| tag.contains(t.as_str()))).count();
                    if tag_hits > 0 { score *= 1.0 + 0.3 * tag_hits as f64; }
                    Some((score, idx))
                }).collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(cap);
            scored.iter().map(|&(score, idx)| {
                let e = filtered[idx];
                let mut lines = vec![format!("## {}", e.date_str())];
                for line in e.body.lines() { lines.push(line.to_string()); }
                ScoredResult { name: e.topic.to_string(), lines, score }
            }).collect()
        };
        let mut results = score_mode(filter.mode);
        let mut fb = false;
        if results.is_empty() && filter.mode == SearchMode::And && terms.len() >= 2 {
            results = score_mode(SearchMode::Or);
            fb = !results.is_empty();
        }
        (results, fb)
    })
}

pub fn topic_matches(dir: &Path, terms: &[String], filter: &Filter)
    -> Result<(Vec<(String, usize)>, bool), String>
{
    crate::cache::with_corpus(dir, |cached| {
        let count_fn = |mode: SearchMode| -> Vec<(String, usize)> {
            let mut hits: FxHashMap<&str, usize> = FxHashMap::default();
            for e in cached {
                if let Some(ref t) = filter.topic {
                    if !crate::config::topic_matches_query(e.topic.as_str(), t) { continue; }
                }
                if !passes_filter(e, filter) { continue; }
                if matches_tokens(&e.tf_map, terms, mode) { *hits.entry(&e.topic).or_insert(0) += 1; }
            }
            hits.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
        };
        let mut hits = count_fn(filter.mode);
        let mut fb = false;
        if hits.is_empty() && filter.mode == SearchMode::And && terms.len() >= 2 {
            hits = count_fn(SearchMode::Or);
            fb = !hits.is_empty();
        }
        (hits, fb)
    })
}

pub fn count_matches(dir: &Path, terms: &[String], filter: &Filter)
    -> Result<(usize, usize, bool), String>
{
    crate::cache::with_corpus(dir, |cached| {
        let do_count = |mode: SearchMode| -> (usize, usize) {
            let mut total = 0;
            let mut topics: FxHashSet<&str> = FxHashSet::default();
            for e in cached {
                if let Some(ref t) = filter.topic {
                    if !crate::config::topic_matches_query(e.topic.as_str(), t) { continue; }
                }
                if !passes_filter(e, filter) { continue; }
                if matches_tokens(&e.tf_map, terms, mode) { total += 1; topics.insert(&e.topic); }
            }
            (total, topics.len())
        };
        let (total, topics) = do_count(filter.mode);
        if total > 0 { return (total, topics, false); }
        if filter.mode == SearchMode::And && terms.len() >= 2 {
            let (t, tp) = do_count(SearchMode::Or);
            return (t, tp, t > 0);
        }
        (0, 0, false)
    })
}

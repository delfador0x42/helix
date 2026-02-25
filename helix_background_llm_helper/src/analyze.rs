//! Three targeted analysis functions for the background daemon.
//! Each builds a specific prompt and returns it for the daemon to run.

use crate::datalog;
use std::collections::HashMap;

/// Session synthesis: what changed, connections, contradictions.
pub fn session_synthesis_prompt(
    entries: &[datalog::Entry], last_seen: u32,
) -> Option<String> {
    let new: Vec<&datalog::Entry> = entries.iter()
        .filter(|e| e.offset > last_seen).collect();
    if new.is_empty() { return None; }

    // Sample older entries for context (up to 10)
    let old: Vec<&datalog::Entry> = entries.iter()
        .filter(|e| e.offset <= last_seen).rev().take(10).collect();

    let mut out = String::from("New entries since last session:\n\n");
    let mut budget = 4000usize;
    for e in &new {
        let c = e.content();
        let snip = &c[..c.len().min(250)];
        out.push_str(&format!("## [NEW] [{}] {}\n\n", e.topic, snip));
        budget = budget.saturating_sub(snip.len() + e.topic.len() + 20);
        if budget == 0 { break; }
    }
    if !old.is_empty() {
        out.push_str("\nPrior context (sample of existing entries):\n\n");
        for e in &old {
            let c = e.content();
            let snip = &c[..c.len().min(150)];
            out.push_str(&format!("## [OLD] [{}] {}\n\n", e.topic, snip));
            budget = budget.saturating_sub(snip.len() + e.topic.len() + 20);
            if budget == 0 { break; }
        }
    }
    out.push_str("\nSynthesize: what changed, connections between new and old, contradictions, key takeaways.");
    Some(out)
}

pub const SYNTHESIS_SYSTEM: &str = "\
You are a knowledge base analyst. Given new entries and prior context, \
identify: what changed, connections between new and existing knowledge, \
contradictions, and key takeaways. Be specific and concise.";

/// Drift detection: find topics with overlapping content under different names.
pub fn drift_prompt(entries: &[datalog::Entry]) -> Option<String> {
    // Group entries by topic, take a content sample from each
    let mut by_topic: HashMap<&str, Vec<&str>> = HashMap::new();
    for e in entries {
        by_topic.entry(&e.topic).or_default().push(&e.body);
    }
    if by_topic.len() < 2 { return None; }

    let mut out = String::from("Topics and sample content:\n\n");
    let mut budget = 5000usize;
    let mut topics: Vec<_> = by_topic.iter().collect();
    topics.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    for (&topic, bodies) in &topics {
        let sample = bodies.first().map(|b| &b[..b.len().min(200)]).unwrap_or("");
        out.push_str(&format!("## {} ({} entries)\n{}\n\n", topic, bodies.len(), sample));
        budget = budget.saturating_sub(sample.len() + topic.len() + 30);
        if budget == 0 { break; }
    }
    out.push_str("\nIdentify topics that overlap in content but use different names. \
                   Suggest merges. Flag topics that should be split.");
    Some(out)
}

pub const DRIFT_SYSTEM: &str = "\
You are a knowledge base curator. Analyze topic names and their content samples. \
Find: topics with overlapping content under different names (suggest merges), \
topics that are too broad (suggest splits), naming inconsistencies. Be specific.";

/// Stale detection: find old entries contradicted by newer ones.
pub fn stale_prompt(
    entries: &[datalog::Entry], topic_filter: Option<&str>,
) -> Option<String> {
    let filtered: Vec<&datalog::Entry> = if let Some(f) = topic_filter {
        let fl = f.to_lowercase();
        entries.iter().filter(|e| e.topic.to_lowercase().contains(&fl)).collect()
    } else {
        // Top 5 topics by entry count
        let mut tc: HashMap<&str, usize> = HashMap::new();
        for e in entries { *tc.entry(&e.topic).or_default() += 1; }
        let mut top: Vec<_> = tc.into_iter().collect();
        top.sort_by(|a, b| b.1.cmp(&a.1));
        top.truncate(5);
        let ts: std::collections::HashSet<&str> = top.iter().map(|&(t, _)| t).collect();
        entries.iter().filter(|e| ts.contains(e.topic.as_str())).collect()
    };
    if filtered.len() < 2 { return None; }

    // Sort by timestamp (oldest first) to show evolution
    let mut sorted: Vec<&&datalog::Entry> = filtered.iter().collect();
    sorted.sort_by_key(|e| e.timestamp_min);

    let mut out = String::from("Entries sorted oldest to newest:\n\n");
    let mut budget = 5000usize;
    for e in &sorted {
        let c = e.content();
        let snip = &c[..c.len().min(200)];
        out.push_str(&format!("## [ts={}] [{}] {}\n\n", e.timestamp_min, e.topic, snip));
        budget = budget.saturating_sub(snip.len() + e.topic.len() + 30);
        if budget == 0 { break; }
    }
    out.push_str("\nIdentify entries that are contradicted or superseded by newer ones. \
                   Flag specific stale entries by their topic and content.");
    Some(out)
}

pub const STALE_SYSTEM: &str = "\
You are a knowledge base auditor. Review entries from oldest to newest. \
Find: entries contradicted by newer ones, outdated information, \
entries that should be updated or deleted. Cite specific entries.";

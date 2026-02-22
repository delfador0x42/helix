//! Analysis passes: find topic overlaps, cross-links, duplicates, gaps.
//! Reads entries + cached embeddings, produces human-readable findings.

use std::collections::HashMap;
use crate::{cache::EmbeddingCache, datalog::Entry, embed::EMBED_DIM, similarity};

pub struct Finding {
    pub kind: &'static str,
    pub description: String,
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "[{}] {}", self.kind, self.description)
    }
}

pub fn analyze(entries: &[Entry], cache: &EmbeddingCache) -> Vec<Finding> {
    let mut findings = Vec::new();
    findings.extend(find_topic_overlaps(entries, cache));
    findings.extend(find_near_duplicates(entries, cache));
    findings.extend(find_cross_links(entries, cache));
    findings
}

/// Find topics with high average pairwise similarity (>0.6 = likely overlapping).
fn find_topic_overlaps(entries: &[Entry], cache: &EmbeddingCache) -> Vec<Finding> {
    // Pre-group embeddings by topic (avoids O(topics × entries) filtering)
    let mut by_topic: HashMap<&str, Vec<&[f32]>> = HashMap::new();
    for e in entries {
        if let Some(emb) = cache.get(e.offset) {
            by_topic.entry(e.topic.as_str()).or_default().push(emb);
        }
    }
    let topics: Vec<&str> = by_topic.keys().copied().collect();
    let mut findings = Vec::new();
    for i in 0..topics.len() {
        for j in (i + 1)..topics.len() {
            let a_embs = &by_topic[topics[i]];
            let b_embs = &by_topic[topics[j]];
            let ca = similarity::centroid(a_embs, EMBED_DIM);
            let cb = similarity::centroid(b_embs, EMBED_DIM);
            let sim = similarity::cosine(&ca, &cb);
            if sim > 0.6 {
                findings.push((sim, Finding {
                    kind: "topic-overlap",
                    description: format!(
                        "{} ({} entries) and {} ({} entries): centroid sim {:.3} — consider merging",
                        topics[i], a_embs.len(), topics[j], b_embs.len(), sim
                    ),
                }));
            }
        }
    }
    findings.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    findings.into_iter().map(|(_, f)| f).collect()
}

/// Find entries with >0.92 cosine similarity (near-duplicates).
fn find_near_duplicates(entries: &[Entry], cache: &EmbeddingCache) -> Vec<Finding> {
    let embedded: Vec<(u32, &[f32])> = entries.iter()
        .filter_map(|e| cache.get(e.offset).map(|emb| (e.offset, emb))).collect();
    let pairs = similarity::top_k_pairs(&embedded, 20, 0.92);
    // O(1) offset lookup instead of O(entries) linear scan per pair
    let idx: HashMap<u32, &Entry> = entries.iter().map(|e| (e.offset, e)).collect();
    pairs.iter().filter_map(|&(a, b, sim)| {
        let ea = idx.get(&a)?;
        let eb = idx.get(&b)?;
        let ca = ea.content();
        let cb = eb.content();
        Some(Finding {
            kind: "duplicate",
            description: format!(
                "sim {:.3}: [{}] '{}...' vs [{}] '{}...'",
                sim, ea.topic, &ca[..ca.len().min(50)],
                eb.topic, &cb[..cb.len().min(50)]
            ),
        })
    }).collect()
}

/// Find high-similarity entries in different topics (candidates for cross-linking).
fn find_cross_links(entries: &[Entry], cache: &EmbeddingCache) -> Vec<Finding> {
    let embedded: Vec<(u32, &[f32])> = entries.iter()
        .filter_map(|e| cache.get(e.offset).map(|emb| (e.offset, emb))).collect();
    let pairs = similarity::top_k_pairs(&embedded, 50, 0.75);
    let idx: HashMap<u32, &Entry> = entries.iter().map(|e| (e.offset, e)).collect();
    pairs.iter().filter_map(|&(a, b, sim)| {
        let ea = idx.get(&a)?;
        let eb = idx.get(&b)?;
        if ea.topic == eb.topic { return None; }
        if sim > 0.92 { return None; } // already caught as duplicate
        let ca = ea.content();
        let cb = eb.content();
        Some(Finding {
            kind: "cross-link",
            description: format!(
                "sim {:.3}: [{}] '{}...' ↔ [{}] '{}...'",
                sim, ea.topic, &ca[..ca.len().min(40)],
                eb.topic, &cb[..cb.len().min(40)]
            ),
        })
    }).collect()
}

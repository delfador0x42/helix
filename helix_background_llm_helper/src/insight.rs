//! Analysis passes: find topic overlaps, cross-links, duplicates, gaps.
//! Reads entries + cached embeddings, produces human-readable findings.

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
    let topics: Vec<String> = {
        let mut t: Vec<String> = entries.iter().map(|e| e.topic.clone()).collect();
        t.sort_unstable(); t.dedup(); t
    };
    let mut findings = Vec::new();
    for i in 0..topics.len() {
        for j in (i + 1)..topics.len() {
            let a_embs: Vec<&[f32]> = entries.iter()
                .filter(|e| e.topic == topics[i])
                .filter_map(|e| cache.get(e.offset)).collect();
            let b_embs: Vec<&[f32]> = entries.iter()
                .filter(|e| e.topic == topics[j])
                .filter_map(|e| cache.get(e.offset)).collect();
            if a_embs.is_empty() || b_embs.is_empty() { continue; }
            let ca = similarity::centroid(&a_embs, EMBED_DIM);
            let cb = similarity::centroid(&b_embs, EMBED_DIM);
            let sim = similarity::cosine(&ca, &cb);
            if sim > 0.6 {
                findings.push(Finding {
                    kind: "topic-overlap",
                    description: format!(
                        "{} ({} entries) and {} ({} entries): centroid sim {:.3} — consider merging",
                        topics[i], a_embs.len(), topics[j], b_embs.len(), sim
                    ),
                });
            }
        }
    }
    findings.sort_by(|a, b| b.description.len().cmp(&a.description.len())); // rough sort by relevance
    findings
}

/// Find entries with >0.92 cosine similarity (near-duplicates).
fn find_near_duplicates(entries: &[Entry], cache: &EmbeddingCache) -> Vec<Finding> {
    let embedded: Vec<(u32, &[f32])> = entries.iter()
        .filter_map(|e| cache.get(e.offset).map(|emb| (e.offset, emb))).collect();
    let pairs = similarity::top_k_pairs(&embedded, 20, 0.92);
    pairs.iter().filter_map(|&(a, b, sim)| {
        let ea = entries.iter().find(|e| e.offset == a)?;
        let eb = entries.iter().find(|e| e.offset == b)?;
        Some(Finding {
            kind: "duplicate",
            description: format!(
                "sim {:.3}: [{}] '{}...' vs [{}] '{}...'",
                sim, ea.topic, &ea.content()[..ea.content().len().min(50)],
                eb.topic, &eb.content()[..eb.content().len().min(50)]
            ),
        })
    }).collect()
}

/// Find high-similarity entries in different topics (candidates for cross-linking).
fn find_cross_links(entries: &[Entry], cache: &EmbeddingCache) -> Vec<Finding> {
    let embedded: Vec<(u32, &[f32])> = entries.iter()
        .filter_map(|e| cache.get(e.offset).map(|emb| (e.offset, emb))).collect();
    let pairs = similarity::top_k_pairs(&embedded, 50, 0.75);
    pairs.iter().filter_map(|&(a, b, sim)| {
        let ea = entries.iter().find(|e| e.offset == a)?;
        let eb = entries.iter().find(|e| e.offset == b)?;
        if ea.topic == eb.topic { return None; } // same topic = not a cross-link
        if sim > 0.92 { return None; } // already caught as duplicate
        Some(Finding {
            kind: "cross-link",
            description: format!(
                "sim {:.3}: [{}] '{}...' ↔ [{}] '{}...'",
                sim, ea.topic, &ea.content()[..ea.content().len().min(40)],
                eb.topic, &eb.content()[..eb.content().len().min(40)]
            ),
        })
    }).collect()
}

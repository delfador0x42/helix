//! Vector similarity math. Embeddings are L2-normalized, so cosine = dot product.

/// Cosine similarity between two L2-normalized vectors (= dot product).
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Find top-k most similar pairs across all entries. Returns (idx_a, idx_b, similarity).
pub fn top_k_pairs(embeddings: &[(u32, &[f32])], k: usize, min_sim: f32) -> Vec<(u32, u32, f32)> {
    let n = embeddings.len();
    let mut pairs: Vec<(u32, u32, f32)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine(embeddings[i].1, embeddings[j].1);
            if sim >= min_sim {
                pairs.push((embeddings[i].0, embeddings[j].0, sim));
            }
        }
    }
    pairs.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(k);
    pairs
}

/// Average embedding for a group of vectors.
pub fn centroid(embeddings: &[&[f32]], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];
    if embeddings.is_empty() { return result; }
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() { result[i] += v; }
    }
    let n = embeddings.len() as f32;
    for v in &mut result { *v /= n; }
    result
}

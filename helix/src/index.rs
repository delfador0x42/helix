//! Inverted index: core BM25 search over the binary index format.
//! Builder in index_build.rs, readers in index_read.rs, scoring in score.rs.

use std::sync::Mutex;
use crate::format::*;

// Re-exports so callers can keep using crate::index::*
pub use crate::index_build::rebuild;
pub use crate::index_read::{topic_table, topic_recency, topic_name, resolve_topic,
    xref_edges, entry_snippet_ref, entry_topic_id, sourced_entries, index_info};
pub use crate::score::{ScoredResult, SearchMode, Filter, search_scored,
    topic_matches, count_matches};

// ── Filter + Query State ──

pub enum TopicFilter {
    Any,
    Exact(u16),
    Prefix(Vec<u16>),
}

pub struct FilterPred {
    pub topic_filter: TopicFilter,
    pub after_days: u16,
    pub before_days: u16,
    pub tag_mask: u32,
    pub source_needle: Option<Vec<u8>>,
}

impl FilterPred {
    pub fn none() -> Self {
        Self { topic_filter: TopicFilter::Any, after_days: 0, before_days: u16::MAX, tag_mask: 0, source_needle: None }
    }
    fn passes(&self, m: &EntryMeta, data: &[u8], src_pool: usize) -> bool {
        match &self.topic_filter {
            TopicFilter::Any => {}
            TopicFilter::Exact(t) => { if { m.topic_id } != *t { return false; } }
            TopicFilter::Prefix(ids) => { if !ids.contains(&{ m.topic_id }) { return false; } }
        }
        let ed = { m.epoch_days };
        if ed < self.after_days || (self.before_days < u16::MAX && ed > self.before_days) { return false; }
        if self.tag_mask != 0 && ({ m.tag_bitmap } & self.tag_mask) != self.tag_mask { return false; }
        if let Some(ref needle) = self.source_needle {
            let sl = { m.source_len } as usize;
            if sl == 0 { return false; }
            let so = src_pool + { m.source_off } as usize;
            if so + sl > data.len() || needle.len() > sl { return false; }
            let hay = &data[so..so + sl];
            if !hay.windows(needle.len()).any(|w| w.eq_ignore_ascii_case(needle)) { return false; }
        }
        true
    }
}

struct QueryState {
    generation: u32,
    entry_gen: Vec<u32>,
    scores: Vec<f64>,
    hit_count: Vec<u16>,
}

impl QueryState {
    fn ensure(&mut self, n: usize) {
        if self.entry_gen.len() < n {
            self.entry_gen.resize(n, 0);
            self.scores.resize(n, 0.0);
            self.hit_count.resize(n, 0);
        }
    }
    fn advance(&mut self) -> u32 {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 { self.generation = 1; }
        self.generation
    }
}

static QUERY_STATE: Mutex<QueryState> = Mutex::new(QueryState {
    generation: 0, entry_gen: Vec::new(), scores: Vec::new(), hit_count: Vec::new(),
});

// ── Core Search ──

pub struct SearchHit {
    pub entry_id: u32,
    pub topic_id: u16,
    pub score: f64,
    pub snippet: String,
    pub date_minutes: i32,
    pub log_offset: u32,
}

struct HeapHit {
    score: f64, entry_id: u32, topic_id: u16, date_minutes: i32,
    log_offset: u32, snippet_off: u32, snippet_len: u16,
}
impl PartialEq for HeapHit { fn eq(&self, o: &Self) -> bool { self.score.to_bits() == o.score.to_bits() } }
impl Eq for HeapHit {}
impl PartialOrd for HeapHit { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }
impl Ord for HeapHit { fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.score.partial_cmp(&o.score).unwrap_or(std::cmp::Ordering::Equal) } }

pub fn search_index(data: &[u8], query: &str, filter: &FilterPred, limit: usize, require_all: bool)
    -> Result<Vec<SearchHit>, String>
{
    let hdr = read_header(data)?;
    let terms = crate::text::query_terms(query);
    if terms.is_empty() { return Err("empty query".into()); }

    let num_entries = { hdr.num_entries } as usize;
    let table_cap = { hdr.table_cap } as usize;
    let avgdl = { hdr.avgdl_x100 } as f64 / 100.0;
    let post_off = { hdr.postings_off } as usize;
    let meta_off = { hdr.meta_off } as usize;
    let snip_off = { hdr.snippet_off } as usize;
    let src_pool = { hdr.source_off } as usize;
    let data_len = data.len();
    let meta_end = meta_off + num_entries * std::mem::size_of::<EntryMeta>();
    if post_off > data_len || meta_end > data_len || snip_off > data_len {
        return Err("index.bin truncated".into());
    }
    let mask = table_cap - 1;
    let num_terms = terms.len() as u16;
    let today_days = (crate::time::LocalTime::now().to_minutes() / 1440) as u16;

    let mut state_guard = QUERY_STATE.lock().map_err(|e| e.to_string())?;
    state_guard.ensure(num_entries);
    let gen = state_guard.advance();
    let state = &mut *state_guard;

    // Phase 1: BM25 scoring
    let mut any_hit = false;
    for term in &terms {
        let h = hash_term(term);
        let mut idx = (h as usize) & mask;
        for _ in 0..table_cap {
            let slot = read_slot(data, idx)?;
            let sh = { slot.hash };
            if sh == 0 { break; }
            if sh == h {
                let p_off = { slot.postings_off } as usize;
                let p_len = { slot.postings_len } as usize;
                let base = post_off + p_off * std::mem::size_of::<Posting>();
                let post_end = base + p_len * std::mem::size_of::<Posting>();
                if post_end > data_len { break; }
                let meta_size = std::mem::size_of::<EntryMeta>();
                for i in 0..p_len {
                    let p: Posting = unsafe { read_at_unchecked(data, base + i * std::mem::size_of::<Posting>()) };
                    let eid = { p.entry_id } as usize;
                    if eid >= num_entries { continue; }
                    let m: EntryMeta = unsafe { read_at_unchecked(data, meta_off + eid * meta_size) };
                    if !filter.passes(&m, data, src_pool) { continue; }
                    if state.entry_gen[eid] != gen {
                        state.scores[eid] = 0.0;
                        state.hit_count[eid] = 0;
                        state.entry_gen[eid] = gen;
                    }
                    let doc_len = { m.word_count } as f64;
                    let idf = { p.idf_x1000 } as f64 / 1000.0;
                    let tf = { p.tf } as f64;
                    let len_norm = 1.0 - 0.75 + 0.75 * doc_len / avgdl.max(1.0);
                    let tf_sat = (tf * 2.2) / (tf + 1.2 * len_norm);
                    let conf = { m.confidence } as f64 / 255.0;
                    let ed = { m.epoch_days };
                    let recency = if ed == 0 { 1.0 } else { 1.0 / (1.0 + today_days.saturating_sub(ed) as f64 / 30.0) };
                    state.scores[eid] += idf * tf_sat * conf * recency;
                    state.hit_count[eid] += 1;
                    any_hit = true;
                }
                break;
            }
            idx = (idx + 1) & mask;
        }
    }
    if !any_hit { return Ok(Vec::new()); }

    // Phase 2: Top-K with diversity cap
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;
    let mut heap: BinaryHeap<Reverse<HeapHit>> = BinaryHeap::with_capacity(limit + 1);
    let mut topic_counts = [0u8; 256];
    let diversity_cap: u8 = 3;

    for eid in 0..num_entries {
        if state.entry_gen[eid] != gen { continue; }
        if state.hit_count[eid] < if require_all { num_terms } else { 1 } { continue; }
        let score = state.scores[eid];
        if score <= 0.0 { continue; }
        let m = read_at::<EntryMeta>(data, meta_off + eid * std::mem::size_of::<EntryMeta>())?;
        let tid = { m.topic_id } as usize;
        if heap.len() >= limit && tid < 256 && topic_counts[tid] >= diversity_cap {
            if score <= heap.peek().map(|r| r.0.score).unwrap_or(0.0) * 1.5 { continue; }
        }
        let hit = HeapHit {
            score, entry_id: eid as u32, topic_id: { m.topic_id },
            date_minutes: { m.date_minutes }, log_offset: { m.log_offset },
            snippet_off: { m.snippet_off }, snippet_len: { m.snippet_len },
        };
        if heap.len() < limit {
            heap.push(Reverse(hit));
            if tid < 256 { topic_counts[tid] = topic_counts[tid].saturating_add(1); }
        } else if score > heap.peek().map(|r| r.0.score).unwrap_or(0.0) {
            let ev = heap.pop().unwrap().0;
            let etid = ev.topic_id as usize;
            if etid < 256 { topic_counts[etid] = topic_counts[etid].saturating_sub(1); }
            heap.push(Reverse(hit));
            if tid < 256 { topic_counts[tid] = topic_counts[tid].saturating_add(1); }
        }
    }

    // Phase 3: Extract snippets for final K
    let mut results: Vec<SearchHit> = Vec::with_capacity(heap.len());
    for r in heap.into_vec() {
        let h = r.0;
        let so = snip_off + h.snippet_off as usize;
        let sl = h.snippet_len as usize;
        let snippet = if so + sl <= data_len {
            std::str::from_utf8(&data[so..so + sl]).unwrap_or("").to_string()
        } else { String::new() };
        results.push(SearchHit {
            entry_id: h.entry_id, topic_id: h.topic_id, score: h.score,
            snippet, date_minutes: h.date_minutes, log_offset: h.log_offset,
        });
    }
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    Ok(results)
}

// ── Low-level readers ──

pub fn read_header(data: &[u8]) -> Result<Header, String> {
    if data.len() < std::mem::size_of::<Header>() { return Err("index too small".into()); }
    let hdr: Header = unsafe { std::ptr::read_unaligned(data.as_ptr() as *const Header) };
    if hdr.magic != MAGIC { return Err("bad index magic".into()); }
    if { hdr.version } != VERSION { return Err("index version mismatch — run reindex".into()); }
    Ok(hdr)
}

fn read_slot(data: &[u8], idx: usize) -> Result<TermSlot, String> {
    read_at::<TermSlot>(data, std::mem::size_of::<Header>() + idx * std::mem::size_of::<TermSlot>())
}

pub fn read_at<T: Copy>(data: &[u8], off: usize) -> Result<T, String> {
    if off + std::mem::size_of::<T>() > data.len() { return Err("read out of bounds".into()); }
    Ok(unsafe { std::ptr::read_unaligned(data.as_ptr().add(off) as *const T) })
}

#[inline(always)]
unsafe fn read_at_unchecked<T: Copy>(data: &[u8], off: usize) -> T {
    std::ptr::read_unaligned(data.as_ptr().add(off) as *const T)
}

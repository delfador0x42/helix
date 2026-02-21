//! Inverted index: build, query, score. Merges builder + binary query + BM25 scoring.
//! Three-phase search: BM25 accumulate → heap top-K → deferred snippet extraction.

use std::path::Path;
use std::sync::Mutex;
use crate::format::*;
use crate::fxhash::{FxHashMap, FxHashSet};

// ══════════ Builder ══════════

struct EntryInfo {
    topic_id: u16, word_count: u16, snippet: String,
    date_minutes: i32, source: String, log_offset: u32,
    tags: Vec<String>, confidence: Option<f64>,
}

pub struct IndexBuilder {
    terms: FxHashMap<String, Vec<(u32, u16)>>,
    entries: Vec<EntryInfo>,
    topics: Vec<String>,
    topic_index: FxHashMap<String, u16>,
    total_words: usize,
    tag_freq: FxHashMap<String, usize>,
}

impl IndexBuilder {
    pub fn new() -> Self {
        Self {
            terms: FxHashMap::default(), entries: Vec::new(), topics: Vec::new(),
            topic_index: FxHashMap::default(), total_words: 0, tag_freq: FxHashMap::default(),
        }
    }

    pub fn add_topic(&mut self, name: &str) -> u16 {
        if let Some(&id) = self.topic_index.get(name) { return id; }
        let id = self.topics.len() as u16;
        self.topic_index.insert(name.to_string(), id);
        self.topics.push(name.to_string());
        id
    }

    /// Add entry from cached tf_map — no tokenization needed.
    pub fn add_entry(
        &mut self, topic_id: u16, snippet: &str, date_minutes: i32,
        source: &str, log_offset: u32, tags: &[String],
        tf_map: &FxHashMap<String, usize>, word_count: usize, confidence: Option<f64>,
    ) -> u32 {
        let entry_id = self.entries.len() as u32;
        self.total_words += word_count;
        for (term, &tf) in tf_map {
            if term.len() < 2 { continue; }
            let posting = (entry_id, tf.min(u16::MAX as usize) as u16);
            if let Some(v) = self.terms.get_mut(term.as_str()) { v.push(posting); }
            else { self.terms.insert(term.clone(), vec![posting]); }
        }
        for tag in tags { *self.tag_freq.entry(tag.clone()).or_insert(0) += 1; }
        self.entries.push(EntryInfo {
            topic_id, word_count: word_count.min(u16::MAX as usize) as u16,
            snippet: snippet.to_string(), date_minutes,
            source: source.to_string(), log_offset, tags: tags.to_vec(), confidence,
        });
        entry_id
    }

    fn compute_xrefs(&self) -> Vec<XrefEdge> {
        let mut edges: FxHashMap<(u16, u16), u16> = FxHashMap::default();
        for (i, name) in self.topics.iter().enumerate() {
            let dst = i as u16;
            let tokens = crate::text::tokenize(name);
            let tokens: Vec<&str> = tokens.iter().filter(|t| t.len() >= 2).map(|s| s.as_str()).collect();
            if tokens.is_empty() { continue; }
            let mut candidates: Option<FxHashSet<u32>> = None;
            for token in &tokens {
                if let Some(postings) = self.terms.get(*token) {
                    let ids: FxHashSet<u32> = postings.iter().map(|(eid, _)| *eid).collect();
                    candidates = Some(match candidates {
                        Some(prev) => prev.intersection(&ids).copied().collect(),
                        None => ids,
                    });
                } else { candidates = Some(FxHashSet::default()); break; }
            }
            if let Some(cands) = candidates {
                for eid in cands {
                    let entry = &self.entries[eid as usize];
                    if entry.topic_id == dst { continue; }
                    *edges.entry((entry.topic_id, dst)).or_insert(0) += 1;
                }
            }
        }
        edges.into_iter().map(|((s, d), c)| XrefEdge {
            src_topic: s, dst_topic: d, mention_count: c, _pad: 0,
        }).collect()
    }

    pub fn build(&self) -> Vec<u8> {
        let n = self.entries.len() as f64;
        let avgdl = if n == 0.0 { 100.0 } else { self.total_words as f64 / n };
        let table_cap = (self.terms.len() * 4 / 3 + 1).next_power_of_two().max(16);
        let mask = table_cap - 1;
        let tag_to_bit = self.build_tag_map();

        // Posting lists with pre-computed IDF
        let mut post_buf: Vec<Posting> = Vec::new();
        let mut term_entries: Vec<(u64, u32, u32)> = Vec::new();
        for (term, postings) in &self.terms {
            let h = hash_term(term);
            let off = post_buf.len() as u32;
            let df = postings.len() as f64;
            let idf_x1000 = (((n - df + 0.5) / (df + 0.5) + 1.0).ln() * 1000.0) as u32;
            for &(eid, tf) in postings {
                post_buf.push(Posting { entry_id: eid, tf, idf_x1000, _pad: 0 });
            }
            term_entries.push((h, off, postings.len() as u32));
        }

        // Open-addressing hash table
        let mut table: Vec<TermSlot> = (0..table_cap)
            .map(|_| TermSlot { hash: 0, postings_off: 0, postings_len: 0 }).collect();
        for &(h, off, len) in &term_entries {
            let mut idx = (h as usize) & mask;
            loop {
                if table[idx].hash == 0 { table[idx] = TermSlot { hash: h, postings_off: off, postings_len: len }; break; }
                idx = (idx + 1) & mask;
            }
        }

        // Entry metadata + snippet/source pools
        let mut mtime_cache: FxHashMap<String, Option<std::time::SystemTime>> = FxHashMap::default();
        let mut snippets = Vec::<u8>::new();
        let mut sources = Vec::<u8>::new();
        let mut metas = Vec::<EntryMeta>::new();
        for info in &self.entries {
            let s_off = snippets.len() as u32;
            let sb = info.snippet.as_bytes();
            let s_len = sb.len().min(u16::MAX as usize) as u16;
            snippets.extend_from_slice(&sb[..s_len as usize]);
            let (src_off, src_len) = if info.source.is_empty() { (0u32, 0u16) } else {
                let o = sources.len() as u32;
                let b = info.source.as_bytes();
                let l = b.len().min(u16::MAX as usize) as u16;
                sources.extend_from_slice(&b[..l as usize]);
                (o, l)
            };
            let tag_bitmap = self.entry_tag_bitmap(&info.tags, &tag_to_bit);
            let stale_conf = compute_confidence(&info.source, info.date_minutes, &mut mtime_cache);
            let confidence = match info.confidence {
                Some(c) => ((c.clamp(0.0, 1.0) * 255.0) as u8).min(stale_conf),
                None => stale_conf,
            };
            let epoch_days = if info.date_minutes > 0 { (info.date_minutes as u32 / 1440) as u16 } else { 0 };
            metas.push(EntryMeta {
                topic_id: info.topic_id, word_count: info.word_count,
                snippet_off: s_off, snippet_len: s_len, date_minutes: info.date_minutes,
                source_off: src_off, source_len: src_len, log_offset: info.log_offset,
                tag_bitmap, confidence, epoch_days, _pad: 0,
            });
        }

        // Topic table + name pool
        let mut tname_pool = Vec::<u8>::new();
        let mut ttable = Vec::<TopicEntry>::new();
        let mut tcounts = vec![0u16; self.topics.len()];
        for e in &self.entries { tcounts[e.topic_id as usize] += 1; }
        for (i, name) in self.topics.iter().enumerate() {
            let off = tname_pool.len() as u32;
            let nb = name.as_bytes();
            let len = nb.len().min(u16::MAX as usize) as u16;
            tname_pool.extend_from_slice(&nb[..len as usize]);
            ttable.push(TopicEntry { name_off: off, name_len: len, entry_count: tcounts[i] });
        }

        let xrefs = self.compute_xrefs();
        let tag_names_buf = self.build_tag_names(&tag_to_bit);

        // Section offsets
        let hdr_sz = std::mem::size_of::<Header>();
        let tab_sz = table_cap * std::mem::size_of::<TermSlot>();
        let post_off = hdr_sz + tab_sz;
        let meta_off = post_off + post_buf.len() * std::mem::size_of::<Posting>();
        let snip_off = meta_off + metas.len() * std::mem::size_of::<EntryMeta>();
        let top_off = snip_off + snippets.len();
        let tname_off = top_off + ttable.len() * std::mem::size_of::<TopicEntry>();
        let src_off = tname_off + tname_pool.len();
        let xref_off = src_off + sources.len();
        let tagn_off = xref_off + xrefs.len() * std::mem::size_of::<XrefEdge>();
        let total = tagn_off + tag_names_buf.len();

        let header = Header {
            magic: MAGIC, version: VERSION,
            num_entries: self.entries.len() as u32, num_terms: self.terms.len() as u32,
            num_topics: self.topics.len() as u16, num_xrefs: xrefs.len() as u16,
            table_cap: table_cap as u32, avgdl_x100: (avgdl * 100.0) as u32,
            postings_off: post_off as u32, meta_off: meta_off as u32,
            snippet_off: snip_off as u32, topics_off: top_off as u32,
            topic_names_off: tname_off as u32, source_off: src_off as u32,
            xref_off: xref_off as u32, total_len: total as u32,
            tag_names_off: tagn_off as u32, num_tags: tag_to_bit.len() as u32,
        };

        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(as_bytes(&header));
        for s in &table { buf.extend_from_slice(as_bytes(s)); }
        for p in &post_buf { buf.extend_from_slice(as_bytes(p)); }
        for m in &metas { buf.extend_from_slice(as_bytes(m)); }
        buf.extend_from_slice(&snippets);
        for t in &ttable { buf.extend_from_slice(as_bytes(t)); }
        buf.extend_from_slice(&tname_pool);
        buf.extend_from_slice(&sources);
        for x in &xrefs { buf.extend_from_slice(as_bytes(x)); }
        buf.extend_from_slice(&tag_names_buf);
        buf
    }

    fn build_tag_map(&self) -> Vec<(String, u8)> {
        let mut sorted: Vec<_> = self.tag_freq.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        sorted.iter().take(32).enumerate().map(|(i, (n, _))| ((*n).clone(), i as u8)).collect()
    }

    fn entry_tag_bitmap(&self, tags: &[String], tag_map: &[(String, u8)]) -> u32 {
        let mut bitmap = 0u32;
        for tag in tags {
            if let Some((_, bit)) = tag_map.iter().find(|(n, _)| n == tag) { bitmap |= 1u32 << *bit; }
        }
        bitmap
    }

    fn build_tag_names(&self, tag_map: &[(String, u8)]) -> Vec<u8> {
        let mut buf = vec![tag_map.len() as u8];
        let mut sorted: Vec<_> = tag_map.to_vec();
        sorted.sort_by_key(|(_, bit)| *bit);
        for (name, _) in &sorted {
            let b = name.as_bytes();
            buf.push(b.len().min(255) as u8);
            buf.extend_from_slice(&b[..b.len().min(255)]);
        }
        buf
    }
}

fn compute_confidence(
    source: &str, date_minutes: i32,
    cache: &mut FxHashMap<String, Option<std::time::SystemTime>>,
) -> u8 {
    if source.is_empty() { return 255; }
    let path = source.split(':').next().unwrap_or(source);
    let mtime = cache.entry(path.to_string()).or_insert_with(|| {
        std::fs::metadata(path).and_then(|m| m.modified()).ok()
    });
    match mtime {
        Some(t) => {
            let entry_time = std::time::UNIX_EPOCH + std::time::Duration::from_secs((date_minutes as u64) * 60);
            if *t > entry_time { 178 } else { 255 }
        }
        None => 255,
    }
}

// ══════════ Rebuild ══════════

pub fn rebuild(dir: &Path, persist: bool) -> Result<(String, Vec<u8>), String> {
    crate::datalog::ensure_log(dir)?;
    let (bytes, ne, nt, ntop) = crate::cache::with_corpus(dir, |cached| {
        let mut builder = IndexBuilder::new();
        for e in cached {
            let tid = builder.add_topic(&e.topic);
            let conf = if e.confidence() < 1.0 { Some(e.confidence()) } else { None };
            builder.add_entry(
                tid, &e.snippet, e.timestamp_min,
                e.source().unwrap_or(""), e.offset, e.tags(),
                &e.tf_map, e.word_count, conf,
            );
        }
        let ne = builder.entries.len();
        let nt = builder.terms.len();
        let ntop = builder.topics.len();
        (builder.build(), ne, nt, ntop)
    })?;
    if persist {
        crate::config::atomic_write_bytes(&dir.join("index.bin"), &bytes)?;
    }
    Ok((format!("index: {ne} entries, {nt} terms, {ntop} topics, {} bytes", bytes.len()), bytes))
}

// ══════════ Query: FilterPred + QueryState ══════════

pub struct FilterPred {
    pub topic_id: Option<u16>,
    pub after_days: u16,
    pub before_days: u16,
    pub tag_mask: u32,
    pub source_needle: Option<Vec<u8>>,
}

impl FilterPred {
    pub fn none() -> Self {
        Self { topic_id: None, after_days: 0, before_days: u16::MAX, tag_mask: 0, source_needle: None }
    }
    fn passes(&self, m: &EntryMeta, data: &[u8], src_pool: usize) -> bool {
        if let Some(t) = self.topic_id { if { m.topic_id } != t { return false; } }
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

// ══════════ Query: Core Search ══════════

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

// ══════════ Index Readers ══════════

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

pub fn topic_table(data: &[u8]) -> Result<Vec<(u16, String, u16)>, String> {
    let hdr = read_header(data)?;
    let top_off = { hdr.topics_off } as usize;
    let tname_off = { hdr.topic_names_off } as usize;
    let n = { hdr.num_topics } as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let te = read_at::<TopicEntry>(data, top_off + i * std::mem::size_of::<TopicEntry>())?;
        let no = tname_off + { te.name_off } as usize;
        let nl = { te.name_len } as usize;
        let name = if no + nl <= data.len() {
            std::str::from_utf8(&data[no..no + nl]).unwrap_or("?").to_string()
        } else { "?".into() };
        out.push((i as u16, name, { te.entry_count }));
    }
    Ok(out)
}

/// Get the most recent date_minutes per topic. O(num_entries) sequential scan.
pub fn topic_recency(data: &[u8]) -> Vec<(u16, i32)> {
    let hdr = match read_header(data) { Ok(h) => h, Err(_) => return vec![] };
    let meta_off = { hdr.meta_off } as usize;
    let n = { hdr.num_entries } as usize;
    let ntop = { hdr.num_topics } as usize;
    let meta_size = std::mem::size_of::<EntryMeta>();
    let mut latest = vec![i32::MIN; ntop];
    for i in 0..n {
        if let Ok(m) = read_at::<EntryMeta>(data, meta_off + i * meta_size) {
            let tid = { m.topic_id } as usize;
            let dm = { m.date_minutes };
            if tid < ntop && dm > latest[tid] { latest[tid] = dm; }
        }
    }
    latest.into_iter().enumerate()
        .filter(|(_, d)| *d > i32::MIN)
        .map(|(i, d)| (i as u16, d)).collect()
}

pub fn topic_name(data: &[u8], topic_id: u16) -> Result<String, String> {
    let hdr = read_header(data)?;
    let top_off = { hdr.topics_off } as usize;
    let tname_off = { hdr.topic_names_off } as usize;
    if topic_id as usize >= { hdr.num_topics } as usize { return Err("topic_id out of range".into()); }
    let te = read_at::<TopicEntry>(data, top_off + topic_id as usize * std::mem::size_of::<TopicEntry>())?;
    let no = tname_off + { te.name_off } as usize;
    let nl = { te.name_len } as usize;
    if no + nl > data.len() { return Err("name out of bounds".into()); }
    Ok(std::str::from_utf8(&data[no..no + nl]).unwrap_or("?").to_string())
}

pub fn resolve_topic(data: &[u8], name: &str) -> Option<u16> {
    topic_table(data).ok()?.iter().find(|(_, n, _)| n == name).map(|(id, _, _)| *id)
}

pub fn resolve_tag(data: &[u8], tag_name: &str) -> Option<u8> {
    let hdr = read_header(data).ok()?;
    let off = { hdr.tag_names_off } as usize;
    if off >= data.len() { return None; }
    let count = data[off] as usize;
    let mut pos = off + 1;
    let lower = tag_name.to_lowercase();
    for bit in 0..count {
        if pos >= data.len() { return None; }
        let len = data[pos] as usize;
        pos += 1;
        if pos + len > data.len() { return None; }
        if std::str::from_utf8(&data[pos..pos + len]).ok() == Some(&lower) { return Some(bit as u8); }
        pos += len;
    }
    None
}

pub fn reconstruct_tags(data: &[u8], entry_id: u32) -> Result<Option<String>, String> {
    let hdr = read_header(data)?;
    let meta_off = { hdr.meta_off } as usize;
    if entry_id as usize >= { hdr.num_entries } as usize { return Err("entry_id out of range".into()); }
    let m = read_at::<EntryMeta>(data, meta_off + entry_id as usize * std::mem::size_of::<EntryMeta>())?;
    let bitmap = { m.tag_bitmap };
    if bitmap == 0 { return Ok(None); }
    let tag_names = read_tag_names(data, &hdr)?;
    let mut out = String::with_capacity(64);
    out.push_str("[tags: ");
    let mut first = true;
    for (bit, name) in tag_names.iter().enumerate() {
        if bitmap & (1u32 << bit) != 0 {
            if !first { out.push_str(", "); }
            out.push_str(name);
            first = false;
        }
    }
    if first { return Ok(None); }
    out.push(']');
    Ok(Some(out))
}

fn read_tag_names(data: &[u8], hdr: &Header) -> Result<Vec<String>, String> {
    let off = { hdr.tag_names_off } as usize;
    if off >= data.len() { return Ok(Vec::new()); }
    let count = data[off] as usize;
    let mut pos = off + 1;
    let mut names = Vec::with_capacity(count);
    for _ in 0..count {
        if pos >= data.len() { break; }
        let len = data[pos] as usize;
        pos += 1;
        if pos + len > data.len() { break; }
        names.push(std::str::from_utf8(&data[pos..pos + len]).unwrap_or("").to_string());
        pos += len;
    }
    Ok(names)
}

pub fn xref_edges(data: &[u8]) -> Result<Vec<(u16, u16, u16)>, String> {
    let hdr = read_header(data)?;
    let off = { hdr.xref_off } as usize;
    let n = { hdr.num_xrefs } as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = read_at::<XrefEdge>(data, off + i * std::mem::size_of::<XrefEdge>())?;
        out.push(({ x.src_topic }, { x.dst_topic }, { x.mention_count }));
    }
    Ok(out)
}

pub fn entry_snippet_ref(data: &[u8], entry_id: u32) -> Result<&str, String> {
    let hdr = read_header(data)?;
    let meta_off = { hdr.meta_off } as usize;
    let snip_off = { hdr.snippet_off } as usize;
    if entry_id as usize >= { hdr.num_entries } as usize { return Err("entry_id out of range".into()); }
    let m = read_at::<EntryMeta>(data, meta_off + entry_id as usize * std::mem::size_of::<EntryMeta>())?;
    let so = snip_off + { m.snippet_off } as usize;
    let sl = { m.snippet_len } as usize;
    if so + sl > data.len() { return Ok(""); }
    Ok(std::str::from_utf8(&data[so..so + sl]).unwrap_or(""))
}

pub fn entries_for_topic(data: &[u8], topic_id: u16) -> Result<Vec<u32>, String> {
    let hdr = read_header(data)?;
    let meta_off = { hdr.meta_off } as usize;
    let n = { hdr.num_entries } as usize;
    let mut entries: Vec<(u32, i32)> = Vec::new();
    for i in 0..n {
        let m = read_at::<EntryMeta>(data, meta_off + i * std::mem::size_of::<EntryMeta>())?;
        if { m.topic_id } == topic_id { entries.push((i as u32, { m.date_minutes })); }
    }
    entries.sort_by_key(|&(_, d)| d);
    Ok(entries.into_iter().map(|(id, _)| id).collect())
}

pub fn sourced_entries(data: &[u8]) -> Result<Vec<(u32, u16, String, i32)>, String> {
    let hdr = read_header(data)?;
    let meta_off = { hdr.meta_off } as usize;
    let src_off = { hdr.source_off } as usize;
    let n = { hdr.num_entries } as usize;
    let mut out = Vec::new();
    for i in 0..n {
        let m = read_at::<EntryMeta>(data, meta_off + i * std::mem::size_of::<EntryMeta>())?;
        let sl = { m.source_len } as usize;
        if sl == 0 { continue; }
        let so = src_off + { m.source_off } as usize;
        if so + sl > data.len() { continue; }
        let path = std::str::from_utf8(&data[so..so + sl]).unwrap_or("").to_string();
        out.push((i as u32, { m.topic_id }, path, { m.date_minutes }));
    }
    Ok(out)
}

pub fn index_info(data: &[u8]) -> Result<String, String> {
    let hdr = read_header(data)?;
    Ok(format!("index v3: {} entries, {} terms, {} topics, {} xrefs, {} tags, {} bytes",
        { hdr.num_entries }, { hdr.num_terms }, { hdr.num_topics },
        { hdr.num_xrefs }, { hdr.num_tags }, { hdr.total_len }))
}

// ══════════ Scoring: Unified Search ══════════

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
        let tag_ok = filter.tag.as_ref().map_or(true, |t| resolve_tag(data, t).is_some());
        if tag_ok {
            if let Ok(result) = score_via_index(dir, data, terms, filter, limit, full_body) {
                return Ok(result);
            }
        }
    }
    score_on_cache(dir, terms, filter, limit)
}

fn build_filter_pred(data: &[u8], filter: &Filter) -> FilterPred {
    FilterPred {
        topic_id: filter.topic.as_ref().and_then(|n| resolve_topic(data, n)),
        after_days: filter.after.map(|d| d.max(0) as u16).unwrap_or(0),
        before_days: filter.before.map(|d| d.min(u16::MAX as i64) as u16).unwrap_or(u16::MAX),
        tag_mask: filter.tag.as_ref().and_then(|t| resolve_tag(data, t)).map(|b| 1u32 << b).unwrap_or(0),
        source_needle: filter.source.as_ref().map(|s| s.as_bytes().to_vec()),
    }
}

fn score_via_index(dir: &Path, data: &[u8], terms: &[String], filter: &Filter,
                   limit: Option<usize>, full_body: bool) -> Result<(Vec<ScoredResult>, bool), String>
{
    let pred = build_filter_pred(data, filter);
    let cap = limit.unwrap_or(20);
    let query = terms.join(" ");
    let hits = search_index(data, &query, &pred, cap, true)?;
    if hits.is_empty() && filter.mode == SearchMode::And && terms.len() >= 2 {
        let or_hits = search_index(data, &query, &pred, cap, false)?;
        if !or_hits.is_empty() { return hydrate_hits(dir, data, terms, &or_hits, true, full_body); }
        return Ok((Vec::new(), false));
    }
    hydrate_hits(dir, data, terms, &hits, false, full_body)
}

fn hydrate_hits(dir: &Path, data: &[u8], terms: &[String], hits: &[SearchHit],
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
            Entry::Vacant(e) => match topic_name(data, hit.topic_id) {
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
            let tag_line = reconstruct_tags(data, hit.entry_id).ok().flatten();
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

// ══════════ Cache Scoring (fallback) ══════════

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
                if let Some(ref t) = filter.topic { if *e.topic != **t { return false; } }
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
                if let Some(ref t) = filter.topic { if *e.topic != **t { continue; } }
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
                if let Some(ref t) = filter.topic { if *e.topic != **t { continue; } }
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

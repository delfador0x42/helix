//! In-memory corpus cache with data.log mtime invalidation.
//! Pre-tokenized entries; metadata parsed lazily on first access.

use crate::fxhash::FxHashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use std::path::Path;

/// Interned topic name — Arc<str> dedup across entries.
#[derive(Clone)]
pub struct InternedStr(Arc<str>);

impl InternedStr {
    pub fn new(s: &str) -> Self { Self(Arc::from(s)) }
    pub fn as_str(&self) -> &str { &self.0 }
}

impl std::ops::Deref for InternedStr {
    type Target = str;
    fn deref(&self) -> &str { &self.0 }
}

impl std::fmt::Display for InternedStr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str(&self.0) }
}

pub struct CachedEntry {
    pub topic: InternedStr,
    pub body: String,
    pub timestamp_min: i32,
    pub offset: u32,
    pub tf_map: FxHashMap<String, usize>,
    pub word_count: usize,
    pub snippet: String,
    meta: std::cell::OnceCell<crate::text::EntryMetadata>,
}

impl CachedEntry {
    fn meta(&self) -> &crate::text::EntryMetadata {
        self.meta.get_or_init(|| crate::text::extract_all_metadata(&self.body))
    }
    pub fn tags(&self) -> &[String] { &self.meta().tags }
    pub fn source(&self) -> Option<&str> { self.meta().source.as_deref() }
    pub fn confidence(&self) -> f64 { self.meta().confidence }
    pub fn links(&self) -> &[(String, usize)] { &self.meta().links }
    pub fn has_tag(&self, tag: &str) -> bool { self.tags().iter().any(|t| t == tag) }

    pub fn date_str(&self) -> String {
        crate::time::minutes_to_date_str(self.timestamp_min)
    }
    pub fn day(&self) -> i64 { self.timestamp_min as i64 / 1440 }
    pub fn days_old(&self, now_days: i64) -> i64 { now_days - self.day() }

    /// First non-metadata, non-empty line.
    pub fn preview(&self) -> &str {
        for line in self.body.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !crate::text::is_metadata_line(trimmed) {
                return trimmed;
            }
        }
        ""
    }
    pub fn confidence_u8(&self) -> u8 { (self.confidence().clamp(0.0, 1.0) * 255.0) as u8 }
    pub fn has_links(&self) -> bool { !self.links().is_empty() }
}

struct CachedCorpus {
    mtime: SystemTime,
    entries: Vec<CachedEntry>,
    intern_pool: FxHashMap<String, InternedStr>,
}

static CACHE: Mutex<Option<CachedCorpus>> = Mutex::new(None);

pub fn invalidate() {
    if let Ok(mut g) = CACHE.lock() { *g = None; }
}

/// Access cached corpus. Reloads from data.log only if mtime changed.
pub fn with_corpus<F, R>(dir: &Path, f: F) -> Result<R, String>
where F: FnOnce(&[CachedEntry]) -> R {
    let log_path = crate::config::log_path(dir);
    let cur_mtime = std::fs::metadata(&log_path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);

    let mut guard = CACHE.lock().map_err(|e| e.to_string())?;
    if let Some(ref cache) = *guard {
        if cache.mtime == cur_mtime {
            return Ok(f(&cache.entries));
        }
    }

    let raw = crate::datalog::iter_live(&log_path)?;
    let mut entries = Vec::with_capacity(raw.len());
    let mut intern_pool: FxHashMap<String, InternedStr> = FxHashMap::default();
    for e in raw {
        let topic = match intern_pool.get(e.topic.as_str()) {
            Some(t) => t.clone(),
            None => {
                let t = InternedStr::new(&e.topic);
                intern_pool.insert(e.topic.clone(), t.clone());
                t
            }
        };
        let mut tf_map = crate::fxhash::map_with_capacity(32);
        let word_count = crate::text::tokenize_into_tfmap(&e.body, &mut tf_map);
        let snippet = build_snippet(topic.as_str(), e.timestamp_min, &e.body);
        entries.push(CachedEntry {
            topic, body: e.body, timestamp_min: e.timestamp_min, offset: e.offset,
            tf_map, word_count, snippet, meta: std::cell::OnceCell::new(),
        });
    }

    let result = f(&entries);
    *guard = Some(CachedCorpus { mtime: cur_mtime, entries, intern_pool });
    Ok(result)
}

/// Append to in-memory cache after store (avoids double corpus load).
pub fn append_to_cache(dir: &Path, topic: &str, body: &str, ts_min: i32, offset: u32) {
    let log_path = crate::config::log_path(dir);
    let cur_mtime = std::fs::metadata(&log_path)
        .and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
    let mut guard = match CACHE.lock() { Ok(g) => g, Err(_) => return };
    let cache = match guard.as_mut() { Some(c) => c, None => return };
    let topic_interned = match cache.intern_pool.get(topic) {
        Some(t) => t.clone(),
        None => {
            let t = InternedStr::new(topic);
            cache.intern_pool.insert(topic.to_string(), t.clone());
            t
        }
    };
    let mut tf_map = crate::fxhash::map_with_capacity(32);
    let word_count = crate::text::tokenize_into_tfmap(body, &mut tf_map);
    let snippet = build_snippet(topic, ts_min, body);
    cache.entries.push(CachedEntry {
        topic: topic_interned, body: body.to_string(), timestamp_min: ts_min,
        offset, tf_map, word_count, snippet, meta: std::cell::OnceCell::new(),
    });
    cache.mtime = cur_mtime;
}

pub struct CacheStats { pub entries: usize, pub cached: bool }

pub fn stats() -> CacheStats {
    let guard = CACHE.lock().unwrap();
    match guard.as_ref() {
        Some(c) => CacheStats { entries: c.entries.len(), cached: true },
        None => CacheStats { entries: 0, cached: false },
    }
}

/// "[topic] YYYY-MM-DD HH:MM first_content_lines"
fn build_snippet(topic: &str, ts_min: i32, body: &str) -> String {
    let mut buf = String::with_capacity(topic.len() + 140);
    buf.push('[');
    buf.push_str(topic);
    buf.push_str("] ");
    crate::time::minutes_to_date_str_into(ts_min, &mut buf);
    buf.push(' ');
    let content_start = buf.len();
    let mut lines = 0u8;
    for line in body.lines() {
        if crate::text::is_metadata_line(line) || line.trim().is_empty() { continue; }
        if lines > 0 { buf.push(' '); }
        buf.push_str(line.trim());
        lines += 1;
        if lines >= 2 || buf.len() - content_start >= 120 { break; }
    }
    let content_len = buf.len() - content_start;
    if content_len > 120 {
        let mut boundary = 120;
        while boundary > 0 && !buf.is_char_boundary(content_start + boundary) { boundary -= 1; }
        let trunc_at = buf[content_start..content_start + boundary].rfind(' ').unwrap_or(boundary);
        buf.truncate(content_start + trunc_at);
    }
    buf
}

//! Tokenizer, query terms, metadata parsing. ASCII fast path with Unicode fallback.

const STOP_WORDS: &[&str] = &[
    "that", "this", "with", "from", "have", "been", "were", "will", "when",
    "which", "their", "there", "about", "would", "could", "should", "into",
    "also", "each", "does", "just", "more", "than", "then", "them", "some",
    "only", "other", "very", "after", "before", "most", "same", "both",
];

/// Tokenize text: split on non-alphanumeric, expand CamelCase, lowercase.
pub fn tokenize(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut tokens = Vec::with_capacity(len / 6);
    let mut pos = 0;
    while pos < len {
        while pos < len && !bytes[pos].is_ascii_alphanumeric() && bytes[pos] < 128 { pos += 1; }
        if pos >= len { break; }
        if bytes[pos] >= 128 {
            let start = pos;
            while pos < len && (bytes[pos] >= 128 || bytes[pos].is_ascii_alphanumeric()) { pos += 1; }
            let seg = &text[start..pos];
            let lower = seg.to_lowercase();
            if lower.len() >= 2 { emit_segment(seg, lower, &mut tokens); }
            continue;
        }
        let start = pos;
        while pos < len && bytes[pos].is_ascii_alphanumeric() { pos += 1; }
        let seg = &bytes[start..pos];
        if seg.len() < 2 { continue; }
        let lower = ascii_lower(seg);
        emit_segment(&text[start..pos], lower, &mut tokens);
    }
    tokens
}

/// Build tf_map directly — no intermediate Vec<String>.
pub fn tokenize_into_tfmap(text: &str, tf_map: &mut crate::fxhash::FxHashMap<String, usize>) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let (mut word_count, mut pos) = (0usize, 0);
    let mut lower_buf = Vec::with_capacity(32);
    while pos < len {
        while pos < len && !bytes[pos].is_ascii_alphanumeric() && bytes[pos] < 128 { pos += 1; }
        if pos >= len { break; }
        if bytes[pos] >= 128 {
            let start = pos;
            while pos < len && (bytes[pos] >= 128 || bytes[pos].is_ascii_alphanumeric()) { pos += 1; }
            let seg = &text[start..pos];
            let lower = seg.to_lowercase();
            if lower.len() >= 2 { word_count += emit_tfmap(seg, &lower, tf_map); }
            continue;
        }
        let start = pos;
        while pos < len && bytes[pos].is_ascii_alphanumeric() { pos += 1; }
        let seg = &bytes[start..pos];
        if seg.len() < 2 { continue; }
        lower_buf.clear();
        lower_buf.extend_from_slice(seg);
        lower_buf.make_ascii_lowercase();
        let lower_str = unsafe { std::str::from_utf8_unchecked(&lower_buf) };
        if seg[1..].iter().any(|b| b.is_ascii_uppercase()) {
            let parts = split_compound(&text[start..pos]);
            if parts.len() > 1 {
                for part in &parts {
                    if part.len() >= 2 && part != lower_str {
                        word_count += 1;
                        *tf_map.entry(part.clone()).or_default() += 1;
                    }
                }
            }
        }
        word_count += 1;
        if let Some(c) = tf_map.get_mut(lower_str) { *c += 1; }
        else { tf_map.insert(lower_str.to_string(), 1); }
    }
    word_count
}

/// Search terms: tokenize + filter stop words + dedup.
pub fn query_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::with_capacity(8);
    let mut seen = crate::fxhash::FxHashSet::default();
    for token in tokenize(query) {
        if STOP_WORDS.contains(&token.as_str()) { continue; }
        if seen.insert(token.clone()) { terms.push(token); }
    }
    terms
}

/// Extract field prefixes (tag:X, topic:X, source:X) from a query string.
/// Returns the remaining query with prefixes stripped and the extracted filters.
pub struct QueryFilters {
    pub query: String,
    pub tag: Option<String>,
    pub topic: Option<String>,
    pub source: Option<String>,
}

pub fn parse_query_filters(raw: &str) -> QueryFilters {
    let mut tag = None;
    let mut topic = None;
    let mut source = None;
    let mut remaining = Vec::with_capacity(8);
    for word in raw.split_whitespace() {
        if let Some(val) = word.strip_prefix("tag:") {
            if !val.is_empty() { tag = Some(val.to_lowercase()); }
        } else if let Some(val) = word.strip_prefix("topic:") {
            if !val.is_empty() { topic = Some(val.to_string()); }
        } else if let Some(val) = word.strip_prefix("source:") {
            if !val.is_empty() { source = Some(val.to_string()); }
        } else {
            remaining.push(word);
        }
    }
    QueryFilters { query: remaining.join(" "), tag, topic, source }
}

#[inline]
fn ascii_lower(bytes: &[u8]) -> String {
    let mut v = bytes.to_vec();
    v.make_ascii_lowercase();
    unsafe { String::from_utf8_unchecked(v) }
}

#[inline]
fn emit_segment(original: &str, lower: String, tokens: &mut Vec<String>) {
    let bytes = original.as_bytes();
    if bytes.len() >= 2 && bytes[1..].iter().any(|b| b.is_ascii_uppercase()) {
        let parts = split_compound(original);
        if parts.len() > 1 {
            for part in parts { if part.len() >= 2 && part != lower { tokens.push(part); } }
        }
    }
    tokens.push(lower);
}

#[inline]
fn emit_tfmap(original: &str, lower: &str, tf_map: &mut crate::fxhash::FxHashMap<String, usize>) -> usize {
    let bytes = original.as_bytes();
    let mut count = 0;
    if bytes.len() >= 2 && bytes[1..].iter().any(|b| b.is_ascii_uppercase()) {
        let parts = split_compound(original);
        if parts.len() > 1 {
            for part in &parts {
                if part.len() >= 2 && part != lower {
                    count += 1;
                    *tf_map.entry(part.clone()).or_default() += 1;
                }
            }
        }
    }
    count += 1;
    *tf_map.entry(lower.to_string()).or_default() += 1;
    count
}

/// Split CamelCase and snake_case into parts.
fn split_compound(s: &str) -> Vec<String> {
    let mut parts = Vec::with_capacity(4);
    for segment in s.split(|c: char| c == '_' || c == '-') {
        if segment.is_empty() { continue; }
        let bytes = segment.as_bytes();
        if bytes.iter().all(|b| b.is_ascii()) {
            let mut start = 0;
            for i in 1..bytes.len() {
                if bytes[i].is_ascii_uppercase() {
                    if i > start { parts.push(ascii_lower(&bytes[start..i])); }
                    start = i;
                }
            }
            if bytes.len() > start { parts.push(ascii_lower(&bytes[start..])); }
        } else {
            let mut cur = String::new();
            let chars: Vec<char> = segment.chars().collect();
            for i in 0..chars.len() {
                if i > 0 && chars[i].is_uppercase() && !cur.is_empty() {
                    parts.push(cur.to_lowercase()); cur = String::new();
                }
                cur.push(chars[i]);
            }
            if !cur.is_empty() { parts.push(cur.to_lowercase()); }
        }
    }
    parts
}

#[inline]
pub fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max { return s; }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) { end -= 1; }
    &s[..end]
}

#[inline]
pub fn is_metadata_line(line: &str) -> bool {
    if !line.starts_with('[') { return false; }
    line.starts_with("[tags:") || line.starts_with("[source:")
        || line.starts_with("[type:") || line.starts_with("[modified:")
        || line.starts_with("[tier:") || line.starts_with("[confidence:")
        || line.starts_with("[links:") || line.starts_with("[linked from:")
}

pub struct EntryMetadata {
    pub source: Option<String>,
    pub tags: Vec<String>,
    pub confidence: f64,
    pub links: Vec<(String, usize)>,
}

pub fn extract_all_metadata(body: &str) -> EntryMetadata {
    let (mut source, mut tags, mut confidence, mut links) = (None, Vec::new(), 1.0, Vec::new());
    for line in body.lines() {
        if !line.starts_with('[') { continue; }
        if let Some(inner) = line.strip_prefix("[tags: ").and_then(|s| s.strip_suffix(']')) {
            tags = inner.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
        } else if let Some(s) = line.strip_prefix("[source: ").and_then(|s| s.strip_suffix(']')) {
            source = Some(s.trim().to_string());
        } else if let Some(c) = line.strip_prefix("[confidence: ")
            .and_then(|s| s.strip_suffix(']')).and_then(|s| s.trim().parse::<f64>().ok()) {
            confidence = c;
        } else if let Some(inner) = line.strip_prefix("[links: ").and_then(|s| s.strip_suffix(']')) {
            links = inner.split_whitespace().filter_map(|pair| {
                let (topic, idx) = pair.rsplit_once(':')?;
                Some((topic.to_string(), idx.parse().ok()?))
            }).collect();
        }
    }
    EntryMetadata { source, tags, confidence, links }
}

pub fn extract_tags(lines: &[impl AsRef<str>]) -> Option<String> {
    for line in lines {
        if let Some(inner) = line.as_ref().strip_prefix("[tags: ").and_then(|s| s.strip_suffix(']')) {
            let tags: Vec<&str> = inner.split(',').map(|t| t.trim()).filter(|t| !t.is_empty()).collect();
            if !tags.is_empty() {
                return Some(tags.iter().map(|t| format!("#{t}")).collect::<Vec<_>>().join(" "));
            }
        }
    }
    None
}

pub fn itoa_push(buf: &mut String, n: u32) {
    if n == 0 { buf.push('0'); return; }
    let mut digits = [0u8; 10];
    let mut i = 0;
    let mut v = n;
    while v > 0 { digits[i] = b'0' + (v % 10) as u8; v /= 10; i += 1; }
    while i > 0 { i -= 1; buf.push(digits[i] as char); }
}

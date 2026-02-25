//! Directory resolution and path helpers.

use std::path::{Path, PathBuf};
use std::fs;

pub fn resolve_dir(explicit: Option<String>) -> PathBuf {
    if let Some(d) = explicit { return PathBuf::from(d); }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".helix-kb")
}

pub fn ensure_dir(dir: &Path) -> Result<(), String> {
    if !dir.exists() {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    Ok(())
}

pub fn sanitize_topic(topic: &str) -> String {
    let bytes = topic.as_bytes();
    let mut raw = String::with_capacity(bytes.len());
    if bytes.iter().all(|b| b.is_ascii()) {
        for &b in bytes {
            raw.push(if b.is_ascii_alphanumeric() || b == b'-' || b == b'/' {
                b.to_ascii_lowercase() as char
            } else { '-' });
        }
    } else {
        for c in topic.to_lowercase().chars() {
            raw.push(if c.is_alphanumeric() || c == '-' || c == '/' { c } else { '-' });
        }
    }
    normalize_slashes(&raw)
}

/// Strip leading/trailing `/`, collapse `//` → `/`, strip trailing `-` per segment.
fn normalize_slashes(s: &str) -> String {
    let trimmed = s.trim_matches('/');
    let mut out = String::with_capacity(trimmed.len());
    let mut last_slash = false;
    for c in trimmed.chars() {
        if c == '/' {
            // Trim trailing dashes from previous segment
            while out.ends_with('-') { out.pop(); }
            if !last_slash && !out.is_empty() { out.push('/'); }
            last_slash = true;
        } else {
            // Skip leading dashes at start of segment
            if last_slash && c == '-' && out.ends_with('/') { continue; }
            out.push(c);
            last_slash = false;
        }
    }
    while out.ends_with('-') || out.ends_with('/') { out.pop(); }
    out
}

/// Does `topic` match `query` in the hierarchy? Exact or prefix.
/// "iris/xnu/regions" matches query "iris/xnu" and "iris" and "iris/xnu/regions".
pub fn topic_matches_query(topic: &str, query: &str) -> bool {
    topic == query || topic.starts_with(query) && topic.as_bytes().get(query.len()) == Some(&b'/')
}

/// Hierarchy distance: 0 for exact match, 1 for direct child, etc.
/// Returns None if topic is not under query at all.
pub fn hierarchy_distance(topic: &str, query: &str) -> Option<usize> {
    if topic == query { return Some(0); }
    if topic.starts_with(query) && topic.as_bytes().get(query.len()) == Some(&b'/') {
        let suffix = &topic[query.len() + 1..];
        return Some(1 + suffix.chars().filter(|&c| c == '/').count());
    }
    None
}

pub fn log_path(dir: &Path) -> PathBuf { dir.join("data.log") }

/// Atomic file write: write to .tmp, fsync, rename.
pub fn atomic_write_bytes(path: &Path, data: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let tmp = path.with_extension("bin.tmp");
    let mut f = fs::File::create(&tmp).map_err(|e| format!("create: {e}"))?;
    f.write_all(data).map_err(|e| format!("write: {e}"))?;
    f.sync_all().map_err(|e| format!("fsync: {e}"))?;
    drop(f);
    fs::rename(&tmp, path).map_err(|e| format!("rename: {e}"))?;
    Ok(())
}

/// Resolve a source path. Try as-is, then one level of CWD subdirectories.
pub fn resolve_source(source: &str) -> Option<PathBuf> {
    let p = PathBuf::from(source);
    if p.exists() { return Some(p); }
    for entry in fs::read_dir(".").ok()?.flatten() {
        if entry.file_type().ok()?.is_dir() {
            let candidate = entry.path().join(source);
            if candidate.exists() { return Some(candidate); }
        }
    }
    None
}

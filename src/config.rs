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
    topic.to_lowercase().chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '-' })
        .collect()
}

pub fn log_path(dir: &Path) -> PathBuf { dir.join("data.log") }

pub fn index_path(dir: &Path) -> PathBuf { dir.join("index.bin") }

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

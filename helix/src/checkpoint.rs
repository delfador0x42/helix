//! Checkpoint: durable "where I left off" that survives across sessions.

use std::path::{Path, PathBuf};

/// Checkpoint persists in checkpoint.json — no TTY binding, no timeout.
/// Replaced atomically on each save; only one active checkpoint at a time.
pub struct Checkpoint {
    pub task: String,
    pub done: Vec<String>,
    pub next: Vec<String>,
    pub hypotheses: Vec<String>,
    pub blocked: String,
    pub files: Vec<String>,
    pub timestamp: u64,
}

fn checkpoint_path(dir: &Path) -> PathBuf { dir.join("checkpoint.json") }

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or(0)
}

impl Checkpoint {
    pub fn new(task: &str) -> Self {
        Checkpoint {
            task: task.to_string(),
            done: Vec::new(), next: Vec::new(), hypotheses: Vec::new(),
            blocked: String::new(), files: Vec::new(),
            timestamp: now_secs(),
        }
    }

    pub fn load(dir: &Path) -> Option<Self> {
        let buf = std::fs::read_to_string(checkpoint_path(dir)).ok()?;
        let val = crate::json::parse(&buf).ok()?;
        Self::from_json(&val)
    }

    pub fn save(&self, dir: &Path) -> Result<(), String> {
        let path = checkpoint_path(dir);
        let tmp = dir.join(".checkpoint.tmp");
        std::fs::write(&tmp, self.to_json())
            .map_err(|e| format!("checkpoint write: {e}"))?;
        std::fs::rename(&tmp, &path)
            .map_err(|e| format!("checkpoint rename: {e}"))?;
        Ok(())
    }

    pub fn clear(dir: &Path) { let _ = std::fs::remove_file(checkpoint_path(dir)); }

    pub fn format_resume(&self) -> String {
        let mut out = String::with_capacity(512);
        out.push_str("## Checkpoint: ");
        out.push_str(&self.task);
        out.push('\n');

        let age = now_secs().saturating_sub(self.timestamp);
        let hours = age / 3600;
        let mins = (age % 3600) / 60;
        if hours > 0 {
            out.push_str("_saved "); crate::text::itoa_push_u64(&mut out, hours);
            out.push_str("h "); crate::text::itoa_push_u64(&mut out, mins);
            out.push_str("m ago_\n");
        } else {
            out.push_str("_saved "); crate::text::itoa_push_u64(&mut out, mins);
            out.push_str("m ago_\n");
        }

        if !self.done.is_empty() {
            out.push_str("\nDone:\n");
            for d in &self.done { out.push_str("  [x] "); out.push_str(d); out.push('\n'); }
        }
        if !self.next.is_empty() {
            out.push_str("\nNext:\n");
            for n in &self.next { out.push_str("  [ ] "); out.push_str(n); out.push('\n'); }
        }
        if !self.hypotheses.is_empty() {
            out.push_str("\nHypotheses:\n");
            for h in &self.hypotheses { out.push_str("  ? "); out.push_str(h); out.push('\n'); }
        }
        if !self.blocked.is_empty() {
            out.push_str("\nBlocked: "); out.push_str(&self.blocked); out.push('\n');
        }
        if !self.files.is_empty() {
            out.push_str("\nFiles: "); out.push_str(&self.files.join(", ")); out.push('\n');
        }
        out
    }

    fn to_json(&self) -> String {
        let mut b = String::with_capacity(512);
        b.push_str("{\"task\":\"");
        crate::json::escape_into(&self.task, &mut b);
        b.push_str("\",\"done\":"); json_str_array(&mut b, &self.done);
        b.push_str(",\"next\":"); json_str_array(&mut b, &self.next);
        b.push_str(",\"hypotheses\":"); json_str_array(&mut b, &self.hypotheses);
        b.push_str(",\"blocked\":\"");
        crate::json::escape_into(&self.blocked, &mut b);
        b.push_str("\",\"files\":"); json_str_array(&mut b, &self.files);
        b.push_str(",\"timestamp\":"); crate::text::itoa_push_u64(&mut b, self.timestamp);
        b.push_str("}\n");
        b
    }

    fn from_json(val: &crate::json::Value) -> Option<Self> {
        let task = val.get("task")?.as_str()?.to_string();
        let done = json_str_vec(val.get("done"));
        let next = json_str_vec(val.get("next"));
        let hypotheses = json_str_vec(val.get("hypotheses"));
        let blocked = val.get("blocked").and_then(|v| v.as_str())
            .unwrap_or("").to_string();
        let files = json_str_vec(val.get("files"));
        let timestamp = val.get("timestamp").and_then(|v| v.as_f64())
            .unwrap_or(0.0) as u64;
        Some(Checkpoint { task, done, next, hypotheses, blocked, files, timestamp })
    }
}

fn json_str_array(buf: &mut String, items: &[String]) {
    buf.push('[');
    for (i, s) in items.iter().enumerate() {
        if i > 0 { buf.push(','); }
        buf.push('"'); crate::json::escape_into(s, buf); buf.push('"');
    }
    buf.push(']');
}

fn json_str_vec(val: Option<&crate::json::Value>) -> Vec<String> {
    match val {
        Some(crate::json::Value::Arr(arr)) =>
            arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect(),
        _ => Vec::new(),
    }
}

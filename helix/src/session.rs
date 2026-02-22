//! Session accumulator: lightweight state tracking what's happening RIGHT NOW.
//!
//! Updated by every hook invocation. Used to:
//! - Dedup injected context across ambient hook calls (injected FxHashSet)
//! - Auto-infer focus topics from injected entry topic frequency
//! - Track build state for phase detection
//! - Track files touched for session summary
//!
//! Session identity: TTY name + 4h idle timeout.
//! Storage: /tmp/helix-session.json (ephemeral, not in KB dir).
//! Concurrency: atomic tmp+rename writes, tolerant of stale reads.

use std::time::{SystemTime, UNIX_EPOCH};

const IDLE_TIMEOUT_SECS: u64 = 4 * 3600;
const SESSION_PATH: &str = "/tmp/helix-session.json";
const SESSION_TMP: &str = "/tmp/helix-session.tmp";

pub struct Session {
    pub id: String,
    pub started: u64,
    pub last_active: u64,
    pub focus_topics: Vec<String>,
    pub phase: Phase,
    pub files_touched: u32,
    pub files_edited: u32,
    pub injected: crate::fxhash::FxHashSet<u32>,
    pub last_build_ok: Option<bool>,
    pub last_build_t: u64,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Phase { Research, Build, Verify, Debug, Unknown }

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn tty_id() -> String {
    extern "C" { fn ttyname(fd: i32) -> *const i8; }
    let ptr = unsafe { ttyname(0) };
    if ptr.is_null() { return "unknown".into(); }
    let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
    cstr.to_str().ok().map(|s| s.rsplit('/').next().unwrap_or("unknown").to_string())
        .unwrap_or_else(|| "unknown".into())
}

impl Session {
    pub fn new() -> Self {
        let now = now_secs();
        let id = format!("{}-{now}", tty_id());
        Session {
            id, started: now, last_active: now,
            focus_topics: Vec::new(), phase: Phase::Unknown,
            files_touched: 0, files_edited: 0,
            injected: crate::fxhash::FxHashSet::default(),
            last_build_ok: None, last_build_t: 0,
        }
    }

    /// Load from /tmp. Returns None if expired, missing, corrupt, or wrong TTY.
    pub fn load() -> Option<Self> {
        let buf = std::fs::read_to_string(SESSION_PATH).ok()?;
        let val = crate::json::parse(&buf).ok()?;
        let s = Self::from_json(&val)?;
        let now = now_secs();
        if now.saturating_sub(s.last_active) > IDLE_TIMEOUT_SECS { return None; }
        // TTY check
        let tty = tty_id();
        if tty != "unknown" && !s.id.starts_with(&tty) { return None; }
        Some(s)
    }

    pub fn load_or_new() -> Self { Self::load().unwrap_or_else(Self::new) }

    /// Save to /tmp via atomic tmp+rename.
    pub fn save(&mut self) {
        self.last_active = now_secs();
        let json = self.to_json();
        if std::fs::write(SESSION_TMP, &json).is_ok() {
            std::fs::rename(SESSION_TMP, SESSION_PATH).ok();
        }
    }

    pub fn mark_injected(&mut self, eid: u32) { self.injected.insert(eid); }
    pub fn was_injected(&self, eid: u32) -> bool { self.injected.contains(&eid) }

    pub fn track_file(&mut self, is_edit: bool) {
        self.files_touched += 1;
        if is_edit { self.files_edited += 1; }
    }

    pub fn record_tool(&mut self, tool: &str) {
        // Phase is primarily build-state-driven; tool names are secondary signal
        self.phase = self.detect_phase(tool);
    }

    pub fn record_build(&mut self, ok: bool) {
        self.last_build_ok = Some(ok);
        self.last_build_t = now_secs();
        self.phase = if ok { Phase::Verify } else { Phase::Debug };
    }

    pub fn add_focus_topic(&mut self, topic: &str) {
        if !self.focus_topics.iter().any(|t| t == topic) {
            self.focus_topics.push(topic.to_string());
        }
    }

    /// Phase detection: build state dominates (within 5 min), tool heuristic as fallback.
    fn detect_phase(&self, tool: &str) -> Phase {
        let now = now_secs();
        if let Some(ok) = self.last_build_ok {
            if now.saturating_sub(self.last_build_t) < 300 {
                return if ok { Phase::Verify } else { Phase::Debug };
            }
        }
        // Simple tool-based signal (no sliding window needed — single tool per hook call)
        match tool {
            "Edit" | "Write" | "NotebookEdit" => Phase::Build,
            "Bash" => if self.phase == Phase::Build { Phase::Verify } else { self.phase },
            "Read" | "Grep" | "Glob" => {
                if self.phase == Phase::Unknown { Phase::Research } else { self.phase }
            }
            _ => self.phase,
        }
    }

    // --- Serialization ---

    fn to_json(&self) -> String {
        let mut b = String::with_capacity(512);
        b.push_str("{\"id\":\""); crate::json::escape_into(&self.id, &mut b);
        b.push_str("\",\"started\":"); push_u64(&mut b, self.started);
        b.push_str(",\"last_active\":"); push_u64(&mut b, self.last_active);
        b.push_str(",\"focus\":[");
        for (i, t) in self.focus_topics.iter().enumerate() {
            if i > 0 { b.push(','); }
            b.push('"'); crate::json::escape_into(t, &mut b); b.push('"');
        }
        b.push_str("],\"phase\":\""); b.push_str(self.phase.as_str());
        b.push_str("\",\"files_touched\":"); push_u64(&mut b, self.files_touched as u64);
        b.push_str(",\"files_edited\":"); push_u64(&mut b, self.files_edited as u64);
        b.push_str(",\"injected\":[");
        let mut sorted: Vec<u32> = self.injected.iter().copied().collect();
        sorted.sort_unstable();
        for (i, id) in sorted.iter().enumerate() {
            if i > 0 { b.push(','); }
            crate::text::itoa_push(&mut b, *id);
        }
        b.push_str("],\"build_ok\":");
        match self.last_build_ok {
            None => b.push_str("null"),
            Some(true) => b.push_str("true"),
            Some(false) => b.push_str("false"),
        }
        b.push_str(",\"build_t\":"); push_u64(&mut b, self.last_build_t);
        b.push_str("}\n");
        b
    }

    fn from_json(val: &crate::json::Value) -> Option<Self> {
        let id = val.get("id")?.as_str()?.to_string();
        let started = val.get("started")?.as_f64()? as u64;
        let last_active = val.get("last_active")?.as_f64()? as u64;

        let focus_topics = match val.get("focus") {
            Some(crate::json::Value::Arr(arr)) =>
                arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect(),
            _ => Vec::new(),
        };

        let phase = val.get("phase").and_then(|v| v.as_str())
            .map(Phase::from_str).unwrap_or(Phase::Unknown);

        let files_touched = val.get("files_touched").and_then(|v| v.as_f64()).unwrap_or(0.0) as u32;
        let files_edited = val.get("files_edited").and_then(|v| v.as_f64()).unwrap_or(0.0) as u32;

        let injected = match val.get("injected") {
            Some(crate::json::Value::Arr(arr)) =>
                arr.iter().filter_map(|v| v.as_f64().map(|n| n as u32)).collect(),
            _ => crate::fxhash::FxHashSet::default(),
        };

        let last_build_ok = match val.get("build_ok") {
            Some(crate::json::Value::Bool(b)) => Some(*b),
            _ => None,
        };
        let last_build_t = val.get("build_t").and_then(|v| v.as_f64()).unwrap_or(0.0) as u64;

        Some(Session {
            id, started, last_active, focus_topics, phase,
            files_touched, files_edited, injected,
            last_build_ok, last_build_t,
        })
    }
}

impl Phase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Phase::Research => "research", Phase::Build => "build",
            Phase::Verify => "verify", Phase::Debug => "debug",
            Phase::Unknown => "unknown",
        }
    }
    fn from_str(s: &str) -> Self {
        match s {
            "research" => Phase::Research, "build" => Phase::Build,
            "verify" => Phase::Verify, "debug" => Phase::Debug,
            _ => Phase::Unknown,
        }
    }
}

fn push_u64(buf: &mut String, n: u64) {
    if n == 0 { buf.push('0'); return; }
    let mut digits = [0u8; 20];
    let mut i = 0;
    let mut v = n;
    while v > 0 { digits[i] = b'0' + (v % 10) as u8; v /= 10; i += 1; }
    while i > 0 { i -= 1; buf.push(digits[i] as char); }
}

// ── Checkpoint: durable "where I left off" that survives across sessions ──

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
            out.push_str("_saved "); push_u64(&mut out, hours);
            out.push_str("h "); push_u64(&mut out, mins);
            out.push_str("m ago_\n");
        } else {
            out.push_str("_saved "); push_u64(&mut out, mins);
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
        b.push_str(",\"timestamp\":"); push_u64(&mut b, self.timestamp);
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

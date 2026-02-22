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

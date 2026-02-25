//! Topics import/export — JSON serialization of the knowledge base.

use crate::json::Value;
use std::fmt::Write;
use std::path::Path;

pub(crate) fn dispatch_export(dir: &Path) -> Result<String, String> {
    crate::cache::with_corpus(dir, |cached| {
        let mut out = String::from("{\n"); let mut cur = String::new(); let mut first_t = true; let mut in_arr = false;
        for e in cached {
            if e.topic.as_str() != cur {
                if in_arr { out.push_str("\n  ]"); }
                if !first_t { out.push_str(",\n"); }
                out.push_str("  \""); crate::json::escape_into(&e.topic, &mut out); out.push_str("\": [");
                cur = e.topic.to_string(); first_t = false; in_arr = true;
            } else { out.push(','); }
            let _ = write!(out, "\n    {{\"ts\":{},\"body\":\"", e.timestamp_min);
            crate::json::escape_into(&e.body, &mut out); out.push_str("\"}");
        }
        if in_arr { out.push_str("\n  ]"); }
        out.push_str("\n}\n"); out
    })
}

pub(crate) fn dispatch_import(args: Option<&Value>, dir: &Path) -> Result<String, String> {
    let json = crate::topics::arg(args, "json");
    let data = crate::json::parse(json)?;
    let _lock = crate::lock::FileLock::acquire(dir)?;
    crate::config::ensure_dir(dir)?;
    let log_path = crate::datalog::ensure_log(dir)?;
    let mut f = std::fs::OpenOptions::new().append(true).open(&log_path)
        .map_err(|e| format!("open data.log: {e}"))?;
    let mut count = 0;
    if let Value::Obj(pairs) = data {
        for (topic, entries) in &pairs {
            if let Value::Arr(items) = entries {
                for item in items {
                    let body = item.get("body").and_then(|v| v.as_str()).unwrap_or("");
                    let ts = item.get("ts").and_then(|v| v.as_f64()).unwrap_or(0.0) as i32;
                    if !body.is_empty() { crate::datalog::append_entry_to(&mut f, topic, body, ts)?; count += 1; }
                }
            }
        }
    }
    let _ = f.sync_all(); drop(f); drop(_lock);
    crate::mcp::after_write(dir); crate::cache::invalidate();
    Ok(format!("imported {count} entries"))
}

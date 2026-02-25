//! High-level codegraph queries: file symbols, symbol tracing, pattern detection.

use std::path::Path;
use crate::codegraph::{extract_symbols, walk_source_files, find_usages, relative_path, truncate};

/// Analyze a single file: show all defined symbols.
pub fn file_symbols(root: &Path, file_path: &Path) -> String {
    let content = match std::fs::read_to_string(file_path) { Ok(c) => c, Err(e) => return format!("error: {e}") };
    let rel = relative_path(file_path, root);
    let syms = extract_symbols(&content, &rel);
    if syms.is_empty() { return format!("no symbols found in {rel}"); }
    let mut out = String::with_capacity(256);
    out.push_str(&rel); out.push_str(":\n");
    for s in &syms {
        out.push_str("  "); out.push_str(s.kind);
        out.push(' '); out.push_str(&s.name);
        out.push_str(" (line "); crate::text::itoa_push(&mut out, s.line);
        out.push_str("): "); out.push_str(&s.signature); out.push('\n');
    }
    out
}

/// Trace a symbol: find definition + all usages across the codebase.
pub fn trace_symbol(root: &Path, symbol: &str) -> String {
    let files = walk_source_files(root);
    let mut def: Option<crate::codegraph::Symbol> = None;
    let mut all_syms: Vec<crate::codegraph::Symbol> = Vec::new();
    for file in &files {
        let content = match std::fs::read_to_string(file) { Ok(c) => c, Err(_) => continue };
        let rel = relative_path(file, root);
        let syms = extract_symbols(&content, &rel);
        for s in syms {
            if s.name == symbol {
                if def.is_none() { def = Some(s.clone()); }
                all_syms.push(s);
            }
        }
    }
    let mut out = String::with_capacity(512);
    match &def {
        None => {
            out.push_str("symbol '"); out.push_str(symbol); out.push_str("' not found in definitions.\n");
            out.push_str("searching usages...\n");
            let usages = find_usages(root, &files, symbol, "", 0);
            if usages.is_empty() {
                out.push_str("no usages found either.\n");
            } else {
                format_usages(&mut out, &usages);
            }
        }
        Some(d) => {
            out.push_str("DEFINED: "); out.push_str(d.kind); out.push(' ');
            out.push_str(&d.name); out.push_str(" @ ");
            out.push_str(&d.file); out.push(':');
            crate::text::itoa_push(&mut out, d.line); out.push('\n');
            out.push_str("  "); out.push_str(&d.signature); out.push('\n');
            if all_syms.len() > 1 {
                out.push_str("\nALSO DEFINED:\n");
                for s in &all_syms[1..] {
                    out.push_str("  "); out.push_str(s.kind); out.push_str(" @ ");
                    out.push_str(&s.file); out.push(':');
                    crate::text::itoa_push(&mut out, s.line); out.push('\n');
                }
            }
            let usages = find_usages(root, &files, symbol, &d.file, d.line);
            if usages.is_empty() {
                out.push_str("\nNO USAGES (dead code?)\n");
            } else {
                out.push_str("\nUSAGES (");
                crate::text::itoa_push(&mut out, usages.len() as u32);
                out.push_str("):\n");
                format_usages(&mut out, &usages);
            }
        }
    }
    out
}

fn format_usages(out: &mut String, usages: &[crate::codegraph::Usage]) {
    let mut by_file: Vec<(&str, Vec<&crate::codegraph::Usage>)> = Vec::new();
    for u in usages {
        if let Some(entry) = by_file.iter_mut().find(|(f, _)| *f == u.file.as_str()) {
            entry.1.push(u);
        } else {
            by_file.push((&u.file, vec![u]));
        }
    }
    for (file, refs) in &by_file {
        out.push_str("  "); out.push_str(file); out.push_str(" (");
        crate::text::itoa_push(out, refs.len() as u32);
        out.push_str(" refs):\n");
        for r in refs.iter().take(5) {
            out.push_str("    :"); crate::text::itoa_push(out, r.line);
            out.push_str("  "); out.push_str(&r.context); out.push('\n');
        }
        if refs.len() > 5 {
            out.push_str("    ... +");
            crate::text::itoa_push(out, (refs.len() - 5) as u32);
            out.push_str(" more\n");
        }
    }
}

/// Scan file content for code patterns: unsafe, FFI, alloc, error handling, perf, locks.
pub fn extract_patterns(content: &str, rel_path: &str) -> Vec<crate::codegraph::Pattern> {
    let mut pats = Vec::with_capacity(8);
    for (i, line) in content.lines().enumerate() {
        let t = line.trim();
        if t.starts_with("//") || t.starts_with("/*") || t.starts_with("* ") { continue; }
        let lineno = i as u32 + 1;
        if t.contains("unsafe ") || t.starts_with("unsafe ") {
            pats.push(crate::codegraph::Pattern { file: rel_path.into(), line: lineno,
                kind: "unsafe", context: truncate(t, 100).into() });
        }
        if t.contains("extern \"C\"") || t.contains("#[no_mangle]") || t.contains("dlopen")
            || t.contains("dlsym") || t.contains("@_silgen_name") || t.contains("@objc") {
            pats.push(crate::codegraph::Pattern { file: rel_path.into(), line: lineno,
                kind: "ffi", context: truncate(t, 100).into() });
        }
        if t.contains("mmap(") || t.contains("mmap ") || t.contains("from_raw_parts")
            || t.contains("UnsafeRawPointer") || t.contains("UnsafeMutablePointer") {
            pats.push(crate::codegraph::Pattern { file: rel_path.into(), line: lineno,
                kind: "mmap", context: truncate(t, 100).into() });
        }
        if t.contains("Mutex") || t.contains("RwLock") || t.contains("AtomicU")
            || t.contains("os_unfair_lock") || t.contains("DispatchQueue")
            || t.contains("NSLock") || t.contains("flock(") {
            pats.push(crate::codegraph::Pattern { file: rel_path.into(), line: lineno,
                kind: "lock", context: truncate(t, 100).into() });
        }
        if t.contains("simd_sum") || t.contains("simdgroup") || t.contains("threadgroup")
            || t.contains("dispatchThreadgroups") || t.contains("commandBuffer")
            || t.contains("MTLComputeCommandEncoder") {
            pats.push(crate::codegraph::Pattern { file: rel_path.into(), line: lineno,
                kind: "perf", context: truncate(t, 100).into() });
        }
    }
    pats
}

/// Extract import/dependency statements from source.
pub fn extract_imports(content: &str, ext: &str) -> Vec<String> {
    let mut imports = Vec::with_capacity(8);
    for line in content.lines() {
        let t = line.trim();
        match ext {
            "rs" => {
                if let Some(rest) = t.strip_prefix("use ") {
                    let module = rest.split("::").next().unwrap_or("").trim_start_matches("crate");
                    if module.len() >= 2 { imports.push(module.to_string()); }
                } else if let Some(rest) = t.strip_prefix("mod ") {
                    let name = rest.trim_end_matches(';').trim();
                    if name.len() >= 2 && !name.contains('{') { imports.push(name.to_string()); }
                }
            }
            "swift" => {
                if let Some(rest) = t.strip_prefix("import ") {
                    imports.push(rest.trim().to_string());
                }
            }
            "metal" | "h" | "c" | "m" => {
                if t.starts_with("#include") || t.starts_with("#import") {
                    if let Some(start) = t.find(|c: char| c == '"' || c == '<') {
                        let end = t[start+1..].find(|c: char| c == '"' || c == '>').unwrap_or(t.len()-start-1);
                        imports.push(t[start+1..start+1+end].to_string());
                    }
                }
            }
            _ => {}
        }
    }
    imports.sort();
    imports.dedup();
    imports
}

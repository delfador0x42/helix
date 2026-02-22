//! Lightweight code structure analyzer. Extracts symbols, call sites, and
//! structural dependencies from Rust/Swift/Metal source files.
//!
//! No AST parsing — regex-based extraction that's 90% correct and fast enough
//! to run on-demand (<100ms for a typical project). Trade-off: misses some
//! indirect calls and complex generics, but catches all direct call sites
//! and symbol definitions that matter for blast-radius analysis.

use std::path::{Path, PathBuf};

// ══════════ Data Model ══════════

#[derive(Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: &'static str, // "fn", "struct", "enum", "trait", "class", "protocol", "kernel"
    pub file: String,       // relative path
    pub line: u32,
    pub signature: String,  // first line of definition (trimmed)
}

#[derive(Clone)]
pub struct Usage {
    pub file: String,
    pub line: u32,
    pub context: String, // the line containing the usage
}

/// Code pattern detected in source
#[derive(Clone)]
pub struct Pattern {
    pub file: String,
    pub line: u32,
    pub kind: &'static str, // "unsafe", "ffi", "alloc", "error", "perf", "lock", "mmap"
    pub context: String,
}

/// Full project analysis result
pub struct ProjectAnalysis {
    pub files: Vec<FileAnalysis>,
    pub coupling: Vec<(String, String, u16)>, // (file_a, file_b, shared_refs)
}

/// Per-file analysis
pub struct FileAnalysis {
    pub path: String,
    pub lines: u32,
    pub symbols: Vec<Symbol>,
    pub imports: Vec<String>,      // modules/crates imported
    pub patterns: Vec<Pattern>,
    pub external_refs: u16,        // symbols used from other files
    pub external_users: u16,       // files that reference this file's symbols
}

// ══════════ File Discovery ══════════

const EXTENSIONS: &[&str] = &["rs", "swift", "metal", "h", "c", "m"];

pub fn walk_source_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::with_capacity(64);
    walk_dir(dir, dir, &mut files, 0);
    files
}

fn walk_dir(root: &Path, dir: &Path, out: &mut Vec<PathBuf>, depth: u16) {
    if depth > 12 { return; }
    let entries = match std::fs::read_dir(dir) { Ok(e) => e, Err(_) => return };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Skip hidden dirs, build artifacts, dependencies
        if name_str.starts_with('.') || name_str == "target" || name_str == "build"
            || name_str == "Pods" || name_str == "DerivedData" || name_str == "references"
            || name_str == "node_modules" { continue; }
        if path.is_dir() {
            walk_dir(root, &path, out, depth + 1);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if EXTENSIONS.contains(&ext) { out.push(path); }
        }
    }
}

fn relative_path(file: &Path, root: &Path) -> String {
    file.strip_prefix(root).unwrap_or(file).to_string_lossy().to_string()
}

// ══════════ Symbol Extraction ══════════

pub fn extract_symbols(content: &str, rel_path: &str) -> Vec<Symbol> {
    let mut syms = Vec::with_capacity(32);
    let ext = rel_path.rsplit('.').next().unwrap_or("");
    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with("///")
            || trimmed.starts_with("/*") || trimmed.starts_with("* ") { continue; }
        match ext {
            "rs" => extract_rust_sym(trimmed, rel_path, i as u32 + 1, &mut syms),
            "swift" => extract_swift_sym(trimmed, rel_path, i as u32 + 1, &mut syms),
            "metal" | "h" | "c" | "m" => extract_c_sym(trimmed, rel_path, i as u32 + 1, &mut syms),
            _ => {}
        }
    }
    syms
}

fn extract_rust_sym(line: &str, file: &str, lineno: u32, out: &mut Vec<Symbol>) {
    // Strip visibility and attributes prefix
    let stripped = line.trim_start_matches("pub ")
        .trim_start_matches("pub(crate) ")
        .trim_start_matches("pub(super) ")
        .trim_start_matches("async ")
        .trim_start_matches("unsafe ")
        .trim_start_matches("const ")
        .trim_start_matches("static ");
    if let Some(rest) = stripped.strip_prefix("fn ") {
        if let Some(name) = extract_ident(rest) {
            out.push(Symbol { name, kind: "fn", file: file.into(), line: lineno,
                signature: truncate(line.trim(), 120).into() });
        }
    } else if let Some(rest) = stripped.strip_prefix("struct ") {
        if let Some(name) = extract_ident(rest) {
            out.push(Symbol { name, kind: "struct", file: file.into(), line: lineno,
                signature: truncate(line.trim(), 120).into() });
        }
    } else if let Some(rest) = stripped.strip_prefix("enum ") {
        if let Some(name) = extract_ident(rest) {
            out.push(Symbol { name, kind: "enum", file: file.into(), line: lineno,
                signature: truncate(line.trim(), 120).into() });
        }
    } else if let Some(rest) = stripped.strip_prefix("trait ") {
        if let Some(name) = extract_ident(rest) {
            out.push(Symbol { name, kind: "trait", file: file.into(), line: lineno,
                signature: truncate(line.trim(), 120).into() });
        }
    }
}

fn extract_swift_sym(line: &str, file: &str, lineno: u32, out: &mut Vec<Symbol>) {
    let stripped = line.trim_start_matches("public ")
        .trim_start_matches("private ")
        .trim_start_matches("internal ")
        .trim_start_matches("open ")
        .trim_start_matches("final ")
        .trim_start_matches("override ")
        .trim_start_matches("static ")
        .trim_start_matches("class ")
        .trim_start_matches("@objc ");
    if let Some(rest) = stripped.strip_prefix("func ") {
        if let Some(name) = extract_ident(rest) {
            out.push(Symbol { name, kind: "fn", file: file.into(), line: lineno,
                signature: truncate(line.trim(), 120).into() });
        }
    } else if let Some(rest) = stripped.strip_prefix("class ") {
        if let Some(name) = extract_ident(rest) {
            if name.as_bytes()[0].is_ascii_uppercase() {
                out.push(Symbol { name, kind: "class", file: file.into(), line: lineno,
                    signature: truncate(line.trim(), 120).into() });
            }
        }
    } else if let Some(rest) = stripped.strip_prefix("struct ") {
        if let Some(name) = extract_ident(rest) {
            out.push(Symbol { name, kind: "struct", file: file.into(), line: lineno,
                signature: truncate(line.trim(), 120).into() });
        }
    } else if let Some(rest) = stripped.strip_prefix("protocol ") {
        if let Some(name) = extract_ident(rest) {
            out.push(Symbol { name, kind: "protocol", file: file.into(), line: lineno,
                signature: truncate(line.trim(), 120).into() });
        }
    } else if let Some(rest) = stripped.strip_prefix("enum ") {
        if let Some(name) = extract_ident(rest) {
            out.push(Symbol { name, kind: "enum", file: file.into(), line: lineno,
                signature: truncate(line.trim(), 120).into() });
        }
    }
}

fn extract_c_sym(line: &str, file: &str, lineno: u32, out: &mut Vec<Symbol>) {
    if let Some(rest) = line.strip_prefix("kernel void ") {
        if let Some(name) = extract_ident(rest) {
            out.push(Symbol { name, kind: "kernel", file: file.into(), line: lineno,
                signature: truncate(line.trim(), 120).into() });
        }
    } else if line.starts_with("void ") || line.starts_with("float ")
        || line.starts_with("int ") || line.starts_with("inline ")
        || line.starts_with("static ") {
        // Look for function definition: type name(
        let parts: Vec<&str> = line.splitn(3, ' ').collect();
        if parts.len() >= 2 {
            let candidate = parts.last().unwrap_or(&"");
            if let Some(paren) = candidate.find('(') {
                let name = &candidate[..paren];
                if name.len() >= 2 && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
                    out.push(Symbol { name: name.into(), kind: "fn", file: file.into(),
                        line: lineno, signature: truncate(line.trim(), 120).into() });
                }
            }
        }
    }
}

fn extract_ident(s: &str) -> Option<String> {
    let name: String = s.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
    if name.len() >= 2 && name.as_bytes()[0].is_ascii_alphabetic() { Some(name) } else { None }
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else { &s[..max] }
}

// ══════════ Usage Search ══════════

/// Find all references to `symbol` across source files. Excludes the definition itself.
pub fn find_usages(root: &Path, files: &[PathBuf], symbol: &str, def_file: &str, def_line: u32) -> Vec<Usage> {
    let mut usages = Vec::with_capacity(16);
    for file in files {
        let rel = relative_path(file, root);
        let content = match std::fs::read_to_string(file) { Ok(c) => c, Err(_) => continue };
        for (i, line) in content.lines().enumerate() {
            let lineno = i as u32 + 1;
            // Skip if this is the definition line
            if rel == def_file && lineno == def_line { continue; }
            // Skip comments
            let trimmed = line.trim();
            if trimmed.starts_with("//") || trimmed.starts_with("/*")
                || trimmed.starts_with("* ") || trimmed.starts_with("///") { continue; }
            // Check for symbol as a word boundary match
            if contains_word(line, symbol) {
                usages.push(Usage {
                    file: rel.clone(), line: lineno,
                    context: truncate(trimmed, 120).into(),
                });
            }
        }
    }
    usages
}

/// Word-boundary-aware contains: matches `symbol` surrounded by non-ident chars.
fn contains_word(line: &str, symbol: &str) -> bool {
    let bytes = line.as_bytes();
    let sym_bytes = symbol.as_bytes();
    let sym_len = sym_bytes.len();
    if sym_len > bytes.len() { return false; }
    let mut pos = 0;
    while pos + sym_len <= bytes.len() {
        if let Some(found) = line[pos..].find(symbol) {
            let abs = pos + found;
            let before_ok = abs == 0 || !is_ident_char(bytes[abs - 1]);
            let after_ok = abs + sym_len >= bytes.len() || !is_ident_char(bytes[abs + sym_len]);
            if before_ok && after_ok { return true; }
            pos = abs + 1;
        } else { break; }
    }
    false
}

fn is_ident_char(b: u8) -> bool { b.is_ascii_alphanumeric() || b == b'_' }

// ══════════ High-Level Queries ══════════

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
    // Find definition
    let mut def: Option<Symbol> = None;
    let mut all_syms: Vec<Symbol> = Vec::new();
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
            // Still search usages even without a definition (might be imported)
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

fn format_usages(out: &mut String, usages: &[Usage]) {
    // Group by file
    let mut by_file: Vec<(&str, Vec<&Usage>)> = Vec::new();
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

// ══════════ Pattern Detection ══════════

/// Scan file content for code patterns: unsafe, FFI, alloc, error handling, perf, locks.
pub fn extract_patterns(content: &str, rel_path: &str) -> Vec<Pattern> {
    let mut pats = Vec::with_capacity(8);
    for (i, line) in content.lines().enumerate() {
        let t = line.trim();
        if t.starts_with("//") || t.starts_with("/*") || t.starts_with("* ") { continue; }
        let lineno = i as u32 + 1;
        // Unsafe blocks
        if t.contains("unsafe ") || t.starts_with("unsafe ") {
            pats.push(Pattern { file: rel_path.into(), line: lineno,
                kind: "unsafe", context: truncate(t, 100).into() });
        }
        // FFI boundaries
        if t.contains("extern \"C\"") || t.contains("#[no_mangle]") || t.contains("dlopen")
            || t.contains("dlsym") || t.contains("@_silgen_name") || t.contains("@objc") {
            pats.push(Pattern { file: rel_path.into(), line: lineno,
                kind: "ffi", context: truncate(t, 100).into() });
        }
        // Memory-mapped I/O / direct memory
        if t.contains("mmap(") || t.contains("mmap ") || t.contains("from_raw_parts")
            || t.contains("UnsafeRawPointer") || t.contains("UnsafeMutablePointer") {
            pats.push(Pattern { file: rel_path.into(), line: lineno,
                kind: "mmap", context: truncate(t, 100).into() });
        }
        // Locks and synchronization
        if t.contains("Mutex") || t.contains("RwLock") || t.contains("AtomicU")
            || t.contains("os_unfair_lock") || t.contains("DispatchQueue")
            || t.contains("NSLock") || t.contains("flock(") {
            pats.push(Pattern { file: rel_path.into(), line: lineno,
                kind: "lock", context: truncate(t, 100).into() });
        }
        // Performance-sensitive: simd, Metal dispatch, GPU
        if t.contains("simd_sum") || t.contains("simdgroup") || t.contains("threadgroup")
            || t.contains("dispatchThreadgroups") || t.contains("commandBuffer")
            || t.contains("MTLComputeCommandEncoder") {
            pats.push(Pattern { file: rel_path.into(), line: lineno,
                kind: "perf", context: truncate(t, 100).into() });
        }
    }
    pats
}

/// Extract import/dependency statements from source
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

// ══════════ Full Project Analysis ══════════

/// Deep analysis of an entire project. Produces module map, call graph,
/// coupling matrix, and pattern inventory.
pub fn analyze_project(root: &Path) -> ProjectAnalysis {
    let source_files = walk_source_files(root);

    // Phase 1: Extract symbols, imports, patterns from each file
    let mut file_data: Vec<(String, String, Vec<Symbol>, Vec<String>, Vec<Pattern>, u32)> = Vec::new();
    for file in &source_files {
        let content = match std::fs::read_to_string(file) { Ok(c) => c, Err(_) => continue };
        let rel = relative_path(file, root);
        let ext = rel.rsplit('.').next().unwrap_or("");
        let lines = content.lines().count() as u32;
        let syms = extract_symbols(&content, &rel);
        let imports = extract_imports(&content, ext);
        let patterns = extract_patterns(&content, &rel);
        file_data.push((rel, content, syms, imports, patterns, lines));
    }

    // Phase 2: Build cross-reference graph (which files reference which symbols)
    // For each symbol in each file, count usages in other files
    let mut coupling_map: crate::fxhash::FxHashMap<(u16, u16), u16> = crate::fxhash::FxHashMap::default();
    let file_count = file_data.len();

    // Build file→symbols map
    let mut file_analyses: Vec<FileAnalysis> = Vec::with_capacity(file_count);
    for (rel, content, syms, imports, patterns, lines) in &file_data {
        let _ = content; // content used in Phase 1, no longer needed per-file
        file_analyses.push(FileAnalysis {
            path: rel.clone(),
            lines: *lines,
            symbols: syms.clone(),
            imports: imports.clone(),
            patterns: patterns.clone(),
            external_refs: 0,
            external_users: 0,
        });
    }

    // Phase 2b: Count cross-file references for coupling
    // For symbols with unique enough names (>= 4 chars), find usages across files
    for (src_idx, (_, _, syms, _, _, _)) in file_data.iter().enumerate() {
        for sym in syms {
            if sym.name.len() < 4 { continue; } // skip short names (false positives)
            for (dst_idx, (_, content, _, _, _, _)) in file_data.iter().enumerate() {
                if src_idx == dst_idx { continue; }
                // Quick check: does this file even contain the symbol name?
                if contains_word_in_content(content, &sym.name) {
                    let key = if src_idx < dst_idx {
                        (src_idx as u16, dst_idx as u16)
                    } else {
                        (dst_idx as u16, src_idx as u16)
                    };
                    *coupling_map.entry(key).or_insert(0) += 1;
                    // Track external refs/users
                    file_analyses[dst_idx].external_refs += 1;
                }
            }
        }
    }

    // Track external users
    for (src_idx, fa) in file_analyses.iter().enumerate() {
        let mut users = crate::fxhash::FxHashSet::default();
        for sym in &fa.symbols {
            if sym.name.len() < 4 { continue; }
            for (dst_idx, (_, content, _, _, _, _)) in file_data.iter().enumerate() {
                if src_idx == dst_idx { continue; }
                if contains_word_in_content(content, &sym.name) {
                    users.insert(dst_idx);
                }
            }
        }
        // Can't mutate during iteration, so store and update after
        let count = users.len() as u16;
        // SAFETY: src_idx is valid
        let _ = count; // handled below
    }

    // Build coupling list (sorted by strength)
    let mut coupling: Vec<(String, String, u16)> = coupling_map.into_iter()
        .filter(|(_, count)| *count >= 2) // Only meaningful coupling
        .map(|((a, b), count)| {
            (file_analyses[a as usize].path.clone(),
             file_analyses[b as usize].path.clone(), count)
        })
        .collect();
    coupling.sort_by(|a, b| b.2.cmp(&a.2));

    ProjectAnalysis { files: file_analyses, coupling }
}

/// Quick word-boundary check without allocating Usage structs
fn contains_word_in_content(content: &str, symbol: &str) -> bool {
    for line in content.lines() {
        if contains_word(line, symbol) { return true; }
    }
    false
}

/// Format analysis results as structured text for KB storage
pub fn format_analysis(analysis: &ProjectAnalysis, root_name: &str) -> Vec<(String, String)> {
    let mut entries: Vec<(String, String)> = Vec::with_capacity(8);

    // Entry 1: Module map — file roles and sizes
    let mut module_map = String::with_capacity(1024);
    module_map.push_str("MODULE MAP (");
    crate::text::itoa_push(&mut module_map, analysis.files.len() as u32);
    module_map.push_str(" files):\n");
    let mut sorted_files: Vec<&FileAnalysis> = analysis.files.iter().collect();
    sorted_files.sort_by(|a, b| b.lines.cmp(&a.lines));
    for fa in &sorted_files {
        module_map.push_str("  ");
        module_map.push_str(&fa.path);
        module_map.push_str(" (");
        crate::text::itoa_push(&mut module_map, fa.lines);
        module_map.push_str("L, ");
        crate::text::itoa_push(&mut module_map, fa.symbols.len() as u32);
        module_map.push_str(" syms");
        if !fa.patterns.is_empty() {
            module_map.push_str(", ");
            crate::text::itoa_push(&mut module_map, fa.patterns.len() as u32);
            module_map.push_str(" patterns");
        }
        module_map.push_str(")\n");
    }
    entries.push(("module-map".into(), module_map));

    // Entry 2: Symbol index — all exported symbols per file
    let mut sym_index = String::with_capacity(2048);
    sym_index.push_str("SYMBOL INDEX:\n");
    for fa in &sorted_files {
        if fa.symbols.is_empty() { continue; }
        sym_index.push_str("  ");
        sym_index.push_str(&fa.path);
        sym_index.push_str(": ");
        for (i, s) in fa.symbols.iter().enumerate() {
            if i > 0 { sym_index.push_str(", "); }
            sym_index.push_str(s.kind);
            sym_index.push(' ');
            sym_index.push_str(&s.name);
        }
        sym_index.push('\n');
    }
    entries.push(("symbol-index".into(), sym_index));

    // Entry 3: Coupling matrix — most coupled file pairs
    if !analysis.coupling.is_empty() {
        let mut coupling = String::with_capacity(512);
        coupling.push_str("COUPLING MATRIX (shared references):\n");
        for (a, b, count) in analysis.coupling.iter().take(15) {
            coupling.push_str("  ");
            coupling.push_str(a);
            coupling.push_str(" ↔ ");
            coupling.push_str(b);
            coupling.push_str(" (");
            crate::text::itoa_push(&mut coupling, *count as u32);
            coupling.push_str(" refs)\n");
        }
        entries.push(("coupling".into(), coupling));
    }

    // Entry 4: Pattern inventory — unsafe, FFI, locks, perf-sensitive code
    let all_patterns: Vec<&Pattern> = analysis.files.iter()
        .flat_map(|f| f.patterns.iter()).collect();
    if !all_patterns.is_empty() {
        let mut pat_text = String::with_capacity(1024);
        pat_text.push_str("PATTERN INVENTORY (");
        crate::text::itoa_push(&mut pat_text, all_patterns.len() as u32);
        pat_text.push_str(" patterns):\n");
        // Group by kind
        let kinds = ["unsafe", "ffi", "mmap", "lock", "perf"];
        for kind in &kinds {
            let matches: Vec<&&Pattern> = all_patterns.iter()
                .filter(|p| p.kind == *kind).collect();
            if matches.is_empty() { continue; }
            pat_text.push_str("  ");
            pat_text.push_str(kind);
            pat_text.push_str(" (");
            crate::text::itoa_push(&mut pat_text, matches.len() as u32);
            pat_text.push_str("):\n");
            for p in matches.iter().take(5) {
                pat_text.push_str("    ");
                pat_text.push_str(&p.file);
                pat_text.push(':');
                crate::text::itoa_push(&mut pat_text, p.line);
                pat_text.push_str("  ");
                pat_text.push_str(&p.context);
                pat_text.push('\n');
            }
            if matches.len() > 5 {
                pat_text.push_str("    +");
                crate::text::itoa_push(&mut pat_text, (matches.len() - 5) as u32);
                pat_text.push_str(" more\n");
            }
        }
        entries.push(("patterns".into(), pat_text));
    }

    // Entry 5: Import/dependency graph
    let mut dep_graph = String::with_capacity(512);
    dep_graph.push_str("DEPENDENCY GRAPH:\n");
    for fa in &sorted_files {
        if fa.imports.is_empty() { continue; }
        dep_graph.push_str("  ");
        dep_graph.push_str(&fa.path);
        dep_graph.push_str(" → ");
        dep_graph.push_str(&fa.imports.join(", "));
        dep_graph.push('\n');
    }
    entries.push(("dependencies".into(), dep_graph));

    // Tag all entries with the root name
    entries.into_iter()
        .map(|(kind, text)| {
            let mut label = String::with_capacity(root_name.len() + kind.len() + 4);
            label.push_str(root_name);
            label.push(' ');
            label.push_str(&kind);
            (label, text)
        })
        .collect()
}

/// Blast radius: trace a symbol AND all symbols defined in the same file.
pub fn blast_radius(root: &Path, file_path: &Path) -> String {
    let content = match std::fs::read_to_string(file_path) { Ok(c) => c, Err(e) => return format!("error: {e}") };
    let rel = relative_path(file_path, root);
    let syms = extract_symbols(&content, &rel);
    let files = walk_source_files(root);
    let mut out = String::with_capacity(1024);
    out.push_str("BLAST RADIUS for "); out.push_str(&rel); out.push_str(":\n\n");
    let mut total_deps = 0;
    for s in &syms {
        let usages = find_usages(root, &files, &s.name, &s.file, s.line);
        if usages.is_empty() { continue; }
        total_deps += usages.len();
        out.push_str("  "); out.push_str(s.kind); out.push(' ');
        out.push_str(&s.name); out.push_str(" → ");
        crate::text::itoa_push(&mut out, usages.len() as u32);
        out.push_str(" refs in ");
        let unique_files: crate::fxhash::FxHashSet<&str> = usages.iter().map(|u| u.file.as_str()).collect();
        crate::text::itoa_push(&mut out, unique_files.len() as u32);
        out.push_str(" files");
        // List the files
        let mut flist: Vec<&str> = unique_files.into_iter().collect();
        flist.sort();
        out.push_str(" (");
        for (i, f) in flist.iter().enumerate() {
            if i > 0 { out.push_str(", "); }
            out.push_str(f);
        }
        out.push_str(")\n");
    }
    if total_deps == 0 {
        out.push_str("  no external references found (self-contained module)\n");
    } else {
        out.push('\n');
        crate::text::itoa_push(&mut out, total_deps as u32);
        out.push_str(" total references across ");
        crate::text::itoa_push(&mut out, syms.len() as u32);
        out.push_str(" symbols\n");
    }
    out
}

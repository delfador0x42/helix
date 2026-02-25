//! Lightweight code structure analyzer: symbols, call sites, structural dependencies.
//! No AST parsing — regex-based extraction, 90% correct, fast (<100ms).

use std::path::{Path, PathBuf};

// Re-exports so callers can keep using crate::codegraph::*
pub use crate::codegraph_query::{file_symbols, trace_symbol};
pub use crate::analysis::{analyze_project, format_analysis, blast_radius};

// ── Data Model ──

#[derive(Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: &'static str,
    pub file: String,
    pub line: u32,
    pub signature: String,
}

#[derive(Clone)]
pub struct Usage {
    pub file: String,
    pub line: u32,
    pub context: String,
}

#[derive(Clone)]
pub struct Pattern {
    pub file: String,
    pub line: u32,
    pub kind: &'static str,
    pub context: String,
}

pub struct ProjectAnalysis {
    pub files: Vec<FileAnalysis>,
    pub coupling: Vec<(String, String, u16)>,
}

pub struct FileAnalysis {
    pub path: String,
    pub lines: u32,
    pub symbols: Vec<Symbol>,
    pub imports: Vec<String>,
    pub patterns: Vec<Pattern>,
    pub external_refs: u16,
}

// ── File Discovery ──

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

pub fn relative_path(file: &Path, root: &Path) -> String {
    file.strip_prefix(root).unwrap_or(file).to_string_lossy().to_string()
}

// ── Symbol Extraction ──

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
    let stripped = line.trim_start_matches("pub ")
        .trim_start_matches("pub(crate) ").trim_start_matches("pub(super) ")
        .trim_start_matches("async ").trim_start_matches("unsafe ")
        .trim_start_matches("const ").trim_start_matches("static ");
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
        .trim_start_matches("private ").trim_start_matches("internal ")
        .trim_start_matches("open ").trim_start_matches("final ")
        .trim_start_matches("override ").trim_start_matches("static ")
        .trim_start_matches("class ").trim_start_matches("@objc ");
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

pub fn truncate(s: &str, max: usize) -> &str {
    crate::text::truncate(s, max)
}

// ── Usage Search ──

pub fn find_usages(root: &Path, files: &[PathBuf], symbol: &str, def_file: &str, def_line: u32) -> Vec<Usage> {
    let mut usages = Vec::with_capacity(16);
    for file in files {
        let rel = relative_path(file, root);
        let content = match std::fs::read_to_string(file) { Ok(c) => c, Err(_) => continue };
        for (i, line) in content.lines().enumerate() {
            let lineno = i as u32 + 1;
            if rel == def_file && lineno == def_line { continue; }
            let trimmed = line.trim();
            if trimmed.starts_with("//") || trimmed.starts_with("/*")
                || trimmed.starts_with("* ") || trimmed.starts_with("///") { continue; }
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

pub fn contains_word(line: &str, symbol: &str) -> bool {
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

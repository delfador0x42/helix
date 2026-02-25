//! Ambient context formatting + file symbol extraction (Layer 2).

use std::borrow::Cow;

pub(crate) fn format_ambient_output(
    pool: &[Cow<str>], code_blast: &[String], code_analysis: &[String],
    counts: &[usize; 6], l6: usize, l7: usize,
    filename: &str, topic_prefix: Option<&str>, stem: &str, syms: &[&str],
    project_name: Option<&str>, file_path: &str,
) -> String {
    let est = pool.iter().map(|s| s.len() + 4).sum::<usize>() + 8 * 40
        + code_blast.iter().map(|s| s.len() + 4).sum::<usize>()
        + code_analysis.iter().map(|s| s.len() + 4).sum::<usize>();
    let mut out = String::with_capacity(est);
    let mut pool_idx = 0;
    for (i, &count) in counts.iter().enumerate() {
        if count == 0 { continue; }
        if !out.is_empty() { out.push_str("---\n"); }
        match i {
            0 => { out.push_str("source-linked ("); out.push_str(filename); out.push_str("):\n"); }
            1 => { out.push_str("topic context ("); out.push_str(topic_prefix.unwrap_or("?")); out.push_str("/*):\n"); }
            2 => out.push_str("symbol context:\n"),
            3 => { out.push_str("related ("); out.push_str(stem); out.push_str("):\n"); }
            4 => out.push_str("structural coupling:\n"),
            5 => {
                out.push_str("REFACTOR IMPACT (symbols modified: ");
                for (j, sym) in syms.iter().enumerate() {
                    if j > 0 { out.push_str(", "); }
                    out.push_str(sym);
                }
                out.push_str("):\n");
            }
            _ => {}
        }
        for _ in 0..count {
            out.push_str("  "); out.push_str(&pool[pool_idx]); out.push('\n');
            pool_idx += 1;
        }
    }
    if l6 > 0 {
        if !out.is_empty() { out.push_str("---\n"); }
        out.push_str("CODE BLAST RADIUS (callers/references that may break):\n");
        for line in code_blast { out.push_str("  "); out.push_str(line); out.push('\n'); }
    }
    if l7 > 0 {
        if !out.is_empty() { out.push_str("---\n"); }
        let pname = project_name.unwrap_or("project");
        out.push_str("PROJECT ANALYSIS ("); out.push_str(pname); out.push_str("):\n");
        for line in code_analysis { out.push_str("  "); out.push_str(line); out.push('\n'); }
    }
    if pool_idx < pool.len() {
        let dir_name = std::path::Path::new(file_path).parent()
            .and_then(|p| p.file_name()).and_then(|f| f.to_str()).unwrap_or("directory");
        if !out.is_empty() { out.push_str("---\n"); }
        out.push_str("directory context ("); out.push_str(dir_name); out.push_str("):\n");
        while pool_idx < pool.len() {
            out.push_str("  "); out.push_str(&pool[pool_idx]); out.push('\n');
            pool_idx += 1;
        }
    }
    out
}

// ══════════ Symbol Extraction (Layer 2) ══════════

fn extract_file_symbols(path: &str) -> Vec<String> {
    let content = match std::fs::read_to_string(path) { Ok(c) => c, Err(_) => return vec![] };
    static KEYWORDS: &[&str] = &[
        "fn ", "struct ", "enum ", "trait ",
        "func ", "class ", "protocol ", "extension ",
    ];
    let mut symbols = Vec::with_capacity(16);
    for line in content.lines().take(500) {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || trimmed.starts_with("///")
            || trimmed.starts_with('#') || trimmed.starts_with("/*") { continue; }
        for kw in KEYWORDS {
            if let Some(pos) = trimmed.find(kw) {
                let rest = &trimmed[pos + kw.len()..];
                let rest = if *kw == "fn " || *kw == "func " { rest }
                else {
                    rest.trim_start_matches(|c: char| c == '<' || c == '\'')
                        .split(|c: char| c == '>' || c == ' ')
                        .next().unwrap_or(rest)
                };
                let name: String = rest.chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                if name.len() >= 3 && name.as_bytes()[0].is_ascii_alphabetic() {
                    symbols.push(name);
                }
            }
        }
    }
    symbols.sort(); symbols.dedup(); symbols.truncate(20);
    symbols
}

const SYM_CACHE_PATH: &str = "/tmp/helix-sym-cache";

pub(crate) fn cached_file_symbols(path: &str) -> Vec<String> {
    let mtime = match std::fs::metadata(path) {
        Ok(m) => m.modified().ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs()).unwrap_or(0),
        Err(_) => return vec![],
    };
    if let Ok(cache) = std::fs::read_to_string(SYM_CACHE_PATH) {
        let mut lines = cache.lines();
        if let (Some(cp), Some(cm)) = (lines.next(), lines.next()) {
            if cp == path { if let Ok(mt) = cm.parse::<u64>() {
                if mt == mtime { return lines.map(|l| l.to_string()).collect(); }
            }}
        }
    }
    let syms = extract_file_symbols(path);
    let mut buf = String::with_capacity(path.len() + 32 + syms.len() * 20);
    buf.push_str(path); buf.push('\n'); crate::text::itoa_push_u64(&mut buf, mtime);
    for sym in &syms { buf.push('\n'); buf.push_str(sym); }
    std::fs::write(SYM_CACHE_PATH, buf.as_bytes()).ok();
    syms
}

pub(crate) fn build_symbol_query(symbols: &[String], stem: &str) -> String {
    let stem_lower = stem.to_lowercase();
    let mut terms = Vec::with_capacity(symbols.len());
    for sym in symbols {
        for tok in crate::text::tokenize(sym) {
            if tok.len() >= 3 && tok != stem_lower { terms.push(tok); }
        }
    }
    terms.sort(); terms.dedup(); terms.truncate(15);
    terms.join(" ")
}

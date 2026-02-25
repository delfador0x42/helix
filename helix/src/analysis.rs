//! Full project analysis: module map, call graph, coupling matrix, pattern inventory.

use std::path::Path;
use crate::codegraph::*;
use crate::codegraph_query::{extract_patterns, extract_imports};

/// Deep analysis of an entire project.
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

    // Phase 2: Build cross-reference graph
    let mut coupling_map: crate::fxhash::FxHashMap<(u16, u16), u16> = crate::fxhash::FxHashMap::default();
    let file_count = file_data.len();

    let mut file_analyses: Vec<FileAnalysis> = Vec::with_capacity(file_count);
    for (rel, _content, syms, imports, patterns, lines) in &file_data {
        file_analyses.push(FileAnalysis {
            path: rel.clone(),
            lines: *lines,
            symbols: syms.clone(),
            imports: imports.clone(),
            patterns: patterns.clone(),
            external_refs: 0,
        });
    }

    for (src_idx, (_, _, syms, _, _, _)) in file_data.iter().enumerate() {
        for sym in syms {
            if sym.name.len() < 4 { continue; }
            for (dst_idx, (_, content, _, _, _, _)) in file_data.iter().enumerate() {
                if src_idx == dst_idx { continue; }
                if contains_word_in_content(content, &sym.name) {
                    let key = if src_idx < dst_idx {
                        (src_idx as u16, dst_idx as u16)
                    } else {
                        (dst_idx as u16, src_idx as u16)
                    };
                    *coupling_map.entry(key).or_insert(0) += 1;
                    file_analyses[dst_idx].external_refs += 1;
                }
            }
        }
    }

    let mut coupling: Vec<(String, String, u16)> = coupling_map.into_iter()
        .filter(|(_, count)| *count >= 2)
        .map(|((a, b), count)| {
            (file_analyses[a as usize].path.clone(),
             file_analyses[b as usize].path.clone(), count)
        })
        .collect();
    coupling.sort_by(|a, b| b.2.cmp(&a.2));

    ProjectAnalysis { files: file_analyses, coupling }
}

/// Quick word-boundary check without allocating Usage structs.
fn contains_word_in_content(content: &str, symbol: &str) -> bool {
    for line in content.lines() {
        if contains_word(line, symbol) { return true; }
    }
    false
}

/// Format analysis results as structured text for KB storage.
pub fn format_analysis(analysis: &ProjectAnalysis, _root_name: &str) -> Vec<(String, String)> {
    let mut entries: Vec<(String, String)> = Vec::with_capacity(8);

    // Entry 1: Module map
    let mut module_map = String::with_capacity(1024);
    module_map.push_str("MODULE MAP (");
    crate::text::itoa_push(&mut module_map, analysis.files.len() as u32);
    module_map.push_str(" files):\n");
    let mut sorted_files: Vec<&FileAnalysis> = analysis.files.iter().collect();
    sorted_files.sort_by(|a, b| b.lines.cmp(&a.lines));
    for fa in &sorted_files {
        module_map.push_str("  "); module_map.push_str(&fa.path);
        module_map.push_str(" ("); crate::text::itoa_push(&mut module_map, fa.lines);
        module_map.push_str("L, "); crate::text::itoa_push(&mut module_map, fa.symbols.len() as u32);
        module_map.push_str(" syms");
        if !fa.patterns.is_empty() {
            module_map.push_str(", "); crate::text::itoa_push(&mut module_map, fa.patterns.len() as u32);
            module_map.push_str(" patterns");
        }
        module_map.push_str(")\n");
    }
    entries.push(("module-map".into(), module_map));

    // Entry 2: Symbol index
    let mut sym_index = String::with_capacity(2048);
    sym_index.push_str("SYMBOL INDEX:\n");
    for fa in &sorted_files {
        if fa.symbols.is_empty() { continue; }
        sym_index.push_str("  "); sym_index.push_str(&fa.path); sym_index.push_str(": ");
        for (i, s) in fa.symbols.iter().enumerate() {
            if i > 0 { sym_index.push_str(", "); }
            sym_index.push_str(s.kind); sym_index.push(' '); sym_index.push_str(&s.name);
        }
        sym_index.push('\n');
    }
    entries.push(("symbol-index".into(), sym_index));

    // Entry 3: Coupling matrix
    if !analysis.coupling.is_empty() {
        let mut coupling = String::with_capacity(512);
        coupling.push_str("COUPLING MATRIX (shared references):\n");
        for (a, b, count) in analysis.coupling.iter().take(15) {
            coupling.push_str("  "); coupling.push_str(a);
            coupling.push_str(" ↔ "); coupling.push_str(b);
            coupling.push_str(" ("); crate::text::itoa_push(&mut coupling, *count as u32);
            coupling.push_str(" refs)\n");
        }
        entries.push(("coupling".into(), coupling));
    }

    // Entry 4: Pattern inventory
    let all_patterns: Vec<&Pattern> = analysis.files.iter()
        .flat_map(|f| f.patterns.iter()).collect();
    if !all_patterns.is_empty() {
        let mut pat_text = String::with_capacity(1024);
        pat_text.push_str("PATTERN INVENTORY (");
        crate::text::itoa_push(&mut pat_text, all_patterns.len() as u32);
        pat_text.push_str(" patterns):\n");
        let kinds = ["unsafe", "ffi", "mmap", "lock", "perf"];
        for kind in &kinds {
            let matches: Vec<&&Pattern> = all_patterns.iter()
                .filter(|p| p.kind == *kind).collect();
            if matches.is_empty() { continue; }
            pat_text.push_str("  "); pat_text.push_str(kind);
            pat_text.push_str(" ("); crate::text::itoa_push(&mut pat_text, matches.len() as u32);
            pat_text.push_str("):\n");
            for p in matches.iter().take(5) {
                pat_text.push_str("    "); pat_text.push_str(&p.file); pat_text.push(':');
                crate::text::itoa_push(&mut pat_text, p.line);
                pat_text.push_str("  "); pat_text.push_str(&p.context); pat_text.push('\n');
            }
            if matches.len() > 5 {
                pat_text.push_str("    +"); crate::text::itoa_push(&mut pat_text, (matches.len() - 5) as u32);
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
        dep_graph.push_str("  "); dep_graph.push_str(&fa.path);
        dep_graph.push_str(" → "); dep_graph.push_str(&fa.imports.join(", "));
        dep_graph.push('\n');
    }
    entries.push(("dependencies".into(), dep_graph));

    // Tag all entries with the root name
    entries.into_iter()
        .map(|(kind, text)| {
            let mut label = String::with_capacity(_root_name.len() + kind.len() + 4);
            label.push_str(_root_name);
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

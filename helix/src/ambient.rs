//! Smart ambient context engine — 7-layer knowledge injection on file access.

use std::path::Path;

/// Extract symbols removed by an Edit (refactor impact detection).
pub(crate) fn extract_removed_syms(input: &crate::json::Value, stem: &str) -> Vec<String> {
    let ti = input.get("tool_input");
    let old = ti.and_then(|t| t.get("old_string")).and_then(|v| v.as_str()).unwrap_or("");
    let new_str = ti.and_then(|t| t.get("new_string")).and_then(|v| v.as_str()).unwrap_or("");
    if old.len() < 8 { return vec![]; }
    let extract = |s: &str| -> crate::fxhash::FxHashSet<String> {
        s.split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| w.len() >= 4 && w.bytes().any(|b| b.is_ascii_alphabetic()))
            .map(|w| w.to_lowercase()).collect()
    };
    let old_tokens: crate::fxhash::FxHashSet<String> = extract(old)
        .into_iter().filter(|t| t != stem).collect();
    let new_tokens = extract(new_str);
    let mut removed: Vec<String> = old_tokens.into_iter()
        .filter(|t| !new_tokens.contains(t)).collect();
    removed.sort();
    removed.truncate(3);
    removed
}

/// 7-layer smart ambient context with deduplication.
/// Cow<str>: Layer 1 borrows from mmap (zero alloc), other layers own.
pub(crate) fn query_ambient(data: &[u8], stem: &str, file_path: &str, syms: &[&str], mut session: Option<&mut crate::session::Session>) -> String {
    let filename = std::path::Path::new(file_path)
        .file_name().and_then(|f| f.to_str()).unwrap_or(stem);
    let mut seen = crate::fxhash::FxHashSet::default();
    if let Some(ref s) = session {
        for &eid in s.injected.iter() { seen.insert(eid); }
    }
    let mut pool: Vec<std::borrow::Cow<str>> = Vec::with_capacity(32);

    // Layer 1: Source-path matches
    let l1_start = pool.len();
    let source_ids = source_entries_for_file(data, filename);
    for &eid in &source_ids {
        seen.insert(eid);
        if let Ok(snip) = crate::index::entry_snippet_ref(data, eid) {
            if !snip.is_empty() { pool.push(std::borrow::Cow::Borrowed(snip)); }
        }
    }
    let l1 = pool.len() - l1_start;

    // Layer 1.5: Topic hierarchy
    let l1h_start = pool.len();
    let topic_prefix = derive_topic_prefix(file_path);
    if let Some(ref prefix) = topic_prefix {
        for h in topic_prefix_entries(data, prefix, 5) {
            if seen.insert(h.entry_id) {
                pool.push(std::borrow::Cow::Owned(h.snippet));
                if pool.len() - l1h_start >= 3 { break; }
            }
        }
    }
    let l1h = pool.len() - l1h_start;

    // Layer 2: Symbol-based OR search (skip if L1 >= 5)
    let l2_start = pool.len();
    if source_ids.len() < 5 {
        let file_symbols = crate::ambient_fmt::cached_file_symbols(file_path);
        if !file_symbols.is_empty() {
            let query = crate::ambient_fmt::build_symbol_query(&file_symbols, stem);
            if !query.is_empty() {
                for h in idx_search_or(data, &query, 8) {
                    if seen.insert(h.entry_id) {
                        pool.push(std::borrow::Cow::Owned(h.snippet));
                        if pool.len() - l2_start >= 5 { break; }
                    }
                }
            }
        }
    }
    let l2 = pool.len() - l2_start;

    // Layer 3: Global BM25 (stem keyword)
    let l3_start = pool.len();
    for h in idx_search(data, stem, 5) {
        if seen.insert(h.entry_id) {
            pool.push(std::borrow::Cow::Owned(h.snippet));
            if pool.len() - l3_start >= 3 { break; }
        }
    }
    let l3 = pool.len() - l3_start;

    // Layer 4: Structural coupling
    let l4_start = pool.len();
    let mut sq_buf = [0u8; 128];
    let sq_prefix = b"structural ";
    let sq_len = sq_prefix.len() + stem.len();
    let structural = if sq_len <= sq_buf.len() {
        sq_buf[..sq_prefix.len()].copy_from_slice(sq_prefix);
        sq_buf[sq_prefix.len()..sq_len].copy_from_slice(stem.as_bytes());
        let sq = unsafe { std::str::from_utf8_unchecked(&sq_buf[..sq_len]) };
        idx_search(data, sq, 3)
    } else {
        idx_search(data, &format!("structural {stem}"), 3)
    };
    for h in structural {
        if seen.insert(h.entry_id) { pool.push(std::borrow::Cow::Owned(h.snippet)); }
    }
    let l4 = pool.len() - l4_start;

    // Layer 5: Refactor impact (Edit only, KB search)
    let l5_start = pool.len();
    for sym in syms {
        for hit in idx_search(data, sym, 3) {
            if seen.insert(hit.entry_id) { pool.push(std::borrow::Cow::Owned(hit.snippet)); }
        }
    }
    let l5 = pool.len() - l5_start;

    // Project root: compute once for Layer 6 + Layer 7
    let project_root = find_project_root(file_path);
    let project_name = project_root.as_ref()
        .and_then(|root| root.file_name().and_then(|n| n.to_str()).map(|s| s.to_string()));

    // Layer 6: Code blast radius (Edit only, codegraph)
    let mut code_blast: Vec<String> = Vec::new();
    if !syms.is_empty() {
        if let Some(ref root) = project_root {
            let files = crate::codegraph::walk_source_files(root);
            for sym in syms {
                let usages = crate::codegraph::find_usages(root, &files, sym, "", 0);
                if usages.is_empty() { continue; }
                let mut line = String::with_capacity(64);
                line.push_str(sym); line.push_str(" → ");
                crate::text::itoa_push(&mut line, usages.len() as u32);
                line.push_str(" refs: ");
                let mut shown_files = crate::fxhash::FxHashSet::default();
                let mut count = 0;
                for u in &usages {
                    if count >= 3 { break; }
                    if shown_files.insert(u.file.as_str()) {
                        if count > 0 { line.push_str(", "); }
                        line.push_str(&u.file); line.push(':');
                        crate::text::itoa_push(&mut line, u.line);
                        count += 1;
                    }
                }
                if usages.len() > 3 {
                    line.push_str(" +");
                    crate::text::itoa_push(&mut line, (usages.len() - 3) as u32);
                    line.push_str(" more");
                }
                code_blast.push(line);
            }
        }
    }
    let l6 = code_blast.len();

    // Layer 7: Project code analysis — inject on first touch per project
    let mut code_analysis: Vec<String> = Vec::new();
    if let Some(ref pname) = project_name {
        let already = session.as_ref().map(|s| s.project_analyzed(pname)).unwrap_or(false);
        if !already {
            let sanitized = pname.to_lowercase().replace('_', "-");
            let topic = format!("code-{sanitized}");
            for hit in idx_search(data, &topic, 10) {
                if seen.insert(hit.entry_id) { code_analysis.push(hit.snippet); }
            }
            if code_analysis.is_empty() {
                let query = format!("code-analysis {pname}");
                for hit in idx_search(data, &query, 8) {
                    if seen.insert(hit.entry_id) { code_analysis.push(hit.snippet); }
                }
            }
            if let Some(ref mut sess) = session {
                sess.mark_project_analyzed(pname);
            }
        }
    }
    let l7 = code_analysis.len();

    // Directory fallback
    if pool.is_empty() || (l1 == 0 && l3 == 0) {
        let dir_name = std::path::Path::new(file_path).parent()
            .and_then(|p| p.file_name()).and_then(|f| f.to_str()).unwrap_or("");
        if dir_name.len() >= 3 && dir_name != stem {
            for h in idx_search(data, dir_name, 3) {
                if seen.insert(h.entry_id) {
                    pool.push(std::borrow::Cow::Owned(h.snippet));
                }
            }
        }
    }
    if pool.is_empty() { return String::new(); }

    // Session bookkeeping
    if let Some(sess) = session {
        for &eid in &seen {
            if !sess.was_injected(eid) { sess.mark_injected(eid); }
        }
        let mut topic_hits: crate::fxhash::FxHashMap<u16, u16> = crate::fxhash::FxHashMap::default();
        for &eid in &seen {
            if let Ok(tid) = crate::index::entry_topic_id(data, eid) {
                *topic_hits.entry(tid).or_insert(0) += 1;
            }
        }
        for (&tid, &count) in &topic_hits {
            if count >= 3 {
                if let Ok(name) = crate::index::topic_name(data, tid) {
                    sess.add_focus_topic(&name);
                }
            }
        }
    }

    // Format output
    crate::ambient_fmt::format_ambient_output(&pool, &code_blast, &code_analysis,
        &[l1, l1h, l2, l3, l4, l5], l6, l7,
        filename, topic_prefix.as_deref(), stem, syms,
        project_name.as_deref(), file_path)
}

// ══════════ Helpers ══════════

fn idx_search(data: &[u8], query: &str, limit: usize) -> Vec<crate::index::SearchHit> {
    crate::index::search_index(data, query, &crate::index::FilterPred::none(), limit, true)
        .unwrap_or_default()
}

fn idx_search_or(data: &[u8], query: &str, limit: usize) -> Vec<crate::index::SearchHit> {
    crate::index::search_index(data, query, &crate::index::FilterPred::none(), limit, false)
        .unwrap_or_default()
}

fn derive_topic_prefix(file_path: &str) -> Option<String> {
    let p = std::path::Path::new(file_path);
    let markers = ["Cargo.toml", "Package.swift", ".git", "Makefile", "build.rs"];
    let mut dir = p.parent()?;
    for _ in 0..8 {
        for m in &markers {
            if dir.join(m).exists() {
                return dir.file_name().and_then(|n| n.to_str())
                    .map(|s| crate::config::sanitize_topic(s));
            }
        }
        dir = dir.parent()?;
    }
    None
}

fn source_entries_for_file(data: &[u8], filename: &str) -> Vec<u32> {
    crate::index::sourced_entries(data).unwrap_or_default()
        .into_iter().filter(|(_, _, path, _)| path.contains(filename))
        .map(|(eid, _, _, _)| eid).collect()
}

fn topic_prefix_entries(data: &[u8], prefix: &str, limit: usize) -> Vec<crate::index::SearchHit> {
    let topics = crate::index::topic_table(data).unwrap_or_default();
    let matching_ids: Vec<u16> = topics.iter()
        .filter(|(_, name, _)| crate::config::topic_matches_query(name, prefix))
        .map(|(id, _, _)| *id).collect();
    if matching_ids.is_empty() { return Vec::new(); }
    let pred = crate::index::FilterPred {
        topic_filter: crate::index::TopicFilter::Prefix(matching_ids),
        after_days: 0, before_days: u16::MAX, tag_mask: 0, source_needle: None,
    };
    let leaf = prefix.rsplit('/').next().unwrap_or(prefix);
    crate::index::search_index(data, leaf, &pred, limit, false).unwrap_or_default()
}

fn find_project_root(file_path: &str) -> Option<std::path::PathBuf> {
    let markers = ["Cargo.toml", "Package.swift", ".git", "Makefile", "build.rs"];
    let mut dir = Path::new(file_path).parent()?;
    for _ in 0..8 {
        for m in &markers { if dir.join(m).exists() { return Some(dir.to_path_buf()); } }
        dir = dir.parent()?;
    }
    None
}


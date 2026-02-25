//! Brief formatting: classification by tag, category display, entry rendering.

use std::fmt::Write;
use std::collections::BTreeSet;

// ══════════ Categories + Classification ══════════

const CATEGORIES: &[(&str, &[&str])] = &[
    ("ARCHITECTURE", &["architecture", "module-map", "overview"]),
    ("DATA FLOW", &["pipeline", "data-flow"]),
    ("INVARIANTS", &["invariant", "constraint"]),
    ("GOTCHAS", &["gotcha"]),
    ("DECISIONS", &["decision"]),
    ("HOW-TO", &["how-to", "workflow"]),
    ("PERFORMANCE", &["performance", "zero-alloc"]),
    ("GAPS", &["gap", "missing"]),
];

struct Classification {
    categories: Vec<(&'static str, Vec<usize>)>,
    untagged: Vec<usize>,
}

fn classify(entries: &[crate::brief::Compressed]) -> Classification {
    let fc_lower: Vec<String> = entries.iter()
        .map(|e| crate::brief::first_content(&e.body).to_lowercase()).collect();
    let mut assigned = vec![false; entries.len()];
    let mut categories: Vec<(&'static str, Vec<usize>)> = Vec::new();
    for &(cat, patterns) in CATEGORIES {
        let mut group = Vec::new();
        for (i, e) in entries.iter().enumerate() {
            if assigned[i] { continue; }
            let tag_match = e.tags.iter().any(|t| patterns.contains(&t.as_str()));
            let kw_match = patterns.iter().any(|p| fc_lower[i].contains(p));
            if tag_match || kw_match { group.push(i); assigned[i] = true; }
        }
        if !group.is_empty() { categories.push((cat, group)); }
    }
    let untagged: Vec<usize> = (0..entries.len()).filter(|i| !assigned[*i]).collect();
    Classification { categories, untagged }
}

// ══════════ Formatting ══════════

pub(crate) fn format_output(
    entries: &[crate::brief::Compressed], query: &str, raw_count: usize,
    primary: &[String], detail: crate::brief::Detail, since: Option<u64>,
    focus: Option<&[String]>,
) -> String {
    let cls = classify(entries);
    let n_topics = entries.iter().map(|e| e.topic.as_str()).collect::<BTreeSet<_>>().len();
    let mut out = String::new();
    let since_note = since.map(|h| format!(" (since {}h)", h)).unwrap_or_default();
    let _ = writeln!(out, "=== {}{} === {} → {} compressed, {} topics\n",
        query.to_uppercase(), since_note, raw_count, entries.len(), n_topics);

    // Topics line
    let mut info: std::collections::BTreeMap<&str, (usize, i64)> = std::collections::BTreeMap::new();
    for e in entries {
        let (c, d) = info.entry(&e.topic).or_insert((0, i64::MAX));
        *c += 1; if e.days_old < *d { *d = e.days_old; }
    }
    let _ = write!(out, "TOPICS:");
    for t in primary {
        if let Some((c, d)) = info.get(t.as_str()) {
            let fresh = match *d { 0 => ", today", 1 => ", 1d", 2..=7 => ", week", _ => "" };
            let _ = write!(out, " {} ({}{})", t, c, fresh);
        }
    }
    let _ = writeln!(out, "\n");

    match detail {
        crate::brief::Detail::Summary => {
            let _ = write!(out, "CATEGORIES:");
            for (cat, indices) in &cls.categories { let _ = write!(out, " {} {}", cat, indices.len()); }
            if !cls.untagged.is_empty() { let _ = write!(out, " | UNTAGGED {}", cls.untagged.len()); }
            let _ = writeln!(out, "\n");
            let mut hot: Vec<usize> = (0..entries.len()).collect();
            hot.sort_by(|&a, &b| entries[b].relevance.partial_cmp(&entries[a].relevance).unwrap_or(std::cmp::Ordering::Equal));
            let _ = writeln!(out, "HOT:");
            for &i in hot.iter().take(5) { format_oneliner(&mut out, &entries[i]); }
        }
        crate::brief::Detail::Scan => {
            for (cat, indices) in &cls.categories {
                if !cat_matches(cat, focus) { continue; }
                let _ = writeln!(out, "--- {} ({}) ---", cat, indices.len());
                for &i in indices.iter().take(3) { format_oneliner(&mut out, &entries[i]); }
                if indices.len() > 3 { let _ = writeln!(out, "  ... +{} more", indices.len() - 3); }
                let _ = writeln!(out);
            }
            if !cls.untagged.is_empty() && cat_matches("UNTAGGED", focus) {
                let _ = writeln!(out, "--- UNTAGGED ({}) ---", cls.untagged.len());
                for &i in cls.untagged.iter().take(3) { format_oneliner(&mut out, &entries[i]); }
                let _ = writeln!(out);
            }
        }
        crate::brief::Detail::Full => {
            for (cat, indices) in &cls.categories {
                if !cat_matches(cat, focus) { continue; }
                let _ = writeln!(out, "--- {} ({}) ---", cat, indices.len());
                let body_limit = match *cat {
                    "DATA FLOW" | "INVARIANTS" => 10,
                    "DECISIONS" | "ARCHITECTURE" => 8,
                    _ => 5,
                };
                for &i in indices.iter().take(5) { format_entry(&mut out, &entries[i], body_limit); }
                for &i in indices.iter().skip(5).take(10) { format_oneliner(&mut out, &entries[i]); }
                if indices.len() > 15 { let _ = writeln!(out, "  ... +{} more\n", indices.len() - 15); }
            }
            if !cls.untagged.is_empty() && cat_matches("UNTAGGED", focus) {
                let _ = writeln!(out, "--- UNTAGGED ({}) ---", cls.untagged.len());
                for &i in cls.untagged.iter().take(5) { format_oneliner(&mut out, &entries[i]); }
                let _ = writeln!(out);
            }
        }
    }

    let pct = if raw_count > 0 { 100 - (entries.len() * 100 / raw_count) } else { 0 };
    let _ = writeln!(out, "\nSTATS: {} compressed ({}% reduction)", entries.len(), pct);
    out
}

fn format_entry(out: &mut String, e: &crate::brief::Compressed, max_lines: usize) {
    out.push('['); out.push_str(&e.topic); out.push_str("] ");
    out.push_str(&e.date); out.push_str(freshness(e.days_old));
    if let Some(ref s) = e.source { out.push_str(" → "); out.push_str(s); }
    out.push('\n');
    if let Some(ref chain) = e.chain {
        let _ = writeln!(out, "  {}", crate::text::truncate(chain, 120));
    }
    let lines: Vec<&str> = e.body.lines().filter(|l| !crate::text::is_metadata_line(l)).collect();
    for l in lines.iter().take(max_lines) { let _ = writeln!(out, "  {}", l.trim()); }
    if lines.len() > max_lines { let _ = writeln!(out, "  ...({} more lines)", lines.len() - max_lines); }
    let _ = writeln!(out);
}

fn format_oneliner(out: &mut String, e: &crate::brief::Compressed) {
    let fc = crate::text::truncate(crate::brief::first_content(&e.body), 80);
    out.push_str("  ["); out.push_str(&e.topic); out.push_str("] ");
    out.push_str(fc);
    if let Some(ref s) = e.source { out.push_str(" → "); out.push_str(s); }
    match &e.chain {
        Some(c) if c.starts_with("superseded") => out.push_str(" [SUPERSEDED]"),
        Some(c) => { out.push_str(" ("); out.push_str(crate::text::truncate(c, 60)); out.push(')'); }
        None => {}
    }
    out.push_str(freshness(e.days_old));
    out.push('\n');
}

fn freshness(days: i64) -> &'static str {
    match days { 0 => " [TODAY]", 1 => " [1d]", 2..=7 => " [week]", _ => "" }
}

fn cat_matches(cat: &str, focus: Option<&[String]>) -> bool {
    match focus {
        None => true,
        Some(cats) => {
            let up = cat.to_uppercase();
            cats.iter().any(|f| up.contains(f.as_str()) || f.contains(&up))
        }
    }
}

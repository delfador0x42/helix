//! MCP tool schemas — JSON definitions for all 8 tool endpoints.

use crate::json::Value;
use std::sync::{Arc, Mutex};

static TOOL_CACHE: Mutex<Option<Arc<str>>> = Mutex::new(None);

pub(crate) fn tool_list_json() -> Arc<str> {
    if let Ok(g) = TOOL_CACHE.lock() { if let Some(ref c) = *g { return Arc::clone(c); } }
    let result = Value::Obj(vec![("tools".into(), tool_list())]);
    let json: Arc<str> = result.to_string().into();
    if let Ok(mut g) = TOOL_CACHE.lock() { *g = Some(Arc::clone(&json)); }
    json
}

fn tool(name: &str, desc: &str, req: &[&str], props: &[(&str, &str, &str)]) -> Value {
    Value::Obj(vec![
        ("name".into(), Value::Str(name.into())),
        ("description".into(), Value::Str(desc.into())),
        ("inputSchema".into(), Value::Obj(vec![
            ("type".into(), Value::Str("object".into())),
            ("properties".into(), Value::Obj(props.iter().map(|(n, t, d)|
                ((*n).into(), Value::Obj(vec![
                    ("type".into(), Value::Str((*t).into())),
                    ("description".into(), Value::Str((*d).into())),
                ]))
            ).collect())),
            ("required".into(), Value::Arr(req.iter().map(|r| Value::Str((*r).into())).collect())),
        ])),
    ])
}

fn batch_tool() -> Value {
    let entry_schema = Value::Obj(vec![
        ("type".into(), Value::Str("object".into())),
        ("properties".into(), Value::Obj(vec![
            ("topic".into(), Value::Obj(vec![("type".into(), Value::Str("string".into())), ("description".into(), Value::Str("Topic name".into()))])),
            ("text".into(), Value::Obj(vec![("type".into(), Value::Str("string".into())), ("description".into(), Value::Str("Entry content".into()))])),
            ("tags".into(), Value::Obj(vec![("type".into(), Value::Str("string".into())), ("description".into(), Value::Str("Comma-separated tags".into()))])),
            ("source".into(), Value::Obj(vec![("type".into(), Value::Str("string".into())), ("description".into(), Value::Str("Source file: path/to/file:line".into()))])),
        ])),
        ("required".into(), Value::Arr(vec![Value::Str("topic".into()), Value::Str("text".into())])),
    ]);
    Value::Obj(vec![
        ("name".into(), Value::Str("batch".into())),
        ("description".into(), Value::Str("Store multiple entries in one call. Each entry: {topic, text, tags?}. Faster than sequential store calls.".into())),
        ("inputSchema".into(), Value::Obj(vec![
            ("type".into(), Value::Str("object".into())),
            ("properties".into(), Value::Obj(vec![
                ("entries".into(), Value::Obj(vec![
                    ("type".into(), Value::Str("array".into())),
                    ("items".into(), entry_schema),
                    ("description".into(), Value::Str("Array of entries to store".into())),
                ])),
                ("verbose".into(), Value::Obj(vec![("type".into(), Value::Str("string".into())),
                    ("description".into(), Value::Str("Set to 'true' for per-entry details (default: terse count only)".into()))])),
            ])),
            ("required".into(), Value::Arr(vec![Value::Str("entries".into())])),
        ])),
    ])
}

pub(crate) const FILTER_PROPS: &[(&str, &str, &str)] = &[
    ("limit", "string", "Max results to return (default: unlimited)"),
    ("after", "string", "Only entries on/after date (YYYY-MM-DD or 'today'/'yesterday'/'this-week')"),
    ("before", "string", "Only entries on/before date (YYYY-MM-DD or 'today'/'yesterday')"),
    ("days", "string", "Number of days (shortcut for after=N-days-ago)"),
    ("hours", "string", "Number of hours (overrides days)"),
    ("tag", "string", "Only entries with this tag"),
    ("topic", "string", "Limit search to a single topic"),
    ("mode", "string", "Search mode: 'and' (default, all terms must match) or 'or' (any term matches)"),
];

fn tool_list() -> Value {
    let search_props: Vec<(&str, &str, &str)> = [
        ("query", "string", "Search query"),
        ("queries", "string", "JSON array of query strings for batch search (e.g. '[\"ExecPolicy\",\"relay buffer\"]'). Deduplicates results across queries."),
        ("detail", "string", "Result detail level: 'full' (complete entry), 'medium' (default, 2 lines), 'brief' (topic+first line), 'count' (match count only), 'topics' (hits per topic), 'grouped' (results by topic)"),
        ("expand", "string", "Set 'true' to return full entry bodies in medium mode (collapses search+read)"),
        ("lines", "string", "Content lines per result in medium mode (default: 2)"),
    ].into_iter().chain(FILTER_PROPS.iter().copied()).collect();

    Value::Arr(vec![
        tool("store", "Store a timestamped knowledge entry under a topic. Topic auto-inferred from session focus if omitted.",
            &["text"],
            &[("topic", "string", "Topic name"), ("text", "string", "Entry content"),
              ("tags", "string", "Comma-separated tags (e.g. 'bug,p0,iris')"),
              ("force", "string", "Set to 'true' to bypass duplicate detection"),
              ("source", "string", "Source file reference: 'path/to/file:line'. Enables staleness detection."),
              ("terse", "string", "Set to 'true' for minimal response (just first line)"),
              ("confidence", "string", "Confidence level 0.0-1.0 (default: 1.0). Affects search ranking."),
              ("links", "string", "Space-separated references: 'topic:index topic:index'. Creates narrative links.")]),
        batch_tool(),
        tool("search", "Search all knowledge files (case-insensitive). Splits CamelCase/snake_case. Falls back to OR when AND finds nothing.",
            &[], &search_props),
        tool("brief", "One-shot compressed briefing for a topic or pattern. Supports glob patterns like 'iris-*'.",
            &[],
            &[("query", "string", "Topic, keyword, or glob pattern"),
              ("detail", "string", "Output tier: 'summary' (default), 'scan', 'full'"),
              ("since", "string", "Only entries from last N hours"),
              ("focus", "string", "Comma-separated category names to show"),
              ("compact", "string", "Set to 'true' for compact meta-briefing (top 5 topics only)")]),
        tool("read", "Read the full contents of a specific topic file.", &["topic"],
            &[("topic", "string", "Topic name"), ("index", "string", "Fetch a single entry by index (0-based)")]),
        tool("edit", "Modify entries. action: append|revise|delete|tag|rename|merge.",
            &["action", "topic"],
            &[("action", "string", "Operation: append|revise|delete|tag|rename|merge"),
              ("topic", "string", "Topic name (or source topic for rename/merge)"),
              ("text", "string", "Text content (for append/revise)"),
              ("index", "string", "Entry index number"), ("match_str", "string", "Substring to find entry"),
              ("tag", "string", "Append to most recent entry with this tag (append only)"),
              ("tags", "string", "Comma-separated tags to add (tag action)"),
              ("remove", "string", "Comma-separated tags to remove (tag action)"),
              ("all", "string", "Set to 'true' to delete entire topic (delete action)"),
              ("new_name", "string", "New topic name (rename action)"),
              ("into", "string", "Target topic to merge INTO (merge action)")]),
        tool("topics", "Browse & maintain knowledge base. Default: list all topics.",
            &[],
            &[("action", "string", "Operation: list|recent|entries|stats|xref|graph|stale|prune|compact|export|import|reindex|session|checkpoint|resume|clear_checkpoint"),
              ("topic", "string", "Topic name (for entries/xref)"), ("days", "string", "Number of days"),
              ("hours", "string", "Number of hours (overrides days)"),
              ("detail", "string", "Output: default|'tags'|'index' (for stats)"),
              ("index", "string", "Entry index"), ("match_str", "string", "Filter entries matching substring"),
              ("focus", "string", "Glob pattern to filter topics (graph)"),
              ("json", "string", "JSON string to import"),
              ("task", "string", "Task description for checkpoint"),
              ("done", "string", "Semicolon-separated completed steps"),
              ("next", "string", "Semicolon-separated next steps"),
              ("hypotheses", "string", "Semicolon-separated working hypotheses"),
              ("blocked", "string", "What's blocking progress"),
              ("files", "string", "Comma-separated key files")]),
        tool("trace", "Code structure analyzer. Actions: symbols, blast, analyze, or default (trace symbol).",
            &["path"],
            &[("path", "string", "Project directory or file path"),
              ("symbol", "string", "Symbol name to trace"),
              ("action", "string", "Operation: symbols|blast|analyze|omit for trace"),
              ("limit", "string", "Max symbols to return (for symbols action)"),
              ("filter", "string", "Glob filter on filename (for symbols action, e.g. '*.swift')")]),
        tool("_reload", "Re-exec the server binary to pick up code changes.", &[], &[]),
    ])
}

# Helix MCP Tools — Complete Reference

Helix exposes 7 MCP tools via JSON-RPC 2.0 over stdio. In Claude Code, they
appear as `mcp__helix__<tool>`. This document covers every tool, every
parameter, and every edge case.

---

## store

Store a timestamped knowledge entry under a topic.

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `topic` | string | yes | Topic name. Auto-sanitized: lowercased, non-alphanumeric chars become hyphens. |
| `text` | string | yes | Entry content. Can be multi-line. |
| `tags` | string | no | Comma-separated tags (e.g. `"gotcha, p0"`). Auto-singularized (`gotchas` → `gotcha`). |
| `source` | string | no | Source file reference: `"src/cache.rs:42"`. Enables staleness detection and ambient hook context. |
| `confidence` | string | no | `"0.0"` to `"1.0"` (default `"1.0"`). Lower values rank lower in search. |
| `links` | string | no | Narrative links: `"other-topic:3 another-topic:0"`. Space-separated `topic:index` pairs. |
| `force` | string | no | `"true"` to bypass duplicate detection. |
| `terse` | string | no | `"true"` for one-line response. |

### Behavior

1. Creates `~/.helix-kb/` if absent
2. Acquires exclusive file lock
3. Prepends metadata lines to body: `[tags: ...]`, `[source: ...]`, `[confidence: ...]`, `[links: ...]`
4. Checks for duplicates (70% Jaccard token overlap within same topic). Warns but still stores.
5. Appends entry record to data.log (12-byte header + topic + body). fsync.
6. Hot-patches in-memory cache (avoids double corpus load)
7. Marks index dirty (rebuilt on next query)

### Auto-Tag Detection

If `tags` is omitted, helix scans the first line for known prefixes:

| First line starts with | Auto-tag |
|----------------------|----------|
| `gotcha:` or `bug:` | gotcha |
| `invariant:` or `security:` | invariant |
| `decision:` or `design:` | decision |
| `data flow:` or `flow:` | data-flow |
| `perf:` or `benchmark:` | performance |
| `gap:` or `missing:` or `todo:` | gap |
| `how-to:` or `impl:` or `fix:` | how-to |
| `module:` | module-map |
| `overview:` | architecture |
| `coupling:` | coupling |
| `pattern:` | pattern |

### Examples

Basic:
```json
{"name": "store", "arguments": {"topic": "build-gotchas", "text": "arm64 only for FFI bridge"}}
```
→ `stored in build-gotchas`

With metadata:
```json
{"name": "store", "arguments": {
  "topic": "engine",
  "text": "CachedEntry holds pre-tokenized tf_map for zero-alloc search",
  "tags": "architecture, performance",
  "source": "src/cache.rs:27",
  "confidence": "0.9"
}}
```
→ `stored in engine [tags: architecture, performance] (~90%)`

Duplicate warning:
```json
{"name": "store", "arguments": {"topic": "engine", "text": "CachedEntry has pre-tokenized tf_map"}}
```
→ `stored in engine`
→ `  dupe warning: CachedEntry holds pre-tokenized tf_map for zero-alloc search`

---

## batch

Store multiple entries in a single operation.

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `entries` | JSON array | yes | Array of `{topic, text, tags?, source?}` objects |

### Behavior

1. Opens data.log once
2. Writes all entries sequentially (no lock per entry, no dupe check)
3. Single fsync at the end
4. Intra-batch Jaccard dedup: warns if entries within the batch are >70% similar

### Example

```json
{"name": "batch", "arguments": {"entries": [
  {"topic": "api-design", "text": "REST endpoints use /v2/ prefix"},
  {"topic": "api-design", "text": "Auth tokens expire after 1 hour", "tags": "gotcha"},
  {"topic": "network", "text": "Max 8 TCP connections per pool", "source": "src/pool.rs:15"}
]}}
```
→ `3 entries stored across 2 topics`

---

## search

Full-text BM25 search across all entries.

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | yes | Search query. CamelCase and snake_case are split automatically. |
| `detail` | string | no | Output format: `full`, `medium` (default), `brief`, `count`, `topics`, `grouped` |
| `limit` | string | no | Max results. Default varies by mode. |
| `after` | string | no | Date filter: `"YYYY-MM-DD"` or `"today"`, `"yesterday"`, `"this-week"` |
| `before` | string | no | Date filter: `"YYYY-MM-DD"` or `"today"`, `"yesterday"` |
| `days` | string | no | Shortcut: entries from last N days |
| `hours` | string | no | Shortcut: entries from last N hours (overrides `days`) |
| `tag` | string | no | Filter to entries with this tag |
| `topic` | string | no | Filter to entries in this topic |
| `mode` | string | no | `"and"` (default) or `"or"`. AND requires all terms present; OR matches any. |

### Search Behavior

1. Tokenize query: split, lowercase, expand CamelCase, filter stop words, dedup
2. AND mode: search with all terms required. If 0 results, automatically retry with OR.
3. BM25 scoring: `score += tf * idf / (tf + K1*(1-B+B*dl/avgdl))` per term per entry
4. Apply filters (time range, tag bitmap, topic)
5. Top-K extraction via min-heap
6. Format output according to `detail` mode

### Detail Modes

**`full`** — Complete entry bodies with topic headers. Matching terms get `>` prefix.
```
--- build-gotchas ---
[tags: gotcha]
> arm64 only for FFI bridge — Rust lib compiled arm64-only,
  must use -arch arm64 in xcodebuild. The default builds
  for both architectures and the Rust .a is only arm64.

1 matching section(s)
```

**`medium`** (default) — Two-line snippets with tags.
```
  [build-gotchas] 2026-02-21 arm64 only for FFI bridge — Rust lib compiled arm64-only #gotcha
  [engine] 2026-02-20 CachedEntry holds pre-tokenized tf_map for zero-alloc search #architecture #performance
2 match(es)
```

**`brief`** — Topic + first content line. Densest format.
```
  [build-gotchas] arm64 only for FFI bridge
  [engine] CachedEntry holds pre-tokenized tf_map
2 match(es)
```

**`grouped`** — Results organized by topic.
```
--- build-gotchas (2 entries) ---
  arm64 only for FFI bridge
  duplicate symbol: ensure ONLY_ACTIVE_ARCH=YES

--- engine (1 entry) ---
  CachedEntry holds pre-tokenized tf_map
```

**`topics`** — Hit counts per topic, no content.
```
build-gotchas: 5
engine: 3
network: 2
```

**`count`** — Just the number.
```
10 match(es)
```

### Examples

Basic search:
```json
{"name": "search", "arguments": {"query": "FFI bridge"}}
```

Recent gotchas:
```json
{"name": "search", "arguments": {"query": "build failure", "tag": "gotcha", "days": "7"}}
```

Count matches in a topic:
```json
{"name": "search", "arguments": {"query": "cache", "topic": "engine", "detail": "count"}}
```

OR-mode broad search:
```json
{"name": "search", "arguments": {"query": "performance optimization cache", "mode": "or", "limit": "10"}}
```

---

## brief

One-shot compressed briefing. The primary way to load a mental model of a topic.

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | no | Topic name, keyword, or glob pattern. Omit for meta-briefing. |
| `detail` | string | no | `"summary"` (~15 lines, default), `"scan"` (one-liners), `"full"` (everything) |
| `since` | string | no | Only entries from last N hours (e.g. `"24"`) |
| `focus` | string | no | Comma-separated categories to show: `"gotchas,invariants"` |
| `compact` | string | no | `"true"` for compact meta-briefing (top 5 topics only) |

### Behavior

The brief pipeline:

1. **Collect:** Find entries matching the query — exact topic name, substring,
   or glob pattern (`iris-*`). Follow narrative links 1 level deep.
2. **Score:** BM25 relevance + freshness (7-day half-life) + confidence weight
   + link-in boost (+2.0 per reference).
3. **Compress:** Jaccard similarity chains (>40% overlap → supersession).
   Temporal proximity chains (48h buckets, 3+ threshold). Cross-topic dedup.
4. **Classify:** Three-pass categorization:
   - Tag-based: `#gotcha` → GOTCHAS, `#decision` → DECISIONS
   - Keyword-based: content matching on first 3 lines
   - Body rescue: untagged entries classified by content patterns
5. **Format:** Output at the requested detail level.

### No-Query Meta-Briefing

With no `query`, returns a KB overview:

```
TOPICS: engine (75), scanners (72), network (37), build-gotchas (28)

CATEGORIES: architecture 23%, gotchas 18%, data-flow 12%, decisions 9%, gaps 7%

HOT (last 48h):
  engine: 5 new entries
  build-gotchas: 3 new entries

GAPS:
  [engine] Missing: connection timeout configuration
  [network] Todo: implement retry backoff

STATS: 847 entries, 52 topics, newest: today
```

### Examples

Topic briefing:
```json
{"name": "brief", "arguments": {"query": "engine"}}
```

Glob across multiple topics:
```json
{"name": "brief", "arguments": {"query": "iris-*", "detail": "scan"}}
```

Recent activity only:
```json
{"name": "brief", "arguments": {"query": "build-gotchas", "since": "48"}}
```

Focus on specific categories:
```json
{"name": "brief", "arguments": {"query": "engine", "focus": "gotchas,gaps"}}
```

Compact overview:
```json
{"name": "brief", "arguments": {"compact": "true"}}
```

---

## read

Read the full contents of a specific topic.

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `topic` | string | yes | Topic name (exact match) |
| `index` | string | no | Fetch a single entry by 0-based index |

### Output Format

```
[0] 2026-02-21 06:11
  [tags: gotcha]
  arm64 only for FFI bridge — Rust lib compiled arm64-only

[1] 2026-02-20 14:30
  [tags: architecture]
  [source: src/ffi.rs:15]
  FFI bridge design: thin wrapper calling Rust entropy functions

2 entries in build-gotchas
```

Entry indices are stable within a session and can be used with `edit` to
target specific entries.

### Examples

Read all entries:
```json
{"name": "read", "arguments": {"topic": "build-gotchas"}}
```

Read single entry:
```json
{"name": "read", "arguments": {"topic": "build-gotchas", "index": "0"}}
```

---

## edit

Modify existing entries. A compound tool — the `action` parameter selects
the operation.

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `action` | string | yes | `append`, `revise`, `delete`, `tag`, `rename`, `merge` |
| `topic` | string | yes | Topic name (source topic for rename/merge) |
| `text` | string | varies | Content for append/revise |
| `index` | string | no | Target entry by 0-based index |
| `match_str` | string | no | Target entry by substring match |
| `tag` | string | no | Target entry by tag (append only) |
| `tags` | string | no | Tags to add (tag action) |
| `remove` | string | no | Tags to remove (tag action) |
| `all` | string | no | `"true"` to delete entire topic |
| `new_name` | string | no | New name for rename |
| `into` | string | no | Target topic for merge |

### Actions

**`append`** — Add text to an existing entry.

Targets the entry by `index`, `match_str`, `tag`, or defaults to the last
entry. The new text is appended with a newline separator.

```json
{"name": "edit", "arguments": {
  "action": "append", "topic": "engine",
  "text": "Update: also supports async dispatch since v2",
  "index": "3"
}}
```

**`revise`** — Overwrite an entry's text entirely.

The old text is replaced. A `[modified: YYYY-MM-DD HH:MM]` line is prepended
automatically. Requires `index` or `match_str` to identify the target.

```json
{"name": "edit", "arguments": {
  "action": "revise", "topic": "engine",
  "text": "CachedEntry: pre-tokenized tf_map, lazy metadata via OnceCell",
  "index": "5"
}}
```

**`delete`** — Remove entries.

Four modes:
- `all: "true"` — delete entire topic (all entries)
- `index: "3"` — delete entry at index 3
- `match_str: "FFI bridge"` — delete first entry matching substring
- No targeting — deletes the last entry

```json
{"name": "edit", "arguments": {"action": "delete", "topic": "test", "all": "true"}}
```

**`tag`** — Add or remove tags on an entry.

```json
{"name": "edit", "arguments": {
  "action": "tag", "topic": "engine",
  "tags": "performance, hot-path",
  "remove": "draft",
  "index": "2"
}}
```

**`rename`** — Rename a topic.

Rewrites all entries with the new topic name (write new + tombstone old for each).

```json
{"name": "edit", "arguments": {"action": "rename", "topic": "old-name", "new_name": "new-name"}}
```

**`merge`** — Merge one topic into another.

All entries from the source topic are rewritten under the target topic, then
the source entries are tombstoned.

```json
{"name": "edit", "arguments": {"action": "merge", "topic": "network-old", "into": "network"}}
```

### How Edits Work Internally

All edits are append-only. When you revise entry #3:

1. The new version is appended to data.log as a fresh entry record
2. A delete tombstone is appended pointing to entry #3's original offset
3. Next read will show the new version, not the old one

This means edits never corrupt existing data. The trade-off: data.log grows.
Run `topics(action: "compact")` periodically to rewrite without tombstones.

---

## topics

Browse and maintain the knowledge base. Another compound tool.

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `action` | string | no | Operation (default: `"list"`). See table below. |
| `topic` | string | no | Topic name (for `entries`, `xref`) |
| `days` | string | no | Days threshold (for `recent`: default 7, for `prune`: default 30) |
| `hours` | string | no | Hours threshold (overrides `days` for `recent`) |
| `detail` | string | no | `"tags"` or `"index"` (for `stats`) |
| `index` | string | no | Entry index (for `entries`) |
| `match_str` | string | no | Filter entries matching substring |
| `focus` | string | no | Glob pattern to filter topics (for `graph`) |
| `json` | string | no | JSON string to import (for `import`) |
| `refresh` | string | no | `"true"` to show stale entries + current source (for `stale`) |

### Actions

**`list`** (default) — All topics with entry counts, sorted alphabetically.
```
api-design (12), build-gotchas (28), engine (75), network (37)...
52 topics, 847 entries
```

**`recent`** — Topics with activity in the last N days/hours.
```json
{"name": "topics", "arguments": {"action": "recent", "hours": "24"}}
```
```
engine (5 new), build-gotchas (3 new)
```

**`entries`** — List entries in a topic (same as `read` but via topics tool).
```json
{"name": "topics", "arguments": {"action": "entries", "topic": "engine", "match_str": "cache"}}
```

**`stats`** — KB statistics.
```json
{"name": "topics", "arguments": {"action": "stats"}}
```
```
52 topics, 847 entries
oldest: 2026-01-15
newest: 2026-02-21
```
With `detail: "tags"`: tag frequency distribution.
With `detail: "index"`: index stats (terms, postings, size).

**`xref`** — Cross-references for a topic.
```json
{"name": "topics", "arguments": {"action": "xref", "topic": "engine"}}
```
```
engine is referenced by: network (5), scanners (3), proxy-ext (2)
engine references: cache (4), datalog (2)
```

**`graph`** — Topic dependency graph.
```json
{"name": "topics", "arguments": {"action": "graph", "focus": "iris-*"}}
```

**`stale`** — Entries with stale source references (source file modified after entry).
```json
{"name": "topics", "arguments": {"action": "stale", "refresh": "true"}}
```

**`prune`** — Remove entries older than N days.
```json
{"name": "topics", "arguments": {"action": "prune", "days": "90"}}
```

**`compact`** — Rewrite data.log without tombstones.
```json
{"name": "topics", "arguments": {"action": "compact"}}
```
```
compacted: 847 entries, 1393539 → 1287042 bytes
```

**`export`** — Export entire KB as JSON.
```json
{"name": "topics", "arguments": {"action": "export"}}
```

**`import`** — Import from JSON.
```json
{"name": "topics", "arguments": {"action": "import", "json": "[{\"topic\":\"test\",\"text\":\"imported\"}]"}}
```

**`reindex`** — Force full index rebuild.
```json
{"name": "topics", "arguments": {"action": "reindex"}}
```

**`session`** — Session summary (same output as the session hook).
```json
{"name": "topics", "arguments": {"action": "session"}}
```

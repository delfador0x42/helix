# Helix Documentation

Helix is a persistent knowledge base designed for AI coding assistants. It runs as an MCP (Model Context Protocol) server alongside Claude Code, providing 9 tools for storing, searching, retrieving, and analyzing knowledge across sessions. A companion hook system injects relevant context before every file operation with zero perceptible latency.

**Key numbers:** ~6,300 lines of Rust, 19 source files, 9 MCP tools, 9 hook types, zero external dependencies.

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [MCP Tools Reference](#mcp-tools-reference)
4. [Hook System](#hook-system)
5. [Knowledge Organization](#knowledge-organization)
6. [Search Engine](#search-engine)
7. [Briefing System](#briefing-system)
8. [Session and Checkpoints](#session-and-checkpoints)
9. [Storage Format](#storage-format)
10. [Architecture Deep Dive](#architecture-deep-dive)
11. [Performance](#performance)
12. [Maintenance](#maintenance)
13. [Troubleshooting](#troubleshooting)

---

## Installation

### Build from Source

```bash
cd helix
cargo build --release
```

The release binary is at `target/release/helix` (~2.5 MB, statically linked, zero dependencies).

### Install the Binary

```bash
# Copy to a location on your PATH
cp target/release/helix ~/.local/bin/helix

# Code-sign (required on macOS for mmap and fork operations)
codesign -s - -f ~/.local/bin/helix
```

### Install Claude Code Hooks

```bash
helix hooks install
```

This writes hook configuration to `.claude/settings.json` in your project directory. To check hook status:

```bash
helix hooks status
```

To remove hooks:

```bash
helix hooks uninstall
```

### Configure as MCP Server

Add to your Claude Code MCP configuration (`.claude/settings.json` or global settings):

```json
{
  "mcpServers": {
    "helix": {
      "command": "helix",
      "args": ["serve"]
    }
  }
}
```

Helix reads and writes its data from `~/.helix-kb/` by default. To use a different directory:

```json
{
  "mcpServers": {
    "helix": {
      "command": "helix",
      "args": ["serve", "/path/to/your/kb"]
    }
  }
}
```

### Verify Installation

After installing and configuring, restart Claude Code. You should see helix tools available (prefixed with `mcp__helix__` in the tool list). Try:

```
Store a test entry: mcp__helix__store topic="test" text="Hello from helix"
Search for it: mcp__helix__search query="hello"
```

---

## Configuration

### Knowledge Base Directory

Default: `~/.helix-kb/`

Contains:
- `data.log` — append-only binary log of all entries
- `index.bin` — inverted search index (auto-rebuilt)
- `checkpoint.json` — durable task checkpoint (if saved)
- `.lock` — flock file for concurrent access

The directory is created automatically on first run.

### Hook Configuration

Hooks are defined in `.claude/settings.json`. The `helix hooks install` command writes the full configuration. Here is what each hook does:

| Hook Type | Event | Purpose |
|-----------|-------|---------|
| `session` | SessionStart | Inject KB briefing on new session |
| `prompt` | UserPromptSubmit | Auto-search on user questions |
| `ambient` | PreToolUse | 5-layer context injection before Read/Edit/Write/Glob/Grep |
| `post-build` | PostToolUse (Bash) | Capture build errors, search for related gotchas |
| `error-context` | PostToolUseFailure | Search KB for error patterns |
| `pre-compact` | PreCompact | Re-inject summary before context compression |
| `stop` | Stop | Remind to store findings before session ends |
| `subagent` | SubagentStart | Inject topic list into subagent context |
| `approve-mcp` | PermissionRequest | Auto-approve helix MCP tool calls |

---

## MCP Tools Reference

Helix exposes 9 MCP tools via the JSON-RPC stdio protocol. Each tool is designed for LLM consumption — outputs are compressed, context-efficient, and structured for immediate use.

### store

Store a timestamped knowledge entry under a topic.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | string | yes | Topic name (lowercase, alphanumeric + dashes) |
| `text` | string | yes | Entry content (plain text, multi-line) |
| `tags` | string | no | Comma-separated tags (e.g. `"gotcha,performance,metal"`) |
| `source` | string | no | Source file reference for staleness tracking (e.g. `"src/infer.rs:42"`) |
| `confidence` | string | no | Confidence level 0.0–1.0 (default: 1.0). Affects search ranking. |
| `links` | string | no | Space-separated narrative links (e.g. `"build-gotchas:3 iris-engine:12"`) |
| `force` | string | no | Set to `"true"` to bypass duplicate detection |
| `terse` | string | no | Set to `"true"` for minimal response |

**Behavior:**
- Topic names are sanitized (lowercased, non-alphanumeric replaced with dashes)
- Duplicate detection uses Jaccard similarity (>70% token overlap on 6+ tokens warns)
- Tags are auto-detected from content prefixes (e.g. text starting with "gotcha:" auto-tags as `gotcha`)
- Links are validated at store time (warns if target topic doesn't exist)
- The entry is appended to `data.log` and the in-memory cache is updated immediately
- The inverted index is marked dirty and rebuilt asynchronously (50ms debounce)

**Auto-detected tag prefixes:** gotcha, invariant, perf, architecture, data-flow, decision, how-to, gap, friction, todo, bug, fix, pattern, coupling, module-map, interface, boundary, lifecycle, error-handling, concurrency, migration, deprecation, dependency, test, benchmark, config, deploy, security, api, protocol, schema, format, encoding, observation, hypothesis

**Example:**
```
store(topic="build-gotchas", text="Rust lib must use -arch arm64. Without it, xcodebuild silently falls back to x86_64 and linking fails with undefined symbols.", tags="gotcha,build,arm64")
```

### batch

Store multiple entries in a single call. Faster than sequential `store` calls because it holds the file lock once.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entries` | array | yes | Array of `{topic, text, tags?, source?}` objects |
| `verbose` | string | no | Set to `"true"` for per-entry details |

**Behavior:**
- Intra-batch Jaccard deduplication (70% threshold) prevents storing near-duplicates within the same batch
- Each entry is appended sequentially while holding a single file lock
- Index is rebuilt once after the entire batch completes

### search

Full-text search across all knowledge entries using BM25 ranking.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | no | Search query (tokenized, CamelCase split, case-insensitive) |
| `detail` | string | no | Output mode (see below) |
| `limit` | string | no | Max results (default: unlimited) |
| `after` | string | no | Only entries on/after date (`YYYY-MM-DD`, `today`, `yesterday`, `this-week`) |
| `before` | string | no | Only entries on/before date |
| `days` | string | no | Entries from last N days |
| `hours` | string | no | Entries from last N hours (overrides days) |
| `tag` | string | no | Filter to entries with this tag |
| `topic` | string | no | Filter to entries in this topic |
| `mode` | string | no | `"and"` (default, all terms must match) or `"or"` (any term) |

**Detail modes:**

| Mode | Description |
|------|-------------|
| `full` | Complete entry body with matched lines highlighted |
| `medium` | Header line + first 2 body lines (default) |
| `brief` | One-liner: topic, date, first content line, tags |
| `grouped` | Results bucketed by topic |
| `topics` | Hit count per topic (no content) |
| `count` | Just the total match count |

**Search behavior:**
- Queries are tokenized: `"FxHashSet performance"` → `["fx", "hash", "set", "performance"]`
- CamelCase is automatically split: `CachedEntry` → `["cached", "entry"]`
- Stop words are filtered: that, this, with, have, been, which, would, about, their, could, other, there, after, these, where, being, should, still, those, using, before, during, while, between
- AND mode is tried first; if no results, automatically falls back to OR
- Results are ranked by BM25 with recency boost (30-day half-life) and confidence weighting
- Diversity cap: maximum 3 results per topic to prevent any single topic from dominating

**BM25 scoring formula:**
```
score = IDF × (TF × (K1 + 1)) / (TF + K1 × (1 - B + B × (doc_len / avg_doc_len)))
      × confidence
      × recency_boost
```
Where K1=1.2, B=0.75, recency_boost = 1.0 / (1.0 + age_days / 30).

### brief

One-shot compressed briefing — the primary tool for loading a mental model of a topic or area.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | no | Topic name, keyword, or glob pattern (e.g. `"iris-*"`) |
| `detail` | string | no | `"summary"` (default, ~15 lines), `"scan"` (one-liner per category), `"full"` (complete entries) |
| `since` | string | no | Only entries from last N hours |
| `focus` | string | no | Comma-separated category filter (e.g. `"gotchas,invariants"`) |
| `compact` | string | no | Set to `"true"` for top-5 meta-briefing |

**Without a query:** Returns a meta-briefing of the entire knowledge base — top topics by activity, entry velocity, and themes.

**With a topic name:** Compresses all entries in that topic through three passes:
1. **Dedup pass:** Entries with identical content across topics are merged (keep highest relevance, note "also in: X")
2. **Supersession pass:** Entries with >60% Jaccard token overlap where one is significantly newer — the older is dimmed with "[SUPERSEDED]" marker
3. **Temporal chain pass:** Entries in the same topic sharing a dominant keyword are chained chronologically to show evolution

**With a glob pattern (e.g. `iris-*`):** Matches all topics fitting the pattern and produces a unified briefing across all of them.

**Category classification:** Entries are classified into 8 categories based on their tags and content patterns:

| Category | Detection |
|----------|-----------|
| ARCHITECTURE | Tags: architecture, module-map, coupling, interface |
| DATA FLOW | Tags: data-flow, lifecycle, protocol |
| INVARIANTS | Tags: invariant, boundary, concurrency |
| GOTCHAS | Tags: gotcha, bug, error-handling |
| DECISIONS | Tags: decision, deprecation, migration |
| HOW-TO | Tags: how-to, config, deploy |
| PERFORMANCE | Tags: perf, benchmark, optimization |
| GAPS | Tags: gap, friction, todo, missing |

Entries not matching any category appear under UNTAGGED.

### read

Read the complete contents of a specific topic.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `topic` | string | yes | Topic name |
| `index` | string | no | Fetch a single entry by 0-based index |

Returns all entries in the topic with timestamps and metadata. Use `index` to fetch a specific entry when you know its position (visible from `topics action=entries`).

### edit

Unified mutation tool for modifying existing entries.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | string | yes | Operation (see below) |
| `topic` | string | yes | Topic name |
| `text` | string | varies | Content for append/revise |
| `index` | string | no | Target entry by 0-based index |
| `match_str` | string | no | Target entry by substring match |
| `tag` | string | no | Target most recent entry with this tag (append only) |
| `tags` | string | no | Comma-separated tags to add (tag action) |
| `remove` | string | no | Comma-separated tags to remove (tag action) |
| `all` | string | no | Set `"true"` to delete entire topic |
| `new_name` | string | no | New name for rename action |
| `into` | string | no | Target topic for merge action |

**Actions:**

| Action | Description |
|--------|-------------|
| `append` | Add text to an existing entry (no new timestamp). Targets last entry by default, or use index/match_str/tag. |
| `revise` | Replace an entry's text entirely. Adds `[modified: date]` marker. Use index or match_str to target. |
| `delete` | Delete by index, match_str, or `all="true"` for entire topic. Default: deletes last entry. |
| `tag` | Add tags (via `tags` param) or remove tags (via `remove` param) on an entry. |
| `rename` | Rename a topic. All entries preserved. Requires `new_name`. |
| `merge` | Merge all entries from `topic` into the topic specified by `into`. Source topic is deleted after merge. |

### topics

Browse and maintain the knowledge base. Actions are specified via the `action` parameter.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | string | no | Operation (default: `list`) |
| `topic` | string | no | Topic name (for entries/xref) |
| `days` | string | no | Time range (default: 7 for recent, 30 for prune) |
| `hours` | string | no | Override days with hours (for recent) |
| `detail` | string | no | Output detail (for stats) |
| `index` | string | no | Entry index (for entries) |
| `match_str` | string | no | Filter substring (for entries) |
| `focus` | string | no | Glob pattern (for graph) |
| `json` | string | no | JSON string (for import) |
| `refresh` | string | no | Show current source (for stale) |
| `task` | string | no | Task description (for checkpoint) |
| `done` | string | no | Semicolon-separated completed steps (for checkpoint) |
| `next` | string | no | Semicolon-separated next steps (for checkpoint) |
| `hypotheses` | string | no | Semicolon-separated hypotheses (for checkpoint) |
| `blocked` | string | no | Blocking description (for checkpoint) |
| `files` | string | no | Comma-separated key files (for checkpoint) |

**Actions:**

| Action | Description |
|--------|-------------|
| `list` | List all topics with entry counts and last-modified dates (default) |
| `recent` | Show entries from last N days/hours across all topics |
| `entries` | List entries in a topic with index numbers. Pass `index` for a single entry. |
| `stats` | Overview: topic count, entry count, date range. `detail="tags"` for all tags. `detail="index"` for index health. |
| `xref` | Cross-references: entries in other topics that mention this topic |
| `graph` | Topic dependency graph. `focus="iris-*"` for filtered view. |
| `stale` | Scan entries with `[source:]` metadata for changed files. `refresh="true"` shows side-by-side. |
| `prune` | Flag topics with no entries in N days (default: 30) |
| `compact` | Find and merge duplicate entries. `apply="true"` to execute. `log="true"` to compact data.log. `mode="migrate"` to fix timestamps. |
| `export` | Export all topics as structured JSON |
| `import` | Import from JSON (merges with existing). Requires `json` param. |
| `reindex` | Force rebuild the binary inverted index |
| `session` | Show store operations performed this session |
| `checkpoint` | Save working state. Requires `task`. Optional: `done`, `next`, `hypotheses`, `blocked`, `files`. |
| `resume` | Load saved checkpoint. Returns formatted markdown with task, progress, hypotheses, and age. |
| `clear_checkpoint` | Delete the saved checkpoint |

---

## Hook System

Hooks are the zero-latency context injection layer. When Claude Code performs a tool operation (reading a file, running a build, etc.), helix hooks fire automatically and inject relevant knowledge from your KB into the conversation context.

### How Hooks Work

1. Claude Code fires a hook event (e.g., "about to read file X")
2. macOS forks a new process and executes `helix hook <type>`
3. The hook process mmaps `index.bin` directly (no socket, no server required)
4. Relevant entries are extracted and returned as JSON
5. Claude Code injects the response into the LLM's context
6. Total wall time: ~2–3ms (dominated by macOS fork+exec overhead)

### The 5-Layer Ambient Context System

The most important hook is `ambient` (PreToolUse). When Claude reads, edits, or writes a file, helix injects up to 5 layers of relevant context:

**Layer 1 — Source Path Matches:**
Finds entries tagged with `[source: path/to/this/file.rs]`. These are notes specifically about the file being accessed. Highest priority.

**Layer 2 — Symbol-Based Search:**
Parses the file for function/struct/class/enum/trait names (first 500 lines, up to 20 symbols). Runs an OR search across the index for these symbols. Skipped if Layer 1 returned 5+ results.

**Layer 3 — Stem BM25:**
Searches the index for the file stem (e.g., `kernels` from `kernels.rs`). Catches topic-level knowledge.

**Layer 4 — Structural Coupling:**
Searches for entries containing "structural" + the file stem. Finds architectural coupling notes.

**Layer 5 — Refactor Impact (Edit only):**
When editing a file, parses the old content to find removed symbols (function/struct names that disappeared). Searches for entries referencing those symbols. Prevents silent breakage.

Each layer deduplicates against previous layers using entry IDs tracked in the session state. Results are limited to prevent context bloat.

### Session-Based Deduplication

The hook system tracks which entries have been injected during the current session (stored in `/tmp/helix-session.json`). If the same entry was already injected earlier in the session, it's skipped. This prevents the same gotcha from appearing in context repeatedly.

### Auto-Approve Hook

The `approve-mcp` hook automatically approves all helix MCP tool calls, preventing Claude Code from prompting the user for permission on every store/search operation.

---

## Knowledge Organization

### Topics

Topics are the primary organizational unit. Each entry belongs to exactly one topic. Topic names are lowercase alphanumeric with dashes (e.g., `build-gotchas`, `iris-engine`, `ref-xnu-kernel`).

**Naming conventions:**
- `project-subsystem` for project-specific knowledge (e.g., `iris-scanners`)
- `ref-source` for reference material from external sources (e.g., `ref-gamozolabs`)
- Descriptive names for cross-cutting concerns (e.g., `build-gotchas`, `performance-notes`)

### Tags

Tags classify entries within topics. They enable filtered search and drive the briefing system's category classification. Tags are stored as `[tags: tag1, tag2, ...]` metadata in the entry body.

**Tag normalization:** Tags are lowercased, singularized (gotchas → gotcha, entries → entry), sorted, and deduplicated.

### Metadata

Entries support four types of metadata, written as bracketed lines in the body:

```
[tags: gotcha, performance]          — classification
[source: src/kernels.rs:42]          — source file reference for staleness tracking
[confidence: 0.85]                   — reliability score (0.0–1.0, default 1.0)
[links: build-gotchas:3 other:12]    — narrative links to related entries
```

**Confidence** affects BM25 ranking — lower confidence entries rank lower in search results. When a source file changes on disk, entries referencing it are automatically downgraded in confidence.

**Links** create explicit connections between entries. The briefing system follows links one level deep when building mental models.

---

## Search Engine

### Inverted Index

The search engine is built on a binary inverted index (`index.bin`) with these components:

- **Term hash table:** Open-addressing with FNV-1a hashing. Each slot maps a term to its posting list.
- **Posting lists:** Per-term lists of (entry_id, term_frequency, pre-computed IDF).
- **Entry metadata:** Per-entry topic, word count, date, source path offset, tag bitmap, confidence, epoch day.
- **Snippet pool:** Pre-formatted entry previews for fast result rendering.
- **Topic table:** Topic names with entry counts.
- **Cross-reference graph:** Which topics mention which other topics.

### BM25 Ranking

Scoring uses Okapi BM25 with these parameters:
- **K1 = 1.2** — term frequency saturation
- **B = 0.75** — document length normalization

Additional scoring factors:
- **Recency boost:** `1.0 / (1.0 + age_days / 30.0)` — 30-day half-life
- **Confidence weight:** `score *= confidence` (0.0–1.0)
- **Stale penalty:** Entries with changed source files get confidence downgraded to 0.7

### Three-Phase Search

1. **Score accumulation:** For each query term, iterate its posting list and accumulate BM25 scores per entry. Filtered entries (wrong topic, wrong date range, missing tag) are skipped.
2. **Top-K heap:** Maintain a min-heap of the best K results. Diversity cap limits each topic to 3 entries maximum.
3. **Snippet extraction:** Only for the final top-K results, extract the pre-computed snippet from the index. This avoids allocating snippets for entries that won't be returned.

### Query Processing

Queries go through tokenization before search:
1. Lowercase the entire query
2. Split on whitespace and punctuation
3. Split CamelCase tokens (e.g., `FxHashSet` → `fx`, `hash`, `set`)
4. Filter tokens shorter than 2 characters
5. Filter stop words
6. Deduplicate

Special query syntax:
- `tag:gotcha` — filter to entries with this tag
- `topic:build-gotchas` — filter to entries in this topic
- `source:kernels.rs` — filter to entries referencing this file

---

## Briefing System

The `brief` tool produces compressed mental models from raw entries. It is the recommended way to load context about a topic area at the start of a task.

### Compression Pipeline

**Input:** All entries matching the query (topic name, glob pattern, or keyword).

**Pass 1 — Deduplication:**
Entries with identical content across different topics are merged. The highest-relevance copy is kept; others are noted as "also in: topic-name".

**Pass 2 — Supersession:**
Entry pairs with >60% Jaccard token overlap where one is significantly newer (2+ day gap) — the older entry is marked `[SUPERSEDED]` and chained to the newer one. This automatically surfaces the latest version of evolving knowledge.

**Pass 3 — Temporal Chains:**
Within a single topic, entries sharing a dominant keyword (3+ chars, appears in 3+ entries) are grouped chronologically. This shows how a concept evolved over time.

### Classification

After compression, entries are classified into 8 semantic categories using a three-pass classifier:

1. **Tag-based:** Entries with explicit tags are placed in their category.
2. **Keyword-based:** Entry body is scanned for content prefixes (e.g., "gotcha:" → GOTCHAS, "architecture:" → ARCHITECTURE).
3. **Body keyword rescue:** First 3 lines of content are scanned for category-associated keywords.

Entries not matching any category appear under UNTAGGED.

### Output Tiers

**Summary (default, ~15 lines):**
```
TOPICS: topic-a (12), topic-b (8)
CATEGORIES: GOTCHAS 5, ARCHITECTURE 3, DECISIONS 2
HOT: [entry previews for top-5 by recency × relevance]
GAPS: [any gap/friction/todo entries]
```

**Scan:** One-liner per category showing count and first entry preview.

**Full:** 5–15 lines per category with complete entry content.

---

## Session and Checkpoints

### Session State

Helix tracks an ephemeral session in `/tmp/helix-session.json`. Sessions are identified by TTY name and expire after 4 hours of inactivity. The session tracks:

- **Injected entries:** Which KB entries have been injected via hooks (prevents re-injection)
- **Focus topics:** Topics frequently accessed this session (auto-inferred from hook activity)
- **Phase:** Current work phase (Research / Build / Verify / Debug) detected from tool usage patterns
- **Build state:** Last build success/failure and timestamp
- **File counts:** Files touched and edited this session

Phase detection works automatically:
- Edit/Write tools → Build phase
- Bash after Edit → Verify phase
- Build failure → Debug phase
- Read/Grep/Glob at start → Research phase

### Checkpoints

Checkpoints are durable "save points" that survive across sessions. Unlike session state, checkpoints have no TTY binding and no timeout — they persist until explicitly cleared.

**Save a checkpoint:**
```
topics(action="checkpoint", task="Implement Metal attention kernel",
       done="Verified embedding correctness;Built Q4K matvec",
       next="Debug RMSNorm output;Compare against llama.cpp",
       hypotheses="RoPE frequency base may be wrong;Attention mask not applied",
       blocked="Need reference output from correct implementation",
       files="infer.rs,kernels.rs,main.rs")
```

**Resume from checkpoint:**
```
topics(action="resume")
```

Returns formatted markdown:
```
## Checkpoint: Implement Metal attention kernel
_saved 2h 15m ago_

Done:
  [x] Verified embedding correctness
  [x] Built Q4K matvec

Next:
  [ ] Debug RMSNorm output
  [ ] Compare against llama.cpp

Hypotheses:
  ? RoPE frequency base may be wrong
  ? Attention mask not applied

Blocked: Need reference output from correct implementation

Files: infer.rs, kernels.rs, main.rs
```

**Clear checkpoint:**
```
topics(action="clear_checkpoint")
```

---

## Storage Format

### data.log Binary Format

All knowledge is stored in a single binary log file. The format is intentionally simple — any language can read it in ~60 lines of code.

```
┌──────────────────────────────────────────┐
│ Header (8 bytes)                         │
│   magic:   [0x41, 0x4D, 0x52, 0x4C]     │  "AMRL"
│   version: u32 LE = 1                    │
├──────────────────────────────────────────┤
│ Record 1                                 │
│   type:      u8  = 0x01 (entry)          │
│   topic_len: u8                          │
│   body_len:  u32 LE                      │
│   timestamp: i32 LE (minutes since epoch)│
│   pad:       2 bytes                     │
│   topic:     [u8; topic_len]             │
│   body:      [u8; body_len]              │
├──────────────────────────────────────────┤
│ Record 2                                 │
│   type:      u8  = 0x02 (delete)         │
│   reserved:  3 bytes                     │
│   offset:    u32 LE (target entry)       │
├──────────────────────────────────────────┤
│ Record 3 ...                             │
└──────────────────────────────────────────┘
```

**Timestamp epoch:** Minutes since 2024-01-01 00:00 UTC. Computed using the Hinnant civil algorithm (no chrono dependency).

**Concurrency model:** Append-only writes with flock(2) exclusive locking. Multiple readers are always safe. The MCP server holds the lock only during individual write operations.

### index.bin Binary Format

The inverted index is a flat binary file designed for memory-mapped access. It contains:

1. **Header** (17 fields, packed C struct) — counts, offsets, and sizes for all sections
2. **Term hash table** — open-addressing with FNV-1a. Each slot: (hash, postings_offset, postings_length)
3. **Posting lists** — per-entry: (entry_id, term_frequency, IDF × 1000)
4. **Entry metadata** — per-entry: (topic_id, word_count, snippet_offset, date_minutes, source_offset, tag_bitmap, confidence, epoch_days)
5. **Topic table** — per-topic: (name_offset, name_length, entry_count)
6. **Cross-reference edges** — per-pair: (src_topic, dst_topic, mention_count)
7. **Tag bitmap** — top 32 tags by frequency, packed as u32 bits per entry
8. **Snippet pool** — pre-formatted byte strings
9. **Source pool** — source file paths
10. **Topic name pool** — topic name strings

All integers are little-endian. All structs are `#[repr(C, packed)]` for zero-copy access from mmap.

---

## Architecture Deep Dive

### Module Dependency Graph

```
main.rs ──► mcp.rs ──► dispatch ──► write.rs ──► datalog.rs
                │                        │              │
                │                        ▼              ▼
                │                   cache.rs ◄──── config.rs
                │                        │
                │                        ▼
                │                   index.rs ──► format.rs
                │                     │   │
                │                     ▼   ▼
                │                search.rs  brief.rs
                │
                ▼
            hook.rs ──► session.rs
                │
                ▼
            sock.rs
```

**Shared utilities (used by most modules):**
- `json.rs` — JSON parsing/serialization
- `text.rs` — tokenization, metadata extraction
- `time.rs` — date/time computation
- `fxhash.rs` — fast hashing
- `lock.rs` — file locking

### In-Memory State

The MCP server maintains several pieces of global state:

- **Index bytes** (`RwLock<Vec<u8>>`) — the current index.bin contents, served to all search queries
- **Corpus cache** (`Mutex<CachedCorpus>`) — parsed entries with TF maps, invalidated on data.log mtime change
- **Tool schema cache** (`Mutex<Option<Arc<str>>>`) — pre-serialized MCP tool list JSON (~15KB, computed once)
- **Session log** (`Mutex<Vec<String>>`) — log of store operations this session
- **Index dirty flag** (`AtomicBool` + timestamp) — triggers async rebuild with 50ms debounce

### Concurrency Model

- **MCP server:** Single-threaded event loop on stdin. All tool dispatch is synchronous.
- **Index rebuild:** Triggered on write, executes inline with 50ms debounce (coalesces rapid writes).
- **Socket listener:** Background thread for hook queries (Unix domain socket).
- **Hooks:** Fork+exec per event. No shared memory with MCP server. Read index.bin via mmap.
- **File locking:** flock(2) on `.lock` file for write serialization.

---

## Performance

### Latency Targets

| Operation | Target | Achieved |
|-----------|--------|----------|
| Hook execution (ambient) | <5ms | ~2–3ms (macOS fork+exec floor) |
| MCP search | <20ms | ~5–10ms for typical queries |
| MCP store | <10ms | ~3–5ms (write + cache update) |
| Index rebuild | <100ms | ~50–80ms for ~1000 entries |
| Brief (topic) | <50ms | ~20–40ms with compression |

### Zero-Allocation Hot Paths

The codebase is aggressively optimized to minimize heap allocations on frequently-called paths:

- **JSON parsing:** Hand-rolled parser with fast-path for unescaped strings (~95% hit rate, single memcpy)
- **Hook responses:** Direct string formatting, no Value tree construction
- **JSON-RPC IDs:** Stack-allocated `[u8; 24]` buffer (zero heap alloc for 99% of calls)
- **Search scoring:** Reusable `QueryState` buffers across queries (no per-query allocation)
- **Snippet extraction:** Deferred to final top-K results (eliminates 80–90% of String allocations)
- **Topic interning:** `Arc<str>` deduplication prevents unbounded string allocation for topic names
- **TF-map building:** Built inline during tokenization (no intermediate Vec of tokens)

### Hashing

All internal hash maps use FxHash (~3ns per hash) instead of the standard library's SipHash (~20ns). FxHash uses rotate-left XOR and wrapping multiplication, processing 8 bytes at a time. It's not cryptographically secure, but it's 6–7x faster for the token-length strings used in search.

---

## Maintenance

### Compacting the Log

Over time, `data.log` accumulates deleted entry tombstones. To reclaim space:

```
topics(action="compact", log="true", apply="true")
```

- Without `apply="true"`: dry run showing what would be merged
- With `log="true"`: rewrites data.log without tombstones
- With `mode="migrate"`: fixes entries that lack timestamps

### Rebuilding the Index

The index rebuilds automatically on writes. To force a rebuild:

```
topics(action="reindex")
```

Or from the command line:

```bash
helix index
```

### Checking for Stale Entries

Entries with `[source:]` metadata can become stale when the referenced file changes:

```
topics(action="stale")                    # list stale entries
topics(action="stale", refresh="true")    # show entry + current source side-by-side
```

### Exporting and Importing

```
topics(action="export")                   # JSON dump of entire KB
topics(action="import", json="...")       # merge JSON into existing KB
```

Import preserves original timestamps.

### Hot Reload

During development, after building a new helix binary:

```
_reload
```

This copies the release binary from `target/release/helix` to the installed location, code-signs it, and re-execs the server process. No restart needed.

---

## Troubleshooting

### Hooks Not Firing

1. Check hook status: `helix hooks status`
2. Verify `.claude/settings.json` has the hook configuration
3. Ensure the helix binary is on your PATH and code-signed
4. Restart Claude Code after installing hooks

### Search Returns No Results

1. Check that entries exist: `topics(action="list")`
2. Try a simpler query — complex queries require all terms to match (AND mode)
3. Check if the index is fresh: `topics(action="stats", detail="index")`
4. Force rebuild: `topics(action="reindex")`

### Slow Performance

1. Check corpus size: `topics(action="stats")` — if >5000 entries, consider pruning
2. Check data.log size — if >10MB, compact: `topics(action="compact", log="true", apply="true")`
3. Ensure you're running the release build (`cargo build --release`)

### Permission Denied on Binary

```bash
codesign -s - -f $(which helix)
```

macOS requires code-signed binaries for certain operations (mmap, fork from sandboxed processes).

### Session State Issues

Session state lives at `/tmp/helix-session.json`. If hooks behave oddly:

```bash
rm /tmp/helix-session.json
```

This resets the session. The next hook invocation creates a fresh session.

### Checkpoint Not Found

Checkpoints live at `<kb_dir>/checkpoint.json` (default: `~/.helix-kb/checkpoint.json`). They persist until explicitly cleared with `topics(action="clear_checkpoint")`.

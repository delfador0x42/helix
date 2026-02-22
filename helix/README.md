# Helix

**Persistent knowledge store for AI coding agents.**

Helix gives Claude Code a long-term memory that survives across sessions,
restarts, and context compactions. Findings, gotchas, architectural decisions,
and debugging insights are stored in an indexed knowledge base and surfaced
automatically when relevant — before the agent even asks.

```
Zero dependencies. Single binary. ~1.1 MB. 2–3 ms hook latency.
9 MCP tools. 9 hooks. BM25 full-text search. mmap zero-copy index.
```

```
You: "Why does the FFI bridge fail on x86?"

Claude: (ambient hook fires, finds entry stored 3 days ago)
  → "The FFI bridge requires arm64. The Rust lib is compiled
     arm64-only — x86_64 builds fail at link time with undefined
     symbols. See build-gotchas topic."
```

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration Files](#configuration-files) — every file helix reads or writes, global and workspace
3. [How It Works](#how-it-works)
4. [MCP Tools](#mcp-tools)
   - [store](#store) · [batch](#batch) · [search](#search) · [brief](#brief)
   - [read](#read) · [edit](#edit) · [topics](#topics) · [trace](#trace) · [_reload](#_reload)
5. [Hooks](#hooks)
6. [Knowledge Organization](#knowledge-organization)
7. [Data Format](#data-format)
8. [CLAUDE.md Integration](#claudemd-integration)
9. [CLI Reference](#cli-reference)
10. [Troubleshooting](#troubleshooting)
11. [Architecture](#architecture)

---

## Installation

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Rust toolchain** | Any recent stable. Install via [rustup.rs](https://rustup.rs). |
| **Claude Code** | CLI (`claude`) or VS Code extension. |
| **macOS or Linux** | macOS is the primary target. Linux works — the only platform-specific calls are `mmap(2)`, `flock(2)`, and `gettimeofday(2)`. |

### Step 1 — Build

```sh
cd helix
cargo build --release
```

The release binary lands at `target/release/helix`. It is statically linked
with zero external dependencies. LTO and single-codegen-unit produce a ~1.1 MB
binary optimized for latency.

### Step 2 — Register the MCP Server

Claude Code discovers MCP servers from `~/.claude.json`. Add helix under
`mcpServers`:

```json
{
  "mcpServers": {
    "helix": {
      "command": "/absolute/path/to/helix/target/release/helix",
      "args": ["serve"]
    }
  }
}
```

> **Important:** Use the absolute path to the release binary. Relative paths
> and PATH-based lookup both work but are fragile — an absolute path ensures
> Claude Code can always find the server, even when launched from different
> working directories.
>
> On macOS, the MCP server is a long-running process. If you copy the binary
> elsewhere (e.g., `~/.local/bin/`), macOS code signing may kill it. The
> simplest fix is to point directly at the build output in `target/release/`.

### Step 3 — Install Hooks

```sh
./target/release/helix hooks install
```

This writes 9 hook entries to `~/.claude/settings.json`. Hooks make helix
*proactive* — knowledge is injected automatically, not just when the agent
asks.

If helix is on your PATH, you can run `helix hooks install` directly.

### Step 4 — Verify

```sh
helix hooks status     # Should list 9 hook events
helix index            # Should print "0 entries" on a fresh install
```

Start a new Claude Code session. The session hook should fire and you'll see:

```
HELIX KNOWLEDGE STORE: 0 entries across 0 topics.
```

Test the round trip:

```
You: "Store in helix under topic 'test': this is my first entry"
You: "Search helix for 'first entry'"
You: "Delete the test topic in helix"
```

If all three work, helix is fully operational.

### Updating

After making changes to the helix source code:

```sh
cargo build --release
```

If helix is already running as an MCP server inside Claude Code, you can
hot-reload without restarting the session:

```
You: "Run the _reload helix tool"
```

The `_reload` tool copies the new release binary into place, re-signs it, fixes
the MCP server path in `~/.claude.json` if needed, and re-execs the server
process. Zero downtime.

---

## Configuration Files

Helix touches several files during installation and operation. Here is the
complete map — **global** files affect all projects, **workspace** files affect
a single project.

### Claude Code Configuration

| File | Scope | What helix writes | Created by |
|------|-------|-------------------|------------|
| `~/.claude.json` | Global | MCP server registration under `mcpServers.helix` | You (Step 2). `_reload` also patches this to fix stale binary paths. |
| `~/.claude/settings.json` | Global | 9 hook entries under `hooks` | `helix hooks install` (Step 3). |
| `<project>/.claude/settings.json` | Workspace | *(Not written by helix.)* You can install hooks here instead of globally if you prefer per-project hooks. | Manual. |

> **Global vs workspace hooks:** `helix hooks install` writes to
> `~/.claude/settings.json` (global). If you only want helix active in
> specific projects, manually copy the hook entries to
> `<project>/.claude/settings.json` instead.

> **Global vs workspace MCP:** `~/.claude.json` registers helix globally.
> To register per-project, add the `mcpServers.helix` block to
> `<project>/.claude/settings.json` under an `mcpServers` key instead.

### Claude Instructions

| File | Scope | Purpose |
|------|-------|---------|
| `~/.claude/CLAUDE.md` | Global | Add the [helix usage block](#claudemd-integration) here for all projects. |
| `<project>/CLAUDE.md` | Workspace | Add the helix usage block here for a single project. |

### Helix Data

| File | Location | Purpose |
|------|----------|---------|
| `data.log` | `~/.helix-kb/` | Append-only binary knowledge log. **Back this up.** |
| `index.bin` | `~/.helix-kb/` | BM25 inverted index (auto-rebuilt, safe to delete). |
| `hook.sock` | `~/.helix-kb/` | Unix socket for hook→server IPC. Ephemeral. |
| `.lock` | `~/.helix-kb/` | `flock(2)` for write serialization. Ephemeral. |
| `checkpoint.json` | `~/.helix-kb/` | Durable task checkpoint (optional). |
| `helix-session.json` | `/tmp/` | Ephemeral session state (injected-entry dedup, phase detection). Expires after 4 hours. |

The KB directory defaults to `~/.helix-kb/`. Override by passing a directory
argument: `helix serve /path/to/kb` or `helix hook ambient /path/to/kb`.

---

## How It Works

Helix runs as two coordinated processes that share a single knowledge base on
disk:

### MCP Server (long-running)

Claude Code launches `helix serve` as a subprocess when a session starts. The
server speaks [JSON-RPC 2.0](https://www.jsonrpc.org/specification) over stdio
and handles all 9 tools. It holds the search index in memory and rebuilds it
automatically after writes (50 ms debounce to coalesce rapid stores).

### Hook Handlers (per-event)

Claude Code fires hooks by forking and executing `helix hook <type>` as a
short-lived subprocess. Each hook process `mmap(2)`s `index.bin` directly from
disk for zero-copy access — no deserialization, no socket round-trip to the
server. Total wall time is 2–3 ms, dominated by macOS `fork(2)` + `execve(2)`
overhead.

### Data Files

All data lives in `~/.helix-kb/` by default. See
[Configuration Files](#configuration-files) for the complete file map.

The only file you need to back up is `data.log` — everything else regenerates.

```sh
cp ~/.helix-kb/data.log ~/backups/helix-$(date +%Y%m%d).log
```

---

## MCP Tools

Helix exposes 9 MCP tools via the JSON-RPC stdio protocol. Claude Code sees
them as `mcp__helix__<name>`.

### `store`

Store a single knowledge entry under a topic.

```
store(topic: "build-gotchas", text: "Rust lib requires -arch arm64", tags: "gotcha,build")
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `topic` | yes | Topic name. Auto-sanitized to `lowercase-hyphenated`. |
| `text` | yes | Entry content. Plain text, any length. |
| `tags` | | Comma-separated tags (e.g., `"gotcha,perf,metal"`). Tags are auto-singularized (`gotchas` → `gotcha`). |
| `source` | | Source file reference (e.g., `"src/infer.rs:42"`). Enables staleness detection — helix warns when the file changes on disk. |
| `confidence` | | `0.0`–`1.0` (default `1.0`). Lower confidence entries rank lower in search. |
| `links` | | Narrative links to other entries: `"build-gotchas:3 iris-engine:12"`. |
| `force` | | `"true"` to bypass duplicate detection. |
| `terse` | | `"true"` for minimal response (first line only). |

**Duplicate detection:** If a new entry has >70% Jaccard token overlap with an
existing entry in the same topic (on 6+ tokens), helix warns instead of
silently storing a near-duplicate.

**Auto-tagging:** Content prefixes are detected and converted to tags
automatically. Write `"gotcha: the FFI bridge requires arm64"` and it gets
tagged `#gotcha` without passing the `tags` parameter. Recognized prefixes:

| Prefix | Auto-tag |
|--------|----------|
| `gotcha:`, `bug:` | `gotcha` |
| `invariant:`, `security:` | `invariant` |
| `decision:`, `design:` | `decision` |
| `perf:`, `benchmark:` | `performance` |
| `gap:`, `missing:`, `todo:` | `gap` |
| `how-to:`, `impl:`, `fix:` | `how-to` |
| `data flow:`, `flow:` | `data-flow` |
| `module:`, `overview:` | `module-map` / `architecture` |
| `coupling:`, `pattern:` | `coupling` / `pattern` |

### `batch`

Store multiple entries in one call. Holds the file lock once and rebuilds the
index once at the end — significantly faster than sequential `store` calls.

```
batch(entries: [
  {"topic": "api-design", "text": "REST endpoints use /v2/ prefix", "tags": "convention"},
  {"topic": "api-design", "text": "Auth tokens expire after 1 hour", "tags": "gotcha"}
])
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `entries` | yes | JSON array of `{topic, text, tags?, source?}` objects. |
| `verbose` | | `"true"` for per-entry details instead of summary count. |

Intra-batch Jaccard deduplication (70% threshold) prevents storing
near-duplicates within the same batch.

### `search`

Full-text search across all entries using BM25 ranking.

```
search(query: "FFI bridge arm64", detail: "brief", limit: "5")
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `query` | yes | Search query. Tokenized, CamelCase-split, case-insensitive. |
| `detail` | | Output mode (see table below). Default: `medium`. |
| `limit` | | Maximum results to return. |
| `after` | | Only entries on/after this date. Accepts `YYYY-MM-DD`, `today`, `yesterday`, `this-week`. |
| `before` | | Only entries on/before this date. |
| `days` | | Entries from last N days. |
| `hours` | | Entries from last N hours (overrides `days`). |
| `tag` | | Filter to entries with this tag. |
| `topic` | | Filter to entries in this topic. |
| `mode` | | `"and"` (default — all terms must match) or `"or"` (any term). |

**Detail modes:**

| Mode | Description |
|------|-------------|
| `full` | Complete entry body with section headers. |
| `medium` | Header line + first 2 body lines. **(default)** |
| `brief` | One-liner: topic, date, first content line. |
| `grouped` | Results bucketed by topic. |
| `topics` | Hit count per topic (no content). |
| `count` | Total match count only. |

**How search works:**

1. Query is tokenized: `"FxHashSet performance"` → `[fx, hash, set, performance]`.
2. CamelCase and snake_case are split automatically.
3. Stop words are filtered (`this`, `that`, `with`, `have`, etc.).
4. AND mode is tried first. If no results, automatically falls back to OR.
5. Scoring: BM25 (K1=1.2, B=0.75) with recency boost (30-day half-life) and
   confidence weighting.
6. Diversity cap: max 3 results per topic to prevent any single topic from
   dominating.

### `brief`

Compressed mental model of a topic or area. The primary tool for loading
context at the start of a task.

```
brief(query: "build-gotchas")
brief(query: "iris-*")           # glob: aggregate across matching topics
brief()                          # no query: meta-briefing of entire KB
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `query` | | Topic name, keyword, or glob pattern (e.g., `"iris-*"`). Omit for meta-briefing. |
| `detail` | | `"summary"` (~15 lines, default), `"scan"` (one-liner per category), `"full"` (complete entries). |
| `since` | | Only entries from last N hours. |
| `focus` | | Comma-separated categories to include (e.g., `"gotchas,invariants"`). |
| `compact` | | `"true"` for compact meta-briefing (top 5 topics only). |

**Compression pipeline:**

1. **Deduplication** — Identical entries across topics are merged. The
   highest-relevance copy is kept; others are noted as "also in: X".
2. **Supersession** — Entry pairs with >60% Jaccard overlap where one is
   significantly newer (2+ day gap): the older is marked `[SUPERSEDED]`.
3. **Temporal chaining** — Entries in the same topic sharing a dominant keyword
   are grouped chronologically to show evolution.

**Category classification** (based on tags):

| Category | Tags |
|----------|------|
| ARCHITECTURE | `architecture`, `module-map`, `coupling`, `interface` |
| DATA FLOW | `data-flow`, `lifecycle`, `protocol` |
| INVARIANTS | `invariant`, `boundary`, `concurrency` |
| GOTCHAS | `gotcha`, `bug`, `error-handling` |
| DECISIONS | `decision`, `deprecation`, `migration` |
| HOW-TO | `how-to`, `config`, `deploy` |
| PERFORMANCE | `perf`, `benchmark`, `optimization` |
| GAPS | `gap`, `friction`, `todo`, `missing` |

### `read`

Read the raw contents of a topic — all entries with timestamps, tags, and
full text.

```
read(topic: "build-gotchas")
read(topic: "build-gotchas", index: "3")   # single entry by index
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `topic` | yes | Topic name. |
| `index` | | Fetch a single entry by 0-based index. |

Use entry index numbers with `edit` to target specific entries for modification.

### `edit`

Modify existing entries. All mutations are append-only under the hood — a new
entry is written and the old one is tombstoned. This makes edits crash-safe
and undoable (before compaction).

| Parameter | Required | Description |
|-----------|----------|-------------|
| `action` | yes | `append` · `revise` · `delete` · `tag` · `rename` · `merge` |
| `topic` | yes | Topic name (or source topic for rename/merge). |
| `text` | varies | Content for `append`/`revise`. |
| `index` | | Target entry by 0-based index. |
| `match_str` | | Target entry by substring match. |
| `tag` | | Target most recent entry with this tag (`append` only). |
| `tags` | | Comma-separated tags to add (`tag` action). |
| `remove` | | Comma-separated tags to remove (`tag` action). |
| `all` | | `"true"` to delete entire topic (`delete` action). |
| `new_name` | | New topic name (`rename` action). |
| `into` | | Target topic to merge INTO (`merge` action). |

**Actions:**

| Action | Description |
|--------|-------------|
| `append` | Add text to an existing entry without changing its timestamp. Targets the last entry by default, or use `index`/`match_str`/`tag` to target a specific one. |
| `revise` | Replace an entry's text entirely. Adds `[modified: date]` marker. |
| `delete` | Remove by `index`, `match_str`, or `all="true"` for the entire topic. Default: deletes last entry. |
| `tag` | Add tags via `tags` or remove via `remove` on a specific entry. |
| `rename` | Rename a topic. All entries are preserved. Requires `new_name`. |
| `merge` | Move all entries from `topic` into the topic named by `into`. Source topic is deleted after merge. |

### `topics`

Browse and maintain the knowledge base. The `action` parameter selects the
operation (default: `list`).

**Browse actions:**

| Action | Description |
|--------|-------------|
| `list` | All topics with entry counts and last-modified dates. **(default)** |
| `recent` | Entries from last N `days`/`hours` across all topics. |
| `entries` | List entries in a `topic` with index numbers. |
| `stats` | KB statistics. `detail="tags"` for tag frequency, `detail="index"` for index health. |
| `xref` | Cross-references: entries in other topics that mention this `topic`. |
| `graph` | Topic dependency graph. `focus="iris-*"` for glob-filtered view. |
| `session` | Store operations performed this session. |

**Maintenance actions:**

| Action | Description |
|--------|-------------|
| `stale` | Entries with stale `[source:]` references. `refresh="true"` for details. |
| `prune` | Flag topics with no entries in N days (default: 30). |
| `compact` | Deduplicate and merge. `log="true"` to rewrite `data.log` without tombstones. `apply="true"` to execute. |
| `reindex` | Force full index rebuild. |
| `export` | Export entire KB as JSON. |
| `import` | Import from JSON string (`json` parameter). Merges with existing data. |

**Checkpoint actions** (durable save points that survive across sessions):

| Action | Description |
|--------|-------------|
| `checkpoint` | Save working state. Requires `task`. Optional: `done`, `next`, `hypotheses`, `blocked`, `files` (semicolon-separated lists). |
| `resume` | Load saved checkpoint. Returns formatted markdown with task, progress, hypotheses, and age. |
| `clear_checkpoint` | Delete the saved checkpoint. |

### `trace`

Code structure analyzer. Extracts symbols, call sites, and structural
dependencies from Rust, Swift, Metal, C, and Objective-C source files.

```
trace(path: "/path/to/project", action: "symbols")
trace(path: "/path/to/file.rs", action: "blast")
trace(path: "/path/to/project", action: "analyze")
trace(path: "/path/to/project", symbol: "dispatch_matvec")
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `path` | yes | File or directory path. |
| `symbol` | | Symbol name to trace. Required when `action` is omitted. |
| `action` | | `"symbols"` · `"blast"` · `"analyze"` · omit for symbol trace. |

**Actions:**

| Action | Description |
|--------|-------------|
| `symbols` | List all function, struct, enum, trait, class, and protocol definitions in a file or directory. |
| `blast` | Blast radius for a file — all external references to/from this file. |
| `analyze` | Deep project analysis: module map, symbol index, coupling matrix, pattern inventory. Results are automatically stored in the KB under topic `code-<dirname>`. |
| *(omit)* | Trace a specific symbol — find its definition and all call sites across the project. |

### `_reload`

Hot-reload the server binary after a rebuild. Copies the release binary to the
installed location, re-signs it, ensures `~/.claude.json` points to the correct
binary path, and re-execs the server process. No session restart needed.

```
_reload()
```

---

## Hooks

Helix installs 9 Claude Code hooks that make knowledge retrieval automatic.
The right context appears when it's needed — you rarely need to search
explicitly.

### How Hooks Work

1. Claude Code emits a hook event (e.g., "about to read file X").
2. macOS forks and executes `helix hook <type>`.
3. The hook process `mmap(2)`s `index.bin` directly from disk (zero-copy).
4. Relevant entries are extracted, scored, and returned as JSON.
5. Claude Code injects the response into the conversation context.
6. Total wall time: **2–3 ms** (dominated by `fork(2)` + `execve(2)` overhead).

### Hook Reference

#### SessionStart — `session`

Fires when a session begins or resumes. Injects a summary of the knowledge
base: total entries, topics sorted by size, and a reminder to search helix
before starting work.

#### UserPromptSubmit — `prompt`

Fires on every user message. Extracts keywords from your question, runs an
OR-mode BM25 search, and injects the top 3 matching entries. Conservative
filters prevent noise: skips prompts shorter than 10 chars or longer than 500,
requires at least 2 search terms.

#### PreToolUse — `ambient`

**The most impactful hook.** Fires every time Claude touches a file
(Read/Edit/Write/Glob/Grep). Runs a 5-layer context query:

| Layer | What it searches | Priority |
|-------|-----------------|----------|
| 1. Source path | Entries tagged with `[source: this_file]` | Highest — exact file knowledge |
| 2. Symbol search | OR-search for function/struct/class names extracted from the file (first 500 lines, up to 20 symbols) | High — symbol-level knowledge |
| 3. Stem BM25 | File stem as query (e.g., `kernels` from `kernels.rs`) | Medium — topic-level knowledge |
| 4. Structural coupling | Entries containing "structural" + the file stem | Low — architectural coupling notes |
| 5. Refactor impact | *(Edit only)* Searches for entries referencing removed symbols | Safety — prevents silent breakage |

Each layer deduplicates against results already injected in this session.

#### PostToolUse — `post-build`

Fires after Bash commands that look like build operations (`cargo`, `xcodebuild`,
`swift build`, `make`, etc.). Silent on success. On build failure, reminds
Claude to check helix for known gotchas.

#### PostToolUseFailure — `error-context`

Fires when any Bash command fails. Tokenizes the error message and searches
the KB for matching patterns. If you previously stored a gotcha about a
specific error, Claude sees it immediately on the next occurrence.

#### PreCompact — `pre-compact`

Fires before Claude Code compresses the conversation context. Re-injects
the full topic list so knowledge base awareness survives the compaction
boundary.

#### Stop — `stop`

Fires when the session is ending. Reminds Claude to store any unstored
findings before context is lost.

#### SubagentStart — `subagent`

Fires when a subagent is spawned. Injects the topic list and instructions to
search helix before starting work. Without this, subagents have no knowledge
of the KB.

#### PermissionRequest — `approve-mcp`

Auto-approves all `mcp__helix__*` tool calls so Claude doesn't prompt for
permission on every store, search, or read operation.

### Managing Hooks

```sh
helix hooks install       # Write all 9 hooks to ~/.claude/settings.json
helix hooks uninstall     # Remove all helix hooks
helix hooks status        # Show which hooks are currently configured
```

Restart Claude Code after installing or uninstalling hooks.

---

## Knowledge Organization

### Topics

Topics are the primary organizational unit. Each entry belongs to exactly one
topic. Topic names are automatically sanitized to `lowercase-hyphenated` form.

**Naming conventions:**

```
build-gotchas         — cross-cutting build issues
iris-engine           — project-subsystem knowledge
ref-xnu-kernel        — reference material from external sources
gpu-inference         — domain-specific findings
perf-data             — benchmark results and performance notes
```

### Entries

Each entry is a timestamped text blob under a topic. Entries are the atomic
unit of knowledge — each one captures a single fact, finding, decision, or
gotcha.

**Metadata** is stored as bracketed lines in the entry body:

```
[tags: gotcha, performance]          — classification
[source: src/kernels.rs:42]          — file reference (enables staleness detection)
[confidence: 0.85]                   — reliability score (0.0–1.0, default 1.0)
[links: build-gotchas:3 other:12]    — narrative links to related entries
```

### Export and Import

```
topics(action: "export")                  → JSON dump of entire KB
topics(action: "import", json: "...")     → merge JSON into existing KB
```

Import preserves original timestamps.

---

## Data Format

### data.log

Append-only binary log. Magic: `AMRL`, version 1.

```
┌──────────────────────────────────────────────────┐
│ Header (8 bytes)                                 │
│   magic:   0x41 0x4D 0x52 0x4C  ("AMRL")        │
│   version: u32 LE = 1                            │
├──────────────────────────────────────────────────┤
│ Entry Record (variable length)                   │
│   type:      u8  = 0x01                          │
│   topic_len: u8                                  │
│   body_len:  u32 LE                              │
│   timestamp: i32 LE (minutes since 2024-01-01)   │
│   pad:       2 bytes                             │
│   topic:     [u8; topic_len]                     │
│   body:      [u8; body_len]                      │
├──────────────────────────────────────────────────┤
│ Delete Tombstone (8 bytes)                       │
│   type:      u8  = 0x02                          │
│   reserved:  3 bytes                             │
│   offset:    u32 LE (file offset of target)      │
└──────────────────────────────────────────────────┘
```

Edits write a new entry and tombstone the old one. The log only grows.
Run `topics(action: "compact", log: "true", apply: "true")` to rewrite it
without tombstones.

### index.bin

Memory-mapped inverted index. Magic: `AMRN`, version 3. All structs are
`#[repr(C, packed)]` for zero-copy access via pointer casting.

Sections in order: Header → Term Hash Table → Posting Lists → Entry Metadata
→ Snippet Pool → Topic Table → Topic Names → Source Pool → Cross-refs → Tag
Names.

The index is rebuilt automatically after writes (50 ms debounce). It can be
deleted safely — helix regenerates it from `data.log` on next access.

---

## CLAUDE.md Integration

Add this block to `~/.claude/CLAUDE.md` (global) or your project's `CLAUDE.md`
so Claude knows how to use helix effectively:

```markdown
## Memory — Helix

Cross-session knowledge store via MCP tools (prefixed `mcp__helix__`).

**Every session:** `search` helix before starting work.
**During work:** `store` non-obvious findings immediately.
**Before stopping:** store anything learned that isn't obvious from the code.

Tools:
- **Search:** `search(query, detail?, limit?, days?, tag?, topic?)`
- **Write:** `store(topic, text, tags?, source?)` · `batch(entries)` · `edit(action, topic, ...)`
- **Read:** `brief(query?)` — compressed mental model · `read(topic)` — raw entries · `topics()` — browse KB
- **Code:** `trace(path, symbol?, action?)` — symbols, blast radius, project analysis
```

---

## CLI Reference

```
helix serve [dir]             Start MCP server (stdio JSON-RPC 2.0).
                              Default dir: ~/.helix-kb/

helix hook <type> [dir]       Run a hook handler.
                              Types: session, prompt, ambient, post-build,
                              error-context, pre-compact, stop, subagent,
                              approve-mcp

helix hooks install           Install all 9 hooks into ~/.claude/settings.json
helix hooks uninstall         Remove all helix hooks from settings
helix hooks status            Show current hook configuration

helix index [dir]             Force rebuild the search index
```

All commands that accept `[dir]` default to `~/.helix-kb/` when omitted.

---

## Troubleshooting

### MCP tools not appearing

Claude Code can't find the helix binary. Check `~/.claude.json`:

```sh
cat ~/.claude.json | python3 -c "import json,sys; print(json.load(sys.stdin)['mcpServers']['helix']['command'])"
```

The path must be an absolute path to a binary that exists. Verify:

```sh
/path/from/above/helix serve <<< '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
```

You should see a JSON response with `"serverInfo":{"name":"helix"}`. If you get
`killed` or `codesign` errors, point the MCP config at the `target/release/helix`
binary directly instead of a copied binary.

### Hooks not firing

1. Run `helix hooks status` — should show 9 events.
2. Check that the binary path in `~/.claude/settings.json` is correct.
3. Restart Claude Code after installing hooks.
4. Hooks and MCP can use different binaries: hooks are short-lived and work
   from any path. The MCP server is long-running and may get killed by macOS
   codesigning if the binary was copied.

### "No matches" for everything

1. Verify entries exist: ask Claude to run `topics()`.
2. Try a simpler query — AND mode requires all terms to match.
3. Force an index rebuild: `helix index` or ask Claude to run
   `topics(action: "reindex")`.

### Permission prompts on helix tools

The `approve-mcp` hook auto-approves all `mcp__helix__*` calls. If you're
still seeing prompts:

```sh
helix hooks status
```

Should list `PermissionRequest` with matcher `mcp__helix__.*`.

### Slow hooks (>5 ms)

Usually means the binary path in `~/.claude/settings.json` is wrong, causing
shell PATH lookup overhead on every hook invocation. Use a full absolute path.

### data.log growing large

Tombstones from edits and deletes accumulate. Compact it:

```
topics(action: "compact", log: "true", apply: "true")
```

This rewrites `data.log` without tombstones. Safe to run while the server
is active.

### Session deduplication issues

Session state lives at `/tmp/helix-session.json`. If hooks inject stale or
duplicate context:

```sh
rm /tmp/helix-session.json
```

The next hook invocation creates a fresh session.

---

## Architecture

### Module Map

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
            sock.rs          codegraph.rs (trace tool)
```

**Shared utilities:** `json.rs` (hand-rolled parser), `text.rs` (tokenization),
`time.rs` (Hinnant civil algorithm), `fxhash.rs` (fast hashing), `lock.rs`
(flock wrapper).

### Performance

| Operation | Target | Typical |
|-----------|--------|---------|
| Hook execution | <5 ms | 2–3 ms |
| MCP search | <20 ms | 5–10 ms |
| MCP store | <10 ms | 3–5 ms |
| Index rebuild | <100 ms | 50–80 ms (1000 entries) |
| Brief (topic) | <50 ms | 20–40 ms |

Key optimizations: hand-rolled JSON parser with unescaped string fast-path,
stack-allocated JSON-RPC IDs (`[u8; 24]`), deferred snippet extraction (only
for final top-K results), FxHash (~3ns vs SipHash ~20ns), reusable
`QueryState` buffers across queries.

### Detailed Documentation

For deep dives into specific areas:

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Full codebase walkthrough,
  binary format specs, design decisions
- [docs/TOOLS.md](docs/TOOLS.md) — Complete MCP tool reference with examples
  for every parameter
- [docs/HOOKS.md](docs/HOOKS.md) — Deep dive on all 9 hooks, input/output
  protocols, the 5-layer ambient context system

---

## License

MIT

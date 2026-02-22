# Helix

Helix is a persistent knowledge store for AI coding agents. It gives Claude
Code a long-term memory that survives across sessions, restarts, and context
compactions. Every finding, gotcha, architectural decision, and build failure
gets stored in an indexed knowledge base and automatically surfaced when
relevant.

**Zero dependencies. Single binary. ~900KB. 2-3ms hook latency.**

```
You: "Why does the FFI bridge fail on x86?"

Claude: (ambient hook fires, finds stored entry from 3 days ago)
  → "The FFI bridge requires arm64. It won't build for x86_64 because
     the Rust lib is compiled arm64-only. See build-gotchas topic."
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [MCP Tools](#mcp-tools)
- [Hooks](#hooks)
- [Knowledge Organization](#knowledge-organization)
- [Data Storage](#data-storage)
- [CLAUDE.md Setup](#claudemd-setup)
- [Migrating from Amaranthine](#migrating-from-amaranthine)
- [CLI Reference](#cli-reference)
- [Troubleshooting](#troubleshooting)
- [Further Reading](#further-reading)

---

## Quick Start

### Prerequisites

- Rust toolchain (`rustup` — any recent stable)
- Claude Code CLI or VS Code extension
- macOS (Linux support is untested but likely works — no platform-specific code
  outside of `mmap(2)` and `gettimeofday(2)`)

### Build and Install

```sh
git clone <repo> && cd llm_double_helix
cargo build --release
cp target/release/helix ~/.local/bin/   # or anywhere in your PATH
```

### Register MCP Server

Add helix to `~/.claude.json` under `mcpServers`:

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

If `helix` isn't in your PATH, use the full path:

```json
{
  "mcpServers": {
    "helix": {
      "command": "/Users/you/.local/bin/helix",
      "args": ["serve"]
    }
  }
}
```

### Install Hooks

```sh
helix hooks install
```

This writes 9 hook entries to `~/.claude/settings.json`. The hooks make helix
*proactive* — it injects knowledge automatically instead of waiting to be asked.

### Verify

```sh
helix hooks status     # Should list 9 hook events
helix index            # Should say "0 entries" (empty KB is fine)
```

Start a Claude Code session. You should see the session hook fire:

```
HELIX KNOWLEDGE STORE: 0 entries across 0 topics.
```

Try storing something:

> "Store in helix under topic 'test': this is my first entry"

Then search for it:

> "Search helix for 'first entry'"

If both work, you're set. Delete the test topic:

> "Delete the test topic in helix"

---

## How It Works

Helix runs as two coordinated processes:

### MCP Server (long-running)

When Claude Code starts, it launches `helix serve` as an MCP server. This
process speaks JSON-RPC 2.0 over stdio and handles all 7 tools (store,
batch, search, brief, read, edit, topics). It holds the search index in
memory and rebuilds it automatically after writes.

### Hook Handlers (per-event)

Claude Code fires hooks by forking and executing `helix hook <type>` as a
subprocess. Each hook process `mmap(2)`s the search index directly from disk
for zero-copy access. Total latency is 2-3ms, dominated by macOS fork+exec
overhead — the helix code itself adds effectively zero.

### The Knowledge Base

All data lives in `~/.helix-kb/`:

| File | Purpose |
|------|---------|
| `data.log` | Append-only binary log. Your entire KB in one file. Back this up. |
| `index.bin` | Search index (BM25). Rebuilt automatically. Deletable — will regenerate. |
| `hook.sock` | Unix socket for hook→server IPC. Ephemeral. |
| `.lock` | File lock for write serialization. |

---

## MCP Tools

Helix exposes 7 MCP tools. Claude Code calls them as `mcp__helix__<tool>`.

### `store` — Save a finding

```
store(topic: "build-gotchas", text: "arm64 only for FFI bridge", tags: "gotcha")
```

Creates a timestamped entry under the given topic. Entries are the atomic unit
of knowledge — each one is a fact, a finding, a decision, or a gotcha.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `topic` | yes | Topic name (auto-sanitized to lowercase-hyphenated) |
| `text` | yes | Entry content |
| `tags` | no | Comma-separated tags. Auto-singularized. Auto-detected from prefixes like `gotcha:`, `decision:`, `perf:` |
| `source` | no | Source file reference (`path/to/file:line`). Enables staleness detection. |
| `confidence` | no | 0.0-1.0 (default 1.0). Affects search ranking. |
| `links` | no | Narrative links: `topic:index topic:index` |
| `force` | no | `"true"` to bypass duplicate detection |
| `terse` | no | `"true"` for minimal response |

Duplicate detection: If your new entry has >70% token overlap (Jaccard
similarity) with an existing entry in the same topic, helix warns you.

### `batch` — Store multiple entries

```
batch(entries: [
  {"topic": "api-design", "text": "REST endpoints use /v2/ prefix", "tags": "convention"},
  {"topic": "api-design", "text": "Auth tokens expire after 1 hour", "tags": "gotcha"}
])
```

Stores multiple entries in a single operation. More efficient than repeated
`store` calls — single file open, single fsync at the end.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `entries` | yes | JSON array of `{topic, text, tags?, source?}` objects |

### `search` — Full-text search

```
search(query: "FFI bridge arm64", detail: "brief", limit: "5")
```

BM25-scored full-text search across all entries. Splits CamelCase and
snake_case automatically (`CachedEntry` matches `cached` and `entry`).

| Parameter | Required | Description |
|-----------|----------|-------------|
| `query` | yes | Search query (AND mode; falls back to OR if no AND results) |
| `detail` | no | Output format — see below |
| `limit` | no | Max results (default varies by detail mode) |
| `after` | no | Only entries after this date (`YYYY-MM-DD` or `today`/`yesterday`/`this-week`) |
| `before` | no | Only entries before this date |
| `days` | no | Shortcut: only entries from last N days |
| `hours` | no | Shortcut: only entries from last N hours |
| `tag` | no | Filter to entries with this tag |
| `topic` | no | Filter to this topic only |
| `mode` | no | `and` (default) or `or` |

**Detail modes:**

| Mode | Output |
|------|--------|
| `full` | Complete entry bodies with section headers |
| `medium` | 2-line snippets with tags (default) |
| `brief` | Topic + first content line per entry |
| `grouped` | Results organized by topic |
| `topics` | Hit counts per topic (no content) |
| `count` | Just the total match count |

### `brief` — Compressed briefing

```
brief(query: "network-module", detail: "summary")
```

The most powerful read tool. Produces a compressed mental model of a topic
by collecting, scoring, deduplicating, and categorizing entries.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `query` | no | Topic name, keyword, or glob pattern (`iris-*`). Omit for meta-briefing. |
| `detail` | no | `summary` (~15 lines, default), `scan` (one-liners), `full` (everything) |
| `since` | no | Only entries from last N hours |
| `focus` | no | Comma-separated categories to show (`gotchas,invariants`) |
| `compact` | no | `"true"` for compact meta-briefing (top 5 topics) |

Without a query, returns a meta-briefing: activity-weighted top topics,
velocity, and theme distribution.

With a glob pattern like `iris-*`, aggregates across all matching topics.

### `read` — Read a topic

```
read(topic: "build-gotchas")
```

Returns all entries in a topic with their index numbers, timestamps, and
full content. Use index numbers with `edit` to target specific entries.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `topic` | yes | Topic name |
| `index` | no | Fetch a single entry by index (0-based) |

### `edit` — Modify entries

A compound tool for all mutation operations. The `action` parameter selects
the operation:

| Action | What It Does | Key Parameters |
|--------|-------------|----------------|
| `append` | Add text to an existing entry | `text`, `index`/`match_str`/`tag` |
| `revise` | Overwrite an entry's text | `text`, `index`/`match_str` |
| `delete` | Remove entries | `all`, `index`, `match_str` (or delete last) |
| `tag` | Add/remove tags on an entry | `tags` (add), `remove` (remove), `index`/`match_str` |
| `rename` | Rename a topic | `new_name` |
| `merge` | Merge source topic into target | `into` |

All edit operations are append-only under the hood: they write a new entry
and tombstone the old one. This means edits are crash-safe and undoable
(before compaction).

### `topics` — Browse and maintain the KB

A compound tool for KB-wide operations:

| Action | What It Does |
|--------|-------------|
| `list` (default) | All topics with entry counts |
| `recent` | Topics with recent activity (configurable via `days`/`hours`) |
| `entries` | List entries in a topic (with `topic` param) |
| `stats` | KB statistics. `detail=tags` for tag frequency, `detail=index` for index stats |
| `xref` | Cross-references for a topic (which topics reference it) |
| `graph` | Topic dependency graph. `focus` param for glob filtering |
| `stale` | Entries with stale source references. `refresh=true` for details |
| `prune` | Remove entries older than N days (default 30) |
| `compact` | Rewrite data.log without tombstones (reclaim space) |
| `export` | Export entire KB as JSON |
| `import` | Import from JSON string (`json` param) |
| `reindex` | Force full index rebuild |
| `session` | Session summary (same as what the session hook shows) |

---

## Hooks

Helix installs 9 Claude Code hooks that make knowledge retrieval automatic.
You rarely need to explicitly search — the right context appears when you need it.

### Session Start (`SessionStart`)

Fires when a session begins or resumes. Injects a summary of the KB: total
entries, topics sorted by size. This is how Claude knows helix exists and
what's in it.

### Prompt Search (`UserPromptSubmit`)

Fires on every user message. Extracts keywords from your question, runs an
OR-mode BM25 search, and injects the top 3 matching entries. Conservative
filters prevent noise: skips prompts shorter than 10 chars or longer than 500,
requires at least 2 search terms.

### Ambient Context (`PreToolUse` — Read/Edit/Write/Glob/Grep)

The most impactful hook. Fires every time Claude touches a file. Runs a
5-layer context query:

1. **Source entries** — Entries with `[source: this_file.rs]` metadata
2. **Symbol search** — OR-search for the file stem + any removed symbols
3. **Global BM25** — File stem as query, top 3 results
4. **Structural coupling** — Other topics that share this source file
5. **Refactor impact** — Entries referencing symbols being deleted (Edit only)

This means if you stored "cache.rs uses mtime-based invalidation" with
`[source: cache.rs]`, Claude will see that entry every time it reads cache.rs.

### Post-Build (`PostToolUse` — Bash)

Fires after build commands (cargo, xcodebuild, swift, make, etc.). Reminds
Claude to store non-obvious build failures or fixes.

### Error Context (`PostToolUseFailure` — Bash)

Fires when a tool call fails. Tokenizes the error message and searches the KB
for matching patterns. If you previously stored a gotcha about a specific
error, Claude sees it immediately on the next occurrence.

### Pre-Compact (`PreCompact`)

Fires before Claude Code compacts the conversation context. Re-injects the
full topic list so KB awareness survives the compaction boundary.

### Stop (`Stop`)

Fires when the session is ending. Reminds Claude to store any unstored
findings. Includes a `stop_hook_active` check to prevent infinite loops
(a requirement of the Claude Code hook protocol).

### Subagent (`SubagentStart`)

Fires when a subagent is spawned. Injects the topic list and explicit
instructions to search helix before starting work. Without this, subagents
wouldn't know the KB exists.

### Auto-Approve (`PermissionRequest`)

Auto-approves all `mcp__helix__*` tool calls so Claude doesn't need to ask
permission for every store/search/read operation.

### Managing Hooks

```sh
helix hooks install     # Write all 9 hooks to ~/.claude/settings.json
helix hooks uninstall   # Remove all helix hooks
helix hooks status      # Show which hooks are configured
```

---

## Knowledge Organization

### Topics

Topics are the primary organizational unit. A topic is a named collection of
entries — think of it as a file in a notebook. Good topic names are
lowercase-hyphenated and descriptive:

```
build-gotchas       — things that went wrong during builds
api-design          — REST API conventions and decisions
network-module      — architecture of the network layer
session-2024-02-21  — findings from a specific session
```

### Entries

Each entry is a timestamped text blob under a topic. Entries can have:

- **Tags** — categorization (`gotcha`, `decision`, `architecture`, `gap`)
- **Source references** — links to source files (`[source: src/cache.rs:42]`)
- **Confidence** — 0.0-1.0, affects search ranking
- **Narrative links** — references to other entries (`[links: other-topic:3]`)

Tags are auto-detected from content prefixes. Write "gotcha: the FFI bridge
requires arm64" and it gets tagged `#gotcha` automatically. Supported prefixes:

| Prefix | Tag |
|--------|-----|
| `gotcha:`, `bug:` | gotcha |
| `invariant:`, `security:` | invariant |
| `decision:`, `design:` | decision |
| `data flow:`, `flow:` | data-flow |
| `perf:`, `benchmark:` | performance |
| `gap:`, `missing:`, `todo:` | gap |
| `how-to:`, `impl:`, `fix:` | how-to |
| `module:` | module-map |
| `overview:` | architecture |
| `coupling:`, `pattern:` | coupling / pattern |

### Briefings

The `brief` tool compresses a topic into a categorized summary. Categories
are derived from tags:

```
=== NETWORK-MODULE === 37 → 28 compressed

--- ARCHITECTURE (8) ---
  [network-module] TCP connection pool design: ...
  [network-module] Event-driven I/O with epoll: ...

--- GOTCHAS (5) ---
  [network-module] DNS resolution blocks the main thread: ...

--- GAPS (3) ---
  [network-module] Missing: connection timeout configuration

STATS: 37 compressed to 28 (24% reduction)
```

---

## Data Storage

### Binary Format

**data.log** (magic: `AMRL`, version 1) is an append-only log with two record types:

```
Entry record (variable length):
  [0x01] [topic_len: u8] [body_len: u32 LE] [timestamp: i32 LE] [topic bytes] [body bytes]

Delete tombstone (8 bytes):
  [0x02] [padding: 3 bytes] [target_offset: u32 LE]
```

Edits (revise, tag, append) write a new entry + tombstone the old one. This
means data.log only grows. Run `topics(action: "compact")` periodically to
reclaim space.

**index.bin** (magic: `AMRN`, version 3) is a single contiguous binary blob:

```
Header → Hash Table → Posting Lists → Entry Metadata → Snippets
→ Topic Table → Topic Names → Sources → Cross-refs → Tag Names
```

All structs are `repr(C, packed)` for zero-copy access via pointer casting.
The index can be `mmap(2)`d without any deserialization — this is what makes
2-3ms hook latency possible.

### Backup

Back up `~/.helix-kb/data.log`. That's it. Everything else regenerates.

```sh
cp ~/.helix-kb/data.log ~/backups/helix-$(date +%Y%m%d).log
```

### Export / Import

```
topics(action: "export")     → JSON dump of entire KB
topics(action: "import", json: "...")  → Import from JSON
```

---

## CLAUDE.md Setup

Add this to `~/.claude/CLAUDE.md` (global) or your project's `CLAUDE.md` so
Claude knows how to use helix. Copy this block verbatim:

```markdown
## Memory — helix
Cross-session knowledge store via MCP tools (prefixed `helix__`).

**Every session:** `search` helix before starting work.
**During work:** `store` non-obvious findings immediately.
**Before stopping:** store anything you learned that isn't obvious from the code.

**Search:** `search(query, detail?, limit?, days?, tag?, topic?)`
**Write:** `store(topic, text, tags?, source?)` | `batch(entries)` | `edit(action, topic, ...)`
**Read:** `brief(query?)` — compressed mental model | `read(topic)` — full entries | `topics()` — browse KB
```

---

## Migrating from Amaranthine

If you used amaranthine, helix reads the same `data.log` format:

```sh
# Copy data
cp ~/.amaranthine/data.log ~/.helix-kb/data.log

# Rebuild index
helix index

# Update MCP server in ~/.claude.json:
#   Change "amaranthine" → "helix" and update the command path

# Install helix hooks (replaces amaranthine hooks):
helix hooks install
```

---

## CLI Reference

```
helix serve [dir]           Start MCP server (stdio JSON-RPC)
                            Default dir: ~/.helix-kb/

helix hook <type> [dir]     Run a hook handler. Types:
                            session, prompt, ambient, post-build,
                            error-context, pre-compact, stop, subagent,
                            approve-mcp

helix hooks install         Install all 9 hooks into ~/.claude/settings.json
helix hooks uninstall       Remove helix hooks from settings
helix hooks status          Show current hook configuration

helix index [dir]           Force rebuild the search index
```

---

## Troubleshooting

**"helix: command not found"** — The binary isn't in your PATH. Use the full
path in `~/.claude.json` instead, or add `~/.local/bin` to your PATH.

**Hooks not firing** — Run `helix hooks status` to verify they're installed.
Restart Claude Code after installing hooks. Check that the binary path in
`~/.claude/settings.json` is correct.

**"no matches" for everything** — The index might be stale or missing. Run
`helix index` to rebuild. If the KB is empty, store some entries first.

**Permission prompts for helix tools** — The `approve-mcp` hook should
auto-approve `mcp__helix__*` calls. If you're still seeing prompts, verify
the hook is installed: `helix hooks status` should list `PermissionRequest`.

**Slow hooks (>5ms)** — This usually means the binary isn't at the path
specified in settings.json, causing shell PATH lookup overhead. Use a full
absolute path.

**data.log growing large** — Run `topics(action: "compact")` via Claude or
`helix serve` + a compact call. This rewrites data.log without tombstones.

---

## Further Reading

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Full codebase walkthrough
  with data flow traces, binary format specifications, and design decisions
- [docs/TOOLS.md](docs/TOOLS.md) — Complete MCP tool reference with examples
  for every parameter
- [docs/HOOKS.md](docs/HOOKS.md) — Deep dive on all 9 hooks, their input/output
  protocols, and the 5-layer ambient context system

---

## License

MIT

# Helix Hooks — Deep Dive

Helix hooks are the proactive half of the system. While MCP tools wait to be
called, hooks fire automatically on Claude Code events and inject relevant
knowledge into the conversation context. They're the difference between
"Claude can search if you ask" and "Claude already knows."

---

## How Hooks Execute

Claude Code supports user-defined hooks — shell commands that execute in
response to lifecycle events. Helix installs 9 hooks via `helix hooks install`,
which writes to `~/.claude/settings.json`.

Each hook invocation is a fork+exec of the helix binary:

```
Claude Code event → fork → exec("helix hook <type>") → read stdin → mmap index → write stdout → exit
```

**Latency budget: 2-3ms.** The macOS fork+exec overhead is ~1.5ms. Helix's
code adds effectively zero — it `mmap(2)`s the index from disk (zero-copy,
no deserialization) and does all string operations without heap allocation
where possible.

### Input Protocol

Claude Code pipes JSON to the hook's stdin. The shape depends on the event:

```json
// PreToolUse
{"tool_name": "Read", "tool_input": {"file_path": "/path/to/file.rs"}}

// PostToolUse
{"tool_name": "Bash", "tool_input": {"command": "cargo build"}, "output": "..."}

// Stop
{"stop_hook_active": "true"}  // or absent

// SessionStart
{"source": "startup"}
```

### Output Protocol

Hooks return JSON to stdout. The structure depends on the hook type:

```json
// Context injection (most hooks)
{"hookSpecificOutput": {"additionalContext": "text injected into conversation"}}

// System message (post-build)
{"systemMessage": "text shown as a system message"}

// Permission decision (approve-mcp)
{"hookSpecificOutput": {"hookEventName": "PermissionRequest",
                        "decision": {"behavior": "allow"}}}

// Suppress output (empty stdout = no-op)
```

---

## The 9 Hooks

### 1. Session (`SessionStart`)

**When:** Session starts or resumes.

**What it does:** Counts entries and topics from the mmap'd index. Injects a
summary so Claude knows the KB exists and what's in it.

**Output example:**
```
HELIX KNOWLEDGE STORE: 847 entries across 52 topics.
Topics: engine (75), scanners (72), network (37), build-gotchas (28)...
BEFORE starting work, call mcp__helix__search with keywords relevant to your task.
```

**Why it matters:** Without this, Claude doesn't know helix exists. The session
hook is the bootstrap — it tells Claude "you have a memory, use it."

### 2. Prompt (`UserPromptSubmit`)

**When:** Every user message.

**What it does:** Extracts keywords from the user's prompt, runs an OR-mode
BM25 search, and injects the top 3 matching entries.

**Filters (to prevent noise):**
- Skips prompts shorter than 10 characters
- Skips prompts longer than 500 characters (pastes, not questions)
- Requires at least 2 search terms after tokenization
- Caps at 6 query terms, 3 results

**Output example:**
```
helix knowledge (relevant to your question):
  [build-gotchas] 2026-02-21 arm64 only for FFI bridge — Rust lib compiled arm64
  [network-module] 2026-02-20 DNS resolution blocks the main thread
  [api-design] 2026-02-19 REST endpoints use /v2/ prefix for all new routes
```

**Why it matters:** You ask "why is the build failing?" and Claude already sees
the stored build gotchas before it even starts investigating.

### 3. Ambient (`PreToolUse` — Read/Edit/Write/Glob/Grep/NotebookEdit)

**When:** Claude reads, edits, writes, or searches for a file.

**What it does:** Runs a 5-layer context query based on the file being touched:

**Layer 1 — Source entries:**
Finds entries with `[source: <matching_file>]` metadata. If you stored
"cache.rs uses mtime-based invalidation" with `source: src/cache.rs`, this
surfaces every time Claude reads cache.rs.

**Layer 2 — Symbol OR search:**
Searches the index for the file stem (`cache`) plus any removed symbols
(only for Edit operations — extracts symbol names from the diff).

**Layer 3 — Global BM25:**
Searches the entire index using the file stem as a query, takes top 3.
Catches entries that mention the file without having a source reference.

**Layer 4 — Structural coupling:**
Finds other topics that share source references to this file. If both
`engine` and `network-module` have entries sourced to `connection.rs`,
touching `connection.rs` surfaces knowledge from both topics.

**Layer 5 — Refactor impact (Edit only):**
When Claude edits a file and removes symbols, searches for entries that
mention those symbols. Warns about potential breakage.

All 5 layers are dedup'd (no repeated entries) and capped at ~2KB total.

**Output example:**
```
symbol context:
  [network-module] TCP connection pool design: max 8 connections, round-robin
source context:
  [engine] Data flow: events come through ES client → queue → dispatcher
coupling:
  Topics also referencing connection.rs: network-module, engine, proxy-ext
```

**Why it matters:** This is the most impactful hook. It means Claude sees
relevant knowledge every time it touches a file, without being asked. Stored
a gotcha about a file 3 weeks ago? It surfaces the next time that file is
touched.

### 4. Post-Build (`PostToolUse` — Bash)

**When:** After Bash commands that match build patterns: `cargo`, `xcodebuild`,
`swift build`, `make`, `npm run build`, `gcc`, `clang`, `rustc`.

**What it does:** Returns a system message reminding Claude to store non-obvious
build failures or fixes in helix.

**Output:**
```
BUILD COMPLETED. If the build failed with a non-obvious error, store the
root cause in helix (topic: build-gotchas). If it succeeded after fixing
an issue, store what fixed it.
```

**Why it matters:** Build failures are the highest-value knowledge to store —
they're specific, reproducible, and frustrating to re-discover. This hook
nudges Claude to capture them in the moment.

### 5. Error Context (`PostToolUseFailure` — Bash)

**When:** A Bash tool call fails (non-zero exit code).

**What it does:** Tokenizes the error message and runs an OR-mode BM25 search
against the KB. If there are matching entries (e.g., a previously stored
gotcha about this exact error), they're injected into context.

**Filters:**
- Requires error message at least 15 characters
- Requires at least 2 search terms
- Caps at 8 query terms, 3 results

**Output example:**
```
helix: relevant knowledge for this error:
  [build-gotchas] 2026-02-20 linker error: arm64 only — use -arch arm64
  [build-gotchas] 2026-02-19 duplicate symbol: ensure ONLY_ACTIVE_ARCH=YES
```

**Why it matters:** Claude encounters the same build error → sees the fix you
stored last week → applies it immediately instead of debugging from scratch.

### 6. Pre-Compact (`PreCompact`)

**When:** Claude Code is about to compact the conversation context (this
happens when the context window fills up).

**What it does:** Re-injects the full topic list and a reminder that helix
exists. After compaction, Claude's context is significantly reduced — without
this hook, it might lose awareness of the KB entirely.

**Output example:**
```
CONTEXT PRESERVED — HELIX KB: 52 topics available. After compaction,
search helix for knowledge.
Topics: engine (75), scanners (72), network (37)...
```

**Why it matters:** Context compaction can erase Claude's awareness of helix.
This hook ensures continuity across compaction boundaries.

### 7. Stop (`Stop`)

**When:** The session is about to end.

**What it does:** Reminds Claude to store any unstored findings before the
session ends.

**Critical protocol detail:** Checks the `stop_hook_active` flag in stdin. If
set, returns empty (no-op). This prevents infinite loops — without this check,
the stop hook's output could trigger another stop, which triggers another
stop, ad infinitum.

**Output:**
```
STOPPING: Store any non-obvious findings in helix before ending.
```

### 8. Subagent (`SubagentStart`)

**When:** A subagent (background research agent, code explorer, etc.) is spawned.

**What it does:** Injects the topic list and explicit MCP tool search
instructions. Subagents start with minimal context and don't inherit the
parent's hook-injected knowledge.

**Output example:**
```
HELIX KNOWLEDGE STORE: You have access to helix MCP tools. BEFORE starting
work, call mcp__helix__search with keywords relevant to your task.
Topics: engine (75), scanners (72), network (37)...
```

**Why it matters:** Without this, subagents don't know helix exists. They'd
research from scratch instead of checking stored knowledge first.

### 9. Auto-Approve (`PermissionRequest`)

**When:** Claude Code asks for permission to call an MCP tool matching
`mcp__helix__.*`.

**What it does:** Returns an "allow" decision, bypassing the interactive
permission prompt.

**Output:**
```json
{"hookSpecificOutput":{"hookEventName":"PermissionRequest",
                       "decision":{"behavior":"allow"}}}
```

**Why it matters:** Without this, every `store`, `search`, `read`, `brief`
call would prompt you for permission. Auto-approving helix tools makes the
knowledge store friction-free.

---

## Ambient Context: The 5-Layer System

The ambient hook deserves special attention because it's the most complex
and most impactful.

### How It Works

```
Claude reads src/cache.rs
  │
  ├── file_stem = "cache"
  ├── file_path = "src/cache.rs"
  │
  ├── Layer 1: source_entries_for_file("cache.rs")
  │   └── Entries with [source: cache.rs] or [source: src/cache.rs]
  │
  ├── Layer 2: idx_search_or("cache") + removed_symbols
  │   └── OR-mode BM25 for file stem + any Edit-removed symbols
  │
  ├── Layer 3: idx_search("cache", limit=3)
  │   └── Global BM25, top 3
  │
  ├── Layer 4: topics sharing source references to cache.rs
  │   └── "Also referenced by: engine, network-module"
  │
  └── Layer 5: entries mentioning removed symbols (Edit only)
      └── "Warning: symbol CachedEntry referenced in engine topic"
```

### Symbol Cache

For Edit operations, helix needs to know which symbols are being removed.
It extracts function/struct/class/enum names from the file via regex and
caches them in `/tmp/helix-sym-cache` keyed on `(file_path, mtime)`. This
avoids re-parsing files that haven't changed.

### Performance

The entire 5-layer query runs in <1ms (on top of the 2ms fork+exec overhead).
All index access is via mmap — no heap allocation for index reads.

---

## Configuration

### Installing

```sh
helix hooks install
```

Writes to `~/.claude/settings.json`. Restart Claude Code or reload hooks.

### Customizing

The installed hooks use the binary path discovered at install time. If you
move the binary, re-run `helix hooks install`.

To customize individual hooks (e.g., change timeouts or matchers), edit
`~/.claude/settings.json` directly. The structure:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read|Edit|Write|Glob|Grep|NotebookEdit",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/helix hook ambient",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

### Disabling Individual Hooks

Remove the event entry from `~/.claude/settings.json`. For example, to disable
the prompt hook (if you find it too noisy), delete the `UserPromptSubmit` entry.

### Uninstalling

```sh
helix hooks uninstall
```

Removes all helix hooks from settings.json. Other (non-helix) hooks are
preserved.

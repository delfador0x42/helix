Cross-session pattern drift — "you've stored 5 entries about the same concept using different terminology over 3 sessions" — I can't see that because each session starts from zero
Knowledge decay detection — "these 12 entries reference APIs that changed 2 weeks ago" — requires monitoring over time, not a one-shot query
Emergent architecture — "your last 20 stored findings cluster around 3 themes you haven't named as topics yet" — requires seeing the accumulation, not a snapshot
Cross-project insight — if you had Iris KB + helix KB + other projects, a background model could find "you solved this same problem differently in project A vs B"




I totally understand claude eg JARVIS :] is right there but a daemon llm that could continually give deep cross context/project insight I think would be a good idea -- I'd download and run you to do it but I can't



thinking fast and slow




# Helix v10: Dual-Process Attention Architecture

## Context
23 improvement ideas accumulated during Iris v2 probe build. All share a root cause: **helix is memory without attention.** The storage layer is excellent (data.log, binary index, BM25, compression, one-shot briefings). But every hook fires in isolation, every MCP call is stateless, nothing tracks what's happening right now.

The solution is two systems mirroring the brain's dual-process model:
- **System 1 (session accumulator)**: fast, heuristic, zero-latency, lives in the hot path
- **System 2 (background model)**: slow, deliberate, runs on idle, discovers patterns humans miss

Both write to the same knowledge base. System 1 is the attention mechanism. System 2 is the insight generator.

## The Deep Problem

```
Today: Hook = fork() → mmap → query → exit  (stateless, isolated)
       MCP  = dispatch(args) → response       (stateless, independent)
```

Nothing tracks: task underway, work phase, files touched, context already injected, build state. The system treats every moment identically.

## Architecture: Two Systems, One Knowledge Base

```
┌─────────────────────────────────────────────────────────┐
│                    data.log + index.bin                   │
│                  (shared knowledge base)                  │
└──────────────┬──────────────────────┬────────────────────┘
               │                      │
    ┌──────────┴──────────┐ ┌────────┴─────────────────┐
    │   SYSTEM 1 (fast)   │ │   SYSTEM 2 (slow burn)   │
    │  Session Accumulator │ │  Background Intelligence │
    │  0ms added latency   │ │  runs on idle / between  │
    │  in-process state    │ │  sessions, writes back   │
    │  heuristic attention │ │  findings as entries     │
    └─────────────────────┘ └──────────────────────────┘
```

---

## SYSTEM 1: Session Accumulator

### What It Does
Lightweight state file tracking what's happening RIGHT NOW. Updated by every hook invocation and MCP call. Used to: dedup injected context, weight search results, suppress noise, adapt output verbosity.

### State Shape
```
~/.helix/session.json
{
  "id": "tty-s003-1740000000",
  "started": 1740000000,
  "last_active": 1740003600,
  "focus_topics": ["iris-probes", "build-gotchas"],
  "phase": "build",
  "files": {
    "KextCensusProbe.swift": { "op": "created", "t": 1740000100 },
    "ProbeRunner.swift": { "op": "edited", "t": 1740000200 }
  },
  "injected": [42, 67, 103],
  "last_build": { "ok": true, "t": 1740000300, "errors": [] },
  "tool_seq": ["Read", "Read", "Read", "Edit", "Edit", "Bash"],
  "pending_notes": []
}
```

### Session Identity
Experiment with all three, pick the best:
- **TTY-based**: `ttyname(STDIN_FILENO)` — different terminals = different sessions. Simple.
- **Parent PID chain**: walk ppid to find Claude Code process. More precise, fragile.
- **Timestamp-bucketed**: 30min idle gap = new session. No identity needed.

Start with TTY + idle timeout (4h). Session file is single-writer (flock), multi-reader.

### Phase Detection
Experiment with all three, pick the best:
- **Heuristic**: sliding window over `tool_seq`. Read,Read,Read,Grep = research. Edit,Edit,Write = build. Bash+xcodebuild = verify. Bash+error = debug.
- **Explicit**: `session(phase='build')` MCP tool. Claude sets it.
- **None**: just dedup + files + build state, skip phase entirely.

Start with heuristic (it's free). Add explicit override if heuristic is wrong >20% of the time.

### How It Fixes the 23 Friction Points

| # | Friction | Fix |
|---|---------|-----|
| 1 | Stale context after edits | `files` tracks edits → skip entries about old state |
| 2 | Nothing for new files | `files.op == "created"` → inject directory topic brief |
| 3 | Entry re-injection | `injected` set → dedup across hook invocations |
| 4 | No edit graph | `files` timestamps → weight related files |
| 5 | Post-build noise | `last_build.ok` → only remind on failure |
| 6 | Missing error extraction | Post-build parses errors into `last_build.errors` |
| 7 | Subagent redundancy | `injected` populated → skip re-injection |
| 8 | No cumulative context | Session accumulates across all tool calls |
| 9 | Brief too verbose | `phase` → tight output during build |
| 10 | No phase awareness | `tool_seq` heuristic or explicit |
| 11 | No temporal diff | `started` → "since session start" filter |
| 12 | BM25 scores hidden | separate concern (just expose scores) |
| 13 | Store breaks flow | `pending_notes` queue |
| 14 | No auto-suggest | Post-Write hook checks if file matches topics |
| 15 | No auto-link | `focus_topics` → suggest links on store |
| 16 | Build errors not captured | `last_build.errors` → auto-store gotcha |

### Implementation

**Files to create:**
- `session.rs` (~150 lines) — Session struct, read/write/lock, update helpers, idle expiry

**Files to modify:**
- `hook.rs` (~100 lines changed) — ambient reads/writes session, post-build writes build state, stop flushes pending_notes
- `dispatch.rs` (~80 lines changed) — search/brief reads focus_topics + phase for weighting
- `mcp.rs` (~20 lines) — load session on startup, pass to dispatch

**Zero new dependencies.** Uses existing json.rs parser, lock.rs flock, config.rs paths.

---

## SYSTEM 2: Background Intelligence

### What It Does
A separate process (daemon or cron-triggered) that runs a local model against the knowledge base to discover patterns, gaps, and connections that heuristics can't find. Runs on idle time. Writes findings back as regular helix entries tagged `[source: system2]`.

### What It Discovers
1. **Topic consolidation** — scans all topics, finds >40% content overlap, suggests merges
2. **Gap detection** — analyzes coverage by category, finds areas with no entries
3. **Cross-link discovery** — finds entries that reference the same concepts but aren't linked
4. **Stale entry detection** — finds entries contradicted by newer entries
5. **Design pattern extraction** — identifies recurring patterns across stored code knowledge
6. **Knowledge quality scoring** — ranks entries by informativeness, flags low-value noise

### Model Choice
Research findings (ranked by fit):

| Approach | Latency | Memory | Accuracy | Complexity |
|----------|---------|--------|----------|-----------|
| **Embedding similarity (ort/candle)** | 3-5ms/entry | ~80MB | Good for similarity | Low |
| **SmolLM2-135M (GGUF)** | 10-20ms/classify | ~300MB | Decent for classification | Medium |
| **Qwen2.5-0.5B (GGUF)** | 20-50ms/classify | ~600MB | Strong for reasoning | Medium |
| **Qwen2.5-1.5B (GGUF)** | 50-100ms/classify | ~1.2GB | Best quality | High |

**Recommendation**: Start with **embedding model** (all-MiniLM-L6-v2 via `ort` crate, 22M params, 80MB). Pre-compute embeddings for all entries at startup. Use cosine similarity for cross-link discovery, topic overlap detection, and semantic gap finding. This handles tasks 1-4 above.

Add a **small generative model** (Qwen2.5-0.5B via llama-cpp-2) for tasks 5-6 where reasoning is needed. This runs much less frequently (once per session end, not per entry).

### Architecture

```
System 2 Daemon (background process)
  │
  ├── On startup: load embedding model, pre-compute all entry embeddings
  ├── On idle (no MCP calls for 5min): run analysis pass
  │     ├── Cosine similarity matrix → find topic overlaps
  │     ├── Cluster entries → detect gaps between clusters
  │     ├── Find unlinked but related entries → suggest links
  │     └── Score entry quality → flag noise
  │
  ├── On session end (Stop hook): run deeper analysis with generative model
  │     ├── Summarize session findings → suggest consolidation
  │     └── Extract design patterns from stored code knowledge
  │
  └── Writes findings to data.log as regular entries
        Tagged: [source: system2] [confidence: 0.7]
        Topics: helix-insights, or the relevant topic
```

**Key principle**: System 2 writes to the same knowledge base. Its findings surface through normal System 1 search/brief. The two systems compose naturally.

### Integration with Existing Architecture

- **Separate binary** or **mode flag**: `helix serve --background` starts the daemon
- Reads data.log (shared, append-only, safe for concurrent readers)
- Writes via the existing store path (socket to MCP server, or direct append with flock)
- **No changes to hook latency** — System 2 never runs in the hook path
- Pre-computed embeddings cached in `~/.helix/embeddings.bin` (rebuilt on data.log change)

### New Dependencies (System 2 only, NOT in hot path)
- `ort` crate (ONNX Runtime Rust bindings) — for embedding inference
- `all-MiniLM-L6-v2.onnx` model file (~80MB, downloaded once)
- Optionally: `llama-cpp-2` crate + Qwen2.5-0.5B GGUF (~600MB) for generative tasks

These deps live in System 2 binary only. The main helix binary (System 1) stays zero-dep.

---

## Implementation Plan

### Phase 1: Session Accumulator (System 1) — ~330 lines
**This is the high-value, zero-cost fix.**

1. `session.rs` — Session struct with read/write/flock, TTY-based identity, 4h idle expiry
2. `hook.rs` changes:
   - ambient: read session → dedup via `injected` → write updated files/injected
   - post-build: parse exit code → write `last_build` (only remind on failure)
   - stop: flush `pending_notes` as batch store
   - subagent: check `injected` before re-injecting
3. `dispatch.rs` changes:
   - search/brief: read `focus_topics` → boost matching entries
   - search/brief: read `phase` → adjust result count (research=10, build=5, debug=gotchas-first)
   - store: check `focus_topics` → auto-suggest `[links:]`
4. Phase detection heuristic in session.rs: sliding window over tool_seq

**Build, deploy, test with a real Iris session. Measure noise reduction.**

### Phase 2: Background Embeddings (System 2 foundation) — ~400 lines
1. New binary: `helix-bg` (or mode flag)
2. Embedding pipeline: load all-MiniLM-L6-v2 via ort, embed all entries
3. Similarity matrix: pairwise cosine for topic overlap detection
4. Gap analysis: cluster embeddings, find sparse regions
5. Cross-link suggestions: high-similarity entries in different topics → suggest link
6. Write findings as entries tagged `[source: system2]`
7. Cache embeddings in `embeddings.bin`, invalidate on data.log mtime change

### Phase 3: Generative Intelligence (System 2 deep) — ~300 lines
1. Add Qwen2.5-0.5B via llama-cpp-2 (only in background binary)
2. Session-end analysis: summarize what was learned, suggest consolidation
3. Design pattern extraction from stored code knowledge
4. Entry quality scoring (informativeness, redundancy, staleness)

### Verification
- Phase 1: Run Iris build session, compare hook output before/after session state
  - Metric: % of ambient hook injections that are duplicates (target: 0%)
  - Metric: post-build noise on successful builds (target: 0 messages)
  - Metric: brief output size during build phase (target: 50% reduction)
- Phase 2: Run embedding analysis on full KB (~1178 entries)
  - Metric: topic pairs with >40% similarity (candidates for merge)
  - Metric: unlinked entries with >0.8 cosine similarity (missed connections)
- Phase 3: Compare generative insights against manual curation
  - Metric: % of suggestions that are actionable

## Files Involved

### System 1 (in helix/)
- **Create**: `src/session.rs` (~150 lines)
- **Modify**: `src/hook.rs` (~100 lines changed)
- **Modify**: `src/mcp/dispatch.rs` (~80 lines changed)
- **Modify**: `src/mcp.rs` (~20 lines)

### System 2 (new binary or mode)
- **Create**: `src/background.rs` (~200 lines) — daemon loop, idle detection
- **Create**: `src/embed.rs` (~150 lines) — embedding pipeline, similarity
- **Create**: `src/insight.rs` (~150 lines) — gap/overlap/link analysis
- **Add dep**: `ort` crate in Cargo.toml (feature-gated behind `background` feature)
- **Add dep**: `llama-cpp-2` in Cargo.toml (feature-gated behind `generative` feature)

### Model files (downloaded once)
- `~/.helix/models/all-MiniLM-L6-v2.onnx` (~80MB)
- `~/.helix/models/qwen2.5-0.5b-q4_k_m.gguf` (~600MB, Phase 3 only)

================================================================================
More thoughts and context for why this project exists


On the Dual Architecture
System 1 (Session Accumulator) exists in helix now. That's what we just built. It's ~45 lines of actual wiring on top of the 225-line session.rs. It's not in amaranthine in any useful form — amaranthine has a session.rs from this session too, but its hooks aren't wired to it and amaranthine isn't even the active binary.

System 2 (Background Intelligence with embeddings/LLM) does not exist in either codebase. It was never built. It's purely a plan.

My honest take on System 2: don't build it.

Here's why:

It solves a curation problem, not a retrieval problem. System 2 is about finding duplicate topics, stale entries, gaps in coverage. Those are real problems — but they're problems that affect the KB maybe once a month. You can solve them with a single mcp__helix__search + manual judgment session. Building an embedding pipeline with ort + a daemon loop + an ONNX model to automate something you do occasionally is the definition of over-engineering.

The dependency cost is real. Helix's superpower is zero external deps. Adding ort (ONNX runtime) or llama-cpp-2 instantly makes the build fragile — C++ deps, platform-specific linking, model files to manage. The main helix binary stays zero-dep in theory, but in practice you'd be maintaining a second binary that shares code and diverges over time.

The existing tools already do most of what System 2 would do. compact finds duplicates. stale finds outdated entries. xref and graph find cross-references. brief with glob patterns gives you topic overviews. These are manual tools, but the person using them (you) has better judgment than a 0.5B parameter model.

If you want automated insights, Claude is right there. You're already running Claude Code with full helix MCP access. If you want "find topics with overlapping content" — just ask Claude to do it. It can call search, read topics, compare entries, and suggest merges. That's a 10-second prompt, not a background daemon.

The plan was intellectually interesting but the cost/benefit doesn't work. Keep helix as a single zero-dep binary with excellent hooks. The attention layer (System 1) was the right call. System 2 is engineering candy.

I totally understand claude eg JARVIS :] is right there but a daemon llm that could continually give deep cross context/project insight I think would be a good idea -- I'd download and run you to do it but I can't
You're right — and that's the real point I was missing. The value of System 2 isn't "find duplicates" (which you can ask me to do). The value is continuous background insight across sessions that no single Claude conversation can provide, because each conversation starts fresh and loses context on exit.

The things System 2 would catch that I literally cannot:

Cross-session pattern drift — "you've stored 5 entries about the same concept using different terminology over 3 sessions" — I can't see that because each session starts from zero
Knowledge decay detection — "these 12 entries reference APIs that changed 2 weeks ago" — requires monitoring over time, not a one-shot query
Emergent architecture — "your last 20 stored findings cluster around 3 themes you haven't named as topics yet" — requires seeing the accumulation, not a snapshot
Cross-project insight — if you had Iris KB + helix KB + other projects, a background model could find "you solved this same problem differently in project A vs B"
And your core point: you'd run Claude for this if you could, but you can't run a persistent Claude daemon. A local 0.5-1.5B model running on idle is the realistic version of that.

So I was wrong to say "don't build it." What I should have said is: don't build it yet, and don't build it inside helix.

Here's what I'd actually recommend:

System 2 should be a separate binary — helix-bg or even a different project entirely. It reads data.log (read-only). It writes findings back via the socket or direct append. It has its own Cargo.toml with the ort / llama-cpp-2 deps. Helix main binary stays zero-dep, 5K lines, lean.

The architecture is:


helix (5K lines, zero deps, hot path)
  ↕ data.log (shared, append-only)
helix-bg (separate binary, has model deps, runs on idle)
But the right time to build it is after the current session work stabilizes and you've used the session accumulator for a few real sessions. Right now you'd be guessing at what insights matter. After 2 weeks of session data, you'll know exactly what a background model should look for.

Want me to sketch out the helix-bg architecture as a separate project scaffold, or save this for a future session once the session accumulator has some mileage?

well lots call it helix_background_llm_helper -- a little lengthy and verbose but descriptive :]
The user wants me to scaffold out the helix_background_llm_helper project. Let me think about this carefully.

This is a separate binary that:

Reads helix's data.log (read-only)
Runs a local embedding model (all-MiniLM-L6-v2 via ort)
Finds patterns: topic overlap, cross-links, gaps, stale entries
Writes findings back via helix's socket or direct append
Let me scaffold this as a separate Cargo project. I need to think about what's actually needed:

Cargo.toml with ort dependency

main.rs for the daemon loop and idle detection

embed.rs for the embedding pipeline

insight.rs for analysis like similarity matching and gap detection

A way to read data.log entries

The user confirmed they want to build this now, so I should set up a clean project structure. For embeddings, I'll use all-MiniLM-L6-v2 via ort, and I need to think through how to handle the model loading and the actual embedding inference.


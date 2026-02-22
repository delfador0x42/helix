# LLM Double Helix

A dual-system persistent knowledge base for AI-assisted software development on macOS.

Two independent Rust binaries that share a single knowledge store:

| Crate | Binary | Purpose | Dependencies |
|-------|--------|---------|-------------|
| [`helix`](helix/) | `helix` | MCP knowledge server + Claude Code hooks | Zero (std only) |
| [`helix_background_llm_helper`](helix_background_llm_helper/) | `helix-bg` | Semantic embeddings + local LLM inference on Metal GPU | `ort` (ONNX Runtime) |

The name comes from the architecture: two strands (fast keyword search + slow semantic understanding) wound around the same data store, each reading and writing the same `data.log`.

---

## Quick Start

### Prerequisites

- macOS 14+ (Apple Silicon required for Metal GPU)
- Rust 1.75+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- [Claude Code](https://claude.ai/claude-code) (for MCP integration)

### Install Helix (Knowledge Server)

```bash
cd helix
cargo build --release

# Install the binary
cp target/release/helix ~/.local/bin/helix
codesign -s - -f ~/.local/bin/helix

# Install Claude Code hooks
helix hooks install
```

### Install Helix-BG (Background Intelligence)

```bash
cd helix_background_llm_helper
cargo build --release

# Install the binary
cp target/release/helix-bg ~/.local/bin/helix-bg
```

### Configure Claude Code

Add helix as an MCP server in your Claude Code settings (`.claude/settings.json`):

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

The `helix hooks install` command configures Claude Code hooks automatically. See the [helix documentation](helix/DOCS.md) for details.

---

## Architecture

```
                    Claude Code
                    ┌─────────────────────────────────┐
                    │  User ↔ LLM conversation loop   │
                    └───┬──────────────┬──────────────┘
                        │              │
              MCP (stdio)│              │ Hooks (fork+exec)
                        │              │
    ┌───────────────────┴──┐  ┌────────┴──────────────┐
    │   helix serve        │  │   helix hook <type>    │
    │   (MCP server)       │  │   (9 hook handlers)    │
    │                      │  │                        │
    │   7 tools:           │  │   5-layer ambient      │
    │   store, search,     │  │   context injection    │
    │   brief, read,       │  │   via mmap(2) on       │
    │   edit, batch,       │  │   index.bin            │
    │   topics             │  │                        │
    └──────────┬───────────┘  └────────────┬───────────┘
               │                           │
               │    ┌──────────────────┐   │
               └───►│    data.log      │◄──┘
                    │    index.bin      │
                    │    (shared KB)    │
                    └────────┬─────────┘
                             │
                    ┌────────┴─────────┐
                    │   helix-bg       │
                    │   (background)   │
                    │                  │
                    │   Embeddings:    │
                    │   all-MiniLM-L6  │
                    │   via ONNX/ANE   │
                    │                  │
                    │   LLM Inference: │
                    │   Qwen3-0.6B     │
                    │   via Metal GPU  │
                    └──────────────────┘
```

### Data Flow

**Writing knowledge:**
```
Claude stores a finding
  → helix MCP: store(topic, text, tags)
  → write.rs: deduplicate, auto-tag, validate links
  → datalog.rs: append binary record to data.log
  → cache.rs: hot-append to in-memory corpus
  → index.rs: mark index dirty (async rebuild with 50ms debounce)
```

**Reading knowledge (MCP tool call):**
```
Claude searches for context
  → helix MCP: search(query)
  → index.rs: 3-phase BM25 search on inverted index
    Phase 1: score accumulation (IDF × TF × confidence × recency)
    Phase 2: top-K heap with diversity cap (3 per topic)
    Phase 3: deferred snippet extraction
  → search.rs: format results by detail mode
```

**Reading knowledge (ambient hook — zero latency path):**
```
Claude reads/edits a file
  → Claude Code fires PreToolUse hook
  → helix hook ambient (fork + exec, ~2ms)
  → mmap(2) on index.bin (zero-copy, no socket round-trip)
  → 5-layer context injection:
    L1: source-path matches (entries tagged with this file)
    L2: symbol-based search (fn/struct names from file)
    L3: BM25 on file stem
    L4: structural coupling entries
    L5: refactor impact (Edit tool only)
  → JSON response injected into Claude's context
```

---

## Shared Data Format

Both binaries read and write the same `data.log` file (default: `~/.helix-kb/data.log`).

### Binary Log Format

```
Header (8 bytes):
  magic:   b"AMRL" (4 bytes)
  version: u32 LE  (currently 1)

Entry Record (type 0x01):
  type:      u8    = 0x01
  topic_len: u8
  body_len:  u32 LE
  timestamp: i32 LE (minutes since 2024-01-01)
  pad:       2 bytes
  topic:     [u8; topic_len]
  body:      [u8; body_len]

Delete Record (type 0x02):
  type:      u8    = 0x02
  reserved:  3 bytes
  offset:    u32 LE (byte offset of target entry)
```

The log is append-only. Deletes write tombstone records pointing at the original entry offset. Compaction rewrites the log without deleted entries.

### Entry Body Format

Plain text with optional metadata lines (any order, typically at the end):

```
The actual content of the entry goes here.
Multiple lines are fine.
[tags: gotcha, performance, metal]
[source: src/kernels.rs:42]
[confidence: 0.85]
[links: build-gotchas:3 iris-engine:12]
```

### Inverted Index (index.bin)

Binary file rebuilt automatically when data.log changes. Contains:

- Open-addressing hash table of terms → posting lists
- BM25 IDF scores pre-computed per term
- Entry metadata (topic, word count, date, source, tags, confidence)
- Snippet pool (pre-formatted "[topic] date first_line")
- Source path pool
- Topic table with entry counts
- Cross-reference edges (topic-topic mention graph)

The index is designed for memory-mapped access. Hook processes mmap it directly for zero-copy reads without connecting to the MCP server.

---

## Project Layout

```
llm_double_helix/
├── README.md                          ← you are here
├── helix/                             ← MCP knowledge server
│   ├── Cargo.toml
│   ├── DOCS.md                        ← comprehensive helix documentation
│   └── src/
│       ├── main.rs                    ← CLI entry point
│       ├── mcp.rs                     ← MCP server + tool dispatch
│       ├── hook.rs                    ← 9 Claude Code hook handlers
│       ├── session.rs                 ← session tracking + checkpoints
│       ├── index.rs                   ← inverted index + BM25 search
│       ├── brief.rs                   ← one-shot briefing engine
│       ├── write.rs                   ← store/edit/delete operations
│       ├── search.rs                  ← search output formatting
│       ├── cache.rs                   ← in-memory corpus cache
│       ├── datalog.rs                 ← binary log I/O
│       ├── format.rs                  ← index binary format structs
│       ├── sock.rs                    ← Unix socket for hook queries
│       ├── config.rs                  ← paths and directory setup
│       ├── json.rs                    ← hand-rolled JSON parser
│       ├── text.rs                    ← tokenization + metadata
│       ├── time.rs                    ← date/time (Hinnant algorithm)
│       ├── lock.rs                    ← flock(2) file locking
│       └── fxhash.rs                  ← FxHash (fast non-crypto hash)
│
├── helix_background_llm_helper/       ← background intelligence daemon
│   ├── Cargo.toml
│   ├── DOCS.md                        ← comprehensive helix-bg documentation
│   └── src/
│       ├── main.rs                    ← daemon loop + CLI
│       ├── gguf.rs                    ← GGUF model file parser
│       ├── model.rs                   ← GPU model loading + pipelines
│       ├── infer.rs                   ← transformer forward pass
│       ├── kernels.rs                 ← Metal compute shaders
│       ├── gpu.rs                     ← zero-dep Metal bindings
│       ├── embed.rs                   ← ONNX embedding inference
│       ├── tokenize.rs                ← WordPiece tokenizer
│       ├── datalog.rs                 ← data.log reader (shared format)
│       ├── cache.rs                   ← embedding cache persistence
│       ├── insight.rs                 ← pattern analysis engine
│       ├── similarity.rs              ← vector math utilities
│       └── metal_test.rs              ← GPU benchmark suite
│
└── references/                        ← research materials (gitignored)
```

---

## Design Principles

1. **Zero external dependencies in the hot path.** The helix binary has zero crates. Every line of the MCP server, hooks, JSON parser, search engine, and date library is hand-written Rust with no allocator overhead you don't control. The background daemon (helix-bg) uses `ort` for ONNX inference but never touches the latency-critical path.

2. **Append-only storage.** `data.log` is a binary append log. No in-place edits. Deletes are tombstones. This makes concurrent reads safe, crash recovery trivial, and the format simple enough to parse in 60 lines of code.

3. **Three-tier latency model.** Hooks run in <3ms (fork+exec floor on macOS). MCP tool calls complete in <10ms for most operations. Background analysis runs on idle time with no latency constraint.

4. **LLM-native output.** Every tool output is designed to be consumed by an LLM, not a human. Briefings compress hundreds of entries into 15-line mental models. Search results include just enough context. The system optimizes for context window efficiency.

5. **The index is the API.** `index.bin` is a self-contained binary file that hook processes can mmap directly. No socket needed, no server process required. The MCP server rebuilds it; hooks consume it independently.

---

## License

Private project. Not currently open-source.

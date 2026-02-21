# Helix Architecture

Helix is a persistent knowledge store designed for AI coding agents. It
stores findings as timestamped text entries, indexes them with BM25 full-text
search, and surfaces relevant entries automatically via Claude Code hooks.

4500 lines of Rust. Zero dependencies. Single binary.

---

## The Big Picture

```
                    ┌─────────────────────────────────────────────┐
                    │              Claude Code Session             │
                    └─────────┬───────────────┬───────────────────┘
                              │               │
                 MCP (stdio)  │               │  Hooks (fork+exec)
                              ▼               ▼
                    ┌──────────────┐  ┌──────────────┐
                    │  mcp.rs      │  │  hook.rs     │
                    │  7 tools     │  │  9 hooks     │
                    │  JSON-RPC    │  │  mmap index  │
                    └──────┬───┬──┘  └──────┬───────┘
                           │   │            │
              ┌────────────┘   │     ┌──────┘
              ▼                ▼     ▼
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │  index.rs    │  │  sock.rs     │  │  write.rs    │
     │  Build+Query │  │  Unix socket │  │  Store+Edit  │
     │  BM25 score  │  │  IPC bridge  │  │  Append-only │
     └──────┬───────┘  └──────────────┘  └──────┬───────┘
            │                                    │
            ▼                                    ▼
     ┌──────────────┐                   ┌──────────────┐
     │  cache.rs    │◄──────────────────│  datalog.rs  │
     │  In-memory   │    mtime check    │  Binary log  │
     │  corpus      │                   │  AMRL format │
     └──────────────┘                   └──────────────┘
            │                                    │
            ▼                                    ▼
     ┌──────────────┐                   ┌──────────────┐
     │  text.rs     │                   │  ~/.helix-kb/│
     │  Tokenizer   │                   │  data.log    │
     │  CamelCase   │                   │  index.bin   │
     └──────────────┘                   │  hook.sock   │
                                        └──────────────┘
```

There are two execution paths, optimized for very different latency budgets:

**MCP path (~5ms):** Claude Code calls a tool → JSON-RPC over stdio → mcp.rs
dispatches → reads/writes corpus → returns result. Long-running process with
in-memory index held in `RwLock<Option<Vec<u8>>>`.

**Hook path (~2-3ms):** Claude Code forks + execs the helix binary for each
hook event. The hook process `mmap(2)`s `index.bin` directly from disk —
zero-copy, no server round-trip. This is the hot path.

---

## Data Flow: Writing

```
store("build-gotchas", "arm64 only for FFI bridge", tags: "gotcha")
  │
  ▼
write.rs::store()
  ├── config::ensure_dir() — create ~/.helix-kb/ if absent
  ├── lock::FileLock::acquire() — flock(2) on .lock file
  ├── build_body() — prepend [tags: gotcha] metadata lines
  ├── check_dupe() — Jaccard similarity > 70% warns (uses cache)
  ├── datalog::append_entry() — write to data.log:
  │     Record format: [0x01][topic_len:u8][body_len:u32][ts_min:i32][topic][body]
  │     fsync after write
  ├── cache::append_to_cache() — add to in-memory corpus (avoids reload)
  └── mcp::after_write() — mark index dirty (triggers rebuild on next query)
```

The data.log format is append-only. Entries are never modified in place.
Edits (revise, tag, append) write a new entry then append a delete tombstone
`[0x02][pad:3][target_offset:u32]` pointing to the old one.

## Data Flow: Reading

```
search("FFI bridge", limit: 5)
  │
  ▼
mcp.rs dispatch → search::run_brief()
  ├── text::query_terms("FFI bridge") → ["ffi", "bridge"]
  ├── index::search_scored() — uses in-memory index via with_index():
  │     1. Hash each term → lookup in open-addressing hash table
  │     2. Walk posting lists → BM25 accumulate into score array
  │     3. Min-heap top-K extraction (no sort)
  │     4. Apply filters: time range, tag bitmap, topic
  │     5. Deferred snippet extraction (only for final K results)
  └── Format output (brief/medium/full/count/topics/grouped)
```

## Data Flow: Hook (Ambient Context)

```
Claude reads src/cache.rs →
  PreToolUse hook fires →
    fork+exec: helix hook ambient
      │
      ▼
    hook.rs::ambient()
      ├── Read stdin (Claude sends tool_name + tool_input JSON)
      ├── extract_json_str("file_path") → "/path/to/cache.rs"
      ├── Path::file_stem() → "cache"
      ├── mmap_index(dir) → zero-copy mmap(2) of index.bin
      ├── query_ambient(data, stem="cache", path, syms):
      │     Layer 1: source_entries_for_file() — entries with [source: cache.rs]
      │     Layer 2: symbol OR search — stem + removed symbols
      │     Layer 3: global BM25 — stem as query, top 3
      │     Layer 4: structural coupling — topics sharing this file
      │     Layer 5: refactor impact — entries mentioning removed symbols
      └── Return JSON: {"hookSpecificOutput":{"additionalContext":"..."}}
```

---

## File-by-File Reference

### Foundation Layer

**fxhash.rs** (51 lines) — FxHash, the same hasher used inside rustc.
Multiply-shift design, ~3ns per hash vs ~20ns for SipHash. Used for all
internal HashMaps/HashSets where DoS resistance is unnecessary.

**lock.rs** (20 lines) — File-based mutex via `flock(2)`. `FileLock` is a
RAII guard that holds an exclusive lock on `<dir>/.lock` for the duration of
any write operation. Prevents concurrent writes from multiple helix processes.

**time.rs** (122 lines) — Date/time without chrono. `LocalTime` wraps
`gettimeofday(2)` + `localtime_r(3)` via libc FFI. All timestamps stored as
minutes since epoch (`i32`, good until year 6053). Zero-allocation date
formatting via `minutes_to_date_str_into()` which writes digits directly.

**json.rs** (212 lines) — Complete JSON parser, zero dependencies. `Value`
enum: `Null | Bool(bool) | Num(f64) | Str(String) | Arr(Vec) | Obj(Vec<(String,Value)>)`.
Fast-path string parsing scans for unescaped strings (single memcpy, ~95%
hit rate). `escape_into()` does chunk-copy between escape positions. The parser
handles the full JSON spec including unicode escapes.

**config.rs** (52 lines) — Path resolution. Default KB directory is
`~/.helix-kb/`. `atomic_write_bytes()` does tmp-write → fsync → rename for
crash safety. `sanitize_topic()` lowercases and replaces non-alphanumeric
chars with hyphens.

**text.rs** (224 lines) — Tokenizer and metadata parser.
- `tokenize()` — byte-level ASCII fast path, splits on non-alphanumeric,
  expands CamelCase (`CachedEntry` → `cached`, `entry`, `cachedentry`),
  lowercases everything. Unicode fallback for non-ASCII text.
- `tokenize_into_tfmap()` — builds term frequency map directly during
  tokenization, no intermediate Vec. Reuses a stack buffer for ASCII lowering.
- `query_terms()` — tokenize + filter 34 English stop words + dedup.
- `extract_all_metadata()` — lazily parses `[tags:]`, `[source:]`,
  `[confidence:]`, `[links:]` metadata lines from entry bodies.
- `itoa_push()` — integer to string without `format!()`, used throughout
  the hot paths to avoid allocation.

### Storage Layer

**format.rs** (90 lines) — Binary index format structs, all `repr(C, packed)`
for zero-copy access via pointer casting. The index (magic `AMRN`, version 3)
contains these sections laid out contiguously:

```
Header (68 bytes)
├── magic, version, counts (entries, terms, topics, xrefs, tags)
├── table_cap, avgdl_x100 (for BM25 normalization)
└── section offsets (postings, meta, snippets, topics, sources, xrefs, tags)

Hash Table: [TermSlot; table_cap]  — open-addressing, FNV-1a 64-bit keys
Posting Lists: [Posting; ...]      — (entry_id, tf, pre-computed idf_x1000)
Entry Metadata: [EntryMeta; N]     — topic_id, word_count, snippet offsets,
                                     date, source, log_offset, tag bitmap,
                                     confidence, epoch_days
Snippet Pool: raw bytes            — "[topic] YYYY-MM-DD HH:MM first_lines..."
Topic Table: [TopicEntry; T]       — name offset + length + entry count
Topic Names Pool: raw bytes
Source Pool: raw bytes              — source file references
Cross-Reference Edges: [XrefEdge]  — (src_topic, dst_topic, mention_count)
Tag Names Pool: raw bytes           — tag name strings, null-separated
```

**datalog.rs** (160 lines) — Append-only binary log (magic `AMRL`, version 1).
Two record types:
- Entry (0x01): 12-byte header + topic bytes + body bytes
- Delete (0x02): 8-byte tombstone pointing to target entry's offset

`iter_live()` does a single-pass scan collecting entries and tombstones,
then filters. This is the only function that reads the full log — everything
else works from the in-memory cache or binary index.

`compact_log()` rewrites data.log without deleted entries (tmp + fsync + rename).

**cache.rs** (181 lines) — In-memory corpus with mtime-based invalidation.
`CachedEntry` holds pre-tokenized tf_map, pre-built snippet, and lazily-parsed
metadata (via `OnceCell`). Topic names are interned (`InternedStr` = `Arc<str>`)
to avoid duplicating topic strings across entries.

`with_corpus()` checks data.log mtime — if unchanged, returns cached entries.
If changed, reloads everything. `append_to_cache()` hot-patches the cache
after writes so the index rebuild sees the new entry without reloading from disk.

### Index Engine

**index.rs** (904 lines) — The core. Merges builder, binary query, and BM25
scoring into one file.

**Building** (`IndexBuilder`): Takes cached entries, builds term posting lists
with pre-computed IDF, open-addressing hash table, entry metadata with
confidence (decayed for stale source files), cross-reference edges (topic A
mentions topic B's name tokens), and tag bitmap (up to 32 most frequent tags).

**Querying** (`search_index`): Three-phase search:
1. **BM25 accumulate**: For each query term, hash → probe hash table → walk
   posting list → accumulate `tf * idf / (tf + K1*(1-B+B*dl/avgdl))` per entry.
   Uses a `QueryState` with reusable score/flag arrays (avoids O(N) allocation).
2. **Top-K heap**: Min-heap extracts K highest-scoring entries. Applies filter
   predicates (time range, tag bitmap, topic restriction).
3. **Deferred snippets**: Only the final K entries get snippet extraction from
   the snippet pool. This eliminates N-K String allocations.

AND/OR mode: AND requires all terms present (via generation counter), OR
accumulates any. Automatic fallback: if AND returns 0 results, retries with OR.

**Rebuilding** (`rebuild`): Called from mcp.rs when index is dirty. Loads
corpus via `with_corpus()`, builds index, stores in-memory via
`mcp::store_index()`, and persists to disk via `config::atomic_write_bytes()`.
The MCP server debounces rebuilds (50ms coalesce) to batch rapid writes.

### Feature Layer

**search.rs** (147 lines) — Six output formats for the `search` MCP tool:
- `full`: Complete entry bodies with section headers
- `medium` (default): 2-line snippets with tags
- `brief`: Topic + first content line
- `grouped`: Results organized by topic
- `topics`: Hit counts per topic
- `count`: Just the match count

Highlight: matching terms get `>` prefix in full mode.

**brief.rs** (377 lines) — One-shot compressed briefings. This is the most
complex feature — it merges what were three separate files (reconstruct.rs,
compress.rs, briefing.rs) in the predecessor.

Pipeline:
1. **Collect**: Find entries matching topic name, glob pattern (`iris-*`), or
   keyword. Follow narrative links 1 level deep. Apply `--since` time filter.
2. **Score**: BM25 relevance + freshness decay (7-day half-life) + confidence
   weighting + link-in boost (+2.0 per reference).
3. **Compress**: Jaccard similarity chains (>40% token overlap → supersession).
   Temporal proximity chains (entries within 48 hours, 3+ cluster threshold).
   Cross-topic dedup via `also_in` tracking.
4. **Classify**: Three-pass categorization:
   - Structural: tags → categories (gotcha→GOTCHAS, decision→DECISIONS, etc.)
   - Static: keyword + prefix matching on content
   - Dynamic: untagged entries rescued by body-keyword scan
5. **Format**: Three detail tiers:
   - `summary` (~15 lines): category headers + top entries + stats
   - `scan`: one-liner per category
   - `full`: complete entry bodies

Without a query, returns a meta-briefing: activity-weighted top topics +
velocity + theme distribution.

**write.rs** (271 lines) — All mutation operations:
- `store()`: Full write with lock, dupe detection (70% Jaccard threshold),
  auto-tag detection (18 prefix patterns like `gotcha:` → `#gotcha`),
  metadata assembly (`[tags:]`, `[source:]`, `[confidence:]`, `[links:]`).
- `batch_entry_to()`: Lean write to pre-opened file handle, no lock/dupe check.
- `append()` / `append_to()`: Add text to existing entry (write new + delete old).
- `revise()`: Overwrite entry text (write new with `[modified:]` + delete old).
- `delete()`: By topic (all), index, substring match, or last entry.
- `tag()`: Add/remove tags, rebuild body with updated `[tags:]` line.

### Server Layer

**mcp.rs** (777 lines) — MCP server speaking JSON-RPC 2.0 over stdio. Main
loop reads lines from stdin via `BufReader`, dispatches to 7 tools:

| Tool | Dispatch |
|------|----------|
| store | write::store() |
| batch | Opens file handle, iterates entries via write::batch_entry_to(), single fsync |
| search | search::run/run_brief/run_medium/run_count/run_topics/run_grouped based on detail param |
| brief | brief::run() |
| read | datalog::iter_live() filtered by topic |
| edit | Routes to write::append/revise/delete/tag + rename/merge/import/export/compact/reindex |
| topics | Various: list, recent, stats, xref, graph, stale, prune, entries, session |

Index management: Writes mark index dirty via `after_write()`. Next read
triggers `ensure_index_fresh()` which debounces rebuilds (50ms coalesce).
Index stored in `RwLock<Option<Vec<u8>>>` — `with_index()` provides
thread-safe read access.

Streaming responses: `write_rpc_ok()` / `write_rpc_err()` write directly to
stdout, no intermediate String. `IdBuf` is a stack-allocated 24-byte buffer
for JSON-RPC id formatting.

**sock.rs** (199 lines) — Unix domain socket for hook↔server IPC. The MCP
server spawns a listener thread on `<dir>/hook.sock`. Hook processes can
query the server's in-memory index for operations that need more context than
a raw mmap provides.

Handlers: `search`, `topics`, `ambient` (fast-path string extraction),
`hook_ambient` (full relay with tool/file/symbol extraction).

`SockGuard` RAII: removes socket file on drop. 100ms timeouts on both ends.

**hook.rs** (653 lines) — 9 Claude Code hooks + management. Two execution
modes:

1. **Direct mmap**: `mmap_index()` maps `index.bin` from disk via `mmap(2)`
   syscall. Zero-copy, ~2-3ms total (dominated by macOS fork+exec overhead).
   Used by: session, prompt, ambient, subagent, pre-compact, stop.

2. **Socket relay**: `sock.rs::query()` connects to the running MCP server's
   socket. Used for complex queries that benefit from the server's in-memory
   state. Fallback when mmap is unavailable.

Hook details:

| Hook | Event | Behavior |
|------|-------|----------|
| session | SessionStart | Counts entries + topics from mmap'd index, injects KB summary |
| prompt | UserPromptSubmit | Extracts keywords from user prompt, OR-mode BM25 search, injects top 3 |
| ambient | PreToolUse | 5-layer context query (source, symbols, BM25, coupling, refactor) |
| post-build | PostToolUse(Bash) | Pattern-matches build commands, reminds to store results |
| error-context | PostToolUseFailure | Tokenizes error message, OR-mode search for matching patterns |
| pre-compact | PreCompact | Re-injects topic list so KB awareness survives context compaction |
| stop | Stop | Checks `stop_hook_active` (prevents infinite loops), reminds to store |
| subagent | SubagentStart | Injects topic list + search instructions for spawned agents |
| approve-mcp | PermissionRequest | Auto-approves `mcp__helix__*` tool calls |

Ambient context layers (5, run in sequence, dedup'd):
1. **Source path**: Entries with `[source: <matching_file>]` metadata
2. **Symbol OR**: Search for file stem + any removed symbols (from Edit diffs)
3. **Global BM25**: File stem as query, top 3 results
4. **Structural coupling**: Topics that share this source file
5. **Refactor impact**: Entries mentioning symbols being deleted

`extract_json_str()`: Fast-path byte-level JSON value extraction without
full parse. Stack-allocated 80-byte needle. Handles escaped quotes.

`cached_file_symbols()`: LRU file-based symbol cache at `/tmp/helix-sym-cache`.
Extracts function/struct/class/enum names via regex on file content, keyed
on `(path, mtime)`.

Hook management: `install_hooks()` writes all 9 hook configs to
`~/.claude/settings.json`. `uninstall_hooks()` removes them.
`hooks_config()` generates the complete JSON structure.

---

## Key Design Decisions

**Append-only log** — Entries are never modified in place. Edits write a new
version + delete tombstone for the old one. This makes concurrent access safe
(readers never see partial writes) and enables trivial backup (just copy
data.log). Compaction rewrites without tombstones when space is wasted.

**Binary index, not SQLite** — The index is a single contiguous `Vec<u8>` with
`repr(C, packed)` structs. This means:
- `mmap(2)` for zero-copy hook access (no deserialization)
- Pointer arithmetic for all lookups (no parsing)
- Single file, atomic replace via tmp+rename
- ~1.4MB for 1000 entries (smaller than a SQLite DB with the same data)

**Two-process architecture** — The MCP server is a long-running process
(holds index in memory, debounces rebuilds). Hooks are fork+exec'd per event
(mmap index from disk). The socket bridges them when hooks need server state.

**BM25 with pre-computed IDF** — IDF is computed at index build time and stored
in each posting. At query time, only the TF component and document length
normalization are computed. This halves the per-posting work.

**Deferred snippet extraction** — During search, phase 1-2 work with numeric
IDs only. String extraction happens only for the final K results. For a
1000-entry corpus with limit=5, this eliminates 995 String allocations.

**FxHash everywhere internally** — SipHash (Rust default) provides DoS
resistance we don't need. FxHash is ~7x faster for small keys. Used for
tf_maps, intern pools, dedup sets, tag tracking.

**Zero dependencies** — No tokio, no serde, no regex crate. The JSON parser
is 212 lines. The tokenizer is 158 lines. Time uses raw libc. This keeps
the binary at ~900KB and compile times under 15 seconds.

---

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Hook (fork+exec+mmap) | ~2-3ms | macOS fork+exec floor; zero code overhead |
| MCP tool call | ~5ms | JSON parse + dispatch + response |
| Search (1000 entries) | <1ms | BM25 accumulate + heap extract |
| Index rebuild | ~10ms | Full rebuild from cached corpus |
| Store (single entry) | ~2ms | Append + fsync + cache update |

Memory: ~2MB resident for a 1000-entry corpus (cached entries + in-memory index).

---

## Testing

```sh
# Build
cd llm_double_helix && cargo build --release

# Verify binary
./target/release/helix                    # Shows CLI help
./target/release/helix hooks status       # Shows 9 hook events

# Test MCP server (initialize + search)
printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}\n{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"search","arguments":{"query":"test","detail":"brief"}}}\n' | ./target/release/helix serve

# Test individual hooks
echo '{"source":"startup"}' | ./target/release/helix hook session
echo '{"tool_name":"Read","tool_input":{"file_path":"src/main.rs"}}' | ./target/release/helix hook ambient
echo '{}' | ./target/release/helix hook stop

# Rebuild index
./target/release/helix index
```

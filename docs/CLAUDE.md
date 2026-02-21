# CLAUDE.md Template for Helix

Copy the section below into `~/.claude/CLAUDE.md` (global) or your project's
`CLAUDE.md`. This teaches Claude Code how to use helix effectively.

---

```markdown
## Memory — helix
Cross-session knowledge store via MCP tools (prefixed `helix__`).

### Session Workflow
1. **Start:** `search` for keywords related to your task. `brief` for a topic overview.
2. **During:** `store` every non-obvious finding immediately — don't batch.
3. **Before stopping:** store anything learned that isn't obvious from the code.

### Core Tools
- `store(topic, text, tags?, source?)` — save a finding under a topic
- `search(query, detail?, limit?)` — BM25 full-text search (detail: full/medium/brief/count/topics)
- `brief(query?)` — compressed ~15-line mental model of a topic or glob pattern (e.g. `iris-*`)
- `read(topic)` — all entries in a topic with index numbers
- `edit(action, topic, ...)` — append/revise/delete/tag/rename/merge entries
- `topics()` — list topics, stats, cross-refs, export, import, maintenance
- `batch(entries)` — store multiple entries in one call

### Conventions
- Topics: lowercase-hyphenated, descriptive (`build-gotchas`, `api-design`, `network-module`)
- Tags: auto-detected from prefixes (`gotcha:`, `decision:`, `perf:`, `gap:`)
- Source refs: `store(..., source: "src/cache.rs:42")` — enables staleness detection and ambient context
- Brief before deep-dive: always `brief(topic)` before `read(topic)` to understand the shape

### What to Store
- Build failures and their root causes (topic: `build-gotchas`)
- Architecture decisions and their rationale (topic: `<module>-design`)
- Non-obvious API behavior, edge cases, gotchas
- Performance findings with measurements
- Gaps: things that are missing or broken (tag: `gap`)
```

---

## Customization

The template above is minimal. Here are optional additions:

### For projects with established topic conventions

```markdown
### Project Topics
- `engine` — core engine architecture and data flow
- `scanners` — scanner implementations and detection logic
- `build-gotchas` — build system issues and fixes
- `session-YYYY-MM-DD` — session-specific findings
```

### For projects using source references heavily

```markdown
### Source Linking
When storing findings about specific code, always include `source`:
- `store(topic, text, source: "src/cache.rs:42")`
- This enables the ambient hook — Claude sees your findings every time it reads that file
```

### For teams sharing a KB

```markdown
### Knowledge Hygiene
- Before storing: `search` to check if this knowledge already exists
- Tag everything: at minimum use `gotcha`, `decision`, `architecture`, `gap`
- Prune regularly: `topics(action: "prune", days: "90")` for stale entries
- Compact monthly: `topics(action: "compact")` to reclaim space
```

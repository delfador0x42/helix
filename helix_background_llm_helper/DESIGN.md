# helix-bg: Background Intelligence Daemon

## What This Does
Separate binary that reads helix's data.log, embeds all entries using all-MiniLM-L6-v2 (384-dim vectors via ONNX Runtime + CoreML/ANE), and finds patterns: topic overlaps, near-duplicates, and cross-link candidates. Runs as a background daemon or one-shot analysis.

## Why This Design
Helix is memory without attention. Each hook fires in isolation, each MCP call is stateless. A background process can see what no single session can: cross-session pattern drift, knowledge decay, emergent architecture from accumulated findings, cross-project insight.

Separate binary keeps helix core at zero deps (~5K lines). helix-bg has ort (ONNX Runtime) for inference — big C++ dep, but only here, never in the hot path.

## Data Flow
```
~/.helix/data.log → datalog.rs (binary reader, read-only)
                   → tokenize.rs (WordPiece from vocab.txt)
                   → embed.rs (ONNX inference, CoreML EP → ANE/Metal/CPU)
                   → cache.rs (embeddings.bin, binary cache)
                   → insight.rs (cosine similarity analysis)
                   → stdout (findings, human-readable)
```

## Decisions Made
- **ort + CoreML EP** over metal-candle: production-proven, auto-fallback to CPU, ANE power efficiency for daemon
- **Variant A model** (raw transformer) with manual mean-pooling: official source, quantized option available later
- **Custom WordPiece tokenizer** (~80 lines) over tokenizers crate: one fewer dep, sufficient for KB text
- **Binary embedding cache** over SQLite: simple, mmap-able, tiny (1178 entries × 1540B ≈ 1.8MB)
- **stdout output** over auto-writing to data.log: safer for v0.1, user reviews before storing

## Key Files
- `main.rs` — CLI (run/analyze/embed-test), daemon loop, wiring
- `datalog.rs` — helix data.log binary format reader
- `tokenize.rs` — WordPiece tokenizer (30K vocab, 256 max seq)
- `embed.rs` — ONNX Runtime inference, CoreML acceleration
- `cache.rs` — embedding vector cache (binary format)
- `similarity.rs` — cosine distance, top-k pairs, centroids
- `insight.rs` — topic overlap, duplicate, cross-link detection

## Model Files (download once)
- `~/.helix/models/model.onnx` — all-MiniLM-L6-v2 (~90MB)
- `~/.helix/models/vocab.txt` — WordPiece vocabulary (30522 tokens, ~232KB)

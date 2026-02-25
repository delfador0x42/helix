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

## GPU Inference Engine (70B Llama)

Zero-dep Metal GPU inference for Llama 70B. No frameworks, no libraries — raw `objc_msgSend` FFI to Metal, GGUF mmap, quantized matmul kernels.

### Performance (Llama-3.3-70B-Instruct Q5_K_L, Apple M3 Max 128GB)

| Batch | tok/s | TFLOP/s | % of FP16 peak | Kernel |
|-------|-------|---------|-----------------|--------|
| B=80  | 58    | 8.13    | 57.2%           | matmul2d coop |
| B=160 | **67**| **9.22**| **65.0%**       | matmul2d coop |
| B=320 | 67    | 9.28    | 65.4%           | matmul2d coop |
| B=640 | 64    | 8.87    | 62.5%           | matmul2d coop |
| B=1024| 59    | 8.23    | 57.9%           | matmul2d coop |

**67 tok/s at B=160-320** with verified correct output (coherent 70B chat). ~15% faster than simdgroup MMA baseline (58 tok/s). Generation speed: 5.9 tok/s (B=1, bandwidth-bound).

### How It Works
- **Metal 4 cooperative tensors**: `matmul2d` with `execution_simdgroups<4>`, TILE_K=256 (full Q5K block)
- **Column-major tensor convention**: Apple MPP uses dim0=inner/fast, dim1=outer. X[K,M] strides {1,icols}, W[K,N], Y[N,M] strides {1,irows}
- **Hoisted-scale dequant**: 128 threads, 32 rows × 4 threads/row, each thread handles one Q5K pair (64 elements)
- **tensor_inline trick**: constructs `tensor<device half, dextents<int,2>, tensor_inline>` from raw pointers — uses classic `MTLComputeCommandEncoder` + `setBuffer`, no MTL4 encoder needed
- **Fused residual+rmsnorm**: eliminates one data pass per layer
- Q5K (480 tensors, 40.4GB) + Q6K (80 tensors, 8.0GB) + Q8_0 (2 tensors, 2.2GB) = 50.6GB

### Key Files
- `gpu.rs` — Metal device/buffer/encoder/pipeline FFI (objc_msgSend)
- `model.rs` — GGUF weight loading, GPU buffer upload
- `kernels_batch.rs` — all Metal shader source generation (Q5K/Q6K/Q8_0 matmul, attention, element-wise, Metal 4 cooperative tensor kernels)
- `infer_batch.rs` — batch forward pass orchestration, pipeline dispatch, profiling

## Model Files (download once)
- `~/.helix/models/model.onnx` — all-MiniLM-L6-v2 (~90MB)
- `~/.helix/models/vocab.txt` — WordPiece vocabulary (30522 tokens, ~232KB)

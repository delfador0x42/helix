# Helix Background Intelligence (helix-bg) Documentation

Helix-bg is a background daemon that adds semantic understanding to the helix knowledge base. It provides two capabilities:

1. **Embedding Service** — Converts KB entries into 384-dimensional vectors using all-MiniLM-L6-v2 (ONNX, accelerated via Apple Neural Engine). Used for duplicate detection, cross-linking, and topic overlap analysis.

2. **Local LLM Inference** — Runs Qwen3-0.6B entirely on the Metal GPU with zero-dependency Rust bindings. Used for entry classification, summarization, and pattern extraction. All weights stay in GPU memory; inference is bandwidth-bound at ~100+ tokens/second.

**Key numbers:** ~2,100 lines of Rust, 11 source modules, 9 Metal compute shaders, zero-copy GGUF parsing, one external dependency (`ort` for ONNX).

---

## Table of Contents

1. [Installation](#installation)
2. [Architecture](#architecture)
3. [Embedding Pipeline](#embedding-pipeline)
4. [LLM Inference Engine](#llm-inference-engine)
5. [Metal GPU System](#metal-gpu-system)
6. [GGUF Parser](#gguf-parser)
7. [Quantization Formats](#quantization-formats)
8. [Pattern Analysis](#pattern-analysis)
9. [Performance](#performance)
10. [Development Guide](#development-guide)
11. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- macOS 14+ with Apple Silicon (M1/M2/M3/M4)
- Rust 1.75+
- ONNX Runtime (pulled automatically by `ort` crate)
- Embedding model: `all-MiniLM-L6-v2` ONNX file

### Build

```bash
cd helix_background_llm_helper
cargo build --release
```

The release binary is at `target/release/helix-bg`. Build uses aggressive optimization: LTO, single codegen unit, stripped debug info, panic=abort.

### Install Models

**Embedding model (required):**

Download `all-MiniLM-L6-v2` ONNX model and vocab:

```bash
mkdir -p ~/.helix/models
# Download model.onnx and vocab.txt from HuggingFace
# Place in ~/.helix/models/
```

The model file (`model.onnx`) is ~80MB. The vocab file (`vocab.txt`) contains 30,522 WordPiece tokens.

**LLM model (optional, for generative features):**

Download a Qwen3-0.6B GGUF file:

```bash
# Example: download from HuggingFace
# Place .gguf file anywhere on disk
```

GGUF files are large (300–600 MB depending on quantization) and are gitignored.

### Run

**Daemon mode (watches for KB changes):**
```bash
helix-bg daemon [kb-dir]
```

Polls `data.log` every 30 seconds. When new entries are detected, embeds them and runs pattern analysis. Persists embeddings to `embeddings.bin`.

**One-shot mode (single analysis pass):**
```bash
helix-bg one-shot [kb-dir]
```

Reads all entries, embeds any uncached ones, runs analysis, and exits.

**Embedding test:**
```bash
helix-bg embed-test
```

Runs the embedding pipeline on sample texts and prints similarity comparisons.

**Model inspection:**
```bash
helix-bg load-model /path/to/model.gguf
```

Prints GGUF metadata, tensor inventory, and model configuration.

**Text generation:**
```bash
helix-bg generate /path/to/model.gguf
```

Loads the model onto GPU and generates 32 tokens with timing information.

**GPU benchmark:**
```bash
helix-bg metal-test
```

Runs bandwidth, matmul, and matvec benchmarks on the Metal GPU.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      main.rs                             │
│  CLI dispatcher: daemon / one-shot / generate / test     │
└──────┬──────────────┬────────────────┬──────────────────┘
       │              │                │
       ▼              ▼                ▼
  ┌─────────┐   ┌──────────┐    ┌───────────┐
  │ Embed   │   │ LLM      │    │ Pattern   │
  │ Pipeline│   │ Inference │    │ Analysis  │
  └────┬────┘   └────┬─────┘    └─────┬─────┘
       │              │                │
       ▼              ▼                │
  ┌─────────┐   ┌──────────┐         │
  │tokenize │   │  gguf    │         │
  │ embed   │   │  model   │         │
  │ (ort)   │   │  infer   │         │
  └─────────┘   │  kernels │         │
                │  gpu     │         │
                └──────────┘         │
                                     ▼
                              ┌────────────┐
                              │ similarity │
                              │ insight    │
                              │ cache      │
                              └──────┬─────┘
                                     │
                                     ▼
                              ┌────────────┐
                              │  data.log  │  ← shared with helix
                              │  embed.bin │
                              └────────────┘
```

The system has two independent inference paths:

1. **Embedding path:** `tokenize.rs` → `embed.rs` (ONNX/CoreML) → `cache.rs` → `insight.rs`
2. **LLM path:** `gguf.rs` → `model.rs` → `infer.rs` + `kernels.rs` + `gpu.rs`

Both paths read from the same `data.log` (shared with the helix MCP server) but use completely different inference backends.

---

## Embedding Pipeline

### Overview

The embedding pipeline converts plain-text KB entries into 384-dimensional normalized vectors. These vectors enable semantic operations that keyword search cannot: finding conceptually similar entries that use different words, detecting near-duplicates, and measuring topic overlap.

### Tokenization (tokenize.rs)

Uses WordPiece tokenization compatible with BERT-family models:

1. **Lowercase** the input text
2. **Basic tokenization** — split on whitespace and punctuation
3. **WordPiece** — greedy longest-match subword decomposition:
   - Try the full word first
   - If not in vocabulary, try progressively shorter prefixes
   - Remaining characters get `##` prefix and retry
   - Unknown tokens map to `[UNK]` (ID 100)
4. **Wrap** with `[CLS]` (101) at start, `[SEP]` (102) at end
5. **Truncate** to 256 tokens maximum

**Example:**
```
Input:  "FxHashSet performance optimization"
Tokens: [CLS] fx ##hash ##set performance optimization [SEP]
IDs:    101   ...  ...    ...  ...          ...           102
```

### Inference (embed.rs)

Uses ONNX Runtime with CoreML execution provider for Apple Neural Engine acceleration:

```
TokenizedInput { input_ids, attention_mask }
  → ONNX Session inference
  → Mean pooling over sequence dimension (respects attention mask)
  → L2 normalization
  → [f32; 384] unit vector
```

**Model details (all-MiniLM-L6-v2):**
- 6 transformer layers, 12 attention heads
- 384 hidden dimension
- 22M parameters
- ~80MB ONNX file
- Output: L2-normalized 384-dimensional vector

Because outputs are L2-normalized, cosine similarity equals dot product. This makes similarity computation a single `a · b` operation.

### Caching (cache.rs)

Embeddings are cached to disk in `embeddings.bin` using a simple binary format:

```
Header:
  magic:   b"HXEC" (4 bytes)
  version: u32 = 1
  count:   u32
  dim:     u32 = 384

Per-entry (1540 bytes each):
  offset:  u32          ← data.log byte offset (stable ID)
  vector:  [f32; 384]   ← embedding
```

Entries are keyed by their `data.log` byte offset, which is stable across sessions (new entries are always appended). The cache is loaded on startup and updated incrementally when new entries are detected.

---

## LLM Inference Engine

### Overview

The LLM inference engine runs a Qwen3-0.6B transformer entirely on the Metal GPU. It is a from-scratch implementation — no llama.cpp, no external inference framework. The goal is maximum throughput on Apple Silicon with full control over the compute pipeline.

### Model Architecture (Qwen3-0.6B)

```
Token ID
  │
  ▼
Embedding Lookup (Q6K dequant, vocab=151936, dim=1024)
  │
  ▼
┌─────────────────────────────────────────┐
│ Transformer Layer × 28                   │
│                                          │
│  RMSNorm(x)                             │
│    │                                     │
│    ├── Q = W_q @ x   (1024 → 1024, Q4K) │
│    ├── K = W_k @ x   (1024 → 512,  Q4K) │
│    └── V = W_v @ x   (1024 → 512,  Q4K) │
│    │                                     │
│    ├── QK-Norm (per-head RMSNorm on Q,K) │
│    ├── RoPE(Q, K, position)              │
│    ├── KV Cache Store                    │
│    │                                     │
│    ▼                                     │
│  Multi-Head Attention (GQA 2:1)          │
│    16 query heads, 8 KV heads            │
│    head_dim = 64                         │
│    │                                     │
│    ▼                                     │
│  O = W_o @ attn_out  (1024 → 1024, Q4K) │
│  x = x + O           (residual)         │
│    │                                     │
│  RMSNorm(x)                             │
│    │                                     │
│    ├── gate = W_gate @ x (1024→2816,Q4K)│
│    └── up   = W_up @ x  (1024→2816,Q4K)│
│    │                                     │
│    ▼                                     │
│  SiLU(gate) * up      (SwiGLU)          │
│    │                                     │
│    ▼                                     │
│  down = W_down @ mid  (2816→1024, Q4K)  │
│  x = x + down         (residual)        │
│                                          │
└─────────────────────────────────────────┘
  │
  ▼
Final RMSNorm(x)
  │
  ▼
Logits = W_embed^T @ x  (1024 → 151936, Q6K, tied weights)
  │
  ▼
Argmax → next token ID
```

### Key Parameters

| Parameter | Value |
|-----------|-------|
| Layers | 28 |
| Hidden dimension | 1024 |
| FFN intermediate | 2816 |
| Attention heads (Q) | 16 |
| KV heads | 8 |
| Head dimension | 64 |
| GQA ratio | 2:1 |
| Vocabulary | 151,936 |
| RoPE frequency base | 1,000,000 |
| RMSNorm epsilon | 1e-6 |
| Total parameters | ~600M |
| Quantized size | ~390 MB (Q4K body + Q6K embeddings) |

### Grouped Query Attention (GQA)

Qwen3-0.6B uses grouped query attention with a 2:1 ratio: 16 query heads share 8 KV heads. Each pair of adjacent query heads reads from the same KV head. This halves KV cache size and KV projection cost while maintaining attention quality.

```
Q heads:  [0] [1] | [2] [3] | [4] [5] | ... | [14] [15]
KV heads:   [0]   |   [1]   |   [2]   | ... |    [7]
```

### Forward Pass Execution (infer.rs)

Each token generates a single Metal command buffer with chained compute encoders:

```
Command Buffer
  ├── Encoder: embed_q6k (token → hidden state)
  │
  ├── [Layer 0]
  │   ├── Encoder: rmsnorm (attention norm)
  │   ├── Encoder: matvec_q4k (Q projection)
  │   ├── Encoder: matvec_q4k (K projection)
  │   ├── Encoder: matvec_q4k (V projection)
  │   ├── Encoder: rmsnorm × 16 (Q heads) + rmsnorm × 8 (K heads)
  │   ├── Encoder: rope (Q)
  │   ├── Encoder: rope (K)
  │   ├── Encoder: kv_store
  │   ├── Encoder: attention (16 heads)
  │   ├── Encoder: matvec_q4k (O projection)
  │   ├── Encoder: residual_add
  │   ├── Encoder: rmsnorm (FFN norm)
  │   ├── Encoder: matvec_q4k (gate)
  │   ├── Encoder: matvec_q4k (up)
  │   ├── Encoder: silu_mul
  │   ├── Encoder: matvec_q4k (down)
  │   └── Encoder: residual_add
  │
  ├── [Layer 1] ... [Layer 27]  (same structure)
  │
  ├── Encoder: rmsnorm (final)
  └── Encoder: matvec_q6k (logits, tied weights)
```

Each layer creates ~17 compute encoders. With 28 layers plus overhead, a single forward pass has ~450+ encoder transitions. Each transition has ~15–20 microseconds of GPU idle time on Apple Silicon, contributing approximately 8ms of overhead per token.

### KV Cache

The KV cache stores key and value vectors for all previous positions, enabling autoregressive generation without recomputing past attention:

```
Per layer: 2 buffers (K and V)
  K cache: [MAX_SEQ × kv_dim] = [2048 × 512] × 4 bytes = 4 MB
  V cache: [MAX_SEQ × kv_dim] = [2048 × 512] × 4 bytes = 4 MB

Total: 28 layers × 2 × 4 MB = 224 MB
```

At each position, the `kv_store` kernel writes the new K and V vectors into the cache at the current position index. The attention kernel reads all positions up to and including the current one.

---

## Metal GPU System

### Zero-Dependency Bindings (gpu.rs)

The Metal GPU bindings are implemented from scratch using direct Objective-C runtime calls — no `metal-rs`, no `objc` crate, no `cocoa`. This is ~424 lines of pure Rust FFI.

**How it works:**

```rust
// Objective-C message send is the universal dispatch mechanism
extern "C" { fn objc_msgSend(); }

// Cast to the specific signature needed for each call
type MsgSend_id_void = unsafe extern "C" fn(*const c_void, *const c_void) -> *const c_void;
type MsgSend_id_u64 = unsafe extern "C" fn(*const c_void, *const c_void) -> u64;
// ... etc

// Helper macros
macro_rules! msg {
    ($obj:expr, $sel:expr) => {
        (objc_msgSend as MsgSend_void_void)($obj, sel($sel))
    };
}
```

**Type hierarchy:**

| Rust Type | Metal Type | Ownership |
|-----------|------------|-----------|
| `Device` | `MTLDevice` | Owned (released on Drop) |
| `Library` | `MTLLibrary` | Owned |
| `Function` | `MTLFunction` | Owned |
| `Pipeline` | `MTLComputePipelineState` | Owned |
| `Buffer` | `MTLBuffer` | Owned |
| `CommandQueue` | `MTLCommandQueue` | Owned |
| `CommandBuffer` | `MTLCommandBuffer` | Autoreleased + retained |
| `ComputeEncoder` | `MTLComputeCommandEncoder` | Autoreleased + retained |

**Buffer creation:**

All buffers use `MTLStorageModeShared` — unified memory on Apple Silicon means CPU and GPU share the same physical memory. No explicit copies needed.

```rust
let buf = device.new_buffer(size_bytes);
let ptr = buf.contents() as *mut f32;  // Direct CPU access to GPU memory
```

### Compute Shaders (kernels.rs)

Nine Metal compute shaders are compiled from source at runtime:

#### rmsnorm

Root mean square normalization with parallel reduction:

```
y[i] = x[i] / sqrt(mean(x²) + eps) × weight[i]
```

Uses simdgroup reduction (32-wide SIMD on Apple GPU) followed by threadgroup reduction. Each threadgroup handles one normalization across the full dimension.

#### matvec_q4k

Matrix-vector multiply with Q4_K dequantization. This is the most performance-critical kernel — it runs 8 times per layer (Q, K, V, O, gate, up, down projections).

**Q4_K block layout (144 bytes per 256 values):**
```
Offset  Size  Content
0       2     f16 super-block scale (d)
2       2     f16 super-block minimum (dmin)
4       12    6-bit scale pairs (12 bytes for 8 sub-blocks)
16      128   4-bit quantized values (2 values per byte)
```

Each thread processes one block (256 values), dequantizes inline, multiplies by the input vector, and accumulates into a partial sum. Simdgroup and threadgroup reductions combine partial sums into the final output value per row.

#### matvec_q6k

Matrix-vector multiply with Q6_K dequantization. Used for embedding lookup and the final logits projection (tied weights).

**Q6_K block layout (210 bytes per 256 values):**
```
Offset  Size  Content
0       128   4-bit low quantized values (ql)
128     64    2-bit high quantized values (qh)
192     16    per-sub-block scales (int8)
208     2     f16 super-block scale (d)
```

Dequantization reconstructs 6-bit values from separate low (4-bit) and high (2-bit) arrays, subtracts 32, and multiplies by the per-sub-block scale and super-block scale.

#### embed_q6k

Embedding table lookup with Q6_K dequantization. Given a token ID, reads the corresponding row from the quantized embedding table and writes dequantized f32 values.

One thread per output dimension. No reduction needed.

#### rope

Rotary position embeddings. Applies position-dependent rotation to pairs of dimensions in Q and K vectors:

```
θ_i = position / (freq_base ^ (2i / head_dim))
(x_0, x_1) → (x_0 × cos(θ) - x_1 × sin(θ),
               x_0 × sin(θ) + x_1 × cos(θ))
```

Qwen3-0.6B uses `freq_base = 1,000,000` (high frequency base for long-context support).

#### kv_store

Writes K and V vectors into the KV cache at the current position:

```
k_cache[pos × kv_dim + i] = k[i]
v_cache[pos × kv_dim + i] = v[i]
```

Simple element-wise copy. One thread per dimension.

#### attention

Multi-head attention with GQA support. Each threadgroup handles one query head:

1. **Score computation:** `score[j] = sum(Q[h,i] × K_cache[j,kv_h,i]) / sqrt(head_dim)` for all positions j ≤ pos
2. **Softmax:** Thread-parallel max-reduce, exp, sum-reduce for numerical stability
3. **Weighted sum:** `out[i] = sum(softmax[j] × V_cache[j,kv_h,i])`

The GQA mapping (`kv_head = query_head / gqa_ratio`) is computed inline in the kernel.

**Thread organization:** 128 threads per threadgroup, one threadgroup per query head. Threads cooperate on the reduction operations using threadgroup shared memory.

#### silu_mul

SwiGLU activation: `output[i] = gate[i] / (1 + exp(-gate[i])) × up[i]`

Element-wise, no reduction. This is the standard SiLU (Swish) activation used in modern LLMs.

#### residual_add

Simple element-wise addition: `x[i] += y[i]`

Used after attention output projection and FFN down projection to add the residual connection.

---

## GGUF Parser

### Overview (gguf.rs)

The GGUF parser reads quantized model weights from the GGUF file format (used by llama.cpp and compatible tools). The implementation is zero-dependency and uses memory-mapped I/O for zero-copy tensor access.

### File Format

```
┌──────────────────────────────────────┐
│ Header                               │
│   magic:          u32 = 0x46554747   │  "GGUF" in LE
│   version:        u32 (2 or 3)      │
│   tensor_count:   u64               │
│   metadata_count: u64               │
├──────────────────────────────────────┤
│ Metadata Key-Value Pairs             │
│   key:   string (u64 len + bytes)    │
│   value: typed (u32 type + data)     │
│   ... × metadata_count              │
├──────────────────────────────────────┤
│ Tensor Info Table                    │
│   name:    string                    │
│   ndim:    u32                       │
│   dims:    [u64; ndim]              │
│   type:    u32 (GGMLType enum)      │
│   offset:  u64 (from tensor_data)   │
│   ... × tensor_count                │
├──────────────────────────────────────┤
│ Alignment Padding                    │
│   (to general.alignment, default 32) │
├──────────────────────────────────────┤
│ Tensor Data                          │
│   raw bytes, packed by type          │
│   accessed via mmap + offset         │
└──────────────────────────────────────┘
```

### Supported Metadata Types

| Type ID | Type | Size |
|---------|------|------|
| 0 | u8 | 1 byte |
| 1 | i8 | 1 byte |
| 2 | u16 | 2 bytes |
| 3 | i16 | 2 bytes |
| 4 | u32 | 4 bytes |
| 5 | i32 | 4 bytes |
| 6 | f32 | 4 bytes |
| 7 | bool | 1 byte |
| 8 | string | u64 len + bytes |
| 9 | array | u32 type + u64 count + data |
| 10 | u64 | 8 bytes |
| 11 | i64 | 8 bytes |
| 12 | f64 | 8 bytes |

### Model Configuration Extraction

The parser reads these metadata keys to build the model configuration:

| Metadata Key | Maps To |
|-------------|---------|
| `*.block_count` | n_layers |
| `*.embedding_length` | hidden_dim |
| `*.attention.head_count` | n_heads |
| `*.attention.head_count_kv` | n_kv_heads |
| `*.feed_forward_length` | ffn_dim |
| `*.vocabulary_size` or vocab tensor dim | vocab_size |
| `*.rope.freq_base` | rope_freq_base |
| `*.attention.layer_norm_rms_epsilon` | rms_norm_eps |
| `tokenizer.ggml.tokens` | vocab tokens array |
| `tokenizer.ggml.bos_token_id` | bos_token |

### Zero-Copy Tensor Access

Tensor data is accessed directly from the memory-mapped file:

```rust
pub fn tensor_data(&self, info: &TensorInfo) -> &[u8] {
    let start = self.tensor_data_start + info.offset as usize;
    let size = info.dtype.tensor_bytes(info.n_elements());
    &self.mmap.data[start..start + size]
}
```

No copies, no allocations. The OS manages paging from disk. When the model is loaded onto GPU, the mmap data is copied once into a unified Metal buffer.

---

## Quantization Formats

### Q4_K (4-bit, K-quantization)

- **Block size:** 256 values
- **Bytes per block:** 144
- **Bits per value:** 4.5 effective
- **Used for:** All attention and FFN weight matrices

**Dequantization:**
```
For each block of 256 values:
  d    = f16_to_f32(block[0:2])       // super-block scale
  dmin = f16_to_f32(block[2:4])       // super-block minimum
  scales[0..7] = unpack(block[4:16])  // 6-bit sub-block scales
  qs[0..127] = block[16:144]          // 4-bit quantized values

  For sub-block j (32 values each):
    sc = scales[j] & 0x3F
    m  = scales[j+4] & 0x3F  (or upper bits)
    For each value i in sub-block:
      q = (qs[i/2] >> (4 * (i%2))) & 0xF
      dequant[j*32+i] = d * sc * q - dmin * m
```

### Q6_K (6-bit, K-quantization)

- **Block size:** 256 values
- **Bytes per block:** 210
- **Bits per value:** 6.5625 effective
- **Used for:** Embedding table, output projection (tied weights)

**Dequantization:**
```
For each block of 256 values:
  ql[0..127] = block[0:128]      // 4-bit low parts
  qh[0..63]  = block[128:192]    // 2-bit high parts
  scales[0..15] = block[192:208] // int8 per-sub-block scales
  d = f16_to_f32(block[208:210]) // super-block scale

  For sub-block j (16 values each):
    For each value i in sub-block:
      low  = (ql[...] >> shift) & 0xF    // 4-bit extraction
      high = (qh[...] >> shift) & 0x3    // 2-bit extraction
      q6   = low | (high << 4)           // reconstruct 6-bit
      dequant[j*16+i] = d * scales[j] * (q6 - 32)
```

### Why Different Quantizations?

The embedding table and output projection use Q6_K (higher precision) because:
1. These layers are accessed once per token (not bandwidth-critical)
2. Vocabulary is huge (151,936 entries) — quantization errors compound across many classes
3. The output projection directly determines token probabilities — precision matters

Internal layers use Q4_K (lower precision) because:
1. These are accessed repeatedly (28 layers × 7 matrices = 196 matvecs per token)
2. Internal representations are redundant enough to tolerate 4-bit quantization
3. Lower precision means less memory bandwidth → faster inference

---

## Pattern Analysis

### Insight Engine (insight.rs)

The insight engine runs semantic analysis passes over embedded entries:

**Topic Overlap Detection:**
1. Compute centroid vector for each topic (average of all entry embeddings)
2. Pairwise cosine similarity between topic centroids
3. Pairs with similarity > 0.6 are merge candidates

**Near-Duplicate Detection:**
1. All-pairs cosine similarity between entries
2. Pairs with similarity > 0.92 are near-duplicates
3. Top 20 pairs returned

**Cross-Link Discovery:**
1. All-pairs cosine similarity
2. Filter: different topics, similarity 0.75–0.92
3. These are semantically related entries that should be linked but aren't
4. Top 50 pairs returned

### Similarity Math (similarity.rs)

Because embeddings are L2-normalized, cosine similarity is just the dot product:

```
cosine(a, b) = a · b = Σ(a[i] × b[i])
```

Range: -1.0 (opposite) to 1.0 (identical). In practice, knowledge base entries cluster between 0.3 (unrelated) and 1.0 (duplicate).

---

## Performance

### Theoretical Limits

On Apple M3 Max (400 GB/s memory bandwidth):

| Operation | Data | Time | Throughput |
|-----------|------|------|-----------|
| Read all weights (1 token) | 390 MB | 0.98 ms | 400 GB/s |
| Minimum per-token latency | — | ~1.0 ms | ~1000 tok/s |
| With encoder overhead (~8ms) | — | ~9 ms | ~111 tok/s |
| Measured | — | ~10 ms | ~100 tok/s |

The bottleneck is memory bandwidth. Every token requires reading all ~390 MB of model weights through the GPU cores. Even at maximum bandwidth, this takes about 1 ms. The additional ~8 ms comes from Metal compute encoder transition overhead (~450 transitions × ~15-20 microseconds each).

### Embedding Performance

| Metric | Value |
|--------|-------|
| Model load time | ~1 second (CoreML compilation cache) |
| Per-entry embedding | ~5–15 ms |
| Batch throughput | ~100 entries/second |
| Vector dimension | 384 |
| Similarity computation | ~0.1 microsecond per pair |

### GPU Benchmark Results (metal-test)

The `metal-test` command runs four benchmark suites:

1. **Bandwidth:** Measures peak GB/s for different buffer sizes
2. **GEMM:** Matrix-matrix multiply using simdgroup_float8x8
3. **Matvec FP32:** Dense matrix-vector multiply
4. **Matvec Q4_0:** Quantized matrix-vector multiply with inline dequant

---

## Development Guide

### Adding a New Metal Kernel

1. Write the kernel source in `kernels.rs` as a string constant
2. Compile it in `model.rs` via `device.compile_library(source)`
3. Create a pipeline from the function name
4. Add encoding helper in `infer.rs`
5. Dispatch with appropriate thread/threadgroup dimensions

**Thread dispatch pattern:**
```rust
// For element-wise operations (no reduction):
let threads = MTLSize { width: n_elements as u64, height: 1, depth: 1 };
let tg_size = MTLSize { width: pipeline.thread_width() as u64, height: 1, depth: 1 };
encoder.dispatch_threads(threads, tg_size);

// For row-parallel matvec (one result per row):
let n_groups = (n_blocks + threads_per_row - 1) / threads_per_row;
let threads = MTLSize { width: n_groups as u64, height: rows as u64, depth: 1 };
let tg_size = MTLSize { width: n_groups.min(256) as u64, height: 1, depth: 1 };
encoder.dispatch_threadgroups(threads, tg_size);
```

### Debugging Model Output

1. **Verify embeddings:** Run `embed-test` to check embedding pipeline produces reasonable similarity scores
2. **Compare CPU vs GPU:** The `cpu_dequant_q6k_row()` function in `main.rs` provides a CPU reference implementation for Q6_K dequantization
3. **Layer-by-layer validation:** Add GPU→CPU readbacks after each layer to compare against a reference implementation (e.g., llama.cpp with `-ngl 0`)
4. **Check tensor shapes:** Use `load-model` to inspect GGUF tensor metadata and verify dimensions match expectations

### Project Structure

```
src/
├── main.rs          ← CLI + daemon loop
├── gguf.rs          ← GGUF file parser (zero-copy mmap)
├── model.rs         ← GPU model loading + pipeline compilation
├── infer.rs         ← Transformer forward pass
├── kernels.rs       ← Metal shader source code
├── gpu.rs           ← Zero-dep Metal GPU bindings
├── embed.rs         ← ONNX embedding inference
├── tokenize.rs      ← WordPiece tokenizer
├── datalog.rs       ← data.log reader (shared format with helix)
├── cache.rs         ← Embedding persistence
├── insight.rs       ← Pattern analysis
├── similarity.rs    ← Vector math
└── metal_test.rs    ← GPU benchmark suite
```

---

## Troubleshooting

### ONNX Runtime Errors

**"CoreML EP not available":**
The `ort` crate needs the CoreML feature. Check Cargo.toml includes:
```toml
ort = { version = "=2.0.0-rc.10", features = ["coreml"] }
```

**"Model file not found":**
Place `model.onnx` and `vocab.txt` in `~/.helix/models/`.

### Metal GPU Errors

**"MTLCreateSystemDefaultDevice returned NULL":**
Not running on Apple Silicon. Metal compute requires M1 or later.

**Shader compilation errors:**
Check `kernels.rs` for syntax errors. Metal shaders are compiled at runtime from string constants. Errors appear on stderr during model loading.

### Model Loading Issues

**"Unknown GGUF version":**
Only versions 2 and 3 are supported. Check the model file was downloaded correctly (not truncated).

**"Tensor not found: token_embd.weight":**
The model file may use a different naming convention. Use `load-model` to inspect available tensor names.

### Poor Generation Quality

If the model produces gibberish:
1. Verify embedding correctness first (`generate` command prints CPU vs GPU comparison)
2. Check that RoPE frequency base matches the model's metadata (Qwen3 uses 1,000,000)
3. Ensure Q4K/Q6K dequantization matches the reference (compare against `cpu_dequant_q6k_row()`)
4. Verify attention mask is applied correctly (causal mask: attend only to positions ≤ current)
5. Check head dimension and GQA ratio match the model config

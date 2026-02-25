//! 70B-optimized batch forward pass for Llama decoder-only transformer.
//! All matmul uses MMA staging: dequant quantized weights in threadgroup memory,
//! then simdgroup_multiply_accumulate for hardware FP16 MMA throughput.
//! Supports Q5K, Q6K, Q8_0 quantization types (the 70B GGUF quant mix).

use crate::gpu::*;
use crate::gguf::GGMLType;
use crate::model::Model;
use crate::kernels_batch;
use std::ffi::c_void;

/// Grid tile size for MMA kernels. 80 = 10 batch groups per staging pass.
/// Optimal on M3 Max: 96→47tok/s (occupancy loss), 80→59tok/s, 160 hangs.
const TILE_B: u32 = 80;

/// Maximum batch size supported (tokens per forward pass).
const MAX_BATCH: u32 = 1024;
const MAX_BATCH_PADDED: u32 = (MAX_BATCH + TILE_B - 1) / TILE_B * TILE_B; // 1040

/// Maximum sequence length for KV cache.
const MAX_SEQ: u32 = 2048;

/// Batch inference state: FP16 buffers + batch pipelines.
pub struct BatchState {
    pub batch_size: u32,
    // MMA staging grid pipelines (primary batch path)
    pub p_q5k_grid: Pipeline,
    pub p_q5k_grid_add: Pipeline, // Fused Y += W×X (residual add in matmul)
    pub p_q6k_grid: Pipeline,
    pub p_q6k_grid_add: Pipeline, // Fused Y += W×X for Q6K weights
    pub p_q6k_grid_silu_add: Pipeline, // Fused Y += Wdown × silu(Gate) * Up
    pub p_q8_0_grid: Pipeline,
    // Scalar matvec (n=1 decode, bandwidth-optimal for single token)
    pub p_matvec_q5k: Pipeline,
    pub p_matvec_q6k: Pipeline,
    pub p_matvec_q8_0: Pipeline,
    // Embedding (quantized lookup)
    pub p_embed_q5k: Pipeline,
    pub p_embed_q6k: Pipeline,
    pub p_embed_q8_0: Pipeline,
    // Element-wise pipelines
    pub p_rmsnorm: Pipeline,
    pub p_residual_rmsnorm: Pipeline,
    pub p_rope: Pipeline,
    pub p_kv_store: Pipeline,
    pub p_silu_mul: Pipeline,
    pub p_residual_add: Pipeline,
    pub p_attn_causal: Pipeline,
    pub p_argmax: Pipeline,
    // FP16 batch buffers: [B × dim] in half precision
    pub tokens_buf: Buffer,
    pub x: Buffer,
    pub norm_out: Buffer,
    pub q: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub attn_out: Buffer,
    pub gate: Buffer,
    pub up: Buffer,
    pub ffn_mid: Buffer,
    pub logits: Buffer,
    pub argmax_buf: Buffer,
    // FP16 KV cache: per layer
    pub k_cache: Vec<Buffer>,
    pub v_cache: Vec<Buffer>,
}

impl BatchState {
    pub fn new(model: &Model) -> Result<Self, String> {
        let bs = MAX_BATCH;
        let cfg = &model.cfg;

        let src = kernels_batch::all_batch_kernels();
        let lib = model.device.new_library_with_source(&src)?;

        let pipe = |name: &str| -> Result<Pipeline, String> {
            let f = lib.get_function(name)?;
            model.device.new_compute_pipeline(&f)
        };

        // MMA staging grid: batch matmul for Q5K/Q6K/Q8_0
        let p_q5k_grid = pipe("matmul_q5k_grid")?;
        let p_q5k_grid_add = pipe("matmul_q5k_grid_add")?;
        let p_q6k_grid = pipe("matmul_q6k_grid")?;
        let p_q6k_grid_add = pipe("matmul_q6k_grid_add")?;
        let p_q6k_grid_silu_add = pipe("matmul_q6k_grid_silu_add")?;
        let p_q8_0_grid = pipe("matmul_q8_0_grid")?;
        // Scalar matvec: n=1 decode
        let p_matvec_q5k = pipe("matvec_q5k_f16")?;
        let p_matvec_q6k = pipe("matvec_q6k_f16")?;
        let p_matvec_q8_0 = pipe("matvec_q8_0_f16")?;
        // Embedding
        let p_embed_q5k = pipe("embed_batch_q5k")?;
        let p_embed_q6k = pipe("embed_batch_q6k")?;
        let p_embed_q8_0 = pipe("embed_batch_q8_0")?;
        // Element-wise
        let p_rmsnorm = pipe("rmsnorm_batch")?;
        let p_residual_rmsnorm = pipe("residual_rmsnorm_batch")?;
        let p_rope = pipe("rope_batch")?;
        let p_kv_store = pipe("kv_store_batch")?;
        let p_silu_mul = pipe("silu_mul_batch")?;
        let p_residual_add = pipe("residual_add_batch")?;
        let p_attn_causal = pipe("attention_batch_causal")?;
        let p_argmax = pipe("argmax_batch")?;

        let dev = &model.device;
        let h = |n: u64| dev.new_buffer(n * 2);
        let bp = MAX_BATCH_PADDED as u64;
        let hidden = cfg.hidden_dim as u64;
        let q_dim = model.q_dim as u64;
        let kv_dim = model.kv_dim as u64;
        let ffn = cfg.ffn_dim as u64;
        let vocab = cfg.vocab_size as u64;

        let tokens_buf = dev.new_buffer(bp * 4);
        let x = h(bp * hidden);
        let norm_out = h(bp * hidden);
        let q = h(bp * q_dim);
        let k = h(bp * kv_dim);
        let v = h(bp * kv_dim);
        let attn_out = h(bp * q_dim);
        let gate = h(bp * ffn);
        let up = h(bp * ffn);
        let ffn_mid = h(bp * ffn);
        let logits = h(bp * vocab);
        let argmax_buf = dev.new_buffer(bp * 4);

        let kv_size = MAX_SEQ as u64 * kv_dim * 2;
        let mut k_cache = Vec::with_capacity(cfg.n_layers as usize);
        let mut v_cache = Vec::with_capacity(cfg.n_layers as usize);
        for _ in 0..cfg.n_layers {
            k_cache.push(dev.new_buffer(kv_size));
            v_cache.push(dev.new_buffer(kv_size));
        }

        let buf_mb = bp as f64 * (hidden + hidden + q_dim + kv_dim + kv_dim + q_dim + ffn + ffn + ffn + vocab) as f64 * 2.0 / 1e6;
        eprintln!("batch: max_B={bs} (padded={bp}), tile_b={TILE_B}, bufs={buf_mb:.0}MB, KV={:.0}MB",
            kv_size as f64 * 2.0 * cfg.n_layers as f64 / 1e6);

        Ok(BatchState {
            batch_size: bs,
            p_q5k_grid, p_q5k_grid_add, p_q6k_grid, p_q6k_grid_add,
            p_q6k_grid_silu_add, p_q8_0_grid,
            p_matvec_q5k, p_matvec_q6k, p_matvec_q8_0,
            p_embed_q5k, p_embed_q6k, p_embed_q8_0,
            p_rmsnorm, p_residual_rmsnorm,
            p_rope, p_kv_store, p_silu_mul,
            p_residual_add, p_attn_causal, p_argmax,
            tokens_buf, x, norm_out, q, k, v, attn_out,
            gate, up, ffn_mid, logits, argmax_buf,
            k_cache, v_cache,
        })
    }
}

/// Run batch forward pass: n tokens at positions [start_pos..start_pos+n-1].
pub fn forward_batch(
    model: &Model, batch: &BatchState, tokens: &[u32], start_pos: u32,
) {
    forward_batch_n(model, batch, tokens, start_pos, tokens.len() as u32);
}

/// Variable-count batch forward: process `n` tokens (1..=MAX_BATCH).
pub fn forward_batch_n(
    model: &Model, batch: &BatchState, tokens: &[u32],
    start_pos: u32, n: u32,
) {
    let cfg = &model.cfg;
    let b = n;
    assert!(tokens.len() >= n as usize);
    assert!(n <= MAX_BATCH, "n={n} exceeds MAX_BATCH={MAX_BATCH}");

    unsafe {
        let dst = batch.tokens_buf.contents() as *mut u32;
        std::ptr::copy_nonoverlapping(tokens.as_ptr(), dst, n as usize);
    }

    let cmd = model.queue.new_command_buffer();
    let enc = cmd.new_compute_encoder();

    // ── Embedding → x[n × hidden_dim] ──
    dispatch_embed(&enc, model, batch, b);

    // ── Transformer layers ──
    for layer in 0..cfg.n_layers as usize {
        let lo = &model.layers[layer];

        // Attention RMSNorm: x → norm_out
        dispatch_rmsnorm(&enc, batch, &batch.x, &batch.norm_out,
            &model.weights, lo.attn_norm, cfg.hidden_dim, b);

        // Q/K/V matmul (MMA staging for Q5K/Q6K/Q8_0)
        dispatch_matmul(&enc, model, batch, lo.attn_q_type,
            lo.attn_q, &batch.norm_out, &batch.q, cfg.hidden_dim, model.q_dim, b);
        dispatch_matmul(&enc, model, batch, lo.attn_k_type,
            lo.attn_k, &batch.norm_out, &batch.k, cfg.hidden_dim, model.kv_dim, b);
        dispatch_matmul(&enc, model, batch, lo.attn_v_type,
            lo.attn_v, &batch.norm_out, &batch.v, cfg.hidden_dim, model.kv_dim, b);

        // RoPE
        dispatch_rope(&enc, model, batch, &batch.q,
            model.head_dim, start_pos, cfg.n_heads, model.q_dim, b);
        dispatch_rope(&enc, model, batch, &batch.k,
            model.head_dim, start_pos, cfg.n_kv_heads, model.kv_dim, b);

        // KV store
        dispatch_kv_store(&enc, batch, model.kv_dim, layer, start_pos, b);

        // Attention (causal)
        dispatch_attention(&enc, model, batch, layer, start_pos, b);

        // O projection → norm_out, then fused: x += norm_out; norm_out = rmsnorm(x, ffn_norm)
        dispatch_matmul(&enc, model, batch, lo.attn_output_type,
            lo.attn_output, &batch.attn_out, &batch.norm_out, model.q_dim, cfg.hidden_dim, b);
        dispatch_residual_rmsnorm(&enc, batch, &batch.x, &batch.norm_out,
            &model.weights, lo.ffn_norm, cfg.hidden_dim, b);

        // Gate + Up matmul
        dispatch_matmul(&enc, model, batch, lo.ffn_gate_type,
            lo.ffn_gate, &batch.norm_out, &batch.gate, cfg.hidden_dim, cfg.ffn_dim, b);
        dispatch_matmul(&enc, model, batch, lo.ffn_up_type,
            lo.ffn_up, &batch.norm_out, &batch.up, cfg.hidden_dim, cfg.ffn_dim, b);

        // SiLU(gate) * up
        dispatch_silu(&enc, batch, cfg.ffn_dim as u64 * b as u64);

        // Fused down + residual: x += W_down @ ffn_mid (single kernel, zero extra BW)
        dispatch_matmul_add(&enc, model, batch, lo.ffn_down_type,
            lo.ffn_down, &batch.ffn_mid, &batch.x, cfg.ffn_dim, cfg.hidden_dim, b);
    }

    // Final RMSNorm
    dispatch_rmsnorm(&enc, batch, &batch.x, &batch.norm_out,
        &model.weights, model.out_norm_off, cfg.hidden_dim, b);

    // Output projection
    dispatch_matmul(&enc, model, batch, model.output_type,
        model.output_off, &batch.norm_out, &batch.logits, cfg.hidden_dim, cfg.vocab_size, b);

    // Argmax
    dispatch_argmax(&enc, batch, cfg.vocab_size, b);

    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

/// Read batch argmax results from GPU.
pub fn argmax_batch(batch: &BatchState, n: u32) -> Vec<u32> {
    let ptr = batch.argmax_buf.contents() as *const u32;
    (0..n as usize).map(|i| unsafe { *ptr.add(i) }).collect()
}

/// Read single argmax result (batch position 0).
pub fn argmax_one(batch: &BatchState) -> u32 {
    unsafe { *(batch.argmax_buf.contents() as *const u32) }
}

/// Prefill a prompt using batch forward pass.
pub fn prefill(model: &Model, batch: &BatchState, prompt: &[u32]) -> u32 {
    let bs = MAX_BATCH as usize;
    let mut pos = 0u32;
    let full_chunks = prompt.len() / bs;
    for c in 0..full_chunks {
        let chunk = &prompt[c * bs..(c + 1) * bs];
        forward_batch_n(model, batch, chunk, pos, MAX_BATCH);
        pos += MAX_BATCH;
    }
    let rem = prompt.len() % bs;
    if rem > 0 {
        forward_batch_n(model, batch, &prompt[full_chunks * bs..], pos, rem as u32);
        pos += rem as u32;
    }
    pos
}

/// Single-token decode step using batch infrastructure.
pub fn decode_step(model: &Model, batch: &BatchState, token: u32, position: u32) -> u32 {
    forward_batch_n(model, batch, &[token], position, 1);
    argmax_one(batch)
}

/// Profiled forward pass: times each operation group separately.
pub fn forward_profiled(
    model: &Model, batch: &BatchState, tokens: &[u32],
    start_pos: u32, n: u32,
) -> (f64, f64, f64, f64, f64) {
    let cfg = &model.cfg;
    let b = n;
    unsafe {
        let dst = batch.tokens_buf.contents() as *mut u32;
        std::ptr::copy_nonoverlapping(tokens.as_ptr(), dst, n as usize);
    }

    let mut matmul_us = 0.0f64;
    let mut attn_us = 0.0f64;
    let mut elem_us = 0.0f64;
    let mut output_us = 0.0f64;

    macro_rules! timed {
        ($acc:expr, $body:expr) => {{
            let cmd = model.queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            $body(&enc);
            enc.end_encoding();
            let t = std::time::Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            $acc += t.elapsed().as_nanos() as f64 / 1000.0;
        }};
    }

    timed!(elem_us, |enc: &ComputeEncoder| {
        dispatch_embed(enc, model, batch, b);
    });

    for layer in 0..cfg.n_layers as usize {
        let lo = &model.layers[layer];

        timed!(elem_us, |enc: &ComputeEncoder| {
            dispatch_rmsnorm(enc, batch, &batch.x, &batch.norm_out,
                &model.weights, lo.attn_norm, cfg.hidden_dim, b);
        });

        timed!(matmul_us, |enc: &ComputeEncoder| {
            dispatch_matmul(enc, model, batch, lo.attn_q_type,
                lo.attn_q, &batch.norm_out, &batch.q, cfg.hidden_dim, model.q_dim, b);
            dispatch_matmul(enc, model, batch, lo.attn_k_type,
                lo.attn_k, &batch.norm_out, &batch.k, cfg.hidden_dim, model.kv_dim, b);
            dispatch_matmul(enc, model, batch, lo.attn_v_type,
                lo.attn_v, &batch.norm_out, &batch.v, cfg.hidden_dim, model.kv_dim, b);
        });

        timed!(elem_us, |enc: &ComputeEncoder| {
            dispatch_rope(enc, model, batch, &batch.q,
                model.head_dim, start_pos, cfg.n_heads, model.q_dim, b);
            dispatch_rope(enc, model, batch, &batch.k,
                model.head_dim, start_pos, cfg.n_kv_heads, model.kv_dim, b);
            dispatch_kv_store(enc, batch, model.kv_dim, layer, start_pos, b);
        });

        timed!(attn_us, |enc: &ComputeEncoder| {
            dispatch_attention(enc, model, batch, layer, start_pos, b);
        });

        timed!(matmul_us, |enc: &ComputeEncoder| {
            dispatch_matmul(enc, model, batch, lo.attn_output_type,
                lo.attn_output, &batch.attn_out, &batch.norm_out, model.q_dim, cfg.hidden_dim, b);
        });

        timed!(elem_us, |enc: &ComputeEncoder| {
            dispatch_residual_rmsnorm(enc, batch, &batch.x, &batch.norm_out,
                &model.weights, lo.ffn_norm, cfg.hidden_dim, b);
        });

        timed!(matmul_us, |enc: &ComputeEncoder| {
            dispatch_matmul(enc, model, batch, lo.ffn_gate_type,
                lo.ffn_gate, &batch.norm_out, &batch.gate, cfg.hidden_dim, cfg.ffn_dim, b);
            dispatch_matmul(enc, model, batch, lo.ffn_up_type,
                lo.ffn_up, &batch.norm_out, &batch.up, cfg.hidden_dim, cfg.ffn_dim, b);
        });

        timed!(elem_us, |enc: &ComputeEncoder| {
            dispatch_silu(enc, batch, cfg.ffn_dim as u64 * b as u64);
        });

        timed!(matmul_us, |enc: &ComputeEncoder| {
            dispatch_matmul_add(enc, model, batch, lo.ffn_down_type,
                lo.ffn_down, &batch.ffn_mid, &batch.x, cfg.ffn_dim, cfg.hidden_dim, b);
        });
    }

    timed!(elem_us, |enc: &ComputeEncoder| {
        dispatch_rmsnorm(enc, batch, &batch.x, &batch.norm_out,
            &model.weights, model.out_norm_off, cfg.hidden_dim, b);
    });

    timed!(output_us, |enc: &ComputeEncoder| {
        dispatch_matmul(enc, model, batch, model.output_type,
            model.output_off, &batch.norm_out, &batch.logits, cfg.hidden_dim, cfg.vocab_size, b);
    });

    timed!(elem_us, |enc: &ComputeEncoder| {
        dispatch_argmax(enc, batch, cfg.vocab_size, b);
    });

    let total = matmul_us + attn_us + elem_us + output_us;
    (total / 1000.0, matmul_us / 1000.0, attn_us / 1000.0, elem_us / 1000.0, output_us / 1000.0)
}

// ── Dispatch helpers ──

/// Embedding: routes Q5K/Q6K/Q8_0 to quantized embed kernel.
fn dispatch_embed(enc: &ComputeEncoder, model: &Model, batch: &BatchState, bs: u32) {
    let dim = model.cfg.hidden_dim;
    let pipe = match model.embd_type {
        GGMLType::Q5K  => &batch.p_embed_q5k,
        GGMLType::Q6K  => &batch.p_embed_q6k,
        GGMLType::Q8_0 => &batch.p_embed_q8_0,
        _ => panic!("unsupported embed dtype: {:?}", model.embd_type),
    };
    enc.set_pipeline(pipe);
    enc.set_buffer(0, &model.weights, model.embd_off);
    enc.set_buffer(1, &batch.x, 0);
    enc.set_buffer(2, &batch.tokens_buf, 0);
    enc.set_bytes(3, &dim as *const u32 as *const c_void, 4);
    enc.set_bytes(4, &bs as *const u32 as *const c_void, 4);
    enc.dispatch_threads(
        MTLSize::new(dim as u64, bs as u64, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Matmul dispatch: n=1 → scalar matvec, n>1 → MMA staging grid.
fn dispatch_matmul(
    enc: &ComputeEncoder, model: &Model, batch: &BatchState,
    dtype: GGMLType, quant_off: u64,
    x: &Buffer, y: &Buffer, cols: u32, rows: u32, b: u32,
) {
    if b <= 1 {
        dispatch_matvec(enc, model, batch, dtype, quant_off, x, y, cols, rows);
    } else {
        let pipe = match dtype {
            GGMLType::Q5K  => &batch.p_q5k_grid,
            GGMLType::Q6K  => &batch.p_q6k_grid,
            GGMLType::Q8_0 => &batch.p_q8_0_grid,
            _ => panic!("unsupported batch matmul dtype: {:?}", dtype),
        };
        dispatch_mma_grid(enc, model, pipe, quant_off, x, y, cols, rows, b);
    }
}

/// Fused matmul+residual_add: y += W × x. Accumulators initialized from y.
/// For n=1, falls back to matvec then explicit add (no fused scalar variant yet).
fn dispatch_matmul_add(
    enc: &ComputeEncoder, model: &Model, batch: &BatchState,
    dtype: GGMLType, quant_off: u64,
    x: &Buffer, y: &Buffer, cols: u32, rows: u32, b: u32,
) {
    if b <= 1 {
        // Scalar path: matvec to norm_out then add. TODO: fused scalar variant.
        dispatch_matvec(enc, model, batch, dtype, quant_off, x, &batch.norm_out, cols, rows);
        dispatch_residual_add(enc, batch, y, &batch.norm_out, rows as u64);
    } else {
        let pipe = match dtype {
            GGMLType::Q5K => &batch.p_q5k_grid_add,
            GGMLType::Q6K => &batch.p_q6k_grid_add,
            _ => panic!("matmul_add unsupported dtype: {:?}", dtype),
        };
        dispatch_mma_grid(enc, model, pipe, quant_off, x, y, cols, rows, b);
    }
}

/// Fused silu + Wdown + residual: x += Wdown × silu(Gate) * Up.
/// Batch path uses single kernel. Scalar (n=1) falls back to silu then matvec+add.
fn dispatch_matmul_silu_add(
    enc: &ComputeEncoder, model: &Model, batch: &BatchState,
    quant_off: u64, cols: u32, rows: u32, b: u32,
) {
    if b <= 1 {
        // Scalar path: explicit silu then matvec+add
        dispatch_silu(enc, batch, cols as u64);
        dispatch_matvec(enc, model, batch, GGMLType::Q6K, quant_off,
            &batch.ffn_mid, &batch.norm_out, cols, rows);
        dispatch_residual_add(enc, batch, &batch.x, &batch.norm_out, rows as u64);
    } else {
        let pipe = &batch.p_q6k_grid_silu_add;
        enc.set_pipeline(pipe);
        enc.set_buffer(0, &model.weights, quant_off);
        enc.set_buffer(1, &batch.gate, 0);
        enc.set_buffer(2, &batch.up, 0);
        enc.set_buffer(3, &batch.x, 0);
        enc.set_bytes(4, &cols as *const u32 as *const c_void, 4);
        enc.set_bytes(5, &rows as *const u32 as *const c_void, 4);
        enc.set_bytes(6, &b as *const u32 as *const c_void, 4);
        let row_groups = ((rows + 31) / 32) as u64;
        let batch_tiles = ((b as u64 + TILE_B as u64 - 1) / TILE_B as u64) as u64;
        enc.dispatch_threadgroups(
            MTLSize::new(row_groups, batch_tiles, 1),
            MTLSize::new(128, 1, 1),
        );
    }
}

/// Scalar matvec: n=1, simd_sum reduction. Bandwidth-optimal for single token.
fn dispatch_matvec(
    enc: &ComputeEncoder, model: &Model, batch: &BatchState,
    dtype: GGMLType, quant_off: u64,
    x: &Buffer, y: &Buffer, cols: u32, rows: u32,
) {
    let pipe = match dtype {
        GGMLType::Q5K  => &batch.p_matvec_q5k,
        GGMLType::Q6K  => &batch.p_matvec_q6k,
        GGMLType::Q8_0 => &batch.p_matvec_q8_0,
        _ => panic!("unsupported matvec dtype: {:?}", dtype),
    };
    let b = 1u32;
    enc.set_pipeline(pipe);
    enc.set_buffer(0, &model.weights, quant_off);
    enc.set_buffer(1, x, 0);
    enc.set_buffer(2, y, 0);
    enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
    enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &b as *const u32 as *const c_void, 4);
    let grid_x = (rows + 7) / 8;
    enc.dispatch_threadgroups(
        MTLSize::new(grid_x as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// M3 Max L2 cache: ~32MB. When X exceeds this, split dispatch into waves.
const L2_CACHE_BYTES: u64 = 32 * 1024 * 1024;

/// Grid-tiled MMA staging: dequant in threadgroup memory → simdgroup MMA.
/// Grid: (ceil(rows/32), ceil(b/tile_b)), TG: 128 (4 SGs × 8 rows).
/// L2 wave tiling: if X (b × cols × 2) exceeds L2, split batch into waves
/// that each fit in L2. Prevents cache thrashing on FFN Down (28672 cols).
fn dispatch_mma_grid(
    enc: &ComputeEncoder, model: &Model, pipe: &Pipeline,
    quant_off: u64, x: &Buffer, y: &Buffer, cols: u32, rows: u32, b: u32,
) {
    let x_bytes = b as u64 * cols as u64 * 2;
    let row_groups = ((rows + 31) / 32) as u64;

    if x_bytes <= L2_CACHE_BYTES {
        // Fast path: X fits in L2, single dispatch
        enc.set_pipeline(pipe);
        enc.set_buffer(0, &model.weights, quant_off);
        enc.set_buffer(1, x, 0);
        enc.set_buffer(2, y, 0);
        enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
        enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
        enc.set_bytes(5, &b as *const u32 as *const c_void, 4);
        let batch_tiles = ((b as u64 + TILE_B as u64 - 1) / TILE_B as u64) as u64;
        enc.dispatch_threadgroups(
            MTLSize::new(row_groups, batch_tiles, 1),
            MTLSize::new(128, 1, 1),
        );
    } else {
        // L2 wave tiling: split batch into waves that fit in L2.
        // Each wave processes wave_b tokens. Waves execute sequentially
        // (Metal serial dispatch) so L2 contains only one wave's X data.
        let max_wave_b = (L2_CACHE_BYTES / (cols as u64 * 2)) as u32;
        // Round down to TILE_B boundary for clean dispatch
        let wave_b = (max_wave_b / TILE_B) * TILE_B;
        let wave_b = if wave_b == 0 { TILE_B } else { wave_b };

        let mut offset = 0u32;
        while offset < b {
            let this_b = std::cmp::min(wave_b, b - offset);
            let x_off = offset as u64 * cols as u64 * 2;
            let y_off = offset as u64 * rows as u64 * 2;
            enc.set_pipeline(pipe);
            enc.set_buffer(0, &model.weights, quant_off);
            enc.set_buffer(1, x, x_off);
            enc.set_buffer(2, y, y_off);
            enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
            enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
            enc.set_bytes(5, &this_b as *const u32 as *const c_void, 4);
            let batch_tiles = ((this_b as u64 + TILE_B as u64 - 1) / TILE_B as u64) as u64;
            enc.dispatch_threadgroups(
                MTLSize::new(row_groups, batch_tiles, 1),
                MTLSize::new(128, 1, 1),
            );
            offset += this_b;
        }
    }
}

fn dispatch_rmsnorm(
    enc: &ComputeEncoder, batch: &BatchState,
    x: &Buffer, y: &Buffer, weights: &Buffer, weight_off: u64, dim: u32, bs: u32,
) {
    let eps: f32 = 1e-6;
    enc.set_pipeline(&batch.p_rmsnorm);
    enc.set_buffer(0, x, 0);
    enc.set_buffer(1, y, 0);
    enc.set_buffer(2, weights, weight_off);
    enc.set_bytes(3, &dim as *const u32 as *const c_void, 4);
    enc.set_bytes(4, &eps as *const f32 as *const c_void, 4);
    enc.set_bytes(5, &bs as *const u32 as *const c_void, 4);
    enc.dispatch_threadgroups(
        MTLSize::new(bs as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

fn dispatch_residual_rmsnorm(
    enc: &ComputeEncoder, batch: &BatchState,
    res: &Buffer, add_norm: &Buffer,
    weights: &Buffer, weight_off: u64, dim: u32, bs: u32,
) {
    let eps: f32 = 1e-6;
    enc.set_pipeline(&batch.p_residual_rmsnorm);
    enc.set_buffer(0, res, 0);
    enc.set_buffer(1, add_norm, 0);
    enc.set_buffer(2, add_norm, 0);
    enc.set_buffer(3, weights, weight_off);
    enc.set_bytes(4, &dim as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &eps as *const f32 as *const c_void, 4);
    enc.set_bytes(6, &bs as *const u32 as *const c_void, 4);
    enc.dispatch_threadgroups(
        MTLSize::new(bs as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

fn dispatch_rope(
    enc: &ComputeEncoder, model: &Model, _batch: &BatchState, vec: &Buffer,
    head_dim: u32, start_pos: u32, n_heads: u32, total_dim: u32, bs: u32,
) {
    let freq_base = model.cfg.rope_freq_base;
    let n_pairs = n_heads * head_dim / 2;
    enc.set_pipeline(&_batch.p_rope);
    enc.set_buffer(0, vec, 0);
    enc.set_bytes(1, &head_dim as *const u32 as *const c_void, 4);
    enc.set_bytes(2, &start_pos as *const u32 as *const c_void, 4);
    enc.set_bytes(3, &freq_base as *const f32 as *const c_void, 4);
    enc.set_bytes(4, &total_dim as *const u32 as *const c_void, 4);
    enc.dispatch_threads(
        MTLSize::new(n_pairs as u64, bs as u64, 1),
        MTLSize::new(64, 1, 1),
    );
}

fn dispatch_kv_store(
    enc: &ComputeEncoder, batch: &BatchState, kv_dim: u32,
    layer: usize, start_pos: u32, bs: u32,
) {
    enc.set_pipeline(&batch.p_kv_store);
    enc.set_buffer(0, &batch.k, 0);
    enc.set_buffer(1, &batch.v, 0);
    enc.set_buffer(2, &batch.k_cache[layer], 0);
    enc.set_buffer(3, &batch.v_cache[layer], 0);
    enc.set_bytes(4, &kv_dim as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &start_pos as *const u32 as *const c_void, 4);
    enc.set_bytes(6, &bs as *const u32 as *const c_void, 4);
    enc.dispatch_threads(
        MTLSize::new(kv_dim as u64, bs as u64, 1),
        MTLSize::new(64, 1, 1),
    );
}

fn dispatch_attention(
    enc: &ComputeEncoder, model: &Model, batch: &BatchState,
    layer: usize, start_pos: u32, bs: u32,
) {
    let hd = model.head_dim;
    let n_kv = model.cfg.n_kv_heads;
    let gqa = model.gqa_ratio;
    let kv_dim = model.kv_dim;
    let total_q_dim = model.q_dim;
    let n_heads = model.cfg.n_heads;
    enc.set_pipeline(&batch.p_attn_causal);
    enc.set_buffer(0, &batch.q, 0);
    enc.set_buffer(1, &batch.k_cache[layer], 0);
    enc.set_buffer(2, &batch.v_cache[layer], 0);
    enc.set_buffer(3, &batch.attn_out, 0);
    enc.set_bytes(4, &start_pos as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &hd as *const u32 as *const c_void, 4);
    enc.set_bytes(6, &n_kv as *const u32 as *const c_void, 4);
    enc.set_bytes(7, &gqa as *const u32 as *const c_void, 4);
    enc.set_bytes(8, &kv_dim as *const u32 as *const c_void, 4);
    enc.set_bytes(9, &total_q_dim as *const u32 as *const c_void, 4);
    enc.set_bytes(10, &bs as *const u32 as *const c_void, 4);
    enc.dispatch_threadgroups(
        MTLSize::new(n_heads as u64, bs as u64, 1),
        MTLSize::new(128, 1, 1),
    );
}

fn dispatch_silu(enc: &ComputeEncoder, batch: &BatchState, n_elems: u64) {
    enc.set_pipeline(&batch.p_silu_mul);
    enc.set_buffer(0, &batch.gate, 0);
    enc.set_buffer(1, &batch.up, 0);
    enc.set_buffer(2, &batch.ffn_mid, 0);
    enc.dispatch_threads(
        MTLSize::new(n_elems, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

fn dispatch_residual_add(
    enc: &ComputeEncoder, batch: &BatchState,
    res: &Buffer, add: &Buffer, n_elems: u64,
) {
    enc.set_pipeline(&batch.p_residual_add);
    enc.set_buffer(0, res, 0);
    enc.set_buffer(1, add, 0);
    enc.dispatch_threads(
        MTLSize::new(n_elems, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

fn dispatch_argmax(enc: &ComputeEncoder, batch: &BatchState, vocab_size: u32, bs: u32) {
    enc.set_pipeline(&batch.p_argmax);
    enc.set_buffer(0, &batch.logits, 0);
    enc.set_buffer(1, &batch.argmax_buf, 0);
    enc.set_bytes(2, &vocab_size as *const u32 as *const c_void, 4);
    enc.set_bytes(3, &bs as *const u32 as *const c_void, 4);
    enc.dispatch_threadgroups(
        MTLSize::new(bs as u64, 1, 1),
        MTLSize::new(1024, 1, 1),
    );
}

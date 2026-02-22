//! Batch forward pass for Llama decoder-only transformer.
//! Processes B tokens simultaneously using FP16 compute path.
//! All batch data in half precision, matmuls use explicit scalar accumulators.

use crate::gpu::*;
use crate::gguf::GGMLType;
use crate::model::Model;
use crate::kernels_batch;
use std::ffi::c_void;

/// Batch inference state: FP16 buffers + batch pipelines layered on top of Model.
pub struct BatchState {
    pub batch_size: u32,
    // Batch pipelines (compiled from kernels_batch)
    pub p_matmul_q4_0_f16: Pipeline,
    pub p_matmul_q4_0_mma: Pipeline,
    pub p_matmul_q4_1_f16: Pipeline,
    pub p_matmul_q6k_f16: Pipeline,
    pub p_embed_q4_0: Pipeline,
    pub p_embed_q6k: Pipeline,
    pub p_rmsnorm: Pipeline,
    pub p_rope: Pipeline,
    pub p_kv_store: Pipeline,
    pub p_silu_mul: Pipeline,
    pub p_residual_add: Pipeline,
    pub p_attn_causal: Pipeline,
    pub p_argmax: Pipeline,
    // FP16 batch buffers: [B × dim] in half precision
    pub tokens_buf: Buffer,    // [B] u32 token IDs
    pub x: Buffer,             // [B × hidden_dim] half
    pub norm_out: Buffer,      // [B × hidden_dim] half
    pub q: Buffer,             // [B × q_dim] half
    pub k: Buffer,             // [B × kv_dim] half
    pub v: Buffer,             // [B × kv_dim] half
    pub attn_out: Buffer,      // [B × q_dim] half
    pub gate: Buffer,          // [B × ffn_dim] half
    pub up: Buffer,            // [B × ffn_dim] half
    pub ffn_mid: Buffer,       // [B × ffn_dim] half
    pub logits: Buffer,        // [B × vocab_size] half
    pub argmax_buf: Buffer,    // [B] u32
    // FP16 KV cache: per layer
    pub k_cache: Vec<Buffer>,  // [MAX_SEQ × kv_dim] half per layer
    pub v_cache: Vec<Buffer>,
}

const MAX_SEQ: u32 = 2048;

impl BatchState {
    pub fn new(model: &Model) -> Result<Self, String> {
        let bs = kernels_batch::BATCH_SIZE;
        let cfg = &model.cfg;

        // Compile batch shader library
        let src = kernels_batch::all_batch_kernels();
        let lib = model.device.new_library_with_source(&src)?;

        let pipe = |name: &str| -> Result<Pipeline, String> {
            let f = lib.get_function(name)?;
            model.device.new_compute_pipeline(&f)
        };

        let p_matmul_q4_0_f16 = pipe("matmul_q4_0_f16")?;
        let p_matmul_q4_0_mma = pipe("matmul_q4_0_mma")?;
        let p_matmul_q4_1_f16 = pipe("matmul_q4_1_f16")?;
        let p_matmul_q6k_f16 = pipe("matmul_q6k_f16")?;
        let p_embed_q4_0 = pipe("embed_batch_q4_0")?;
        let p_embed_q6k = pipe("embed_batch_q6k")?;
        let p_rmsnorm = pipe("rmsnorm_batch")?;
        let p_rope = pipe("rope_batch")?;
        let p_kv_store = pipe("kv_store_batch")?;
        let p_silu_mul = pipe("silu_mul_batch")?;
        let p_residual_add = pipe("residual_add_batch")?;
        let p_attn_causal = pipe("attention_batch_causal")?;
        let p_argmax = pipe("argmax_batch")?;

        let dev = &model.device;

        // FP16 buffers: 2 bytes per element
        // Use BATCH_PADDED (56) for allocation so MMA kernel can safely
        // read/write full 8-wide batch groups without OOB.
        let h = |n: u64| dev.new_buffer(n * 2);
        let bp = kernels_batch::BATCH_PADDED as u64;
        let hidden = cfg.hidden_dim as u64;
        let q_dim = model.q_dim as u64;
        let kv_dim = model.kv_dim as u64;
        let ffn = cfg.ffn_dim as u64;
        let vocab = cfg.vocab_size as u64;

        let tokens_buf = dev.new_buffer(bp * 4); // u32
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
        let argmax_buf = dev.new_buffer(bp * 4); // u32

        // FP16 KV cache
        let kv_size = MAX_SEQ as u64 * kv_dim * 2;
        let mut k_cache = Vec::with_capacity(cfg.n_layers as usize);
        let mut v_cache = Vec::with_capacity(cfg.n_layers as usize);
        for _ in 0..cfg.n_layers {
            k_cache.push(dev.new_buffer(kv_size));
            v_cache.push(dev.new_buffer(kv_size));
        }

        eprintln!("batch: B={bs}, FP16 buffers allocated, KV cache {:.1}MB",
            kv_size as f64 * 2.0 * cfg.n_layers as f64 / 1e6);

        Ok(BatchState {
            batch_size: bs,
            p_matmul_q4_0_f16, p_matmul_q4_0_mma, p_matmul_q4_1_f16, p_matmul_q6k_f16,
            p_embed_q4_0, p_embed_q6k,
            p_rmsnorm, p_rope, p_kv_store, p_silu_mul,
            p_residual_add, p_attn_causal, p_argmax,
            tokens_buf, x, norm_out, q, k, v, attn_out,
            gate, up, ffn_mid, logits, argmax_buf,
            k_cache, v_cache,
        })
    }

    /// Select the FP16 batch matmul pipeline for a given quantization type.
    pub fn matmul_pipeline(&self, dtype: GGMLType) -> &Pipeline {
        match dtype {
            GGMLType::Q4_0 => &self.p_matmul_q4_0_f16,
            GGMLType::Q4_1 => &self.p_matmul_q4_1_f16,
            GGMLType::Q6K => &self.p_matmul_q6k_f16,
            _ => panic!("unsupported batch matmul dtype: {:?}", dtype),
        }
    }
}

/// Run batch forward pass: n tokens at positions [start_pos..start_pos+n-1].
/// All GPU dispatches in a single compute encoder.
/// `tokens` slice must have at least `n` elements. `n` defaults to batch_size.
pub fn forward_batch(
    model: &Model,
    batch: &BatchState,
    tokens: &[u32],
    start_pos: u32,
) {
    forward_batch_n(model, batch, tokens, start_pos, batch.batch_size);
}

/// Variable-count batch forward: process `n` tokens (1..=batch_size).
pub fn forward_batch_n(
    model: &Model,
    batch: &BatchState,
    tokens: &[u32],
    start_pos: u32,
    n: u32,
) {
    let cfg = &model.cfg;
    let b = n;
    assert!(tokens.len() >= n as usize);
    assert!(n <= batch.batch_size);

    // Upload token IDs (only n active tokens)
    unsafe {
        let dst = batch.tokens_buf.contents() as *mut u32;
        std::ptr::copy_nonoverlapping(tokens.as_ptr(), dst, n as usize);
    }

    let cmd = model.queue.new_command_buffer();
    let enc = cmd.new_compute_encoder();

    // ── Embedding: n tokens → x[n × hidden_dim] ──
    dispatch_embed_batch(&enc, model, batch, b);

    // ── Transformer layers ──
    for layer in 0..cfg.n_layers as usize {
        let lo = &model.layers[layer];

        // Attention RMSNorm: x → norm_out
        dispatch_rmsnorm_batch(&enc, batch, &batch.x, &batch.norm_out,
            &model.weights, lo.attn_norm, cfg.hidden_dim, b);

        // Q/K/V batch matmul: norm_out → q, k, v (MMA for Q4_0, scalar for others)
        dispatch_matmul(&enc, batch, lo.attn_q_type,
            &model.weights, lo.attn_q, &batch.norm_out,
            &batch.q, cfg.hidden_dim, model.q_dim, b);
        dispatch_matmul(&enc, batch, lo.attn_k_type,
            &model.weights, lo.attn_k, &batch.norm_out,
            &batch.k, cfg.hidden_dim, model.kv_dim, b);
        dispatch_matmul(&enc, batch, lo.attn_v_type,
            &model.weights, lo.attn_v, &batch.norm_out,
            &batch.v, cfg.hidden_dim, model.kv_dim, b);

        // RoPE batch: Q and K with positions [start_pos..start_pos+n-1]
        dispatch_rope_batch(&enc, batch, &batch.q,
            model.head_dim, start_pos, cfg.n_heads, model.q_dim, b);
        dispatch_rope_batch(&enc, batch, &batch.k,
            model.head_dim, start_pos, cfg.n_kv_heads, model.kv_dim, b);

        // KV store batch
        dispatch_kv_store_batch(&enc, batch, model.kv_dim, layer, start_pos, b);

        // Attention batch (causal)
        dispatch_attention_batch(&enc, model, batch, layer, start_pos, b);

        // O projection + residual: x += W_o @ attn_out
        dispatch_matmul(&enc, batch, lo.attn_output_type,
            &model.weights, lo.attn_output, &batch.attn_out,
            &batch.norm_out, model.q_dim, cfg.hidden_dim, b);
        dispatch_residual_add(&enc, batch, &batch.x, &batch.norm_out,
            cfg.hidden_dim as u64 * b as u64);

        // FFN RMSNorm: x → norm_out
        dispatch_rmsnorm_batch(&enc, batch, &batch.x, &batch.norm_out,
            &model.weights, lo.ffn_norm, cfg.hidden_dim, b);

        // Gate + Up batch matmul (MMA for Q4_0, scalar for others)
        dispatch_matmul(&enc, batch, lo.ffn_gate_type,
            &model.weights, lo.ffn_gate, &batch.norm_out,
            &batch.gate, cfg.hidden_dim, cfg.ffn_dim, b);
        dispatch_matmul(&enc, batch, lo.ffn_up_type,
            &model.weights, lo.ffn_up, &batch.norm_out,
            &batch.up, cfg.hidden_dim, cfg.ffn_dim, b);

        // SiLU(gate) * up → ffn_mid
        dispatch_silu_batch(&enc, batch, cfg.ffn_dim as u64 * b as u64);

        // Down projection + residual: x += W_down @ ffn_mid
        dispatch_matmul(&enc, batch, lo.ffn_down_type,
            &model.weights, lo.ffn_down, &batch.ffn_mid,
            &batch.norm_out, cfg.ffn_dim, cfg.hidden_dim, b);
        dispatch_residual_add(&enc, batch, &batch.x, &batch.norm_out,
            cfg.hidden_dim as u64 * b as u64);
    }

    // Final RMSNorm
    dispatch_rmsnorm_batch(&enc, batch, &batch.x, &batch.norm_out,
        &model.weights, model.out_norm_off, cfg.hidden_dim, b);

    // Output projection (MMA for Q4_0, scalar for Q6K tied weights, etc.)
    dispatch_matmul(&enc, batch, model.output_type,
        &model.weights, model.output_off, &batch.norm_out,
        &batch.logits, cfg.hidden_dim, cfg.vocab_size, b);

    // Argmax batch
    dispatch_argmax_batch(&enc, batch, cfg.vocab_size, b);

    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

/// Read batch argmax results from GPU.
pub fn argmax_batch(batch: &BatchState) -> Vec<u32> {
    let ptr = batch.argmax_buf.contents() as *const u32;
    (0..batch.batch_size as usize)
        .map(|i| unsafe { *ptr.add(i) })
        .collect()
}

/// Read single argmax result (batch position 0).
pub fn argmax_one(batch: &BatchState) -> u32 {
    unsafe { *(batch.argmax_buf.contents() as *const u32) }
}

/// Prefill a prompt using batch forward pass.
/// Processes prompt in chunks of batch_size, filling the FP16 KV cache.
/// Returns prompt length (= next decode position).
pub fn prefill(model: &Model, batch: &BatchState, prompt: &[u32]) -> u32 {
    let bs = batch.batch_size as usize;
    let mut pos = 0u32;

    // Full chunks
    let full_chunks = prompt.len() / bs;
    for c in 0..full_chunks {
        let chunk = &prompt[c * bs..(c + 1) * bs];
        forward_batch_n(model, batch, chunk, pos, batch.batch_size);
        pos += batch.batch_size;
    }

    // Remainder (partial chunk)
    let rem = prompt.len() % bs;
    if rem > 0 {
        forward_batch_n(model, batch, &prompt[full_chunks * bs..], pos, rem as u32);
        pos += rem as u32;
    }

    pos
}

/// Single-token decode step using batch FP16 infrastructure.
/// Processes 1 token at the given position using the batch KV cache.
/// Returns the predicted next token.
pub fn decode_step(model: &Model, batch: &BatchState, token: u32, position: u32) -> u32 {
    forward_batch_n(model, batch, &[token], position, 1);
    argmax_one(batch)
}

/// Complete generation: prefill prompt + greedy decode.
/// Returns generated token IDs (not including prompt).
pub fn generate(
    model: &Model,
    batch: &BatchState,
    prompt: &[u32],
    max_tokens: u32,
    eos_token: u32,
) -> Vec<u32> {
    let pos = prefill(model, batch, prompt);
    // First decode token comes from last prefill step
    let mut next = argmax_one(batch);
    let mut generated = Vec::with_capacity(max_tokens as usize);

    for i in 0..max_tokens {
        generated.push(next);
        if next == eos_token { break; }
        next = decode_step(model, batch, next, pos + i);
    }
    generated
}

fn dispatch_embed_batch(enc: &ComputeEncoder, model: &Model, batch: &BatchState, bs: u32) {
    let dim = model.cfg.hidden_dim;
    // Select embed pipeline based on model's embedding quantization type
    let pipe = match model.embd_type {
        crate::gguf::GGMLType::Q6K => &batch.p_embed_q6k,
        _ => &batch.p_embed_q4_0,
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

fn dispatch_rmsnorm_batch(
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
    // One threadgroup per batch item
    enc.dispatch_threadgroups(
        MTLSize::new(bs as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

fn dispatch_batch_matmul_f16(
    enc: &ComputeEncoder, batch: &BatchState, pipeline: &Pipeline,
    weights: &Buffer, weight_off: u64, x: &Buffer,
    y: &Buffer, cols: u32, rows: u32, b: u32,
) {
    enc.set_pipeline(pipeline);
    enc.set_buffer(0, weights, weight_off);
    enc.set_buffer(1, x, 0);
    enc.set_buffer(2, y, 0);
    enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
    enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &b as *const u32 as *const c_void, 4);
    let grid_x = (rows + 7) / 8;
    // N=BATCH_SIZE accumulators handle all batch items; grid_y=1 always
    enc.dispatch_threadgroups(
        MTLSize::new(grid_x as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// MMA v2 dispatch: grid = (ceil(rows/32), 1), TG = 128.
/// Kernel handles all batch groups internally (amortized weight dequant).
fn dispatch_batch_matmul_mma(
    enc: &ComputeEncoder, batch: &BatchState,
    weights: &Buffer, weight_off: u64, x: &Buffer,
    y: &Buffer, cols: u32, rows: u32, b: u32,
) {
    enc.set_pipeline(&batch.p_matmul_q4_0_mma);
    enc.set_buffer(0, weights, weight_off);
    enc.set_buffer(1, x, 0);
    enc.set_buffer(2, y, 0);
    enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
    enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &b as *const u32 as *const c_void, 4);
    let grid_x = (rows + 31) / 32;
    enc.dispatch_threadgroups(
        MTLSize::new(grid_x as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
}

/// Unified matmul dispatch: uses MMA for Q4_0 batch, scalar for others/single.
fn dispatch_matmul(
    enc: &ComputeEncoder, batch: &BatchState,
    dtype: GGMLType,
    weights: &Buffer, weight_off: u64, x: &Buffer,
    y: &Buffer, cols: u32, rows: u32, b: u32,
) {
    if dtype == GGMLType::Q4_0 && b > 1 {
        dispatch_batch_matmul_mma(enc, batch, weights, weight_off, x, y, cols, rows, b);
    } else {
        dispatch_batch_matmul_f16(
            enc, batch, batch.matmul_pipeline(dtype),
            weights, weight_off, x, y, cols, rows, b,
        );
    }
}

fn dispatch_rope_batch(
    enc: &ComputeEncoder, batch: &BatchState, vec: &Buffer,
    head_dim: u32, start_pos: u32, n_heads: u32, total_dim: u32, bs: u32,
) {
    let freq_base: f32 = 500000.0; // Llama 3.2
    let n_pairs = n_heads * head_dim / 2;
    enc.set_pipeline(&batch.p_rope);
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

fn dispatch_kv_store_batch(
    enc: &ComputeEncoder, batch: &BatchState, kv_dim_val: u32,
    layer: usize, start_pos: u32, bs: u32,
) {
    enc.set_pipeline(&batch.p_kv_store);
    enc.set_buffer(0, &batch.k, 0);
    enc.set_buffer(1, &batch.v, 0);
    enc.set_buffer(2, &batch.k_cache[layer], 0);
    enc.set_buffer(3, &batch.v_cache[layer], 0);
    enc.set_bytes(4, &kv_dim_val as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &start_pos as *const u32 as *const c_void, 4);
    enc.set_bytes(6, &bs as *const u32 as *const c_void, 4);
    enc.dispatch_threads(
        MTLSize::new(kv_dim_val as u64, bs as u64, 1),
        MTLSize::new(64, 1, 1),
    );
}

fn dispatch_attention_batch(
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
    // Grid: (n_heads, batch)
    enc.dispatch_threadgroups(
        MTLSize::new(n_heads as u64, bs as u64, 1),
        MTLSize::new(128, 1, 1),
    );
}

fn dispatch_silu_batch(enc: &ComputeEncoder, batch: &BatchState, n_elems: u64) {
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

fn dispatch_argmax_batch(enc: &ComputeEncoder, batch: &BatchState, vocab_size: u32, bs: u32) {
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

//! Transformer forward pass (Qwen3 / Llama / generic decoder-only).
//! Single encoder per forward pass — all dispatches in one encoder.
//! Metal handles implicit barriers between dependent dispatches.

use crate::gpu::*;
use crate::model::Model;
use crate::gguf::GGMLType;
use std::ffi::c_void;

/// Run one forward pass: token_id at position → logits buffer populated.
/// All GPU dispatches encoded in a SINGLE compute encoder.
/// Metal inserts implicit barriers between dispatches that share buffers.
pub fn forward(model: &Model, token_id: u32, position: u32) {
    let cfg = &model.cfg;
    let cmd = model.queue.new_command_buffer();
    let enc = cmd.new_compute_encoder();

    // ── Embedding lookup ──
    dispatch_embed(&enc, model, token_id);

    // ── Transformer layers ──
    for layer in 0..cfg.n_layers as usize {
        let lo = &model.layers[layer];

        // Attention RMSNorm
        dispatch_rmsnorm(&enc, model, &model.x, &model.norm_out, lo.attn_norm);

        // Q/K/V projections
        dispatch_qkv(&enc, model, lo);

        // QK-norm (Qwen3 only)
        if model.has_qk_norm {
            dispatch_qk_norm(&enc, model, lo);
        }

        // RoPE on Q and K
        dispatch_rope(&enc, model, position);

        // Store K/V to cache
        dispatch_kv_store(&enc, model, layer, position);

        // Attention
        dispatch_attention(&enc, model, layer, position + 1);

        // Output projection + residual: attn_out → o, x += o
        dispatch_matvec_add(&enc, model, lo.attn_output_type, lo.attn_output,
            &model.attn_out, &model.o, model.q_dim, cfg.hidden_dim,
            &model.x);

        // FFN RMSNorm
        dispatch_rmsnorm(&enc, model, &model.x, &model.norm_out, lo.ffn_norm);

        // FFN gate + up
        dispatch_gate_up(&enc, model, lo);

        // SiLU(gate) * up
        dispatch_silu(&enc, model, cfg.ffn_dim);

        // FFN down + residual: ffn_mid → down, x += down
        dispatch_matvec_add(&enc, model, lo.ffn_down_type, lo.ffn_down,
            &model.ffn_mid, &model.down, cfg.ffn_dim, cfg.hidden_dim,
            &model.x);
    }

    // Final RMSNorm
    dispatch_rmsnorm(&enc, model, &model.x, &model.norm_out, model.out_norm_off);

    // Output projection: [vocab_size × hidden_dim] @ norm_out → logits
    dispatch_matvec(&enc, model, model.output_type, model.output_off,
        &model.norm_out, &model.logits, cfg.hidden_dim, cfg.vocab_size);

    // GPU argmax
    dispatch_argmax(&enc, model);

    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

/// Greedy decode: read GPU argmax result (4 bytes instead of scanning 128K+ floats).
pub fn argmax(model: &Model) -> u32 {
    unsafe { *(model.argmax_buf.contents() as *const u32) }
}

/// Generate n tokens starting from a prompt (single token for now).
pub fn generate(model: &Model, prompt_token: u32, n_tokens: u32) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(n_tokens as usize + 1);
    tokens.push(prompt_token);
    for pos in 0..n_tokens {
        forward(model, *tokens.last().unwrap(), pos);
        tokens.push(argmax(model));
    }
    tokens
}

/// Generate with per-token timing for the first N tokens.
pub fn generate_timed(model: &Model, prompt_token: u32, n_tokens: u32) -> Vec<u32> {
    use std::time::Instant;
    let mut tokens = Vec::with_capacity(n_tokens as usize + 1);
    tokens.push(prompt_token);
    let print_n = 20.min(n_tokens);
    for pos in 0..n_tokens {
        let t0 = Instant::now();
        forward(model, *tokens.last().unwrap(), pos);
        let fwd_us = t0.elapsed().as_micros();
        tokens.push(argmax(model));
        if pos < print_n {
            eprintln!("    token[{pos:>3}] fwd={fwd_us:>5}µs  seq_len={}", pos + 1);
        }
    }
    tokens
}

/// Per-operation timing: separate command buffer per op category.
/// OVERCOUNTS total time (commit+wait overhead per group), but shows relative cost.
pub fn forward_timed(model: &Model, token_id: u32, position: u32) -> TimingBreakdown {
    use std::time::Instant;
    let cfg = &model.cfg;
    let mut t = TimingBreakdown::default();

    // Embed
    let t0 = Instant::now();
    {
        let cmd = model.queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        dispatch_embed(&enc, model, token_id);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    t.embed_us = t0.elapsed().as_micros() as u64;

    // Layers
    for layer in 0..cfg.n_layers as usize {
        let lo = &model.layers[layer];

        // Attn norm
        let t0 = Instant::now();
        {
            let cmd = model.queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            dispatch_rmsnorm(&enc, model, &model.x, &model.norm_out, lo.attn_norm);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t.rmsnorm_us += t0.elapsed().as_micros() as u64;

        // QKV
        let t0 = Instant::now();
        {
            let cmd = model.queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            dispatch_qkv(&enc, model, lo);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t.qkv_us += t0.elapsed().as_micros() as u64;

        // QK-norm
        if model.has_qk_norm {
            let t0 = Instant::now();
            {
                let cmd = model.queue.new_command_buffer();
                let enc = cmd.new_compute_encoder();
                dispatch_qk_norm(&enc, model, lo);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
            t.qk_norm_us += t0.elapsed().as_micros() as u64;
        }

        // RoPE
        let t0 = Instant::now();
        {
            let cmd = model.queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            dispatch_rope(&enc, model, position);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t.rope_us += t0.elapsed().as_micros() as u64;

        // KV store
        let t0 = Instant::now();
        {
            let cmd = model.queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            dispatch_kv_store(&enc, model, layer, position);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t.kv_store_us += t0.elapsed().as_micros() as u64;

        // Attention
        let t0 = Instant::now();
        {
            let cmd = model.queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            dispatch_attention(&enc, model, layer, position + 1);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t.attn_us += t0.elapsed().as_micros() as u64;

        // Attn output + residual
        let t0 = Instant::now();
        {
            let cmd = model.queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            dispatch_matvec_add(&enc, model, lo.attn_output_type, lo.attn_output,
                &model.attn_out, &model.o, model.q_dim, cfg.hidden_dim, &model.x);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t.attn_proj_us += t0.elapsed().as_micros() as u64;

        // FFN norm
        let t0 = Instant::now();
        {
            let cmd = model.queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            dispatch_rmsnorm(&enc, model, &model.x, &model.norm_out, lo.ffn_norm);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t.rmsnorm_us += t0.elapsed().as_micros() as u64;

        // FFN gate+up
        let t0 = Instant::now();
        {
            let cmd = model.queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            dispatch_gate_up(&enc, model, lo);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t.ffn_proj_us += t0.elapsed().as_micros() as u64;

        // SiLU
        let t0 = Instant::now();
        {
            let cmd = model.queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            dispatch_silu(&enc, model, cfg.ffn_dim);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t.silu_us += t0.elapsed().as_micros() as u64;

        // FFN down + residual
        let t0 = Instant::now();
        {
            let cmd = model.queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            dispatch_matvec_add(&enc, model, lo.ffn_down_type, lo.ffn_down,
                &model.ffn_mid, &model.down, cfg.ffn_dim, cfg.hidden_dim, &model.x);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        t.ffn_down_us += t0.elapsed().as_micros() as u64;
    }

    // Final norm
    let t0 = Instant::now();
    {
        let cmd = model.queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        dispatch_rmsnorm(&enc, model, &model.x, &model.norm_out, model.out_norm_off);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    t.rmsnorm_us += t0.elapsed().as_micros() as u64;

    // Output projection
    let t0 = Instant::now();
    {
        let cmd = model.queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        dispatch_matvec(&enc, model, model.output_type, model.output_off,
            &model.norm_out, &model.logits, cfg.hidden_dim, cfg.vocab_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    t.output_proj_us = t0.elapsed().as_micros() as u64;

    // Argmax
    let t0 = Instant::now();
    {
        let cmd = model.queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        dispatch_argmax(&enc, model);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    t.argmax_us = t0.elapsed().as_micros() as u64;

    t
}

#[derive(Default)]
pub struct TimingBreakdown {
    pub embed_us: u64,
    pub rmsnorm_us: u64,
    pub qkv_us: u64,
    pub qk_norm_us: u64,
    pub rope_us: u64,
    pub kv_store_us: u64,
    pub attn_us: u64,
    pub attn_proj_us: u64,
    pub ffn_proj_us: u64,
    pub silu_us: u64,
    pub ffn_down_us: u64,
    pub output_proj_us: u64,
    pub argmax_us: u64,
}

impl TimingBreakdown {
    pub fn total_us(&self) -> u64 {
        self.embed_us + self.rmsnorm_us + self.qkv_us + self.qk_norm_us
            + self.rope_us + self.kv_store_us + self.attn_us + self.attn_proj_us
            + self.ffn_proj_us + self.silu_us + self.ffn_down_us
            + self.output_proj_us + self.argmax_us
    }

    pub fn print(&self) {
        let total = self.total_us();
        let pct = |v: u64| if total > 0 { v as f64 / total as f64 * 100.0 } else { 0.0 };
        eprintln!("  ┌─ Per-operation timing (separate cmd bufs, overcounts total) ─┐");
        eprintln!("  │ embed:       {:>6}µs  ({:>5.1}%)                             │", self.embed_us, pct(self.embed_us));
        eprintln!("  │ rmsnorm:     {:>6}µs  ({:>5.1}%)  [all {} calls]              │", self.rmsnorm_us, pct(self.rmsnorm_us), "33");
        eprintln!("  │ qkv matvec:  {:>6}µs  ({:>5.1}%)                             │", self.qkv_us, pct(self.qkv_us));
        if self.qk_norm_us > 0 {
            eprintln!("  │ qk_norm:     {:>6}µs  ({:>5.1}%)                             │", self.qk_norm_us, pct(self.qk_norm_us));
        }
        eprintln!("  │ rope:        {:>6}µs  ({:>5.1}%)                             │", self.rope_us, pct(self.rope_us));
        eprintln!("  │ kv_store:    {:>6}µs  ({:>5.1}%)                             │", self.kv_store_us, pct(self.kv_store_us));
        eprintln!("  │ attention:   {:>6}µs  ({:>5.1}%)                             │", self.attn_us, pct(self.attn_us));
        eprintln!("  │ attn_proj:   {:>6}µs  ({:>5.1}%)                             │", self.attn_proj_us, pct(self.attn_proj_us));
        eprintln!("  │ ffn gate+up: {:>6}µs  ({:>5.1}%)                             │", self.ffn_proj_us, pct(self.ffn_proj_us));
        eprintln!("  │ silu:        {:>6}µs  ({:>5.1}%)                             │", self.silu_us, pct(self.silu_us));
        eprintln!("  │ ffn_down:    {:>6}µs  ({:>5.1}%)                             │", self.ffn_down_us, pct(self.ffn_down_us));
        eprintln!("  │ output_proj: {:>6}µs  ({:>5.1}%)                             │", self.output_proj_us, pct(self.output_proj_us));
        eprintln!("  │ argmax:      {:>6}µs  ({:>5.1}%)                             │", self.argmax_us, pct(self.argmax_us));
        eprintln!("  │ TOTAL:       {:>6}µs  (overcounted due to cmd overhead)      │", total);
        eprintln!("  └──────────────────────────────────────────────────────────────┘");
    }
}

// ── Threadgroup sizing ──────────────────────────────────────────────

/// Compute optimal threadgroup size for a matvec kernel.
/// blocks_per_row = cols / block_size. Align up to simdgroup (32), cap at 256.
fn matvec_tgs(dtype: GGMLType, cols: u32) -> u64 {
    let bs = dtype.block_size() as u32;
    let blocks = cols / bs;
    let raw = (blocks + 31) & !31;
    raw.max(32).min(256) as u64
}

// ── Dispatch helpers (no encoder create/end — caller manages encoder) ──

fn dispatch_embed(enc: &ComputeEncoder, m: &Model, token_id: u32) {
    enc.set_pipeline(m.embed_pipeline(m.embd_type));
    enc.set_buffer(0, &m.weights, m.embd_off);
    enc.set_buffer(1, &m.x, 0);
    let dim = m.cfg.hidden_dim;
    enc.set_bytes(2, &dim as *const u32 as *const c_void, 4);
    enc.set_bytes(3, &token_id as *const u32 as *const c_void, 4);
    enc.dispatch_threads(
        MTLSize::new(dim as u64, 1, 1),
        MTLSize::new(256.min(dim as u64), 1, 1),
    );
}

fn dispatch_rmsnorm(enc: &ComputeEncoder, m: &Model, x: &Buffer, y: &Buffer, weight_off: u64) {
    enc.set_pipeline(&m.p_rmsnorm);
    enc.set_buffer(0, x, 0);
    enc.set_buffer(1, y, 0);
    enc.set_buffer(2, &m.weights, weight_off);
    let dim = m.cfg.hidden_dim;
    let eps = m.cfg.rms_norm_eps;
    enc.set_bytes(3, &dim as *const u32 as *const c_void, 4);
    enc.set_bytes(4, &eps as *const f32 as *const c_void, 4);
    enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
}

fn dispatch_qkv(enc: &ComputeEncoder, m: &Model, lo: &crate::model::LayerOffsets) {
    let h = m.cfg.hidden_dim;
    // Q
    enc.set_pipeline(m.matvec_pipeline(lo.attn_q_type));
    enc.set_buffer(0, &m.weights, lo.attn_q);
    enc.set_buffer(1, &m.norm_out, 0);
    enc.set_buffer(2, &m.q, 0);
    enc.set_bytes(3, &h as *const u32 as *const c_void, 4);
    let q_tgs = MTLSize::new(matvec_tgs(lo.attn_q_type, h), 1, 1);
    enc.dispatch_threadgroups(MTLSize::new(m.q_dim as u64, 1, 1), q_tgs);
    // K
    if lo.attn_k_type != lo.attn_q_type {
        enc.set_pipeline(m.matvec_pipeline(lo.attn_k_type));
    }
    enc.set_buffer(0, &m.weights, lo.attn_k);
    enc.set_buffer(2, &m.k, 0);
    let k_tgs = MTLSize::new(matvec_tgs(lo.attn_k_type, h), 1, 1);
    enc.dispatch_threadgroups(MTLSize::new(m.kv_dim as u64, 1, 1), k_tgs);
    // V
    if lo.attn_v_type != lo.attn_k_type {
        enc.set_pipeline(m.matvec_pipeline(lo.attn_v_type));
    }
    enc.set_buffer(0, &m.weights, lo.attn_v);
    enc.set_buffer(2, &m.v, 0);
    enc.dispatch_threadgroups(MTLSize::new(m.kv_dim as u64, 1, 1), k_tgs);
}

fn dispatch_qk_norm(enc: &ComputeEncoder, m: &Model, lo: &crate::model::LayerOffsets) {
    let hd = m.head_dim;
    let eps = m.cfg.rms_norm_eps;
    let q_norm_off = lo.attn_q_norm.unwrap();
    let k_norm_off = lo.attn_k_norm.unwrap();
    enc.set_pipeline(&m.p_rmsnorm);
    // Q heads — all dispatches in same encoder (no encoder creation overhead)
    for head in 0..m.cfg.n_heads {
        enc.set_buffer(0, &m.q, (head * hd * 4) as u64);
        enc.set_buffer(1, &m.q, (head * hd * 4) as u64);
        enc.set_buffer(2, &m.weights, q_norm_off);
        enc.set_bytes(3, &hd as *const u32 as *const c_void, 4);
        enc.set_bytes(4, &eps as *const f32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
    }
    // K heads
    for head in 0..m.cfg.n_kv_heads {
        enc.set_buffer(0, &m.k, (head * hd * 4) as u64);
        enc.set_buffer(1, &m.k, (head * hd * 4) as u64);
        enc.set_buffer(2, &m.weights, k_norm_off);
        enc.set_bytes(3, &hd as *const u32 as *const c_void, 4);
        enc.set_bytes(4, &eps as *const f32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
    }
}

fn dispatch_matvec(
    enc: &ComputeEncoder, m: &Model, dtype: GGMLType, weight_off: u64,
    x: &Buffer, y: &Buffer, cols: u32, rows: u32,
) {
    enc.set_pipeline(m.matvec_pipeline(dtype));
    enc.set_buffer(0, &m.weights, weight_off);
    enc.set_buffer(1, x, 0);
    enc.set_buffer(2, y, 0);
    enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
    let tgs = MTLSize::new(matvec_tgs(dtype, cols), 1, 1);
    enc.dispatch_threadgroups(MTLSize::new(rows as u64, 1, 1), tgs);
}

fn dispatch_rope(enc: &ComputeEncoder, m: &Model, position: u32) {
    let hd = m.head_dim;
    let fb = m.cfg.rope_freq_base;
    enc.set_pipeline(&m.p_rope);
    enc.set_bytes(1, &hd as *const u32 as *const c_void, 4);
    enc.set_bytes(2, &position as *const u32 as *const c_void, 4);
    enc.set_bytes(3, &fb as *const f32 as *const c_void, 4);
    // Q
    enc.set_buffer(0, &m.q, 0);
    let n_pairs_q = m.cfg.n_heads * hd / 2;
    enc.dispatch_threads(MTLSize::new(n_pairs_q as u64, 1, 1), MTLSize::new(64, 1, 1));
    // K
    enc.set_buffer(0, &m.k, 0);
    let n_pairs_k = m.cfg.n_kv_heads * hd / 2;
    enc.dispatch_threads(MTLSize::new(n_pairs_k as u64, 1, 1), MTLSize::new(64, 1, 1));
}

fn dispatch_kv_store(enc: &ComputeEncoder, m: &Model, layer: usize, position: u32) {
    enc.set_pipeline(&m.p_kv_store);
    enc.set_buffer(0, &m.k, 0);
    enc.set_buffer(1, &m.v, 0);
    enc.set_buffer(2, &m.k_cache[layer], 0);
    enc.set_buffer(3, &m.v_cache[layer], 0);
    let kv_dim = m.kv_dim;
    enc.set_bytes(4, &kv_dim as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &position as *const u32 as *const c_void, 4);
    enc.dispatch_threads(MTLSize::new(kv_dim as u64, 1, 1), MTLSize::new(64, 1, 1));
}

fn dispatch_attention(enc: &ComputeEncoder, m: &Model, layer: usize, seq_len: u32) {
    enc.set_pipeline(&m.p_attn);
    enc.set_buffer(0, &m.q, 0);
    enc.set_buffer(1, &m.k_cache[layer], 0);
    enc.set_buffer(2, &m.v_cache[layer], 0);
    enc.set_buffer(3, &m.attn_out, 0);
    let hd = m.head_dim;
    let n_kv = m.cfg.n_kv_heads;
    let gqa = m.gqa_ratio;
    let kv_dim = m.kv_dim;
    enc.set_bytes(4, &seq_len as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &hd as *const u32 as *const c_void, 4);
    enc.set_bytes(6, &n_kv as *const u32 as *const c_void, 4);
    enc.set_bytes(7, &gqa as *const u32 as *const c_void, 4);
    enc.set_bytes(8, &kv_dim as *const u32 as *const c_void, 4);
    enc.dispatch_threadgroups(
        MTLSize::new(m.cfg.n_heads as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
}

fn dispatch_silu(enc: &ComputeEncoder, m: &Model, dim: u32) {
    enc.set_pipeline(&m.p_silu);
    enc.set_buffer(0, &m.gate, 0);
    enc.set_buffer(1, &m.up, 0);
    enc.set_buffer(2, &m.ffn_mid, 0);
    enc.dispatch_threads(MTLSize::new(dim as u64, 1, 1), MTLSize::new(256, 1, 1));
}

fn dispatch_matvec_add(
    enc: &ComputeEncoder, m: &Model, dtype: GGMLType, weight_off: u64,
    x: &Buffer, _y: &Buffer, cols: u32, rows: u32, residual: &Buffer,
) {
    // Fused matvec + residual add: residual[gid] += dot(W[gid], x)
    enc.set_pipeline(m.matvec_add_pipeline(dtype));
    enc.set_buffer(0, &m.weights, weight_off);
    enc.set_buffer(1, x, 0);
    enc.set_buffer(2, residual, 0);
    enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
    let tgs = MTLSize::new(matvec_tgs(dtype, cols), 1, 1);
    enc.dispatch_threadgroups(MTLSize::new(rows as u64, 1, 1), tgs);
}

fn dispatch_gate_up(enc: &ComputeEncoder, m: &Model, lo: &crate::model::LayerOffsets) {
    let h = m.cfg.hidden_dim;
    let grid = MTLSize::new(m.cfg.ffn_dim as u64, 1, 1);
    // Gate
    enc.set_pipeline(m.matvec_pipeline(lo.ffn_gate_type));
    enc.set_buffer(0, &m.weights, lo.ffn_gate);
    enc.set_buffer(1, &m.norm_out, 0);
    enc.set_buffer(2, &m.gate, 0);
    enc.set_bytes(3, &h as *const u32 as *const c_void, 4);
    let gate_tgs = MTLSize::new(matvec_tgs(lo.ffn_gate_type, h), 1, 1);
    enc.dispatch_threadgroups(grid, gate_tgs);
    // Up
    if lo.ffn_up_type != lo.ffn_gate_type {
        enc.set_pipeline(m.matvec_pipeline(lo.ffn_up_type));
    }
    enc.set_buffer(0, &m.weights, lo.ffn_up);
    enc.set_buffer(2, &m.up, 0);
    let up_tgs = MTLSize::new(matvec_tgs(lo.ffn_up_type, h), 1, 1);
    enc.dispatch_threadgroups(grid, up_tgs);
}

fn dispatch_argmax(enc: &ComputeEncoder, m: &Model) {
    enc.set_pipeline(&m.p_argmax);
    enc.set_buffer(0, &m.logits, 0);
    enc.set_buffer(1, &m.argmax_buf, 0);
    let n = m.cfg.vocab_size;
    enc.set_bytes(2, &n as *const u32 as *const c_void, 4);
    enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
}

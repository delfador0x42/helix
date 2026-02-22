//! Transformer forward pass for Qwen3. Single command buffer per token.
//! All operations (matvec, rmsnorm, rope, attention, silu) encoded into one
//! command buffer with implicit barriers between compute encoders.

use crate::gpu::*;
use crate::model::Model;
use crate::gguf::GGMLType;
use std::ffi::c_void;

/// Run one forward pass: token_id at position → logits buffer populated.
/// Returns nothing; read logits from model.logits buffer.
pub fn forward(model: &Model, token_id: u32, position: u32) {
    let cfg = &model.cfg;
    let cmd = model.queue.new_command_buffer();

    // ── Embedding lookup ──
    encode_embed(&cmd, model, token_id);

    // ── Transformer layers ──
    for layer in 0..cfg.n_layers as usize {
        let lo = &model.layers[layer];

        // Attention RMSNorm
        encode_rmsnorm(&cmd, model, &model.x, &model.norm_out, lo.attn_norm);

        // Q/K/V projections (same input, different outputs — one encoder)
        encode_qkv(&cmd, model, lo);

        // QK-norm (per-head RMSNorm on Q and K)
        encode_qk_norm(&cmd, model, lo);

        // RoPE on Q and K
        encode_rope(&cmd, model, position);

        // Store K/V to cache
        encode_kv_store(&cmd, model, layer, position);

        // Attention
        encode_attention(&cmd, model, layer, position + 1);

        // Output projection: attn_out → o
        encode_matvec(&cmd, model, lo.attn_output_type, lo.attn_output,
            &model.attn_out, &model.o, model.q_dim, cfg.hidden_dim);

        // Residual: x += o
        encode_add(&cmd, model, &model.x, &model.o, cfg.hidden_dim);

        // FFN RMSNorm
        encode_rmsnorm(&cmd, model, &model.x, &model.norm_out, lo.ffn_norm);

        // FFN gate + up (same input, different outputs)
        encode_matvec(&cmd, model, lo.ffn_gate_type, lo.ffn_gate,
            &model.norm_out, &model.gate, cfg.hidden_dim, cfg.ffn_dim);
        encode_matvec(&cmd, model, lo.ffn_up_type, lo.ffn_up,
            &model.norm_out, &model.up, cfg.hidden_dim, cfg.ffn_dim);

        // SiLU(gate) * up
        encode_silu(&cmd, model, cfg.ffn_dim);

        // FFN down projection
        encode_matvec(&cmd, model, lo.ffn_down_type, lo.ffn_down,
            &model.ffn_mid, &model.down, cfg.ffn_dim, cfg.hidden_dim);

        // Residual: x += down
        encode_add(&cmd, model, &model.x, &model.down, cfg.hidden_dim);
    }

    // Final RMSNorm
    encode_rmsnorm(&cmd, model, &model.x, &model.norm_out, model.out_norm_off);

    // Output projection (tied weights = token_embd.weight)
    // This is a Q6K matvec: [vocab_size × hidden_dim] @ norm_out → logits
    encode_matvec(&cmd, model, GGMLType::Q6K, model.embd_off,
        &model.norm_out, &model.logits, cfg.hidden_dim, cfg.vocab_size);

    cmd.commit();
    cmd.wait_until_completed();
}

/// Greedy decode: argmax over logits buffer (CPU-side, after forward).
pub fn argmax(model: &Model) -> u32 {
    let n = model.cfg.vocab_size as usize;
    let ptr = model.logits.contents() as *const f32;
    let mut best_id = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for i in 0..n {
        let v = unsafe { *ptr.add(i) };
        if v > best_val { best_val = v; best_id = i as u32; }
    }
    best_id
}

/// Generate n tokens starting from a prompt (single token for now).
pub fn generate(model: &Model, prompt_token: u32, n_tokens: u32) -> Vec<u32> {
    let mut tokens = vec![prompt_token];
    for pos in 0..n_tokens {
        forward(model, *tokens.last().unwrap(), pos);
        let next = argmax(model);
        tokens.push(next);
    }
    tokens
}

// ── Encoder helpers ──────────────────────────────────────────────────

fn encode_embed(cmd: &CommandBuffer, m: &Model, token_id: u32) {
    let enc = cmd.new_compute_encoder();
    enc.set_pipeline(&m.p_embed);
    enc.set_buffer(0, &m.weights, m.embd_off);
    enc.set_buffer(1, &m.x, 0);
    let dim = m.cfg.hidden_dim;
    enc.set_bytes(2, &dim as *const u32 as *const c_void, 4);
    enc.set_bytes(3, &token_id as *const u32 as *const c_void, 4);
    let tg = MTLSize::new(dim as u64, 1, 1);
    let tgs = MTLSize::new(256.min(dim as u64), 1, 1);
    enc.dispatch_threads(tg, tgs);
    enc.end_encoding();
}

fn encode_rmsnorm(cmd: &CommandBuffer, m: &Model, x: &Buffer, y: &Buffer, weight_off: u64) {
    let enc = cmd.new_compute_encoder();
    enc.set_pipeline(&m.p_rmsnorm);
    enc.set_buffer(0, x, 0);
    enc.set_buffer(1, y, 0);
    enc.set_buffer(2, &m.weights, weight_off);
    let dim = m.cfg.hidden_dim;
    let eps = m.cfg.rms_norm_eps;
    enc.set_bytes(3, &dim as *const u32 as *const c_void, 4);
    enc.set_bytes(4, &eps as *const f32 as *const c_void, 4);
    enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
}

fn encode_qkv(cmd: &CommandBuffer, m: &Model, lo: &crate::model::LayerOffsets) {
    let h = m.cfg.hidden_dim;
    // Q: norm_out → q  (output rows = q_dim)
    encode_matvec(cmd, m, lo.attn_q_type, lo.attn_q, &m.norm_out, &m.q, h, m.q_dim);
    // K: norm_out → k  (output rows = kv_dim)
    encode_matvec(cmd, m, lo.attn_k_type, lo.attn_k, &m.norm_out, &m.k, h, m.kv_dim);
    // V: norm_out → v  (output rows = kv_dim)
    encode_matvec(cmd, m, lo.attn_v_type, lo.attn_v, &m.norm_out, &m.v, h, m.kv_dim);
}

fn encode_matvec(
    cmd: &CommandBuffer, m: &Model, dtype: GGMLType, weight_off: u64,
    x: &Buffer, y: &Buffer, cols: u32, rows: u32,
) {
    let enc = cmd.new_compute_encoder();
    enc.set_pipeline(m.matvec_pipeline(dtype));
    enc.set_buffer(0, &m.weights, weight_off);
    enc.set_buffer(1, x, 0);
    enc.set_buffer(2, y, 0);
    enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
    let grid = MTLSize::new(rows as u64, 1, 1);
    let tgs = MTLSize::new(128, 1, 1); // 128 threads per row reduction
    enc.dispatch_threadgroups(grid, tgs);
    enc.end_encoding();
}

fn encode_qk_norm(cmd: &CommandBuffer, m: &Model, lo: &crate::model::LayerOffsets) {
    // Per-head RMSNorm on Q (16 heads of head_dim)
    let hd = m.head_dim;
    let eps = m.cfg.rms_norm_eps;
    for head in 0..m.cfg.n_heads {
        let enc = cmd.new_compute_encoder();
        enc.set_pipeline(&m.p_rmsnorm);
        enc.set_buffer(0, &m.q, (head * hd * 4) as u64);
        enc.set_buffer(1, &m.q, (head * hd * 4) as u64);
        enc.set_buffer(2, &m.weights, lo.attn_q_norm);
        enc.set_bytes(3, &hd as *const u32 as *const c_void, 4);
        enc.set_bytes(4, &eps as *const f32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
    }
    // Per-head RMSNorm on K (8 heads of head_dim)
    for head in 0..m.cfg.n_kv_heads {
        let enc = cmd.new_compute_encoder();
        enc.set_pipeline(&m.p_rmsnorm);
        enc.set_buffer(0, &m.k, (head * hd * 4) as u64);
        enc.set_buffer(1, &m.k, (head * hd * 4) as u64);
        enc.set_buffer(2, &m.weights, lo.attn_k_norm);
        enc.set_bytes(3, &hd as *const u32 as *const c_void, 4);
        enc.set_bytes(4, &eps as *const f32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(128, 1, 1));
        enc.end_encoding();
    }
}

fn encode_rope(cmd: &CommandBuffer, m: &Model, position: u32) {
    let hd = m.head_dim;
    let fb = m.cfg.rope_freq_base;
    // RoPE on Q
    let enc = cmd.new_compute_encoder();
    enc.set_pipeline(&m.p_rope);
    enc.set_buffer(0, &m.q, 0);
    enc.set_bytes(1, &hd as *const u32 as *const c_void, 4);
    enc.set_bytes(2, &position as *const u32 as *const c_void, 4);
    enc.set_bytes(3, &fb as *const f32 as *const c_void, 4);
    let n_pairs_q = m.cfg.n_heads * hd / 2;
    enc.dispatch_threads(MTLSize::new(n_pairs_q as u64, 1, 1), MTLSize::new(64, 1, 1));
    enc.end_encoding();
    // RoPE on K
    let enc = cmd.new_compute_encoder();
    enc.set_pipeline(&m.p_rope);
    enc.set_buffer(0, &m.k, 0);
    enc.set_bytes(1, &hd as *const u32 as *const c_void, 4);
    enc.set_bytes(2, &position as *const u32 as *const c_void, 4);
    enc.set_bytes(3, &fb as *const f32 as *const c_void, 4);
    let n_pairs_k = m.cfg.n_kv_heads * hd / 2;
    enc.dispatch_threads(MTLSize::new(n_pairs_k as u64, 1, 1), MTLSize::new(64, 1, 1));
    enc.end_encoding();
}

fn encode_kv_store(cmd: &CommandBuffer, m: &Model, layer: usize, position: u32) {
    let enc = cmd.new_compute_encoder();
    enc.set_pipeline(&m.p_kv_store);
    enc.set_buffer(0, &m.k, 0);
    enc.set_buffer(1, &m.v, 0);
    enc.set_buffer(2, &m.k_cache[layer], 0);
    enc.set_buffer(3, &m.v_cache[layer], 0);
    let kv_dim = m.kv_dim;
    enc.set_bytes(4, &kv_dim as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &position as *const u32 as *const c_void, 4);
    enc.dispatch_threads(MTLSize::new(kv_dim as u64, 1, 1), MTLSize::new(64, 1, 1));
    enc.end_encoding();
}

fn encode_attention(cmd: &CommandBuffer, m: &Model, layer: usize, seq_len: u32) {
    let enc = cmd.new_compute_encoder();
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
    // One threadgroup per query head, 128 threads each
    enc.dispatch_threadgroups(
        MTLSize::new(m.cfg.n_heads as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
}

fn encode_silu(cmd: &CommandBuffer, m: &Model, dim: u32) {
    let enc = cmd.new_compute_encoder();
    enc.set_pipeline(&m.p_silu);
    enc.set_buffer(0, &m.gate, 0);
    enc.set_buffer(1, &m.up, 0);
    enc.set_buffer(2, &m.ffn_mid, 0);
    enc.dispatch_threads(MTLSize::new(dim as u64, 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
}

fn encode_add(cmd: &CommandBuffer, m: &Model, x: &Buffer, y: &Buffer, dim: u32) {
    let enc = cmd.new_compute_encoder();
    enc.set_pipeline(&m.p_add);
    enc.set_buffer(0, x, 0);
    enc.set_buffer(1, y, 0);
    enc.dispatch_threads(MTLSize::new(dim as u64, 1, 1), MTLSize::new(256, 1, 1));
    enc.end_encoding();
}

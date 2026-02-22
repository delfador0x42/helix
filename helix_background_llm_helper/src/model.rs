//! Loads Qwen3 GGUF weights into GPU buffers and compiles Metal pipelines.
//! Single command buffer architecture: all ops encoded ahead, one commit per token.

use crate::gpu::*;
use crate::gguf::{GGUFFile, GGMLType, ModelConfig};
use crate::kernels;
use std::ffi::c_void;

const MAX_SEQ: u32 = 2048;

pub struct Model {
    pub cfg: ModelConfig,
    pub head_dim: u32,
    pub gqa_ratio: u32,
    pub kv_dim: u32,       // n_kv_heads * head_dim
    pub q_dim: u32,        // n_heads * head_dim
    pub device: Device,
    pub queue: CommandQueue,
    // Pipelines
    pub p_rmsnorm: Pipeline,
    pub p_q4k: Pipeline,
    pub p_q6k: Pipeline,
    pub p_embed: Pipeline,
    pub p_rope: Pipeline,
    pub p_kv_store: Pipeline,
    pub p_attn: Pipeline,
    pub p_silu: Pipeline,
    pub p_add: Pipeline,
    // Weight buffer (all tensor data, one big buffer)
    pub weights: Buffer,
    pub tensor_data_start: usize,
    // Per-layer weight offsets (relative to tensor_data_start)
    pub layers: Vec<LayerOffsets>,
    pub embd_off: u64,      // token_embd.weight offset
    pub out_norm_off: u64,  // output_norm.weight offset
    // Scratch buffers
    pub x: Buffer,          // [hidden_dim] current hidden state
    pub norm_out: Buffer,   // [hidden_dim] after rmsnorm
    pub q: Buffer,          // [q_dim]
    pub k: Buffer,          // [kv_dim]
    pub v: Buffer,          // [kv_dim]
    pub attn_out: Buffer,   // [q_dim]
    pub o: Buffer,          // [hidden_dim]
    pub gate: Buffer,       // [ffn_dim]
    pub up: Buffer,         // [ffn_dim]
    pub ffn_mid: Buffer,    // [ffn_dim]
    pub down: Buffer,       // [hidden_dim]
    pub logits: Buffer,     // [vocab_size]
    // KV cache: per layer
    pub k_cache: Vec<Buffer>, // [MAX_SEQ * kv_dim] per layer
    pub v_cache: Vec<Buffer>,
}

pub struct LayerOffsets {
    pub attn_q: u64,
    pub attn_k: u64,
    pub attn_v: u64,
    pub attn_output: u64,
    pub attn_norm: u64,
    pub attn_q_norm: u64,
    pub attn_k_norm: u64,
    pub ffn_gate: u64,
    pub ffn_up: u64,
    pub ffn_down: u64,
    pub ffn_norm: u64,
    // dtype info for dispatch
    pub attn_q_type: GGMLType,
    pub attn_k_type: GGMLType,
    pub attn_v_type: GGMLType,
    pub attn_output_type: GGMLType,
    pub ffn_gate_type: GGMLType,
    pub ffn_up_type: GGMLType,
    pub ffn_down_type: GGMLType,
}

impl Model {
    pub fn load(gguf: &GGUFFile) -> Result<Self, String> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        eprintln!("model: device={}, compiling shaders...", device.name());

        // Compile all shaders
        let src = kernels::all_kernels();
        let lib = device.new_library_with_source(&src)?;
        let p_rmsnorm = device.new_compute_pipeline(&lib.get_function("rmsnorm")?)?;
        let p_q4k = device.new_compute_pipeline(&lib.get_function("matvec_q4k")?)?;
        let p_q6k = device.new_compute_pipeline(&lib.get_function("matvec_q6k")?)?;
        let p_embed = device.new_compute_pipeline(&lib.get_function("embed_q6k")?)?;
        let p_rope = device.new_compute_pipeline(&lib.get_function("rope")?)?;
        let p_kv_store = device.new_compute_pipeline(&lib.get_function("kv_store")?)?;
        let p_attn = device.new_compute_pipeline(&lib.get_function("attention")?)?;
        let p_silu = device.new_compute_pipeline(&lib.get_function("silu_mul")?)?;
        let p_add = device.new_compute_pipeline(&lib.get_function("residual_add")?)?;
        eprintln!("model: shaders compiled");

        // Upload weight data to GPU
        let td_start = gguf.tensor_data_start;
        let td_len = gguf.data.len() - td_start;
        let weights = device.new_buffer_with_data(
            gguf.data[td_start..].as_ptr() as *const c_void,
            td_len as u64,
        );
        eprintln!("model: weights uploaded ({:.1}MB)", td_len as f64 / 1e6);

        let cfg = &gguf.config;
        // Derive real head_dim from tensor shapes
        let q_tensor = gguf.tensor("blk.0.attn_q.weight")
            .ok_or("missing blk.0.attn_q.weight")?;
        let head_dim = (q_tensor.dims[1] as u32) / cfg.n_heads;
        let q_dim = cfg.n_heads * head_dim;
        let kv_dim = cfg.n_kv_heads * head_dim;
        let gqa_ratio = cfg.n_heads / cfg.n_kv_heads;

        eprintln!("model: head_dim={head_dim}, q_dim={q_dim}, kv_dim={kv_dim}, gqa={gqa_ratio}");

        // Collect per-layer offsets
        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers {
            let get = |name: &str| -> Result<(u64, GGMLType), String> {
                let full = format!("blk.{i}.{name}");
                let t = gguf.tensor(&full).ok_or(format!("missing {full}"))?;
                Ok((t.offset, t.dtype))
            };
            let (attn_q, attn_q_type) = get("attn_q.weight")?;
            let (attn_k, attn_k_type) = get("attn_k.weight")?;
            let (attn_v, attn_v_type) = get("attn_v.weight")?;
            let (attn_output, attn_output_type) = get("attn_output.weight")?;
            let (attn_norm, _) = get("attn_norm.weight")?;
            let (attn_q_norm, _) = get("attn_q_norm.weight")?;
            let (attn_k_norm, _) = get("attn_k_norm.weight")?;
            let (ffn_gate, ffn_gate_type) = get("ffn_gate.weight")?;
            let (ffn_up, ffn_up_type) = get("ffn_up.weight")?;
            let (ffn_down, ffn_down_type) = get("ffn_down.weight")?;
            let (ffn_norm, _) = get("ffn_norm.weight")?;
            layers.push(LayerOffsets {
                attn_q, attn_k, attn_v, attn_output, attn_norm,
                attn_q_norm, attn_k_norm,
                ffn_gate, ffn_up, ffn_down, ffn_norm,
                attn_q_type, attn_k_type, attn_v_type, attn_output_type,
                ffn_gate_type, ffn_up_type, ffn_down_type,
            });
        }

        let embd = gguf.tensor("token_embd.weight").ok_or("missing token_embd.weight")?;
        let out_norm = gguf.tensor("output_norm.weight").ok_or("missing output_norm.weight")?;

        // Scratch buffers
        let f = |n: u32| device.new_buffer(n as u64 * 4);
        let x = f(cfg.hidden_dim);
        let norm_out = f(cfg.hidden_dim);
        let q_buf = f(q_dim);
        let k_buf = f(kv_dim);
        let v_buf = f(kv_dim);
        let attn_out = f(q_dim);
        let o = f(cfg.hidden_dim);
        let gate = f(cfg.ffn_dim);
        let up = f(cfg.ffn_dim);
        let ffn_mid = f(cfg.ffn_dim);
        let down = f(cfg.hidden_dim);
        let logits = f(cfg.vocab_size);

        // KV cache
        let kv_size = MAX_SEQ as u64 * kv_dim as u64 * 4;
        let mut k_cache = Vec::with_capacity(cfg.n_layers as usize);
        let mut v_cache = Vec::with_capacity(cfg.n_layers as usize);
        for _ in 0..cfg.n_layers {
            k_cache.push(device.new_buffer(kv_size));
            v_cache.push(device.new_buffer(kv_size));
        }
        eprintln!("model: KV cache {:.1}MB ({MAX_SEQ} max seq)",
            kv_size as f64 * 2.0 * cfg.n_layers as f64 / 1e6);

        Ok(Model {
            cfg: gguf.config.clone(), head_dim, gqa_ratio, kv_dim, q_dim,
            device, queue,
            p_rmsnorm, p_q4k, p_q6k, p_embed, p_rope, p_kv_store, p_attn, p_silu, p_add,
            weights, tensor_data_start: td_start,
            layers,
            embd_off: embd.offset,
            out_norm_off: out_norm.offset,
            x, norm_out, q: q_buf, k: k_buf, v: v_buf, attn_out, o,
            gate, up, ffn_mid, down, logits,
            k_cache, v_cache,
        })
    }

    /// Get the pipeline for a given quantization type's matvec.
    pub fn matvec_pipeline(&self, dtype: GGMLType) -> &Pipeline {
        match dtype {
            GGMLType::Q4K => &self.p_q4k,
            GGMLType::Q6K => &self.p_q6k,
            _ => panic!("unsupported matvec dtype: {:?}", dtype),
        }
    }
}

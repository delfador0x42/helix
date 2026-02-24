//! Loads GGUF weights into GPU buffer. 70B-only: no FP16 pre-dequant, no pipelines.
//! All inference pipelines are compiled by infer_batch.rs.

use crate::gpu::*;
use crate::gguf::{GGUFFile, GGMLType, ModelConfig};
use std::ffi::c_void;

pub struct Model {
    pub cfg: ModelConfig,
    pub head_dim: u32,
    pub gqa_ratio: u32,
    pub kv_dim: u32,       // n_kv_heads * head_dim
    pub q_dim: u32,        // n_heads * head_dim
    pub device: Device,
    pub queue: CommandQueue,
    pub weights: Buffer,
    pub layers: Vec<LayerOffsets>,
    pub embd_off: u64,
    pub embd_type: GGMLType,
    pub output_off: u64,
    pub output_type: GGMLType,
    pub out_norm_off: u64,
}

pub struct LayerOffsets {
    pub attn_q: u64,
    pub attn_k: u64,
    pub attn_v: u64,
    pub attn_output: u64,
    pub attn_norm: u64,
    pub ffn_gate: u64,
    pub ffn_up: u64,
    pub ffn_down: u64,
    pub ffn_norm: u64,
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
        eprintln!("model: device={}", device.name());

        // Upload quantized weight data to GPU (supports split GGUF)
        let weights = if gguf.is_split() {
            let total = gguf.total_tensor_bytes;
            let buf = device.new_buffer(total);
            let dst = buf.contents() as *mut u8;
            for shard in &gguf.shards {
                let src = &shard.data[shard.tensor_data_start..];
                let len = shard.tensor_data_len;
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst.add(shard.base_offset as usize), len);
                }
                eprintln!("model: shard copied {:.1}GB at offset {:.1}GB",
                    len as f64 / 1e9, shard.base_offset as f64 / 1e9);
            }
            eprintln!("model: split weights uploaded ({:.1}GB total)", total as f64 / 1e9);
            buf
        } else {
            let td_start = gguf.tensor_data_start;
            let td_len = gguf.data.len() - td_start;
            let buf = device.new_buffer_with_data(
                gguf.data[td_start..].as_ptr() as *const c_void,
                td_len as u64,
            );
            eprintln!("model: weights uploaded ({:.1}MB)", td_len as f64 / 1e6);
            buf
        };

        let cfg = &gguf.config;
        let q_tensor = gguf.tensor("blk.0.attn_q.weight")
            .ok_or("missing blk.0.attn_q.weight")?;
        let head_dim = (q_tensor.dims[1] as u32) / cfg.n_heads;
        let q_dim = cfg.n_heads * head_dim;
        let kv_dim = cfg.n_kv_heads * head_dim;
        let gqa_ratio = cfg.n_heads / cfg.n_kv_heads;

        let embd = gguf.tensor("token_embd.weight").ok_or("missing token_embd.weight")?;
        let out_norm = gguf.tensor("output_norm.weight").ok_or("missing output_norm.weight")?;
        let (output_off, output_type) = if let Some(out) = gguf.tensor("output.weight") {
            (out.offset, out.dtype)
        } else {
            eprintln!("model: output.weight not found, using tied embeddings");
            (embd.offset, embd.dtype)
        };

        let mut layers = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers {
            let get = |name: &str| -> Result<(u64, GGMLType), String> {
                let full = format!("blk.{i}.{name}");
                let t = gguf.tensor(&full).ok_or(format!("missing {full}"))?;
                Ok((t.offset, t.dtype))
            };
            let (attn_q, aqt) = get("attn_q.weight")?;
            let (attn_k, akt) = get("attn_k.weight")?;
            let (attn_v, avt) = get("attn_v.weight")?;
            let (attn_output, aot) = get("attn_output.weight")?;
            let (attn_norm, _) = get("attn_norm.weight")?;
            let (ffn_gate, fgt) = get("ffn_gate.weight")?;
            let (ffn_up, fut) = get("ffn_up.weight")?;
            let (ffn_down, fdt) = get("ffn_down.weight")?;
            let (ffn_norm, _) = get("ffn_norm.weight")?;

            layers.push(LayerOffsets {
                attn_q, attn_k, attn_v, attn_output, attn_norm,
                ffn_gate, ffn_up, ffn_down, ffn_norm,
                attn_q_type: aqt, attn_k_type: akt, attn_v_type: avt,
                attn_output_type: aot,
                ffn_gate_type: fgt, ffn_up_type: fut, ffn_down_type: fdt,
            });
        }

        Ok(Model {
            cfg: gguf.config.clone(), head_dim, gqa_ratio, kv_dim, q_dim,
            device, queue, weights, layers,
            embd_off: embd.offset, embd_type: embd.dtype,
            output_off, output_type,
            out_norm_off: out_norm.offset,
        })
    }
}

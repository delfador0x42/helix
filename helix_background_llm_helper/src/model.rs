//! Loads GGUF weights into GPU buffers and compiles Metal pipelines.
//! Llama architecture only (Q4_0/Q4_1/Q4_K/Q5_K/Q6_K quantization types).
//! FP16 pre-dequant is optional — skipped for large models (70B+).

use crate::gpu::*;
use crate::gguf::{GGUFFile, GGMLType, ModelConfig};
use crate::kernels_fp16;
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
    // Single-token pipelines
    pub p_rmsnorm: Pipeline,
    pub p_q4_0: Pipeline,
    pub p_q4_1: Pipeline,
    pub p_q4k: Pipeline,
    pub p_q5k: Pipeline,
    pub p_q6k: Pipeline,
    pub p_q8_0: Pipeline,
    pub p_q4_0_add: Pipeline,
    pub p_q4_1_add: Pipeline,
    pub p_q4k_add: Pipeline,
    pub p_q5k_add: Pipeline,
    pub p_q6k_add: Pipeline,
    pub p_q8_0_add: Pipeline,
    pub p_embed_q4_0: Pipeline,
    pub p_embed_q5k: Pipeline,
    pub p_embed_q6k: Pipeline,
    pub p_embed_q8_0: Pipeline,
    pub p_rope: Pipeline,
    pub p_kv_store: Pipeline,
    pub p_attn: Pipeline,
    pub p_silu: Pipeline,
    pub p_argmax: Pipeline,
    // Quantized weight buffer (kept for norm weights which are F32)
    pub weights: Buffer,
    // FP16 pre-dequantized weight buffer (row-major, all weight tensors expanded)
    pub fp16_buf: Buffer,
    pub fp16_embd_off: u64,
    pub fp16_output_off: u64,
    // Per-layer weight offsets
    pub layers: Vec<LayerOffsets>,
    pub embd_off: u64,
    pub embd_type: GGMLType,
    pub output_off: u64,
    pub output_type: GGMLType,
    pub out_norm_off: u64,
    // Scratch buffers (single-token, FP32)
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
    // KV cache (single-token, FP32)
    pub k_cache: Vec<Buffer>,
    pub v_cache: Vec<Buffer>,
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
    // dtype for single-token dispatch
    pub attn_q_type: GGMLType,
    pub attn_k_type: GGMLType,
    pub attn_v_type: GGMLType,
    pub attn_output_type: GGMLType,
    pub ffn_gate_type: GGMLType,
    pub ffn_up_type: GGMLType,
    pub ffn_down_type: GGMLType,
    // FP16 offsets (into fp16_buf)
    pub fp16_attn_q: u64,
    pub fp16_attn_k: u64,
    pub fp16_attn_v: u64,
    pub fp16_attn_output: u64,
    pub fp16_ffn_gate: u64,
    pub fp16_ffn_up: u64,
    pub fp16_ffn_down: u64,
}

struct DequantJob {
    dtype: GGMLType,
    quant_off: u64,
    fp16_off: u64,
    rows: u32,
    cols: u32,
}

impl Model {
    pub fn load(gguf: &GGUFFile) -> Result<Self, String> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        eprintln!("model: device={}, compiling shaders...", device.name());

        // Compile single-token shaders
        let src = kernels::all_kernels();
        let lib = device.new_library_with_source(&src)?;
        let p_rmsnorm = device.new_compute_pipeline(&lib.get_function("rmsnorm")?)?;
        let p_q4_0 = device.new_compute_pipeline(&lib.get_function("matvec_q4_0")?)?;
        let p_q4_1 = device.new_compute_pipeline(&lib.get_function("matvec_q4_1")?)?;
        let p_q4k = device.new_compute_pipeline(&lib.get_function("matvec_q4k")?)?;
        let p_q5k = device.new_compute_pipeline(&lib.get_function("matvec_q5k")?)?;
        let p_q6k = device.new_compute_pipeline(&lib.get_function("matvec_q6k")?)?;
        let p_q8_0 = device.new_compute_pipeline(&lib.get_function("matvec_q8_0")?)?;
        let p_q4_0_add = device.new_compute_pipeline(&lib.get_function("matvec_q4_0_add")?)?;
        let p_q4_1_add = device.new_compute_pipeline(&lib.get_function("matvec_q4_1_add")?)?;
        let p_q4k_add = device.new_compute_pipeline(&lib.get_function("matvec_q4k_add")?)?;
        let p_q5k_add = device.new_compute_pipeline(&lib.get_function("matvec_q5k_add")?)?;
        let p_q6k_add = device.new_compute_pipeline(&lib.get_function("matvec_q6k_add")?)?;
        let p_q8_0_add = device.new_compute_pipeline(&lib.get_function("matvec_q8_0_add")?)?;
        let p_embed_q4_0 = device.new_compute_pipeline(&lib.get_function("embed_q4_0")?)?;
        let p_embed_q5k = device.new_compute_pipeline(&lib.get_function("embed_q5k")?)?;
        let p_embed_q6k = device.new_compute_pipeline(&lib.get_function("embed_q6k")?)?;
        let p_embed_q8_0 = device.new_compute_pipeline(&lib.get_function("embed_q8_0")?)?;
        let p_rope = device.new_compute_pipeline(&lib.get_function("rope")?)?;
        let p_kv_store = device.new_compute_pipeline(&lib.get_function("kv_store")?)?;
        let p_attn = device.new_compute_pipeline(&lib.get_function("attention")?)?;
        let p_silu = device.new_compute_pipeline(&lib.get_function("silu_mul")?)?;
        let p_argmax = device.new_compute_pipeline(&lib.get_function("argmax")?)?;
        eprintln!("model: shaders compiled");

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
            eprintln!("model: quantized weights uploaded ({:.1}MB)", td_len as f64 / 1e6);
            buf
        };

        let cfg = &gguf.config;
        let q_tensor = gguf.tensor("blk.0.attn_q.weight")
            .ok_or("missing blk.0.attn_q.weight")?;
        let head_dim = (q_tensor.dims[1] as u32) / cfg.n_heads;
        let q_dim = cfg.n_heads * head_dim;
        let kv_dim = cfg.n_kv_heads * head_dim;
        let gqa_ratio = cfg.n_heads / cfg.n_kv_heads;
        let h = cfg.hidden_dim;

        // Collect per-layer offsets
        let mut layers = Vec::with_capacity(cfg.n_layers as usize);

        let embd = gguf.tensor("token_embd.weight").ok_or("missing token_embd.weight")?;
        let out_norm = gguf.tensor("output_norm.weight").ok_or("missing output_norm.weight")?;
        let (output_off, output_type) = if let Some(out) = gguf.tensor("output.weight") {
            (out.offset, out.dtype)
        } else {
            eprintln!("model: output.weight not found, using tied embeddings");
            (embd.offset, embd.dtype)
        };

        // Decide whether to do FP16 pre-dequant (skip for large models / Q5K)
        let has_q5k = gguf.tensors.iter().any(|t| t.dtype == GGMLType::Q5K);
        let fp16_budget = cfg.n_layers as u64 * 7 * (cfg.hidden_dim as u64).max(cfg.ffn_dim as u64)
            * cfg.hidden_dim as u64 * 2;
        let skip_fp16 = has_q5k || fp16_budget > 20_000_000_000; // >20GB = skip
        if skip_fp16 {
            eprintln!("model: skipping FP16 pre-dequant (Q5K or large model)");
        }

        let mut jobs: Vec<DequantJob> = Vec::new();
        let mut cursor: u64 = 0;

        // FP16 layout (only computed if !skip_fp16)
        let fp16_embd_off;
        let fp16_output_off;
        if !skip_fp16 {
            fp16_embd_off = cursor;
            jobs.push(DequantJob { dtype: embd.dtype, quant_off: embd.offset,
                fp16_off: cursor, rows: cfg.vocab_size, cols: h });
            cursor += cfg.vocab_size as u64 * h as u64 * 2;
            fp16_output_off = if output_off == embd.offset {
                fp16_embd_off
            } else {
                let off = cursor;
                jobs.push(DequantJob { dtype: output_type, quant_off: output_off,
                    fp16_off: cursor, rows: cfg.vocab_size, cols: h });
                cursor += cfg.vocab_size as u64 * h as u64 * 2;
                off
            };
        } else {
            fp16_embd_off = 0;
            fp16_output_off = 0;
        }

        // Per-layer weight tensors
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

            let (fp16_aq, fp16_ak, fp16_av, fp16_ao, fp16_fg, fp16_fu, fp16_fd);
            if !skip_fp16 {
                macro_rules! alloc {
                    ($dt:expr, $qoff:expr, $r:expr, $c:expr) => {{
                        let off = cursor;
                        jobs.push(DequantJob { dtype: $dt, quant_off: $qoff,
                            fp16_off: off, rows: $r, cols: $c });
                        cursor += $r as u64 * $c as u64 * 2;
                        off
                    }};
                }
                fp16_aq = alloc!(aqt, attn_q, q_dim, h);
                fp16_ak = alloc!(akt, attn_k, kv_dim, h);
                fp16_av = alloc!(avt, attn_v, kv_dim, h);
                fp16_ao = alloc!(aot, attn_output, h, q_dim);
                fp16_fg = alloc!(fgt, ffn_gate, cfg.ffn_dim, h);
                fp16_fu = alloc!(fut, ffn_up, cfg.ffn_dim, h);
                fp16_fd = alloc!(fdt, ffn_down, h, cfg.ffn_dim);
            } else {
                fp16_aq = 0; fp16_ak = 0; fp16_av = 0; fp16_ao = 0;
                fp16_fg = 0; fp16_fu = 0; fp16_fd = 0;
            }

            layers.push(LayerOffsets {
                attn_q, attn_k, attn_v, attn_output, attn_norm,
                ffn_gate, ffn_up, ffn_down, ffn_norm,
                attn_q_type: aqt, attn_k_type: akt, attn_v_type: avt,
                attn_output_type: aot,
                ffn_gate_type: fgt, ffn_up_type: fut, ffn_down_type: fdt,
                fp16_attn_q: fp16_aq, fp16_attn_k: fp16_ak,
                fp16_attn_v: fp16_av, fp16_attn_output: fp16_ao,
                fp16_ffn_gate: fp16_fg, fp16_ffn_up: fp16_fu,
                fp16_ffn_down: fp16_fd,
            });
        }

        // Allocate FP16 buffer and run GPU dequant (if not skipped)
        let fp16_buf = if cursor > 0 {
            let buf = device.new_buffer(cursor);
            eprintln!("model: FP16 buffer allocated ({:.1}MB, {} tensors to dequant)",
                cursor as f64 / 1e6, jobs.len());

            let dequant_src = kernels_fp16::dequant_source();
            let dequant_lib = device.new_library_with_source(&dequant_src)?;
            let pd_q4_0 = device.new_compute_pipeline(&dequant_lib.get_function("dequant_q4_0")?)?;
            let pd_q4_1 = device.new_compute_pipeline(&dequant_lib.get_function("dequant_q4_1")?)?;
            let pd_q6k = device.new_compute_pipeline(&dequant_lib.get_function("dequant_q6k")?)?;

            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            for job in &jobs {
                let pipe = match job.dtype {
                    GGMLType::Q4_0 => &pd_q4_0,
                    GGMLType::Q4_1 => &pd_q4_1,
                    GGMLType::Q6K => &pd_q6k,
                    _ => return Err(format!("unsupported dequant dtype: {:?}", job.dtype)),
                };
                enc.set_pipeline(pipe);
                enc.set_buffer(0, &weights, job.quant_off);
                enc.set_buffer(1, &buf, job.fp16_off);
                enc.set_bytes(2, &job.cols as *const u32 as *const c_void, 4);
                enc.set_bytes(3, &job.rows as *const u32 as *const c_void, 4);
                enc.dispatch_threads(
                    MTLSize::new(job.cols as u64, job.rows as u64, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            eprintln!("model: FP16 dequant complete");
            buf
        } else {
            eprintln!("model: no FP16 buffer needed (single-token only)");
            device.new_buffer(4) // dummy 4-byte buffer
        };

        // Scratch buffers (single-token, FP32)
        let f = |n: u32| device.new_buffer(n as u64 * 4);
        let x = f(cfg.hidden_dim);
        let norm_out = f(cfg.hidden_dim);
        let q_buf = f(q_dim);
        let k_buf = f(kv_dim);
        let v_buf = f(kv_dim);
        let attn_out = f(q_dim);
        let gate = f(cfg.ffn_dim);
        let up = f(cfg.ffn_dim);
        let ffn_mid = f(cfg.ffn_dim);
        let logits = f(cfg.vocab_size);
        let argmax_buf = device.new_buffer(4);

        // KV cache (single-token, FP32)
        let kv_size = MAX_SEQ as u64 * kv_dim as u64 * 4;
        let mut k_cache = Vec::with_capacity(cfg.n_layers as usize);
        let mut v_cache = Vec::with_capacity(cfg.n_layers as usize);
        for _ in 0..cfg.n_layers {
            k_cache.push(device.new_buffer(kv_size));
            v_cache.push(device.new_buffer(kv_size));
        }

        Ok(Model {
            cfg: gguf.config.clone(), head_dim, gqa_ratio, kv_dim, q_dim,
            device, queue,
            p_rmsnorm, p_q4_0, p_q4_1, p_q4k, p_q5k, p_q6k, p_q8_0,
            p_q4_0_add, p_q4_1_add, p_q4k_add, p_q5k_add, p_q6k_add, p_q8_0_add,
            p_embed_q4_0, p_embed_q5k, p_embed_q6k, p_embed_q8_0,
            p_rope, p_kv_store, p_attn, p_silu, p_argmax,
            weights, fp16_buf, fp16_embd_off, fp16_output_off,
            layers,
            embd_off: embd.offset, embd_type: embd.dtype,
            output_off, output_type,
            out_norm_off: out_norm.offset,
            x, norm_out, q: q_buf, k: k_buf, v: v_buf, attn_out,
            gate, up, ffn_mid, logits, argmax_buf,
            k_cache, v_cache,
        })
    }

    pub fn matvec_pipeline(&self, dtype: GGMLType) -> &Pipeline {
        match dtype {
            GGMLType::Q4_0 => &self.p_q4_0,
            GGMLType::Q4_1 => &self.p_q4_1,
            GGMLType::Q4K => &self.p_q4k,
            GGMLType::Q5K => &self.p_q5k,
            GGMLType::Q6K => &self.p_q6k,
            GGMLType::Q8_0 => &self.p_q8_0,
            _ => panic!("unsupported matvec dtype: {:?}", dtype),
        }
    }

    pub fn matvec_add_pipeline(&self, dtype: GGMLType) -> &Pipeline {
        match dtype {
            GGMLType::Q4_0 => &self.p_q4_0_add,
            GGMLType::Q4_1 => &self.p_q4_1_add,
            GGMLType::Q4K => &self.p_q4k_add,
            GGMLType::Q5K => &self.p_q5k_add,
            GGMLType::Q6K => &self.p_q6k_add,
            GGMLType::Q8_0 => &self.p_q8_0_add,
            _ => panic!("unsupported matvec_add dtype: {:?}", dtype),
        }
    }

    pub fn embed_pipeline(&self, dtype: GGMLType) -> &Pipeline {
        match dtype {
            GGMLType::Q4_0 => &self.p_embed_q4_0,
            GGMLType::Q5K => &self.p_embed_q5k,
            GGMLType::Q6K => &self.p_embed_q6k,
            GGMLType::Q8_0 => &self.p_embed_q8_0,
            _ => panic!("unsupported embed dtype: {:?}", dtype),
        }
    }
}

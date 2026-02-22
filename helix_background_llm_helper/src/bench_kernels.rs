//! Benchmark individual kernels at actual model dimensions.
//! Goal: find which kernel(s) consume the 4.4ms of real GPU time.

use crate::gpu::*;
use crate::kernels;
use std::ffi::c_void;
use std::time::Instant;

const WARMUP: u32 = 5;
const ITERS: u32 = 50;

pub fn run() {
    let dev = Device::system_default().unwrap();
    let queue = dev.new_command_queue();
    let src = kernels::all_kernels();
    let lib = dev.new_library_with_source(&src).unwrap();

    let p_rmsnorm = dev.new_compute_pipeline(&lib.get_function("rmsnorm").unwrap()).unwrap();
    let p_q4_0 = dev.new_compute_pipeline(&lib.get_function("matvec_q4_0").unwrap()).unwrap();
    let p_q6k = dev.new_compute_pipeline(&lib.get_function("matvec_q6k").unwrap()).unwrap();
    let p_rope = dev.new_compute_pipeline(&lib.get_function("rope").unwrap()).unwrap();
    let p_kv_store = dev.new_compute_pipeline(&lib.get_function("kv_store").unwrap()).unwrap();
    let p_attn = dev.new_compute_pipeline(&lib.get_function("attention").unwrap()).unwrap();
    let p_silu = dev.new_compute_pipeline(&lib.get_function("silu_mul").unwrap()).unwrap();
    let p_argmax = dev.new_compute_pipeline(&lib.get_function("argmax").unwrap()).unwrap();

    // Llama-3.2-1B dimensions
    let hidden: u32 = 2048;
    let ffn: u32 = 8192;
    let n_heads: u32 = 32;
    let n_kv: u32 = 8;
    let head_dim: u32 = 64;
    let vocab: u32 = 128256;
    let n_layers: u32 = 16;
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv * head_dim;

    // Buffers at real sizes
    let x = dev.new_buffer(hidden as u64 * 4);
    let norm_out = dev.new_buffer(hidden as u64 * 4);
    let q_buf = dev.new_buffer(q_dim as u64 * 4);
    let k_buf = dev.new_buffer(kv_dim as u64 * 4);
    let v_buf = dev.new_buffer(kv_dim as u64 * 4);
    let attn_out = dev.new_buffer(q_dim as u64 * 4);
    let gate = dev.new_buffer(ffn as u64 * 4);
    let up = dev.new_buffer(ffn as u64 * 4);
    let ffn_mid = dev.new_buffer(ffn as u64 * 4);
    let logits = dev.new_buffer(vocab as u64 * 4);
    let argmax_buf = dev.new_buffer(4);

    // Q4_0: 18 bytes per 32 elements
    let q4_0_row_bytes = |cols: u32| (cols / 32) as u64 * 18;
    // Q6K: 210 bytes per 256 elements
    let q6k_row_bytes = |cols: u32| (cols / 256) as u64 * 210;

    // Weight buffers sized for specific ops
    let w_q4_0_2048x2048 = dev.new_buffer(q4_0_row_bytes(hidden) * hidden as u64);
    let w_q4_0_8192x2048 = dev.new_buffer(q4_0_row_bytes(hidden) * ffn as u64);
    let w_q4_0_2048x8192 = dev.new_buffer(q4_0_row_bytes(ffn) * hidden as u64);
    let w_q6k_vocab = dev.new_buffer(q6k_row_bytes(hidden) * vocab as u64);
    // KV cache for 1 layer
    let kc = dev.new_buffer(2048 * kv_dim as u64 * 4);
    let vc = dev.new_buffer(2048 * kv_dim as u64 * 4);

    let eps: f32 = 1e-6;
    let freq_base: f32 = 500000.0;
    let gqa: u32 = 4;
    let seq_len: u32 = 64; // test at seq_len=64

    eprintln!("\n=== Kernel Benchmark (Llama-3.2-1B dimensions, M3 Max) ===\n");

    // ── RMSNorm ──
    bench("rmsnorm(2048)", ITERS, &queue, |enc| {
        enc.set_pipeline(&p_rmsnorm);
        enc.set_buffer(0, &x, 0);
        enc.set_buffer(1, &norm_out, 0);
        enc.set_buffer(2, &x, 0); // weight (reuse x)
        enc.set_bytes(3, &hidden as *const u32 as *const c_void, 4);
        enc.set_bytes(4, &eps as *const f32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
    });

    // ── MatVec Q4_0: [2048 x 2048] @ x → q (attn projection) ──
    // New: 8 rows per threadgroup (8 simdgroups × 32 threads)
    bench("matvec_q4_0 [2048x2048] (Q proj)", ITERS, &queue, |enc| {
        enc.set_pipeline(&p_q4_0);
        enc.set_buffer(0, &w_q4_0_2048x2048, 0);
        enc.set_buffer(1, &norm_out, 0);
        enc.set_buffer(2, &q_buf, 0);
        enc.set_bytes(3, &hidden as *const u32 as *const c_void, 4);
        enc.dispatch_threadgroups(
            MTLSize::new((q_dim / 8) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    });

    // ── MatVec Q4_0: [512 x 2048] @ x → k (K projection) ──
    bench("matvec_q4_0 [512x2048] (K proj)", ITERS, &queue, |enc| {
        enc.set_pipeline(&p_q4_0);
        enc.set_buffer(0, &w_q4_0_2048x2048, 0);
        enc.set_buffer(1, &norm_out, 0);
        enc.set_buffer(2, &k_buf, 0);
        enc.set_bytes(3, &hidden as *const u32 as *const c_void, 4);
        enc.dispatch_threadgroups(
            MTLSize::new((kv_dim / 8) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    });

    // ── MatVec Q4_0: [8192 x 2048] @ x → gate (FFN gate) ──
    bench("matvec_q4_0 [8192x2048] (FFN gate)", ITERS, &queue, |enc| {
        enc.set_pipeline(&p_q4_0);
        enc.set_buffer(0, &w_q4_0_8192x2048, 0);
        enc.set_buffer(1, &norm_out, 0);
        enc.set_buffer(2, &gate, 0);
        enc.set_bytes(3, &hidden as *const u32 as *const c_void, 4);
        enc.dispatch_threadgroups(
            MTLSize::new((ffn / 8) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    });

    // ── MatVec Q4_0: [2048 x 8192] @ ffn_mid → down (FFN down) ──
    bench("matvec_q4_0 [2048x8192] (FFN down)", ITERS, &queue, |enc| {
        enc.set_pipeline(&p_q4_0);
        enc.set_buffer(0, &w_q4_0_2048x8192, 0);
        enc.set_buffer(1, &ffn_mid, 0);
        enc.set_buffer(2, &x, 0);
        let cols = ffn;
        enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
        enc.dispatch_threadgroups(
            MTLSize::new((hidden / 8) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    });

    // ── MatVec Q6K: [128256 x 2048] @ norm → logits (OUTPUT PROJ) ──
    // vocab=128256 not divisible by 8, round up
    let vocab_tgs = (vocab + 7) / 8;
    bench("matvec_q6k [128256x2048] (output proj)", ITERS, &queue, |enc| {
        enc.set_pipeline(&p_q6k);
        enc.set_buffer(0, &w_q6k_vocab, 0);
        enc.set_buffer(1, &norm_out, 0);
        enc.set_buffer(2, &logits, 0);
        enc.set_bytes(3, &hidden as *const u32 as *const c_void, 4);
        enc.dispatch_threadgroups(
            MTLSize::new(vocab_tgs as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    });

    // ── RoPE ──
    bench("rope(32 heads × 32 pairs)", ITERS, &queue, |enc| {
        enc.set_pipeline(&p_rope);
        enc.set_buffer(0, &q_buf, 0);
        enc.set_bytes(1, &head_dim as *const u32 as *const c_void, 4);
        let pos: u32 = 42;
        enc.set_bytes(2, &pos as *const u32 as *const c_void, 4);
        enc.set_bytes(3, &freq_base as *const f32 as *const c_void, 4);
        let n_pairs = n_heads * head_dim / 2;
        enc.dispatch_threads(MTLSize::new(n_pairs as u64, 1, 1), MTLSize::new(64, 1, 1));
    });

    // ── KV store ──
    bench("kv_store(512 dim)", ITERS, &queue, |enc| {
        enc.set_pipeline(&p_kv_store);
        enc.set_buffer(0, &k_buf, 0);
        enc.set_buffer(1, &v_buf, 0);
        enc.set_buffer(2, &kc, 0);
        enc.set_buffer(3, &vc, 0);
        enc.set_bytes(4, &kv_dim as *const u32 as *const c_void, 4);
        let pos: u32 = 42;
        enc.set_bytes(5, &pos as *const u32 as *const c_void, 4);
        enc.dispatch_threads(MTLSize::new(kv_dim as u64, 1, 1), MTLSize::new(64, 1, 1));
    });

    // ── Attention ──
    bench(&format!("attention(32 heads, seq_len={seq_len})"), ITERS, &queue, |enc| {
        enc.set_pipeline(&p_attn);
        enc.set_buffer(0, &q_buf, 0);
        enc.set_buffer(1, &kc, 0);
        enc.set_buffer(2, &vc, 0);
        enc.set_buffer(3, &attn_out, 0);
        enc.set_bytes(4, &seq_len as *const u32 as *const c_void, 4);
        enc.set_bytes(5, &head_dim as *const u32 as *const c_void, 4);
        enc.set_bytes(6, &n_kv as *const u32 as *const c_void, 4);
        enc.set_bytes(7, &gqa as *const u32 as *const c_void, 4);
        enc.set_bytes(8, &kv_dim as *const u32 as *const c_void, 4);
        enc.dispatch_threadgroups(
            MTLSize::new(n_heads as u64, 1, 1),
            MTLSize::new(128, 1, 1),
        );
    });

    // ── SiLU ──
    bench("silu_mul(8192)", ITERS, &queue, |enc| {
        enc.set_pipeline(&p_silu);
        enc.set_buffer(0, &gate, 0);
        enc.set_buffer(1, &up, 0);
        enc.set_buffer(2, &ffn_mid, 0);
        enc.dispatch_threads(MTLSize::new(ffn as u64, 1, 1), MTLSize::new(256, 1, 1));
    });

    // ── Argmax ──
    bench("argmax(128256)", ITERS, &queue, |enc| {
        enc.set_pipeline(&p_argmax);
        enc.set_buffer(0, &logits, 0);
        enc.set_buffer(1, &argmax_buf, 0);
        enc.set_bytes(2, &vocab as *const u32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
    });

    // ── Now measure a full layer ──
    eprintln!("\n--- Full Layer Simulation ---");
    bench("1 full layer (14 dispatches)", ITERS, &queue, |enc| {
        // rmsnorm
        enc.set_pipeline(&p_rmsnorm);
        enc.set_buffer(0, &x, 0); enc.set_buffer(1, &norm_out, 0);
        enc.set_buffer(2, &x, 0);
        enc.set_bytes(3, &hidden as *const u32 as *const c_void, 4);
        enc.set_bytes(4, &eps as *const f32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        // Q matvec (rows/8 threadgroups)
        enc.set_pipeline(&p_q4_0);
        enc.set_buffer(0, &w_q4_0_2048x2048, 0); enc.set_buffer(1, &norm_out, 0);
        enc.set_buffer(2, &q_buf, 0);
        enc.set_bytes(3, &hidden as *const u32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new((q_dim/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
        // K matvec
        enc.set_buffer(2, &k_buf, 0);
        enc.dispatch_threadgroups(MTLSize::new((kv_dim/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
        // V matvec
        enc.set_buffer(2, &v_buf, 0);
        enc.dispatch_threadgroups(MTLSize::new((kv_dim/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
        // RoPE Q
        enc.set_pipeline(&p_rope);
        enc.set_buffer(0, &q_buf, 0);
        enc.set_bytes(1, &head_dim as *const u32 as *const c_void, 4);
        let pos: u32 = 42;
        enc.set_bytes(2, &pos as *const u32 as *const c_void, 4);
        enc.set_bytes(3, &freq_base as *const f32 as *const c_void, 4);
        enc.dispatch_threads(MTLSize::new((n_heads*head_dim/2) as u64, 1, 1), MTLSize::new(64, 1, 1));
        // RoPE K
        enc.set_buffer(0, &k_buf, 0);
        enc.dispatch_threads(MTLSize::new((n_kv*head_dim/2) as u64, 1, 1), MTLSize::new(64, 1, 1));
        // KV store
        enc.set_pipeline(&p_kv_store);
        enc.set_buffer(0, &k_buf, 0); enc.set_buffer(1, &v_buf, 0);
        enc.set_buffer(2, &kc, 0); enc.set_buffer(3, &vc, 0);
        enc.set_bytes(4, &kv_dim as *const u32 as *const c_void, 4);
        enc.set_bytes(5, &pos as *const u32 as *const c_void, 4);
        enc.dispatch_threads(MTLSize::new(kv_dim as u64, 1, 1), MTLSize::new(64, 1, 1));
        // Attention
        enc.set_pipeline(&p_attn);
        enc.set_buffer(0, &q_buf, 0); enc.set_buffer(1, &kc, 0);
        enc.set_buffer(2, &vc, 0); enc.set_buffer(3, &attn_out, 0);
        enc.set_bytes(4, &seq_len as *const u32 as *const c_void, 4);
        enc.set_bytes(5, &head_dim as *const u32 as *const c_void, 4);
        enc.set_bytes(6, &n_kv as *const u32 as *const c_void, 4);
        enc.set_bytes(7, &gqa as *const u32 as *const c_void, 4);
        enc.set_bytes(8, &kv_dim as *const u32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(128, 1, 1));
        // O proj
        enc.set_pipeline(&p_q4_0);
        enc.set_buffer(0, &w_q4_0_2048x2048, 0); enc.set_buffer(1, &attn_out, 0);
        enc.set_buffer(2, &x, 0);
        enc.set_bytes(3, &(q_dim) as *const u32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new((hidden/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
        // FFN rmsnorm
        enc.set_pipeline(&p_rmsnorm);
        enc.set_buffer(0, &x, 0); enc.set_buffer(1, &norm_out, 0);
        enc.set_buffer(2, &x, 0);
        enc.set_bytes(3, &hidden as *const u32 as *const c_void, 4);
        enc.set_bytes(4, &eps as *const f32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        // gate matvec
        enc.set_pipeline(&p_q4_0);
        enc.set_buffer(0, &w_q4_0_8192x2048, 0); enc.set_buffer(1, &norm_out, 0);
        enc.set_buffer(2, &gate, 0);
        enc.set_bytes(3, &hidden as *const u32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new((ffn/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
        // up matvec
        enc.set_buffer(2, &up, 0);
        enc.dispatch_threadgroups(MTLSize::new((ffn/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
        // silu
        enc.set_pipeline(&p_silu);
        enc.set_buffer(0, &gate, 0); enc.set_buffer(1, &up, 0); enc.set_buffer(2, &ffn_mid, 0);
        enc.dispatch_threads(MTLSize::new(ffn as u64, 1, 1), MTLSize::new(256, 1, 1));
        // down matvec
        enc.set_pipeline(&p_q4_0);
        enc.set_buffer(0, &w_q4_0_2048x8192, 0); enc.set_buffer(1, &ffn_mid, 0);
        enc.set_buffer(2, &x, 0);
        let cols = ffn;
        enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new((hidden/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
    });

    // ── Full forward pass simulation: 16 layers + output proj ──
    bench("16 layers + output proj (full forward sim)", ITERS / 5, &queue, |enc| {
        for _ in 0..n_layers {
            enc.set_pipeline(&p_q4_0);
            // Q
            enc.set_buffer(0, &w_q4_0_2048x2048, 0); enc.set_buffer(1, &norm_out, 0);
            enc.set_buffer(2, &q_buf, 0);
            enc.set_bytes(3, &hidden as *const u32 as *const c_void, 4);
            enc.dispatch_threadgroups(MTLSize::new((q_dim/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
            // K
            enc.set_buffer(2, &k_buf, 0);
            enc.dispatch_threadgroups(MTLSize::new((kv_dim/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
            // V
            enc.set_buffer(2, &v_buf, 0);
            enc.dispatch_threadgroups(MTLSize::new((kv_dim/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
            // O
            enc.set_buffer(1, &attn_out, 0); enc.set_buffer(2, &x, 0);
            enc.dispatch_threadgroups(MTLSize::new((hidden/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
            // gate
            enc.set_buffer(0, &w_q4_0_8192x2048, 0); enc.set_buffer(1, &norm_out, 0);
            enc.set_buffer(2, &gate, 0);
            enc.dispatch_threadgroups(MTLSize::new((ffn/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
            // up
            enc.set_buffer(2, &up, 0);
            enc.dispatch_threadgroups(MTLSize::new((ffn/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
            // down
            enc.set_buffer(0, &w_q4_0_2048x8192, 0); enc.set_buffer(1, &ffn_mid, 0);
            enc.set_buffer(2, &x, 0);
            let cols = ffn;
            enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
            enc.dispatch_threadgroups(MTLSize::new((hidden/8) as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
        // Output projection
        enc.set_pipeline(&p_q6k);
        enc.set_buffer(0, &w_q6k_vocab, 0); enc.set_buffer(1, &norm_out, 0);
        enc.set_buffer(2, &logits, 0);
        enc.set_bytes(3, &hidden as *const u32 as *const c_void, 4);
        enc.dispatch_threadgroups(MTLSize::new(vocab_tgs as u64, 1, 1), MTLSize::new(256, 1, 1));
    });

    // ── Batch matmul Q4_0 ──
    eprintln!("\n--- Batch Matmul (simdgroup_float8x8 tiled) ---");
    {
        let batch_src = crate::kernels_batch::all_batch_kernels();
        let batch_lib = dev.new_library_with_source(&batch_src).unwrap();
        let p_bm_q4_0 = dev.new_compute_pipeline(
            &batch_lib.get_function("matmul_q4_0").unwrap()).unwrap();

        // Test at various batch sizes with Llama-1B dimensions
        // Q proj: [2048 × 2048] @ X[2048 × B] → Y[2048 × B]
        for b in [1u32, 4, 8, 16, 25, 32, 64] {
            let x_batch = dev.new_buffer(hidden as u64 * b as u64 * 4);
            let y_batch = dev.new_buffer(hidden as u64 * b as u64 * 4);
            let rows_val = hidden;
            let cols_val = hidden;
            bench(&format!("matmul_q4_0 [{hidden}x{hidden}] B={b}"), ITERS, &queue, |enc| {
                enc.set_pipeline(&p_bm_q4_0);
                enc.set_buffer(0, &w_q4_0_2048x2048, 0);
                enc.set_buffer(1, &x_batch, 0);
                enc.set_buffer(2, &y_batch, 0);
                enc.set_bytes(3, &cols_val as *const u32 as *const c_void, 4);
                enc.set_bytes(4, &rows_val as *const u32 as *const c_void, 4);
                enc.set_bytes(5, &b as *const u32 as *const c_void, 4);
                let grid_x = (rows_val + 31) / 32;
                let grid_y = (b + 31) / 32;
                enc.dispatch_threadgroups(
                    MTLSize::new(grid_x as u64, grid_y as u64, 1),
                    MTLSize::new(128, 1, 1), // 4 simdgroups × 32 threads
                );
            });
        }

        // FFN gate: [8192 × 2048] @ X[2048 × B] → Y[8192 × B]
        for b in [1u32, 8, 25, 32] {
            let x_batch = dev.new_buffer(hidden as u64 * b as u64 * 4);
            let y_batch = dev.new_buffer(ffn as u64 * b as u64 * 4);
            let rows_val = ffn;
            let cols_val = hidden;
            bench(&format!("matmul_q4_0 [{ffn}x{hidden}] B={b}"), ITERS, &queue, |enc| {
                enc.set_pipeline(&p_bm_q4_0);
                enc.set_buffer(0, &w_q4_0_8192x2048, 0);
                enc.set_buffer(1, &x_batch, 0);
                enc.set_buffer(2, &y_batch, 0);
                enc.set_bytes(3, &cols_val as *const u32 as *const c_void, 4);
                enc.set_bytes(4, &rows_val as *const u32 as *const c_void, 4);
                enc.set_bytes(5, &b as *const u32 as *const c_void, 4);
                let grid_x = (rows_val + 31) / 32;
                let grid_y = (b + 31) / 32;
                enc.dispatch_threadgroups(
                    MTLSize::new(grid_x as u64, grid_y as u64, 1),
                    MTLSize::new(64, 1, 1),
                );
            });
        }
    }

    // ── Raw bandwidth test: just read N bytes through a kernel ──
    eprintln!("\n--- Raw Bandwidth ---");
    for mb in [10, 100, 500, 745] {
        let sz = mb * 1_000_000u64;
        let big = dev.new_buffer(sz);
        let out = dev.new_buffer(4);
        let src = format!(r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void bw_test(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                                uint gid [[thread_position_in_grid]], uint tgs [[threads_per_threadgroup]]) {{
                uint n = {n};
                float sum = 0.0;
                for (uint i = gid; i < n; i += {threads}) sum += x[i];
                if (gid == 0) out[0] = sum;
            }}
        "#, n = sz / 4, threads = 65536);
        let l = dev.new_library_with_source(&src).unwrap();
        let p = dev.new_compute_pipeline(&l.get_function("bw_test").unwrap()).unwrap();
        bench(&format!("read {mb}MB (bandwidth test)"), ITERS, &queue, |enc| {
            enc.set_pipeline(&p);
            enc.set_buffer(0, &big, 0);
            enc.set_buffer(1, &out, 0);
            enc.dispatch_threadgroups(
                MTLSize::new(256, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        });
    }

    eprintln!("\n=== Done ===");
}

fn bench(name: &str, iters: u32, queue: &CommandQueue, f: impl Fn(&ComputeEncoder)) {
    for _ in 0..WARMUP {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        f(&enc);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        f(&enc);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let elapsed = t0.elapsed();
    let per_iter = elapsed.as_nanos() as f64 / iters as f64;
    if per_iter > 1_000_000.0 {
        eprintln!("  {name:<58} {:.2}ms", per_iter / 1_000_000.0);
    } else {
        eprintln!("  {name:<58} {:.1}µs", per_iter / 1_000.0);
    }
}

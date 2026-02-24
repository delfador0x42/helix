//! Comprehensive matmul kernel benchmark: tests ALL kernel variants at ALL
//! relevant matrix sizes and batch counts. Output: TFLOP/s comparison table.
//!
//! Kernel types tested:
//!   v3b MMA    — Q4_0 quantized dequant + simdgroup staging (128-thread TG)
//!   FP16 MMA   — pre-dequant FP16 weights, direct device loads (128/256-thread)
//!   Scalar FP16 — Q4_0 quantized, half accumulators, simd_sum (256-thread)
//!   Matvec n=1  — scalar Q4_0/Q6K half, single-token decode path

use crate::gpu::*;
use crate::kernels_batch;
use crate::kernels_fp16;
use std::ffi::c_void;
use std::time::Instant;

const WARMUP: u32 = 5;
const ITERS: u32 = 30;

// Llama 3.2 1B dimensions
const H: u32 = 2048;
const FFN: u32 = 8192;
const KV_DIM: u32 = 512;
const Q_DIM: u32 = 2048;
const VOCAB: u32 = 128256;

/// Matrix sizes to benchmark: (rows, cols, label)
const SIZES: &[(u32, u32, &str)] = &[
    (Q_DIM, H,     "Q/O proj [2048×2048]"),
    (KV_DIM, H,    "K/V proj [512×2048]"),
    (FFN, H,       "Gate/Up  [8192×2048]"),
    (H, FFN,       "Down     [2048×8192]"),
    (VOCAB, H,     "Output   [128256×2048]"),
];

pub fn run() {
    let dev = Device::system_default().unwrap();
    let queue = dev.new_command_queue();
    eprintln!("Compiling batch kernels...");
    let batch_src = kernels_batch::all_batch_kernels();
    let batch_lib = dev.new_library_with_source(&batch_src).unwrap();

    // Pipelines
    let p_v3b = dev.new_compute_pipeline(&batch_lib.get_function("matmul_q4_0_mma").unwrap()).unwrap();
    let p_fp16_128 = dev.new_compute_pipeline(&batch_lib.get_function("matmul_fp16_mma_128").unwrap()).unwrap();
    let p_fp16_256 = dev.new_compute_pipeline(&batch_lib.get_function("matmul_fp16_mma").unwrap()).unwrap();
    let p_scalar = dev.new_compute_pipeline(&batch_lib.get_function("matmul_q4_0_f16").unwrap()).unwrap();
    let p_matvec = dev.new_compute_pipeline(&batch_lib.get_function("matvec_q4_0_f16").unwrap()).unwrap();
    let p_scalar_q6k = dev.new_compute_pipeline(&batch_lib.get_function("matmul_q6k_f16").unwrap()).unwrap();
    let p_matvec_q6k = dev.new_compute_pipeline(&batch_lib.get_function("matvec_q6k_f16").unwrap()).unwrap();

    eprintln!("\n================================================================================");
    eprintln!("  COMPREHENSIVE MATMUL BENCHMARK — Llama 3.2 1B, M3 Max");
    eprintln!("  FP16 MMA peak: 14.2 TFLOP/s, Memory BW: 400 GB/s");
    eprintln!("================================================================================\n");

    // ═══════════════════════════════════════════════════════════════
    // PART 1: Per-kernel isolated benchmarks at each matrix size
    // ═══════════════════════════════════════════════════════════════
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PART 1: Isolated Kernel Benchmarks (B=80)                      ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝\n");

    let bs = kernels_batch::BATCH_SIZE;

    for &(rows, cols, label) in SIZES {
        let flop = 2.0 * rows as f64 * cols as f64 * bs as f64;

        // Q4_0 weight buffer
        let q4_bytes = q4_0_buf_size(rows, cols);
        let w_q4 = dev.new_buffer(q4_bytes);

        // FP16 weight buffer (row-major, rows × cols × 2 bytes)
        let w_fp16 = dev.new_buffer(rows as u64 * cols as u64 * 2);

        // Q6K weight buffer (for output projection)
        let q6k_bytes = q6k_buf_size(rows, cols);
        let w_q6k = dev.new_buffer(q6k_bytes);

        // X/Y FP16 buffers
        let x16 = dev.new_buffer(cols as u64 * bs as u64 * 2);
        let y16 = dev.new_buffer(rows as u64 * bs as u64 * 2);

        eprintln!("  --- {label} (B={bs}) ---");

        // v3b MMA: grid=(ceil(rows/32), 1), TG=128
        let t = bench_kernel(ITERS, &queue, |enc| {
            enc.set_pipeline(&p_v3b);
            enc.set_buffer(0, &w_q4, 0);
            enc.set_buffer(1, &x16, 0);
            enc.set_buffer(2, &y16, 0);
            enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
            enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
            enc.set_bytes(5, &bs as *const u32 as *const c_void, 4);
            enc.dispatch_threadgroups(
                MTLSize::new(((rows + 31) / 32) as u64, 1, 1),
                MTLSize::new(128, 1, 1),
            );
        });
        print_result("v3b MMA (Q4_0, 128t)", t, flop);

        // FP16 MMA 128-thread: grid=(ceil(rows/32), 1), TG=128
        let t = bench_kernel(ITERS, &queue, |enc| {
            enc.set_pipeline(&p_fp16_128);
            enc.set_buffer(0, &w_fp16, 0);
            enc.set_buffer(1, &x16, 0);
            enc.set_buffer(2, &y16, 0);
            enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
            enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
            enc.set_bytes(5, &bs as *const u32 as *const c_void, 4);
            enc.dispatch_threadgroups(
                MTLSize::new(((rows + 31) / 32) as u64, 1, 1),
                MTLSize::new(128, 1, 1),
            );
        });
        print_result("FP16 MMA (128t)", t, flop);

        // FP16 MMA 256-thread: grid=(ceil(rows/64), 1), TG=256
        let t = bench_kernel(ITERS, &queue, |enc| {
            enc.set_pipeline(&p_fp16_256);
            enc.set_buffer(0, &w_fp16, 0);
            enc.set_buffer(1, &x16, 0);
            enc.set_buffer(2, &y16, 0);
            enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
            enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
            enc.set_bytes(5, &bs as *const u32 as *const c_void, 4);
            enc.dispatch_threadgroups(
                MTLSize::new(((rows + 63) / 64) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        });
        print_result("FP16 MMA (256t)", t, flop);

        // Scalar FP16 B=80 (Q4_0 weights, half accumulators)
        let t = bench_kernel(ITERS, &queue, |enc| {
            enc.set_pipeline(&p_scalar);
            enc.set_buffer(0, &w_q4, 0);
            enc.set_buffer(1, &x16, 0);
            enc.set_buffer(2, &y16, 0);
            enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
            enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
            enc.set_bytes(5, &bs as *const u32 as *const c_void, 4);
            enc.dispatch_threadgroups(
                MTLSize::new(((rows + 7) / 8) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        });
        print_result("Scalar FP16 B=80 (Q4_0)", t, flop);

        // For output projection, also test Q6K variants
        if rows == VOCAB {
            let t = bench_kernel(ITERS, &queue, |enc| {
                enc.set_pipeline(&p_scalar_q6k);
                enc.set_buffer(0, &w_q6k, 0);
                enc.set_buffer(1, &x16, 0);
                enc.set_buffer(2, &y16, 0);
                enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
                enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
                enc.set_bytes(5, &bs as *const u32 as *const c_void, 4);
                enc.dispatch_threadgroups(
                    MTLSize::new(((rows + 7) / 8) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            });
            print_result("Scalar FP16 B=80 (Q6K)", t, flop);
        }

        eprintln!();
    }

    // ═══════════════════════════════════════════════════════════════
    // PART 2: Batch size sweep — v3b MMA vs FP16 MMA at [2048×2048]
    // ═══════════════════════════════════════════════════════════════
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PART 2: Batch Size Sweep [2048×2048] — v3b vs FP16 MMA        ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝\n");

    eprintln!("  {:>4}  {:>12}  {:>12}  {:>12}  {:>12}",
        "B", "v3b (ms)", "FP16-128 (ms)", "FP16-256 (ms)", "Winner");
    eprintln!("  {:->4}  {:->12}  {:->12}  {:->12}  {:->12}", "", "", "", "", "");

    let w_q4 = dev.new_buffer(q4_0_buf_size(H, H));
    let w_fp16 = dev.new_buffer(H as u64 * H as u64 * 2);

    for b in [1u32, 8, 16, 32, 48, 64, 80, 96, 112, 128, 160] {
        let buf_b = b.max(bs); // pad to BATCH_SIZE for unconditional reads
        let x16 = dev.new_buffer(H as u64 * buf_b as u64 * 2);
        let y16 = dev.new_buffer(H as u64 * buf_b as u64 * 2);

        // v3b
        let t_v3b = bench_kernel(ITERS, &queue, |enc| {
            enc.set_pipeline(&p_v3b);
            enc.set_buffer(0, &w_q4, 0);
            enc.set_buffer(1, &x16, 0);
            enc.set_buffer(2, &y16, 0);
            enc.set_bytes(3, &H as *const u32 as *const c_void, 4);
            enc.set_bytes(4, &H as *const u32 as *const c_void, 4);
            enc.set_bytes(5, &b as *const u32 as *const c_void, 4);
            enc.dispatch_threadgroups(
                MTLSize::new(((H + 31) / 32) as u64, 1, 1),
                MTLSize::new(128, 1, 1),
            );
        });

        // FP16 MMA 128-thread
        let t_f128 = bench_kernel(ITERS, &queue, |enc| {
            enc.set_pipeline(&p_fp16_128);
            enc.set_buffer(0, &w_fp16, 0);
            enc.set_buffer(1, &x16, 0);
            enc.set_buffer(2, &y16, 0);
            enc.set_bytes(3, &H as *const u32 as *const c_void, 4);
            enc.set_bytes(4, &H as *const u32 as *const c_void, 4);
            enc.set_bytes(5, &b as *const u32 as *const c_void, 4);
            enc.dispatch_threadgroups(
                MTLSize::new(((H + 31) / 32) as u64, 1, 1),
                MTLSize::new(128, 1, 1),
            );
        });

        // FP16 MMA 256-thread
        let t_f256 = bench_kernel(ITERS, &queue, |enc| {
            enc.set_pipeline(&p_fp16_256);
            enc.set_buffer(0, &w_fp16, 0);
            enc.set_buffer(1, &x16, 0);
            enc.set_buffer(2, &y16, 0);
            enc.set_bytes(3, &H as *const u32 as *const c_void, 4);
            enc.set_bytes(4, &H as *const u32 as *const c_void, 4);
            enc.set_bytes(5, &b as *const u32 as *const c_void, 4);
            enc.dispatch_threadgroups(
                MTLSize::new(((H + 63) / 64) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        });

        let min = t_v3b.min(t_f128).min(t_f256);
        let winner = if min == t_v3b { "v3b" }
            else if min == t_f128 { "FP16-128" }
            else { "FP16-256" };

        eprintln!("  {:>4}  {:>10.3}ms  {:>10.3}ms  {:>10.3}ms  {:>12}",
            b, t_v3b, t_f128, t_f256, winner);
    }

    // ═══════════════════════════════════════════════════════════════
    // PART 3: n=1 matvec benchmark (decode path)
    // ═══════════════════════════════════════════════════════════════
    eprintln!("\n╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PART 3: n=1 Matvec (Decode Path)                              ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝\n");

    let b1: u32 = 1;
    for &(rows, cols, label) in SIZES {
        let flop = 2.0 * rows as f64 * cols as f64;
        let w_q4 = dev.new_buffer(q4_0_buf_size(rows, cols));
        let w_q6k = dev.new_buffer(q6k_buf_size(rows, cols));
        let x16 = dev.new_buffer(cols as u64 * 2);
        let y16 = dev.new_buffer(rows as u64 * 2);

        eprintln!("  {label}:");

        // Matvec Q4_0 n=1
        let t = bench_kernel(ITERS, &queue, |enc| {
            enc.set_pipeline(&p_matvec);
            enc.set_buffer(0, &w_q4, 0);
            enc.set_buffer(1, &x16, 0);
            enc.set_buffer(2, &y16, 0);
            enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
            enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
            enc.set_bytes(5, &b1 as *const u32 as *const c_void, 4);
            enc.dispatch_threadgroups(
                MTLSize::new(((rows + 7) / 8) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        });
        print_result("matvec Q4_0 (n=1)", t, flop);

        if rows == VOCAB {
            let t = bench_kernel(ITERS, &queue, |enc| {
                enc.set_pipeline(&p_matvec_q6k);
                enc.set_buffer(0, &w_q6k, 0);
                enc.set_buffer(1, &x16, 0);
                enc.set_buffer(2, &y16, 0);
                enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
                enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
                enc.set_bytes(5, &b1 as *const u32 as *const c_void, 4);
                enc.dispatch_threadgroups(
                    MTLSize::new(((rows + 7) / 8) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            });
            print_result("matvec Q6K (n=1)", t, flop);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // PART 4: Full forward simulation (matmuls only) — all strategies
    // ═══════════════════════════════════════════════════════════════
    eprintln!("\n╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PART 4: Full Forward Simulation (16 layers + output, matmul)   ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝\n");

    let n_layers = 16u32;
    // Per-token FLOP: 7 matmuls per layer × 16 layers + output
    // Layer: Q(2048×2048) + K(512×2048) + V(512×2048) + O(2048×2048) + gate(8192×2048) + up(8192×2048) + down(2048×8192)
    // = 2*2048*2048 + 2*512*2048 + 2*512*2048 + 2*2048*2048 + 2*8192*2048 + 2*8192*2048 + 2*2048*8192
    // = 2 * (2048² + 512*2048 + 512*2048 + 2048² + 8192*2048 + 8192*2048 + 2048*8192)
    // = 2 * (4194304 + 1048576 + 1048576 + 4194304 + 16777216 + 16777216 + 16777216)
    // = 2 * 60817408 = 121634816 per layer
    let flop_per_layer = 2.0 * (2.0 * H as f64 * H as f64 + 2.0 * KV_DIM as f64 * H as f64
        + 3.0 * FFN as f64 * H as f64) as f64;
    let flop_output = 2.0 * VOCAB as f64 * H as f64;

    // Weight buffers for full forward
    let w_q4_hh = dev.new_buffer(q4_0_buf_size(H, H));
    let w_q4_fh = dev.new_buffer(q4_0_buf_size(FFN, H));
    let w_q4_hf = dev.new_buffer(q4_0_buf_size(H, FFN));
    let w_q4_vh = dev.new_buffer(q4_0_buf_size(VOCAB, H));
    let w_q6k_vh = dev.new_buffer(q6k_buf_size(VOCAB, H));
    let w_fp16_hh = dev.new_buffer(H as u64 * H as u64 * 2);
    let w_fp16_fh = dev.new_buffer(FFN as u64 * H as u64 * 2);
    let w_fp16_hf = dev.new_buffer(H as u64 * FFN as u64 * 2);
    let w_fp16_vh = dev.new_buffer(VOCAB as u64 * H as u64 * 2);

    let fwd_iters = 10u32;

    for b in [1u32, 40, 80, 96, 128] {
        let buf_b = b.max(bs);
        let max_dim = FFN.max(H).max(VOCAB) as u64;
        let x16 = dev.new_buffer(max_dim * buf_b as u64 * 2);
        let y_h = dev.new_buffer(H as u64 * buf_b as u64 * 2);
        let y_kv = dev.new_buffer(KV_DIM as u64 * buf_b as u64 * 2);
        let y_f = dev.new_buffer(FFN as u64 * buf_b as u64 * 2);
        let y_v = dev.new_buffer(VOCAB as u64 * buf_b as u64 * 2);

        let total_flop = (flop_per_layer * n_layers as f64 + flop_output) * b as f64;

        eprintln!("  B={b}:");

        // Strategy 1: v3b MMA for all (current production for Q4_0)
        let t = bench_kernel(fwd_iters, &queue, |enc| {
            for _ in 0..n_layers {
                dispatch_v3b(enc, &p_v3b, &w_q4_hh, &x16, &y_h, H, Q_DIM, b);
                dispatch_v3b(enc, &p_v3b, &w_q4_hh, &x16, &y_kv, H, KV_DIM, b);
                dispatch_v3b(enc, &p_v3b, &w_q4_hh, &x16, &y_kv, H, KV_DIM, b);
                dispatch_v3b(enc, &p_v3b, &w_q4_hh, &x16, &y_h, H, H, b);  // O proj: cols=q_dim=H
                dispatch_v3b(enc, &p_v3b, &w_q4_fh, &x16, &y_f, H, FFN, b);
                dispatch_v3b(enc, &p_v3b, &w_q4_fh, &x16, &y_f, H, FFN, b);
                dispatch_v3b(enc, &p_v3b, &w_q4_hf, &x16, &y_h, FFN, H, b);
            }
            dispatch_v3b(enc, &p_v3b, &w_q4_vh, &x16, &y_v, H, VOCAB, b);
        });
        let tps = b as f64 / (t / 1000.0);
        let tflops = total_flop / (t / 1000.0) / 1e12;
        eprintln!("    v3b all:      {t:>8.2}ms  {tps:>7.0} tok/s  {tflops:.2} TFLOP/s ({:.1}%)",
            tflops / 14.2 * 100.0);

        // Strategy 2: FP16 MMA 128-thread for all
        let t = bench_kernel(fwd_iters, &queue, |enc| {
            for _ in 0..n_layers {
                dispatch_fp16(enc, &p_fp16_128, &w_fp16_hh, &x16, &y_h, H, Q_DIM, b, 32);
                dispatch_fp16(enc, &p_fp16_128, &w_fp16_hh, &x16, &y_kv, H, KV_DIM, b, 32);
                dispatch_fp16(enc, &p_fp16_128, &w_fp16_hh, &x16, &y_kv, H, KV_DIM, b, 32);
                dispatch_fp16(enc, &p_fp16_128, &w_fp16_hh, &x16, &y_h, H, H, b, 32);
                dispatch_fp16(enc, &p_fp16_128, &w_fp16_fh, &x16, &y_f, H, FFN, b, 32);
                dispatch_fp16(enc, &p_fp16_128, &w_fp16_fh, &x16, &y_f, H, FFN, b, 32);
                dispatch_fp16(enc, &p_fp16_128, &w_fp16_hf, &x16, &y_h, FFN, H, b, 32);
            }
            dispatch_fp16(enc, &p_fp16_128, &w_fp16_vh, &x16, &y_v, H, VOCAB, b, 32);
        });
        let tps = b as f64 / (t / 1000.0);
        let tflops = total_flop / (t / 1000.0) / 1e12;
        eprintln!("    FP16-128 all:  {t:>8.2}ms  {tps:>7.0} tok/s  {tflops:.2} TFLOP/s ({:.1}%)",
            tflops / 14.2 * 100.0);

        // Strategy 3: FP16 MMA 256-thread for all
        let t = bench_kernel(fwd_iters, &queue, |enc| {
            for _ in 0..n_layers {
                dispatch_fp16(enc, &p_fp16_256, &w_fp16_hh, &x16, &y_h, H, Q_DIM, b, 64);
                dispatch_fp16(enc, &p_fp16_256, &w_fp16_hh, &x16, &y_kv, H, KV_DIM, b, 64);
                dispatch_fp16(enc, &p_fp16_256, &w_fp16_hh, &x16, &y_kv, H, KV_DIM, b, 64);
                dispatch_fp16(enc, &p_fp16_256, &w_fp16_hh, &x16, &y_h, H, H, b, 64);
                dispatch_fp16(enc, &p_fp16_256, &w_fp16_fh, &x16, &y_f, H, FFN, b, 64);
                dispatch_fp16(enc, &p_fp16_256, &w_fp16_fh, &x16, &y_f, H, FFN, b, 64);
                dispatch_fp16(enc, &p_fp16_256, &w_fp16_hf, &x16, &y_h, FFN, H, b, 64);
            }
            dispatch_fp16(enc, &p_fp16_256, &w_fp16_vh, &x16, &y_v, H, VOCAB, b, 64);
        });
        let tps = b as f64 / (t / 1000.0);
        let tflops = total_flop / (t / 1000.0) / 1e12;
        eprintln!("    FP16-256 all:  {t:>8.2}ms  {tps:>7.0} tok/s  {tflops:.2} TFLOP/s ({:.1}%)",
            tflops / 14.2 * 100.0);

        // Strategy 4: Scalar FP16 B=N (Q4_0 quantized)
        if b <= bs {
            let t = bench_kernel(fwd_iters, &queue, |enc| {
                for _ in 0..n_layers {
                    dispatch_scalar(enc, &p_scalar, &w_q4_hh, &x16, &y_h, H, Q_DIM, b);
                    dispatch_scalar(enc, &p_scalar, &w_q4_hh, &x16, &y_kv, H, KV_DIM, b);
                    dispatch_scalar(enc, &p_scalar, &w_q4_hh, &x16, &y_kv, H, KV_DIM, b);
                    dispatch_scalar(enc, &p_scalar, &w_q4_hh, &x16, &y_h, H, H, b);
                    dispatch_scalar(enc, &p_scalar, &w_q4_fh, &x16, &y_f, H, FFN, b);
                    dispatch_scalar(enc, &p_scalar, &w_q4_fh, &x16, &y_f, H, FFN, b);
                    dispatch_scalar(enc, &p_scalar, &w_q4_hf, &x16, &y_h, FFN, H, b);
                }
                dispatch_scalar(enc, &p_scalar_q6k, &w_q6k_vh, &x16, &y_v, H, VOCAB, b);
            });
            let tps = b as f64 / (t / 1000.0);
            let tflops = total_flop / (t / 1000.0) / 1e12;
            eprintln!("    Scalar FP16:   {t:>8.2}ms  {tps:>7.0} tok/s  {tflops:.2} TFLOP/s ({:.1}%)",
                tflops / 14.2 * 100.0);
        }

        eprintln!();
    }

    // ═══════════════════════════════════════════════════════════════
    // PART 4B: v3b Batch Scaling — real kernel recompilation per B
    // ═══════════════════════════════════════════════════════════════
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PART 4B: v3b Real Batch Scaling (recompile kernel per B)       ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝\n");

    eprintln!("  {:>4}  {:>10}  {:>10}  {:>10}  {:>10}",
        "B", "Time(ms)", "tok/s", "TFLOP/s", "% peak");
    eprintln!("  {:->4}  {:->10}  {:->10}  {:->10}  {:->10}", "", "", "", "", "");

    for b in [80u32, 96, 112, 128, 160, 192, 200, 224, 256, 320] {
        // Generate and compile v3b with this exact batch size
        let mut src = kernels_batch::HEADER_STR.to_string();
        src += &kernels_batch::gen_matmul_q4_0_mma_named("v3b_test", b);
        let lib = match dev.new_library_with_source(&src) {
            Ok(l) => l,
            Err(e) => { eprintln!("  B={b}: compile FAILED: {e}"); continue; }
        };
        let pipe = match lib.get_function("v3b_test") {
            Ok(f) => dev.new_compute_pipeline(&f).unwrap(),
            Err(e) => { eprintln!("  B={b}: pipeline FAILED: {e}"); continue; }
        };

        // Full forward sim with this kernel
        let max_dim = FFN.max(H).max(VOCAB) as u64;
        let x16 = dev.new_buffer(max_dim * b as u64 * 2);
        let y_h = dev.new_buffer(H as u64 * b as u64 * 2);
        let y_kv = dev.new_buffer(KV_DIM as u64 * b as u64 * 2);
        let y_f = dev.new_buffer(FFN as u64 * b as u64 * 2);
        let y_v = dev.new_buffer(VOCAB as u64 * b as u64 * 2);

        let total_flop = (flop_per_layer * n_layers as f64 + flop_output) * b as f64;

        let t = bench_kernel(fwd_iters, &queue, |enc| {
            for _ in 0..n_layers {
                dispatch_v3b(enc, &pipe, &w_q4_hh, &x16, &y_h, H, Q_DIM, b);
                dispatch_v3b(enc, &pipe, &w_q4_hh, &x16, &y_kv, H, KV_DIM, b);
                dispatch_v3b(enc, &pipe, &w_q4_hh, &x16, &y_kv, H, KV_DIM, b);
                dispatch_v3b(enc, &pipe, &w_q4_hh, &x16, &y_h, H, H, b);
                dispatch_v3b(enc, &pipe, &w_q4_fh, &x16, &y_f, H, FFN, b);
                dispatch_v3b(enc, &pipe, &w_q4_fh, &x16, &y_f, H, FFN, b);
                dispatch_v3b(enc, &pipe, &w_q4_hf, &x16, &y_h, FFN, H, b);
            }
            dispatch_v3b(enc, &pipe, &w_q4_vh, &x16, &y_v, H, VOCAB, b);
        });

        let tps = b as f64 / (t / 1000.0);
        let tflops = total_flop / (t / 1000.0) / 1e12;
        let pct = tflops / 14.2 * 100.0;
        eprintln!("  {:>4}  {:>8.2}ms  {:>8.0}  {:>8.2}  {:>8.1}%", b, t, tps, tflops, pct);
    }

    eprintln!();

    // ═══════════════════════════════════════════════════════════════
    // PART 4C: Tiled v3b — constant register pressure, any batch size
    // ═══════════════════════════════════════════════════════════════
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PART 4C: Tiled v3b (weight staged once, batch tiled)           ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝\n");

    eprintln!("  {:>4}  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}",
        "B", "TileB", "Time(ms)", "tok/s", "TFLOP/s", "% peak");
    eprintln!("  {:->4}  {:->6}  {:->10}  {:->10}  {:->10}  {:->10}", "", "", "", "", "", "");

    // Test various B with different tile sizes
    for (b, tile_b) in [(80u32, 8u32), (80, 16), (80, 40), (80, 80),
                         (128, 8), (128, 16), (128, 32), (128, 64),
                         (160, 8), (160, 16), (160, 32), (160, 80),
                         (256, 8), (256, 16), (256, 32), (256, 64),
                         (320, 8), (320, 16), (320, 32), (320, 80)] {
        if b % tile_b != 0 { continue; }
        let name = format!("tiled_v3b_b{b}_t{tile_b}");
        let mut src = kernels_batch::HEADER_STR.to_string();
        src += &kernels_batch::gen_matmul_q4_0_mma_tiled(&name, b, tile_b);
        let lib = match dev.new_library_with_source(&src) {
            Ok(l) => l,
            Err(e) => { eprintln!("  B={b} T={tile_b}: compile FAILED: {e}"); continue; }
        };
        let pipe = match lib.get_function(&name) {
            Ok(f) => dev.new_compute_pipeline(&f).unwrap(),
            Err(e) => { eprintln!("  B={b} T={tile_b}: pipeline FAILED: {e}"); continue; }
        };

        let max_dim = FFN.max(H).max(VOCAB) as u64;
        let x16 = dev.new_buffer(max_dim * b as u64 * 2);
        let y_h = dev.new_buffer(H as u64 * b as u64 * 2);
        let y_kv = dev.new_buffer(KV_DIM as u64 * b as u64 * 2);
        let y_f = dev.new_buffer(FFN as u64 * b as u64 * 2);
        let y_v = dev.new_buffer(VOCAB as u64 * b as u64 * 2);

        let total_flop = (flop_per_layer * n_layers as f64 + flop_output) * b as f64;

        let t = bench_kernel(fwd_iters, &queue, |enc| {
            for _ in 0..n_layers {
                dispatch_v3b(enc, &pipe, &w_q4_hh, &x16, &y_h, H, Q_DIM, b);
                dispatch_v3b(enc, &pipe, &w_q4_hh, &x16, &y_kv, H, KV_DIM, b);
                dispatch_v3b(enc, &pipe, &w_q4_hh, &x16, &y_kv, H, KV_DIM, b);
                dispatch_v3b(enc, &pipe, &w_q4_hh, &x16, &y_h, H, H, b);
                dispatch_v3b(enc, &pipe, &w_q4_fh, &x16, &y_f, H, FFN, b);
                dispatch_v3b(enc, &pipe, &w_q4_fh, &x16, &y_f, H, FFN, b);
                dispatch_v3b(enc, &pipe, &w_q4_hf, &x16, &y_h, FFN, H, b);
            }
            dispatch_v3b(enc, &pipe, &w_q4_vh, &x16, &y_v, H, VOCAB, b);
        });

        let tps = b as f64 / (t / 1000.0);
        let tflops = total_flop / (t / 1000.0) / 1e12;
        let pct = tflops / 14.2 * 100.0;
        eprintln!("  {:>4}  {:>6}  {:>8.2}ms  {:>8.0}  {:>8.2}  {:>8.1}%",
            b, tile_b, t, tps, tflops, pct);
    }

    eprintln!();

    // ═══════════════════════════════════════════════════════════════
    // PART 4D: Grid-tiled v3b — batch tiling via GPU grid_y
    // ═══════════════════════════════════════════════════════════════
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PART 4D: Grid-tiled v3b (batch via grid_y, parallel tiles)    ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝\n");

    eprintln!("  {:>4}  {:>6}  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}",
        "B", "TileB", "TGs", "Time(ms)", "tok/s", "TFLOP/s", "% peak");
    eprintln!("  {:->4}  {:->6}  {:->6}  {:->10}  {:->10}  {:->10}  {:->10}",
        "", "", "", "", "", "", "");

    for (b, tile_b) in [(80u32, 80u32),
                         (160, 80), (240, 80), (320, 80),
                         (480, 80), (640, 80), (800, 80),
                         (960, 80), (1280, 80), (1600, 80),
                         (2400, 80), (3200, 80)] {
        if b % tile_b != 0 { continue; }
        let name = format!("grid_v3b_t{tile_b}");
        let mut src = kernels_batch::HEADER_STR.to_string();
        src += &kernels_batch::gen_matmul_q4_0_mma_grid(&name, tile_b);
        let lib = match dev.new_library_with_source(&src) {
            Ok(l) => l,
            Err(e) => { eprintln!("  B={b} T={tile_b}: compile FAILED: {e}"); continue; }
        };
        let pipe = match lib.get_function(&name) {
            Ok(f) => dev.new_compute_pipeline(&f).unwrap(),
            Err(e) => { eprintln!("  B={b} T={tile_b}: pipeline FAILED: {e}"); continue; }
        };

        let max_dim = FFN.max(H).max(VOCAB) as u64;
        let x16 = dev.new_buffer(max_dim * b as u64 * 2);
        let y_h = dev.new_buffer(H as u64 * b as u64 * 2);
        let y_kv = dev.new_buffer(KV_DIM as u64 * b as u64 * 2);
        let y_f = dev.new_buffer(FFN as u64 * b as u64 * 2);
        let y_v = dev.new_buffer(VOCAB as u64 * b as u64 * 2);

        let total_flop = (flop_per_layer * n_layers as f64 + flop_output) * b as f64;
        let b_tiles = (b + tile_b - 1) / tile_b;

        // Count total TGs for the largest matmul (output: 128256 rows)
        let max_tgs = ((VOCAB + 31) / 32) as u64 * b_tiles as u64;

        let t = bench_kernel(fwd_iters, &queue, |enc| {
            for _ in 0..n_layers {
                dispatch_grid(enc, &pipe, &w_q4_hh, &x16, &y_h, H, Q_DIM, b, tile_b);
                dispatch_grid(enc, &pipe, &w_q4_hh, &x16, &y_kv, H, KV_DIM, b, tile_b);
                dispatch_grid(enc, &pipe, &w_q4_hh, &x16, &y_kv, H, KV_DIM, b, tile_b);
                dispatch_grid(enc, &pipe, &w_q4_hh, &x16, &y_h, H, H, b, tile_b);
                dispatch_grid(enc, &pipe, &w_q4_fh, &x16, &y_f, H, FFN, b, tile_b);
                dispatch_grid(enc, &pipe, &w_q4_fh, &x16, &y_f, H, FFN, b, tile_b);
                dispatch_grid(enc, &pipe, &w_q4_hf, &x16, &y_h, FFN, H, b, tile_b);
            }
            dispatch_grid(enc, &pipe, &w_q4_vh, &x16, &y_v, H, VOCAB, b, tile_b);
        });

        let tps = b as f64 / (t / 1000.0);
        let tflops = total_flop / (t / 1000.0) / 1e12;
        let pct = tflops / 14.2 * 100.0;
        eprintln!("  {:>4}  {:>6}  {:>6}  {:>8.2}ms  {:>8.0}  {:>8.2}  {:>8.1}%",
            b, tile_b, max_tgs, t, tps, tflops, pct);
    }

    eprintln!();

    // ═══════════════════════════════════════════════════════════════
    // PART 5: Bandwidth ceiling test
    // ═══════════════════════════════════════════════════════════════
    eprintln!("╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PART 5: Memory Bandwidth Ceiling                               ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝\n");

    for mb in [100, 500, 745, 2400] {
        let sz = mb as u64 * 1_000_000;
        let big = dev.new_buffer(sz);
        let out = dev.new_buffer(4);
        let n_elems = sz / 2; // half precision
        let threads = 65536u32;
        let src = format!(r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void bw_half(device const half *x [[buffer(0)]], device half *out [[buffer(1)]],
                                uint gid [[thread_position_in_grid]]) {{
                uint n = {n_elems};
                half sum = 0.0h;
                for (uint i = gid; i < n; i += {threads}) sum += x[i];
                if (gid == 0) out[0] = sum;
            }}
        "#);
        let l = dev.new_library_with_source(&src).unwrap();
        let p = dev.new_compute_pipeline(&l.get_function("bw_half").unwrap()).unwrap();
        let t = bench_kernel(ITERS, &queue, |enc| {
            enc.set_pipeline(&p);
            enc.set_buffer(0, &big, 0);
            enc.set_buffer(1, &out, 0);
            enc.dispatch_threadgroups(
                MTLSize::new(256, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        });
        let bw = sz as f64 / (t / 1000.0) / 1e9;
        eprintln!("  Read {mb}MB FP16: {t:.3}ms = {bw:.0} GB/s");
    }

    // ═══════════════════════════════════════════════════════════════
    // PART 6: Theoretical analysis
    // ═══════════════════════════════════════════════════════════════
    eprintln!("\n╔══════════════════════════════════════════════════════════════════╗");
    eprintln!("║  PART 6: Theoretical Ceiling Analysis                           ║");
    eprintln!("╚══════════════════════════════════════════════════════════════════╝\n");

    let flop_per_token = flop_per_layer * n_layers as f64 + flop_output;
    eprintln!("  FLOP per token: {:.3} GFLOP", flop_per_token / 1e9);
    eprintln!("  FP16 MMA peak:  14.2 TFLOP/s → {:.0} tok/s ceiling",
        14.2e12 / flop_per_token);
    eprintln!("  INT8 MMA peak:  ~28.4 TFLOP/s → {:.0} tok/s ceiling",
        28.4e12 / flop_per_token);
    eprintln!("  At 39.5% util: {:.0} tok/s (current v3b efficiency)",
        14.2e12 * 0.395 / flop_per_token);
    eprintln!("  At 60% util:   {:.0} tok/s (aggressive target)",
        14.2e12 * 0.60 / flop_per_token);
    eprintln!("  At 70% util:   {:.0} tok/s (very aggressive target)",
        14.2e12 * 0.70 / flop_per_token);

    // Q4_0 weight bytes per token
    let q4_bytes_per_token = flop_per_token / 2.0 * 18.0 / 32.0 / 8.0;
    // FP16 weight bytes per token
    let fp16_bytes_per_token = flop_per_token; // 2 bytes per element, 2 FLOP per MAC → 1 byte/FLOP
    eprintln!("\n  Weight memory per token:");
    eprintln!("    Q4_0:  {:.1} MB (bandwidth: {:.0} GB/s at 2262 tok/s)",
        q4_bytes_per_token / 1e6, q4_bytes_per_token * 2262.0 / 1e9);
    eprintln!("    FP16:  {:.1} MB (bandwidth: {:.0} GB/s at 2262 tok/s)",
        fp16_bytes_per_token / 1e6, fp16_bytes_per_token * 2262.0 / 1e9);

    eprintln!("\n=== Done ===");
}

// ── Helper functions ──

fn q4_0_buf_size(rows: u32, cols: u32) -> u64 {
    let bpr = cols as u64 / 32;
    bpr * 18 * rows as u64
}

fn q6k_buf_size(rows: u32, cols: u32) -> u64 {
    let bpr = cols as u64 / 256;
    bpr * 210 * rows as u64
}

fn dispatch_v3b(
    enc: &ComputeEncoder, pipe: &Pipeline, w: &Buffer,
    x: &Buffer, y: &Buffer, cols: u32, rows: u32, b: u32,
) {
    enc.set_pipeline(pipe);
    enc.set_buffer(0, w, 0);
    enc.set_buffer(1, x, 0);
    enc.set_buffer(2, y, 0);
    enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
    enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &b as *const u32 as *const c_void, 4);
    enc.dispatch_threadgroups(
        MTLSize::new(((rows + 31) / 32) as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
}

fn dispatch_fp16(
    enc: &ComputeEncoder, pipe: &Pipeline, w: &Buffer,
    x: &Buffer, y: &Buffer, cols: u32, rows: u32, b: u32, rows_per_tg: u32,
) {
    enc.set_pipeline(pipe);
    enc.set_buffer(0, w, 0);
    enc.set_buffer(1, x, 0);
    enc.set_buffer(2, y, 0);
    enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
    enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &b as *const u32 as *const c_void, 4);
    enc.dispatch_threadgroups(
        MTLSize::new(((rows + rows_per_tg - 1) / rows_per_tg) as u64, 1, 1),
        MTLSize::new(if rows_per_tg == 64 { 256 } else { 128 }, 1, 1),
    );
}

fn dispatch_grid(
    enc: &ComputeEncoder, pipe: &Pipeline, w: &Buffer,
    x: &Buffer, y: &Buffer, cols: u32, rows: u32, b: u32, tile_b: u32,
) {
    enc.set_pipeline(pipe);
    enc.set_buffer(0, w, 0);
    enc.set_buffer(1, x, 0);
    enc.set_buffer(2, y, 0);
    enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
    enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &b as *const u32 as *const c_void, 4);
    let row_groups = ((rows + 31) / 32) as u64;
    let batch_tiles = ((b + tile_b - 1) / tile_b) as u64;
    enc.dispatch_threadgroups(
        MTLSize::new(row_groups, batch_tiles, 1),
        MTLSize::new(128, 1, 1),
    );
}

fn dispatch_scalar(
    enc: &ComputeEncoder, pipe: &Pipeline, w: &Buffer,
    x: &Buffer, y: &Buffer, cols: u32, rows: u32, b: u32,
) {
    enc.set_pipeline(pipe);
    enc.set_buffer(0, w, 0);
    enc.set_buffer(1, x, 0);
    enc.set_buffer(2, y, 0);
    enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
    enc.set_bytes(4, &rows as *const u32 as *const c_void, 4);
    enc.set_bytes(5, &b as *const u32 as *const c_void, 4);
    enc.dispatch_threadgroups(
        MTLSize::new(((rows + 7) / 8) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

fn bench_kernel(iters: u32, queue: &CommandQueue, f: impl Fn(&ComputeEncoder)) -> f64 {
    // Warmup
    for _ in 0..WARMUP {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        f(&enc);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    // Measure
    let t0 = Instant::now();
    for _ in 0..iters {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        f(&enc);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    t0.elapsed().as_nanos() as f64 / iters as f64 / 1_000_000.0
}

fn print_result(label: &str, ms: f64, flop: f64) {
    let tflops = flop / (ms / 1000.0) / 1e12;
    let pct = tflops / 14.2 * 100.0;
    if ms < 0.001 {
        eprintln!("    {label:<30} {:.1}µs  {tflops:.2} TFLOP/s ({pct:.1}%)", ms * 1000.0);
    } else {
        eprintln!("    {label:<30} {ms:.3}ms  {tflops:.2} TFLOP/s ({pct:.1}%)");
    }
}

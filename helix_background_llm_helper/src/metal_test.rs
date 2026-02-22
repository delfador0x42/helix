//! Metal GPU benchmark: bandwidth, matmul throughput, and matvec (the LLM bottleneck).
//! tok/s = memory_bandwidth / model_weight_bytes — matvec is the ceiling.

use crate::gpu::*;
use std::ffi::c_void;
use std::time::Instant;

const BANDWIDTH_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void bandwidth_test(
    device const float4 *input  [[buffer(0)]],
    device float4       *output [[buffer(1)]],
    uint                 gid    [[thread_position_in_grid]]
) { output[gid] = input[gid]; }
"#;

/// Generate matmul kernel with N baked as compile-time constant (tinygrad pattern).
/// This eliminates buffer indirection and lets the compiler optimize address math.
fn matmul_kernel(n: u32) -> String {
    format!(r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void matmul_opt(
    device float       *C [[buffer(0)]],
    device const float *A [[buffer(1)]],
    device const float *B [[buffer(2)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {{
    C += gid.x * 32 * {N} + (gid.y * 2 + lid.y) * 32;
    A += gid.x * 32 * {N};
    B += (gid.y * 2 + lid.y) * 32;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = simdgroup_float8x8(0);

    simdgroup_float8x8 a[4], b[4];
    for (uint k = 0; k < {N}; k += 8) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(a[0], A+k+{r0},  {N}, ulong2(0,0));
        simdgroup_load(a[1], A+k+{r8},  {N}, ulong2(0,0));
        simdgroup_load(a[2], A+k+{r16}, {N}, ulong2(0,0));
        simdgroup_load(a[3], A+k+{r24}, {N}, ulong2(0,0));
        simdgroup_load(b[0], B+0+k*{N},  {N}, ulong2(0,0));
        simdgroup_load(b[1], B+8+k*{N},  {N}, ulong2(0,0));
        simdgroup_load(b[2], B+16+k*{N}, {N}, ulong2(0,0));
        simdgroup_load(b[3], B+24+k*{N}, {N}, ulong2(0,0));

        simdgroup_multiply_accumulate(acc[0][0], a[0], b[0], acc[0][0]);
        simdgroup_multiply_accumulate(acc[0][1], a[1], b[0], acc[0][1]);
        simdgroup_multiply_accumulate(acc[0][2], a[2], b[0], acc[0][2]);
        simdgroup_multiply_accumulate(acc[0][3], a[3], b[0], acc[0][3]);
        simdgroup_multiply_accumulate(acc[1][0], a[0], b[1], acc[1][0]);
        simdgroup_multiply_accumulate(acc[1][1], a[1], b[1], acc[1][1]);
        simdgroup_multiply_accumulate(acc[1][2], a[2], b[1], acc[1][2]);
        simdgroup_multiply_accumulate(acc[1][3], a[3], b[1], acc[1][3]);
        simdgroup_multiply_accumulate(acc[2][0], a[0], b[2], acc[2][0]);
        simdgroup_multiply_accumulate(acc[2][1], a[1], b[2], acc[2][1]);
        simdgroup_multiply_accumulate(acc[2][2], a[2], b[2], acc[2][2]);
        simdgroup_multiply_accumulate(acc[2][3], a[3], b[2], acc[2][3]);
        simdgroup_multiply_accumulate(acc[3][0], a[0], b[3], acc[3][0]);
        simdgroup_multiply_accumulate(acc[3][1], a[1], b[3], acc[3][1]);
        simdgroup_multiply_accumulate(acc[3][2], a[2], b[3], acc[3][2]);
        simdgroup_multiply_accumulate(acc[3][3], a[3], b[3], acc[3][3]);
    }}
    simdgroup_store(acc[0][0], C+{s0_0},   {N}, ulong2(0,0));
    simdgroup_store(acc[1][0], C+{s8_0},   {N}, ulong2(0,0));
    simdgroup_store(acc[2][0], C+{s16_0},  {N}, ulong2(0,0));
    simdgroup_store(acc[3][0], C+{s24_0},  {N}, ulong2(0,0));
    simdgroup_store(acc[0][1], C+{s0_8},   {N}, ulong2(0,0));
    simdgroup_store(acc[1][1], C+{s8_8},   {N}, ulong2(0,0));
    simdgroup_store(acc[2][1], C+{s16_8},  {N}, ulong2(0,0));
    simdgroup_store(acc[3][1], C+{s24_8},  {N}, ulong2(0,0));
    simdgroup_store(acc[0][2], C+{s0_16},  {N}, ulong2(0,0));
    simdgroup_store(acc[1][2], C+{s8_16},  {N}, ulong2(0,0));
    simdgroup_store(acc[2][2], C+{s16_16}, {N}, ulong2(0,0));
    simdgroup_store(acc[3][2], C+{s24_16}, {N}, ulong2(0,0));
    simdgroup_store(acc[0][3], C+{s0_24},  {N}, ulong2(0,0));
    simdgroup_store(acc[1][3], C+{s8_24},  {N}, ulong2(0,0));
    simdgroup_store(acc[2][3], C+{s16_24}, {N}, ulong2(0,0));
    simdgroup_store(acc[3][3], C+{s24_24}, {N}, ulong2(0,0));
}}
"#,
        N = n,
        r0 = 0, r8 = 8 * n, r16 = 16 * n, r24 = 24 * n,
        s0_0  = 0,            s8_0  = 8,           s16_0  = 16,          s24_0  = 24,
        s0_8  = 8  * n,       s8_8  = 8 + 8 * n,   s16_8  = 16 + 8 * n,  s24_8  = 24 + 8 * n,
        s0_16 = 16 * n,       s8_16 = 8 + 16 * n,  s16_16 = 16 + 16 * n, s24_16 = 24 + 16 * n,
        s0_24 = 24 * n,       s8_24 = 8 + 24 * n,  s16_24 = 16 + 24 * n, s24_24 = 24 + 24 * n,
    )
}

/// Matvec kernel: W(rows x cols) @ x(cols) = y(rows).
/// THIS is the LLM inference bottleneck. Pure bandwidth-bound.
/// Each threadgroup reduces one chunk of the output vector.
fn matvec_kernel() -> String {
    r#"
#include <metal_stdlib>
using namespace metal;

// Each threadgroup computes one output element.
// Threads within the group cooperatively reduce across cols.
kernel void matvec(
    device const float *W [[buffer(0)]],   // [rows, cols] row-major
    device const float *x [[buffer(1)]],   // [cols]
    device float       *y [[buffer(2)]],   // [rows]
    constant uint      &cols [[buffer(3)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    // Each thread sums a strided slice of the dot product
    device const float *row = W + gid * cols;
    float sum = 0.0;
    for (uint i = lid; i < cols; i += tgs) {
        sum += row[i] * x[i];
    }

    // Simdgroup reduction (warp-level, no shared memory needed)
    sum = simd_sum(sum);

    // First thread in each simdgroup writes partial result to threadgroup memory
    threadgroup float partials[32]; // max 32 simdgroups per threadgroup
    uint sg_idx = lid / 32;
    if (lid % 32 == 0) {
        partials[sg_idx] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First simdgroup reduces partials
    if (lid < 32) {
        float val = (lid < (tgs + 31) / 32) ? partials[lid] : 0.0;
        val = simd_sum(val);
        if (lid == 0) y[gid] = val;
    }
}

// Quantized matvec: W stored as int4 (packed 2 per byte), dequant on the fly
// For Q4_0: blocks of 32 values, 1 f16 scale + 16 bytes of nibbles = 18 bytes/block
kernel void matvec_q4(
    device const uchar *W [[buffer(0)]],   // Q4_0 packed weights
    device const float *x [[buffer(1)]],   // [cols] fp32 input
    device float       *y [[buffer(2)]],   // [rows] fp32 output
    constant uint      &cols [[buffer(3)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    uint blocks_per_row = cols / 32;
    device const uchar *row = W + gid * blocks_per_row * 18;  // 18 bytes per Q4_0 block
    float sum = 0.0;

    for (uint b = lid; b < blocks_per_row; b += tgs) {
        device const uchar *block = row + b * 18;
        // First 2 bytes: f16 scale
        half scale = *((device const half *)block);
        device const uchar *nibbles = block + 2;

        // Unpack 32 int4 values and multiply with x
        for (uint j = 0; j < 16; j++) {
            uchar packed = nibbles[j];
            int lo = int(packed & 0xF) - 8;
            int hi = int(packed >> 4) - 8;
            uint idx = b * 32 + j * 2;
            sum += float(scale) * float(lo) * x[idx];
            sum += float(scale) * float(hi) * x[idx + 1];
        }
    }

    sum = simd_sum(sum);
    threadgroup float partials[32];
    uint sg_idx = lid / 32;
    if (lid % 32 == 0) partials[sg_idx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32) {
        float val = (lid < (tgs + 31) / 32) ? partials[lid] : 0.0;
        val = simd_sum(val);
        if (lid == 0) y[gid] = val;
    }
}
"#.to_string()
}

pub fn run_bandwidth_test() {
    let device = Device::system_default().expect("No Metal device found");
    eprintln!("=== Metal GPU Benchmark ===");
    eprintln!("Device: {}", device.name());
    eprintln!("Working set: {} MB\n",
        device.recommended_working_set_size() / (1024 * 1024));

    let bw_lib = device.new_library_with_source(BANDWIDTH_KERNEL)
        .expect("Failed to compile bandwidth shader");
    bandwidth_test(&device, &bw_lib);

    eprintln!("\n--- Matmul (tinygrad-style, baked constants) ---");
    for &n in &[1024u32, 2048, 4096] {
        matmul_test(&device, n);
    }

    eprintln!("\n--- Matvec (LLM inference bottleneck) ---");
    let mv_src = matvec_kernel();
    let mv_lib = device.new_library_with_source(&mv_src)
        .expect("Failed to compile matvec shader");
    // Llama-3.2-1B relevant sizes: hidden=2048, ffn=8192, vocab=128256
    for &(rows, cols, label) in &[
        (1024u32,  1024,  "attn_proj 1024x1024"),
        (2816,     1024,  "ffn_up 2816x1024"),
        (1024,     2816,  "ffn_down 1024x2816"),
        (4096,     4096,  "generic 4096x4096"),
        (151936,   1024,  "vocab_proj 151936x1024"),
    ] {
        matvec_test(&device, &mv_lib, rows, cols, label);
    }

    // Q4 matvec
    eprintln!("\n--- Matvec Q4_0 (quantized, LLM actual) ---");
    for &(rows, cols, label) in &[
        (1024u32,  1024,  "attn_proj Q4 1024x1024"),
        (2816,     1024,  "ffn_up Q4 2816x1024"),
        (4096,     4096,  "generic Q4 4096x4096"),
    ] {
        matvec_q4_test(&device, &mv_lib, rows, cols, label);
    }

    // Dispatch overhead: how much does each dispatch cost inside a single cmd buffer?
    eprintln!("\n--- Dispatch overhead (single cmd buffer, N trivial dispatches) ---");
    dispatch_overhead_test(&device, &bw_lib);
}

fn bandwidth_test(device: &Device, library: &Library) {
    let func = library.get_function("bandwidth_test").expect("bandwidth_test not found");
    let pipeline = device.new_compute_pipeline(&func).expect("pipeline");
    eprintln!("--- Memory Bandwidth ---");

    for &size_mb in &[1u64, 10, 100, 300, 600, 1000] {
        let n_float4 = size_mb * 1024 * 1024 / 16;
        let byte_size = n_float4 * 16;
        let buf_in = device.new_buffer(byte_size);
        let buf_out = device.new_buffer(byte_size);

        unsafe {
            let ptr = buf_in.contents() as *mut f32;
            for i in 0..std::cmp::min(1024, n_float4 as usize * 4) {
                *ptr.add(i) = i as f32;
            }
        }

        let queue = device.new_command_queue();

        // Warmup
        dispatch_bw(&queue, &pipeline, &buf_in, &buf_out, n_float4);

        let iters = 10u32;
        let start = Instant::now();
        for _ in 0..iters {
            dispatch_bw(&queue, &pipeline, &buf_in, &buf_out, n_float4);
        }
        let elapsed = start.elapsed();
        let total_bytes = byte_size as f64 * 2.0 * iters as f64;
        let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;
        eprintln!("  {:>5}MB: {:>6.1} GB/s  ({:.2}ms/iter)",
            size_mb, gb_per_sec, elapsed.as_secs_f64() * 1000.0 / iters as f64);
    }
}

fn dispatch_bw(queue: &CommandQueue, pipeline: &Pipeline, buf_in: &Buffer, buf_out: &Buffer, n: u64) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_encoder();
    enc.set_pipeline(pipeline);
    enc.set_buffer(0, buf_in, 0);
    enc.set_buffer(1, buf_out, 0);
    enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(pipeline.thread_execution_width(), 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn matmul_test(device: &Device, n: u32) {
    let src = matmul_kernel(n);
    let library = device.new_library_with_source(&src)
        .unwrap_or_else(|e| panic!("Shader compile {n}: {e}"));
    let func = library.get_function("matmul_opt").expect("matmul_opt not found");
    let pipeline = device.new_compute_pipeline(&func).expect("pipeline");

    let bytes = (n as u64) * (n as u64) * 4;
    let buf_c = device.new_buffer(bytes);
    let buf_a = device.new_buffer(bytes);
    let buf_b = device.new_buffer(bytes);

    unsafe {
        let pa = buf_a.contents() as *mut f32;
        let pb = buf_b.contents() as *mut f32;
        for i in 0..(n * n) as usize {
            *pa.add(i) = (i % 17) as f32 * 0.01;
            *pb.add(i) = (i % 13) as f32 * 0.01;
        }
    }

    let queue = device.new_command_queue();
    let tg_grid = MTLSize::new((n / 32) as u64, (n / 64) as u64, 1);
    let tg_size = MTLSize::new(32, 2, 1);

    // Warmup
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        enc.set_pipeline(&pipeline);
        enc.set_buffer(0, &buf_c, 0);
        enc.set_buffer(1, &buf_a, 0);
        enc.set_buffer(2, &buf_b, 0);
        enc.dispatch_threadgroups(tg_grid, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let iters = 20u32;
    let start = Instant::now();
    for _ in 0..iters {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        enc.set_pipeline(&pipeline);
        enc.set_buffer(0, &buf_c, 0);
        enc.set_buffer(1, &buf_a, 0);
        enc.set_buffer(2, &buf_b, 0);
        enc.dispatch_threadgroups(tg_grid, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let elapsed = start.elapsed();
    let flops = 2.0 * (n as f64).powi(3) * iters as f64;
    let tflops = flops / elapsed.as_secs_f64() / 1e12;

    eprintln!("  {:>4}x{:<4}: {:>5.2} TFLOPS  ({:.2}ms/iter)",
        n, n, tflops, elapsed.as_secs_f64() * 1000.0 / iters as f64);
}

fn matvec_test(device: &Device, library: &Library, rows: u32, cols: u32, label: &str) {
    let func = library.get_function("matvec").expect("matvec not found");
    let pipeline = device.new_compute_pipeline(&func).expect("pipeline");

    let w_bytes = (rows as u64) * (cols as u64) * 4;
    let x_bytes = (cols as u64) * 4;
    let y_bytes = (rows as u64) * 4;

    let buf_w = device.new_buffer(w_bytes);
    let buf_x = device.new_buffer(x_bytes);
    let buf_y = device.new_buffer(y_bytes);

    unsafe {
        let pw = buf_w.contents() as *mut f32;
        let px = buf_x.contents() as *mut f32;
        for i in 0..(rows * cols) as usize { *pw.add(i) = (i % 7) as f32 * 0.001; }
        for i in 0..cols as usize { *px.add(i) = (i % 11) as f32 * 0.01; }
    }

    let queue = device.new_command_queue();
    let tg_count = MTLSize::new(rows as u64, 1, 1);
    let tg_size = MTLSize::new(256, 1, 1); // 256 threads per row reduction

    // Warmup
    dispatch_mv(&queue, &pipeline, &buf_w, &buf_x, &buf_y, cols, tg_count, tg_size);

    let iters = 50u32;
    let start = Instant::now();
    for _ in 0..iters {
        dispatch_mv(&queue, &pipeline, &buf_w, &buf_x, &buf_y, cols, tg_count, tg_size);
    }
    let elapsed = start.elapsed();
    let total_bytes = w_bytes as f64 * iters as f64; // bandwidth = weight reads
    let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;
    let us_per = elapsed.as_secs_f64() * 1e6 / iters as f64;

    eprintln!("  {:<28} {:>6.1} GB/s  {:>8.1}us/call", label, gb_per_sec, us_per);
}

fn dispatch_mv(
    queue: &CommandQueue, pipeline: &Pipeline,
    w: &Buffer, x: &Buffer, y: &Buffer, cols: u32,
    tg_count: MTLSize, tg_size: MTLSize,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_encoder();
    enc.set_pipeline(pipeline);
    enc.set_buffer(0, w, 0);
    enc.set_buffer(1, x, 0);
    enc.set_buffer(2, y, 0);
    enc.set_bytes(3, &cols as *const u32 as *const c_void, 4);
    enc.dispatch_threadgroups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn matvec_q4_test(device: &Device, library: &Library, rows: u32, cols: u32, label: &str) {
    let func = library.get_function("matvec_q4").expect("matvec_q4 not found");
    let pipeline = device.new_compute_pipeline(&func).expect("pipeline");

    let blocks_per_row = cols / 32;
    let w_bytes = (rows as u64) * (blocks_per_row as u64) * 18; // Q4_0: 18 bytes/block
    let x_bytes = (cols as u64) * 4;
    let y_bytes = (rows as u64) * 4;

    let buf_w = device.new_buffer(w_bytes);
    let buf_x = device.new_buffer(x_bytes);
    let buf_y = device.new_buffer(y_bytes);

    // Fill with test data
    unsafe {
        let pw = buf_w.contents() as *mut u8;
        for i in 0..w_bytes as usize { *pw.add(i) = (i % 256) as u8; }
        let px = buf_x.contents() as *mut f32;
        for i in 0..cols as usize { *px.add(i) = (i % 11) as f32 * 0.01; }
    }

    let queue = device.new_command_queue();
    let tg_count = MTLSize::new(rows as u64, 1, 1);
    let tg_size = MTLSize::new(256, 1, 1);

    // Warmup
    dispatch_mv(&queue, &pipeline, &buf_w, &buf_x, &buf_y, cols, tg_count, tg_size);

    let iters = 50u32;
    let start = Instant::now();
    for _ in 0..iters {
        dispatch_mv(&queue, &pipeline, &buf_w, &buf_x, &buf_y, cols, tg_count, tg_size);
    }
    let elapsed = start.elapsed();
    let total_bytes = w_bytes as f64 * iters as f64;
    let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;
    let us_per = elapsed.as_secs_f64() * 1e6 / iters as f64;

    eprintln!("  {:<28} {:>6.1} GB/s  {:>8.1}us/call  (weight={:.1}MB)",
        label, gb_per_sec, us_per, w_bytes as f64 / 1e6);
}

/// Measure per-dispatch overhead inside a single command buffer.
/// Dispatches N trivial copies in one cmd buffer, one encoder.
/// The slope of time vs N gives us the per-dispatch cost on the GPU.
fn dispatch_overhead_test(device: &Device, library: &Library) {
    let func = library.get_function("bandwidth_test").expect("bandwidth_test not found");
    let pipeline = device.new_compute_pipeline(&func).expect("pipeline");

    // Small buffer — fits in cache, so we're measuring dispatch overhead not BW
    let n_float4 = 256u64; // 4KB
    let buf_a = device.new_buffer(n_float4 * 16);
    let buf_b = device.new_buffer(n_float4 * 16);
    let queue = device.new_command_queue();
    let tew = pipeline.thread_execution_width();

    for &n_dispatches in &[1u32, 10, 50, 100, 200, 300, 500] {
        // Warmup
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            enc.set_pipeline(&pipeline);
            enc.set_buffer(0, &buf_a, 0);
            enc.set_buffer(1, &buf_b, 0);
            enc.dispatch_threads(MTLSize::new(n_float4, 1, 1), MTLSize::new(tew, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let iters = 20u32;
        let start = Instant::now();
        for _ in 0..iters {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            enc.set_pipeline(&pipeline);
            enc.set_buffer(0, &buf_a, 0);
            enc.set_buffer(1, &buf_b, 0);
            for _ in 0..n_dispatches {
                enc.dispatch_threads(MTLSize::new(n_float4, 1, 1), MTLSize::new(tew, 1, 1));
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let elapsed = start.elapsed();
        let us_per_cmd = elapsed.as_secs_f64() * 1e6 / iters as f64;
        let us_per_dispatch = if n_dispatches > 1 { us_per_cmd / n_dispatches as f64 } else { us_per_cmd };
        eprintln!("  {:>3} dispatches/cmd:  {:>8.1}µs/cmd  {:>6.2}µs/dispatch",
            n_dispatches, us_per_cmd, us_per_dispatch);
    }

    // Same test but with DEPENDENT dispatches (A→B→A→B chain, writes to same buffer)
    eprintln!("  --- with data dependencies (A→B→A→B chain) ---");
    for &n_dispatches in &[1u32, 10, 50, 100, 200, 300] {
        let iters = 20u32;
        let start = Instant::now();
        for _ in 0..iters {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            enc.set_pipeline(&pipeline);
            for i in 0..n_dispatches {
                if i % 2 == 0 {
                    enc.set_buffer(0, &buf_a, 0);
                    enc.set_buffer(1, &buf_b, 0);
                } else {
                    enc.set_buffer(0, &buf_b, 0);
                    enc.set_buffer(1, &buf_a, 0);
                }
                enc.dispatch_threads(MTLSize::new(n_float4, 1, 1), MTLSize::new(tew, 1, 1));
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let elapsed = start.elapsed();
        let us_per_cmd = elapsed.as_secs_f64() * 1e6 / iters as f64;
        let us_per_dispatch = if n_dispatches > 1 { us_per_cmd / n_dispatches as f64 } else { us_per_cmd };
        eprintln!("  {:>3} dep dispatches:  {:>8.1}µs/cmd  {:>6.2}µs/dispatch",
            n_dispatches, us_per_cmd, us_per_dispatch);
    }
}

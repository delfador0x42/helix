//! Microbenchmark: measure actual Metal dispatch overhead on this GPU.
//! Isolates: empty dispatch, pipeline switch, buffer binding, barriers,
//! concurrent vs serial, and batch sizes.

use crate::gpu::*;
use std::ffi::c_void;
use std::time::Instant;

const WARMUP: u32 = 10;
const ITERS: u32 = 100;

fn noop_kernel_src() -> &'static str {
    r#"
    #include <metal_stdlib>
    using namespace metal;
    kernel void noop(device float *x [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
        // intentionally empty — measure pure dispatch overhead
    }
    kernel void write_one(device float *x [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
        x[gid] = 1.0;
    }
    kernel void add_one(device float *x [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
        x[gid] += 1.0;
    }
    kernel void heavy_work(
        device const float *a [[buffer(0)]],
        device float       *b [[buffer(1)]],
        uint gid [[thread_position_in_grid]]
    ) {
        float v = a[gid];
        for (int i = 0; i < 100; i++) v = v * 1.0001 + 0.0001;
        b[gid] = v;
    }
    "#
}

pub fn run() {
    let dev = Device::system_default().unwrap();
    let queue = dev.new_command_queue();
    let lib = dev.new_library_with_source(noop_kernel_src()).unwrap();

    let p_noop = dev.new_compute_pipeline(&lib.get_function("noop").unwrap()).unwrap();
    let p_write = dev.new_compute_pipeline(&lib.get_function("write_one").unwrap()).unwrap();
    let p_add = dev.new_compute_pipeline(&lib.get_function("add_one").unwrap()).unwrap();
    let p_heavy = dev.new_compute_pipeline(&lib.get_function("heavy_work").unwrap()).unwrap();

    let buf = dev.new_buffer(4096 * 4);
    let buf2 = dev.new_buffer(4096 * 4);

    eprintln!("\n=== Metal Dispatch Overhead Microbenchmark ({}) ===\n", dev.name());

    // ── Test 1: Single empty dispatch (commit+wait overhead) ──
    bench("1 empty dispatch (commit+wait)", ITERS, || {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        enc.set_pipeline(&p_noop);
        enc.set_buffer(0, &buf, 0);
        enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    });

    // ── Test 2: N dispatches in ONE encoder, ONE commit ──
    for n in [1, 10, 50, 100, 200, 500] {
        bench(&format!("{n} noops in 1 encoder"), ITERS, || {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            for _ in 0..n {
                enc.set_pipeline(&p_noop);
                enc.set_buffer(0, &buf, 0);
                enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });
    }

    // ── Test 3: N dispatches that WRITE (create real dependencies) ──
    for n in [1, 10, 50, 100, 228] {
        bench(&format!("{n} write_one(4K) in 1 encoder"), ITERS, || {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            for _ in 0..n {
                enc.set_pipeline(&p_write);
                enc.set_buffer(0, &buf, 0);
                enc.dispatch_threads(MTLSize::new(4096, 1, 1), MTLSize::new(256, 1, 1));
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });
    }

    // ── Test 4: N dispatches with READ-AFTER-WRITE dependency (actual barrier) ──
    for n in [1, 10, 50, 100, 228] {
        bench(&format!("{n} add_one(4K) chained in 1 encoder"), ITERS, || {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            for _ in 0..n {
                enc.set_pipeline(&p_add);
                enc.set_buffer(0, &buf, 0);
                enc.dispatch_threads(MTLSize::new(4096, 1, 1), MTLSize::new(256, 1, 1));
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });
    }

    // ── Test 5: Pipeline switch overhead ──
    bench("100 dispatches, SAME pipeline", ITERS, || {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        for _ in 0..100 {
            enc.set_pipeline(&p_write);
            enc.set_buffer(0, &buf, 0);
            enc.dispatch_threads(MTLSize::new(4096, 1, 1), MTLSize::new(256, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    });

    bench("100 dispatches, ALTERNATING 2 pipelines", ITERS, || {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        for i in 0..100 {
            if i % 2 == 0 {
                enc.set_pipeline(&p_write);
            } else {
                enc.set_pipeline(&p_add);
            }
            enc.set_buffer(0, &buf, 0);
            enc.dispatch_threads(MTLSize::new(4096, 1, 1), MTLSize::new(256, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    });

    // ── Test 6: Independent vs dependent dispatches ──
    bench("100 writes to DIFFERENT buffers (independent)", ITERS, || {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        for i in 0..100 {
            enc.set_pipeline(&p_write);
            if i % 2 == 0 {
                enc.set_buffer(0, &buf, 0);
            } else {
                enc.set_buffer(0, &buf2, 0);
            }
            enc.dispatch_threads(MTLSize::new(4096, 1, 1), MTLSize::new(256, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    });

    bench("100 add_one to SAME buffer (chained deps)", ITERS, || {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        for _ in 0..100 {
            enc.set_pipeline(&p_add);
            enc.set_buffer(0, &buf, 0);
            enc.dispatch_threads(MTLSize::new(4096, 1, 1), MTLSize::new(256, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    });

    // ── Test 7: Varying threadgroup sizes (GPU occupancy) ──
    for threads in [1, 32, 256, 1024, 4096, 65536, 524288] {
        bench(&format!("1 dispatch, {threads} threads"), ITERS, || {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            enc.set_pipeline(&p_write);
            enc.set_buffer(0, &buf, 0);
            let tgs = 256.min(threads as u64);
            let grids = ((threads as u64) + tgs - 1) / tgs;
            enc.dispatch_threadgroups(
                MTLSize::new(grids, 1, 1),
                MTLSize::new(tgs, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        });
    }

    // ── Test 8: Commit overhead: N cmd bufs vs 1 cmd buf ──
    bench("228 dispatches in 228 cmd bufs (current forward_timed)", ITERS / 10, || {
        for _ in 0..228 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_encoder();
            enc.set_pipeline(&p_add);
            enc.set_buffer(0, &buf, 0);
            enc.dispatch_threads(MTLSize::new(4096, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    });

    bench("228 dispatches in 1 cmd buf", ITERS / 10, || {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        for _ in 0..228 {
            enc.set_pipeline(&p_add);
            enc.set_buffer(0, &buf, 0);
            enc.dispatch_threads(MTLSize::new(4096, 1, 1), MTLSize::new(256, 1, 1));
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    });

    // ── Test 9: The REAL question — what does 228 matvec-sized dispatches cost? ──
    // Simulate our actual forward pass pattern: mix of tiny and large dispatches
    let big_buf = dev.new_buffer(2048 * 8192 * 4); // big enough for matvec
    bench("228 mixed dispatches (simulating forward pass)", ITERS / 10, || {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        for i in 0..228 {
            match i % 14 {
                // rmsnorm: 1 threadgroup, 256 threads
                0 | 7 => {
                    enc.set_pipeline(&p_add);
                    enc.set_buffer(0, &buf, 0);
                    enc.dispatch_threadgroups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
                }
                // matvec: 2048 threadgroups, 256 threads (Q/K/V/O/gate/up/down)
                1 | 2 | 3 | 6 | 8 | 9 | 10 => {
                    enc.set_pipeline(&p_heavy);
                    enc.set_buffer(0, &big_buf, 0);
                    enc.set_buffer(1, &buf, 0);
                    enc.dispatch_threadgroups(MTLSize::new(2048, 1, 1), MTLSize::new(256, 1, 1));
                }
                // rope: 512 threads
                4 => {
                    enc.set_pipeline(&p_add);
                    enc.set_buffer(0, &buf, 0);
                    enc.dispatch_threads(MTLSize::new(512, 1, 1), MTLSize::new(64, 1, 1));
                }
                // kv_store: 512 threads
                5 => {
                    enc.set_pipeline(&p_write);
                    enc.set_buffer(0, &buf, 0);
                    enc.dispatch_threads(MTLSize::new(512, 1, 1), MTLSize::new(64, 1, 1));
                }
                // attention: 32 threadgroups, 128 threads
                11 => {
                    enc.set_pipeline(&p_heavy);
                    enc.set_buffer(0, &big_buf, 0);
                    enc.set_buffer(1, &buf, 0);
                    enc.dispatch_threadgroups(MTLSize::new(32, 1, 1), MTLSize::new(128, 1, 1));
                }
                // silu: 8192 threads
                12 => {
                    enc.set_pipeline(&p_add);
                    enc.set_buffer(0, &buf, 0);
                    enc.dispatch_threads(MTLSize::new(8192, 1, 1), MTLSize::new(256, 1, 1));
                }
                _ => {
                    enc.set_pipeline(&p_add);
                    enc.set_buffer(0, &buf, 0);
                    enc.dispatch_threads(MTLSize::new(2048, 1, 1), MTLSize::new(256, 1, 1));
                }
            }
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    });

    eprintln!("\n=== Done ===");
}

fn bench(name: &str, iters: u32, f: impl Fn()) {
    // Warmup
    for _ in 0..WARMUP { f(); }
    // Measure
    let t0 = Instant::now();
    for _ in 0..iters { f(); }
    let elapsed = t0.elapsed();
    let per_iter = elapsed.as_nanos() as f64 / iters as f64;
    if per_iter > 1_000_000.0 {
        eprintln!("  {name:<58} {:.2}ms", per_iter / 1_000_000.0);
    } else {
        eprintln!("  {name:<58} {:.1}µs", per_iter / 1_000.0);
    }
}

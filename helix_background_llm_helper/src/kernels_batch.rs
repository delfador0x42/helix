//! Batch matmul kernels for multi-token inference.
//! Q4_0/Q6_K dequant fused into tiled matmul using simdgroup_float8x8.
//! These replace matvec kernels when batch > 1.
//!
//! Pattern: dequant W tile into threadgroup memory, load X tile directly,
//! then simdgroup_multiply_accumulate for the 8×8 sub-tiles.
//!
//! Metal constraint: all position attributes must have same scalar/vector width.
//! We use uint3 for all and derive sgid manually from tid.x.
//!
//! Threadgroup: 128 threads = 4 simdgroups.
//! Output tile: 32 rows × 32 cols. Each SG handles 32 rows × 8 cols.
//! 128 threads load 1024 dequant values = 8 per thread (fast).

/// Q4_0 batch matmul: Y[rows × batch] = W_q4_0[rows × cols] @ X[cols × batch]
pub const MATMUL_Q4_0: &str = r#"
kernel void matmul_q4_0(
    device const uchar *W     [[buffer(0)]],
    device const float *X     [[buffer(1)]],
    device float       *Y     [[buffer(2)]],
    constant uint      &cols  [[buffer(3)]],
    constant uint      &rows  [[buffer(4)]],
    constant uint      &batch [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]],
    uint3 tgs  [[threads_per_threadgroup]]
) {
    uint lid = tid.x;            // 0..127
    uint sgid = lid / 32;       // 0..3

    uint row_base = tgid.x * 32;
    uint col_base = tgid.y * 32;

    // 4 SGs × 4 row-blocks × 1 col-block of 8×8 = 32×8 per SG, 32×32 total
    simdgroup_float8x8 acc[4];
    for (uint i = 0; i < 4; i++)
        acc[i] = simdgroup_float8x8(0);

    threadgroup float W_tile[32][32];
    threadgroup float X_tile[32][32];

    uint blocks_per_row = cols / 32;
    uint row_bytes = blocks_per_row * 18;

    for (uint k = 0; k < cols; k += 32) {
        uint block_idx = k / 32;

        // 128 threads load 1024 values = 8 per thread
        for (uint t = lid; t < 1024; t += 128) {
            uint r = t / 32, c = t % 32;
            uint gr = row_base + r;
            if (gr < rows) {
                device const uchar *rp = W + gr * row_bytes + block_idx * 18;
                float d = float(*((device const half *)(rp)));
                uchar qb = *(rp + 2 + (c < 16 ? c : c - 16));
                float q = (c < 16) ? float(qb & 0xF) - 8.0 : float(qb >> 4) - 8.0;
                W_tile[r][c] = d * q;
            } else { W_tile[r][c] = 0.0; }
        }

        for (uint t = lid; t < 1024; t += 128) {
            uint kk = t / 32, bb = t % 32;
            uint gk = k + kk, gb = col_base + bb;
            X_tile[kk][bb] = (gk < cols && gb < batch) ? X[gb * cols + gk] : 0.0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 w[4], x;
            simdgroup_load(w[0], &W_tile[0][kk],  32);
            simdgroup_load(w[1], &W_tile[8][kk],  32);
            simdgroup_load(w[2], &W_tile[16][kk], 32);
            simdgroup_load(w[3], &W_tile[24][kk], 32);
            simdgroup_load(x,    &X_tile[kk][sgid * 8], 32);

            simdgroup_multiply_accumulate(acc[0], w[0], x, acc[0]);
            simdgroup_multiply_accumulate(acc[1], w[1], x, acc[1]);
            simdgroup_multiply_accumulate(acc[2], w[2], x, acc[2]);
            simdgroup_multiply_accumulate(acc[3], w[3], x, acc[3]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float out_tile[32][32];
    simdgroup_store(acc[0], &out_tile[0][sgid * 8],  32);
    simdgroup_store(acc[1], &out_tile[8][sgid * 8],  32);
    simdgroup_store(acc[2], &out_tile[16][sgid * 8], 32);
    simdgroup_store(acc[3], &out_tile[24][sgid * 8], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = lid; t < 1024; t += 128) {
        uint r = t / 32, c = t % 32;
        uint gr = row_base + r, gc = col_base + c;
        if (gr < rows && gc < batch) {
            Y[gc * rows + gr] = out_tile[r][c];
        }
    }
}
"#;

/// Fused Q4_0 batch matmul + residual add.
pub const MATMUL_Q4_0_ADD: &str = r#"
kernel void matmul_q4_0_add(
    device const uchar *W     [[buffer(0)]],
    device const float *X     [[buffer(1)]],
    device float       *res   [[buffer(2)]],
    constant uint      &cols  [[buffer(3)]],
    constant uint      &rows  [[buffer(4)]],
    constant uint      &batch [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]],
    uint3 tgs  [[threads_per_threadgroup]]
) {
    uint lid = tid.x;
    uint sgid = lid / 32;
    uint row_base = tgid.x * 32;
    uint col_base = tgid.y * 32;

    simdgroup_float8x8 acc[4];
    for (uint i = 0; i < 4; i++) acc[i] = simdgroup_float8x8(0);

    threadgroup float W_tile[32][32];
    threadgroup float X_tile[32][32];
    uint blocks_per_row = cols / 32;
    uint row_bytes = blocks_per_row * 18;

    for (uint k = 0; k < cols; k += 32) {
        uint block_idx = k / 32;
        for (uint t = lid; t < 1024; t += 128) {
            uint r = t / 32, c = t % 32, gr = row_base + r;
            if (gr < rows) {
                device const uchar *rp = W + gr * row_bytes + block_idx * 18;
                float d = float(*((device const half *)(rp)));
                uchar qb = *(rp + 2 + (c < 16 ? c : c - 16));
                float q = (c < 16) ? float(qb & 0xF) - 8.0 : float(qb >> 4) - 8.0;
                W_tile[r][c] = d * q;
            } else { W_tile[r][c] = 0.0; }
        }
        for (uint t = lid; t < 1024; t += 128) {
            uint kk = t / 32, bb = t % 32;
            uint gk = k + kk, gb = col_base + bb;
            X_tile[kk][bb] = (gk < cols && gb < batch) ? X[gb * cols + gk] : 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 w[4], x;
            simdgroup_load(w[0], &W_tile[0][kk],  32);
            simdgroup_load(w[1], &W_tile[8][kk],  32);
            simdgroup_load(w[2], &W_tile[16][kk], 32);
            simdgroup_load(w[3], &W_tile[24][kk], 32);
            simdgroup_load(x,    &X_tile[kk][sgid * 8], 32);
            simdgroup_multiply_accumulate(acc[0], w[0], x, acc[0]);
            simdgroup_multiply_accumulate(acc[1], w[1], x, acc[1]);
            simdgroup_multiply_accumulate(acc[2], w[2], x, acc[2]);
            simdgroup_multiply_accumulate(acc[3], w[3], x, acc[3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float out_tile[32][32];
    simdgroup_store(acc[0], &out_tile[0][sgid * 8],  32);
    simdgroup_store(acc[1], &out_tile[8][sgid * 8],  32);
    simdgroup_store(acc[2], &out_tile[16][sgid * 8], 32);
    simdgroup_store(acc[3], &out_tile[24][sgid * 8], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = lid; t < 1024; t += 128) {
        uint r = t / 32, c = t % 32;
        uint gr = row_base + r, gc = col_base + c;
        if (gr < rows && gc < batch) {
            res[gc * rows + gr] += out_tile[r][c];
        }
    }
}
"#;

/// Q6_K batch matmul.
pub const MATMUL_Q6K: &str = r#"
kernel void matmul_q6k(
    device const uchar *W     [[buffer(0)]],
    device const float *X     [[buffer(1)]],
    device float       *Y     [[buffer(2)]],
    constant uint      &cols  [[buffer(3)]],
    constant uint      &rows  [[buffer(4)]],
    constant uint      &batch [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]],
    uint3 tgs  [[threads_per_threadgroup]]
) {
    uint lid = tid.x;
    uint sgid = lid / 32;
    uint row_base = tgid.x * 32;
    uint col_base = tgid.y * 32;

    simdgroup_float8x8 acc[4];
    for (uint i = 0; i < 4; i++) acc[i] = simdgroup_float8x8(0);

    threadgroup float W_tile[32][32];
    threadgroup float X_tile[32][32];
    uint blocks_per_row = cols / 256;

    for (uint k = 0; k < cols; k += 32) {
        for (uint t = lid; t < 1024; t += 128) {
            uint r = t / 32, c = t % 32;
            uint gr = row_base + r, gc = k + c;
            if (gr < rows && gc < cols) {
                uint bi = gc / 256, vi = gc % 256;
                device const uchar *blk = W + gr * blocks_per_row * 210 + bi * 210;
                device const uchar *ql = blk, *qh = blk + 128;
                device const char *sc = (device const char *)(blk + 192);
                float d = float(*((device const half *)(blk + 208)));
                uint n = vi / 128, rem = vi % 128, l = rem % 32, sub = rem / 32;
                device const uchar *qlp = ql + n * 64, *qhp = qh + n * 32;
                uint sb = n * 8, is = l / 16;
                int q; float s;
                switch (sub) {
                    case 0: q = (int(qlp[l]    & 0xF) | (int((qhp[l] >> 0) & 3) << 4)) - 32; s = d * float(sc[sb+is+0]); break;
                    case 1: q = (int(qlp[l+32] & 0xF) | (int((qhp[l] >> 2) & 3) << 4)) - 32; s = d * float(sc[sb+is+2]); break;
                    case 2: q = (int(qlp[l]     >> 4) | (int((qhp[l] >> 4) & 3) << 4)) - 32; s = d * float(sc[sb+is+4]); break;
                    default: q = (int(qlp[l+32]  >> 4) | (int((qhp[l] >> 6) & 3) << 4)) - 32; s = d * float(sc[sb+is+6]); break;
                }
                W_tile[r][c] = s * float(q);
            } else { W_tile[r][c] = 0.0; }
        }
        for (uint t = lid; t < 1024; t += 128) {
            uint kk = t / 32, bb = t % 32;
            uint gk = k + kk, gb = col_base + bb;
            X_tile[kk][bb] = (gk < cols && gb < batch) ? X[gb * cols + gk] : 0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_float8x8 w[4], x;
            simdgroup_load(w[0], &W_tile[0][kk],  32);
            simdgroup_load(w[1], &W_tile[8][kk],  32);
            simdgroup_load(w[2], &W_tile[16][kk], 32);
            simdgroup_load(w[3], &W_tile[24][kk], 32);
            simdgroup_load(x,    &X_tile[kk][sgid * 8], 32);
            simdgroup_multiply_accumulate(acc[0], w[0], x, acc[0]);
            simdgroup_multiply_accumulate(acc[1], w[1], x, acc[1]);
            simdgroup_multiply_accumulate(acc[2], w[2], x, acc[2]);
            simdgroup_multiply_accumulate(acc[3], w[3], x, acc[3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float out_tile[32][32];
    simdgroup_store(acc[0], &out_tile[0][sgid * 8],  32);
    simdgroup_store(acc[1], &out_tile[8][sgid * 8],  32);
    simdgroup_store(acc[2], &out_tile[16][sgid * 8], 32);
    simdgroup_store(acc[3], &out_tile[24][sgid * 8], 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = lid; t < 1024; t += 128) {
        uint r = t / 32, c = t % 32;
        uint gr = row_base + r, gc = col_base + c;
        if (gr < rows && gc < batch) {
            Y[gc * rows + gr] = out_tile[r][c];
        }
    }
}
"#;

pub fn all_batch_kernels() -> String {
    format!("{HEADER}{MATMUL_Q4_0}{MATMUL_Q4_0_ADD}{MATMUL_Q6K}")
}

const HEADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"#;

//! Metal compute kernels for Llama-3.2 inference.
//! Q4_0/Q4_1/Q6_K dequant fused into matvec. All matvec kernels use simdgroup
//! parallelism: 8 rows per threadgroup (256 threads = 8 simdgroups of 32).
//! Element-wise ops (RMSNorm, RoPE, SiLU, attention) in single command buffer.

pub fn all_kernels() -> String {
    format!("{HEADER}{RMSNORM}{MATVEC_Q4_0}{MATVEC_Q4_1}{MATVEC_Q4K}{MATVEC_Q6K}{MATVEC_Q4_0_ADD}{MATVEC_Q4_1_ADD}{MATVEC_Q4K_ADD}{MATVEC_Q6K_ADD}{EMBED_Q4_0}{EMBED_Q6K}{ROPE}{KV_STORE}{ATTENTION}{SILU_MUL}{ARGMAX}")
}

const HEADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
"#;

/// RMSNorm: y = x / sqrt(mean(x²) + eps) * weight
const RMSNORM: &str = r#"
kernel void rmsnorm(
    device const float *x      [[buffer(0)]],
    device float       *y      [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    constant uint      &dim    [[buffer(3)]],
    constant float     &eps    [[buffer(4)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    float sum_sq = 0.0;
    for (uint i = lid; i < dim; i += tgs) sum_sq += x[i] * x[i];
    sum_sq = simd_sum(sum_sq);
    threadgroup float parts[32];
    if (lid % 32 == 0) parts[lid / 32] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32) {
        float v = (lid < (tgs + 31) / 32) ? parts[lid] : 0.0;
        v = simd_sum(v);
        if (lid == 0) parts[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms = rsqrt(parts[0] / float(dim) + eps);
    for (uint i = lid; i < dim; i += tgs) y[i] = x[i] * rms * weight[i];
}
"#;

/// Q4_0 dequant matvec: y = W_q4_0 @ x. Simplest quantization format.
/// Q4_0: 32 values/block, 18 bytes/block (2B half scale + 16B packed nibbles)
/// Each simdgroup (32 threads) handles one row. Threadgroup processes num_sg rows.
const MATVEC_Q4_0: &str = r#"
kernel void matvec_q4_0(
    device const uchar *W    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device float       *y    [[buffer(2)]],
    constant uint      &cols [[buffer(3)]],
    constant uint      &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint num_sg [[simdgroups_per_threadgroup]]
) {
    uint row_idx = tgid * num_sg + simd_gid;
    if (row_idx >= rows) return;
    uint blocks_per_row = cols / 32;
    uint row_bytes = blocks_per_row * 18;
    device const uchar *row = W + row_idx * row_bytes;
    float sum = 0.0;

    for (uint b = simd_lid; b < blocks_per_row; b += 32) {
        device const uchar *blk = row + b * 18;
        float d = float(*((device const half *)(blk)));
        device const uchar *qs = blk + 2;
        uint base = b * 32;

        for (uint i = 0; i < 16; i++) {
            uchar qb = qs[i];
            sum += d * (float(qb & 0xF) - 8.0) * x[base + i];
            sum += d * (float(qb >> 4) - 8.0) * x[base + 16 + i];
        }
    }

    sum = simd_sum(sum);
    if (simd_lid == 0) y[row_idx] = sum;
}
"#;

/// Q4_1 dequant matvec: y = W_q4_1 @ x. Like Q4_0 but with min offset.
/// Q4_1: 32 values/block, 20 bytes/block (2B half d, 2B half m, 16B packed nibbles)
/// Each simdgroup handles one row. Threadgroup processes num_sg rows.
const MATVEC_Q4_1: &str = r#"
kernel void matvec_q4_1(
    device const uchar *W    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device float       *y    [[buffer(2)]],
    constant uint      &cols [[buffer(3)]],
    constant uint      &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint num_sg [[simdgroups_per_threadgroup]]
) {
    uint row_idx = tgid * num_sg + simd_gid;
    if (row_idx >= rows) return;
    uint blocks_per_row = cols / 32;
    uint row_bytes = blocks_per_row * 20;
    device const uchar *row = W + row_idx * row_bytes;
    float sum = 0.0;

    for (uint b = simd_lid; b < blocks_per_row; b += 32) {
        device const uchar *blk = row + b * 20;
        float d = float(*((device const half *)(blk)));
        float m = float(*((device const half *)(blk + 2)));
        device const uchar *qs = blk + 4;
        uint base = b * 32;

        for (uint i = 0; i < 16; i++) {
            uchar qb = qs[i];
            sum += (d * float(qb & 0xF) + m) * x[base + i];
            sum += (d * float(qb >> 4) + m) * x[base + 16 + i];
        }
    }

    sum = simd_sum(sum);
    if (simd_lid == 0) y[row_idx] = sum;
}
"#;

/// Q6_K dequant matvec: y = W_q6k @ x.
/// Q6_K: 256 values/block, 210 bytes/block (128 ql, 64 qh, 16 scales, 2 d)
/// Each simdgroup (32 threads) handles one row. Threadgroup processes num_sg rows.
const MATVEC_Q6K: &str = r#"
kernel void matvec_q6k(
    device const uchar *W    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device float       *y    [[buffer(2)]],
    constant uint      &cols [[buffer(3)]],
    constant uint      &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint num_sg [[simdgroups_per_threadgroup]]
) {
    uint row_idx = tgid * num_sg + simd_gid;
    if (row_idx >= rows) return;
    uint blocks_per_row = cols / 256;
    device const uchar *row = W + row_idx * blocks_per_row * 210;
    float sum = 0.0;

    for (uint b = simd_lid; b < blocks_per_row; b += 32) {
        device const uchar *blk = row + b * 210;
        device const uchar *ql = blk;
        device const uchar *qh = blk + 128;
        device const char  *sc = (device const char *)(blk + 192);
        float d = float(*((device const half *)(blk + 208)));
        uint base = b * 256;

        for (uint n = 0; n < 2; n++) {
            uint sb = n * 8;
            device const uchar *qlp = ql + n * 64;
            device const uchar *qhp = qh + n * 32;
            uint idx = base + n * 128;
            for (uint h = 0; h < 2; h++) {
                float s0 = d * float(sc[sb+h+0]), s2 = d * float(sc[sb+h+2]);
                float s4 = d * float(sc[sb+h+4]), s6 = d * float(sc[sb+h+6]);
                for (uint l = h * 16; l < h * 16 + 16; l++) {
                    int q1 = (int(qlp[l]    & 0xF) | (int((qhp[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int(qlp[l+32] & 0xF) | (int((qhp[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int(qlp[l]     >> 4) | (int((qhp[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int(qlp[l+32]  >> 4) | (int((qhp[l] >> 6) & 3) << 4)) - 32;
                    sum += s0 * float(q1) * x[idx + l];
                    sum += s2 * float(q2) * x[idx + 32 + l];
                    sum += s4 * float(q3) * x[idx + 64 + l];
                    sum += s6 * float(q4) * x[idx + 96 + l];
                }
            }
        }
    }

    sum = simd_sum(sum);
    if (simd_lid == 0) y[row_idx] = sum;
}
"#;

/// Fused Q4_0 matvec + residual add: res[row] += dot(W[row], x).
/// Uses simdgroup-per-row pattern (same as non-fused) for max efficiency.
const MATVEC_Q4_0_ADD: &str = r#"
kernel void matvec_q4_0_add(
    device const uchar *W    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device float       *res  [[buffer(2)]],
    constant uint      &cols [[buffer(3)]],
    constant uint      &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint num_sg [[simdgroups_per_threadgroup]]
) {
    uint row_idx = tgid * num_sg + simd_gid;
    if (row_idx >= rows) return;
    uint blocks_per_row = cols / 32;
    uint row_bytes = blocks_per_row * 18;
    device const uchar *row = W + row_idx * row_bytes;
    float sum = 0.0;
    for (uint b = simd_lid; b < blocks_per_row; b += 32) {
        device const uchar *blk = row + b * 18;
        float d = float(*((device const half *)(blk)));
        device const uchar *qs = blk + 2;
        uint base = b * 32;
        for (uint i = 0; i < 16; i++) {
            uchar qb = qs[i];
            sum += d * (float(qb & 0xF) - 8.0) * x[base + i];
            sum += d * (float(qb >> 4) - 8.0) * x[base + 16 + i];
        }
    }
    sum = simd_sum(sum);
    if (simd_lid == 0) res[row_idx] += sum;
}
"#;

/// Fused Q4_1 matvec + residual add. Simdgroup-per-row pattern.
const MATVEC_Q4_1_ADD: &str = r#"
kernel void matvec_q4_1_add(
    device const uchar *W    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device float       *res  [[buffer(2)]],
    constant uint      &cols [[buffer(3)]],
    constant uint      &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint num_sg [[simdgroups_per_threadgroup]]
) {
    uint row_idx = tgid * num_sg + simd_gid;
    if (row_idx >= rows) return;
    uint blocks_per_row = cols / 32;
    uint row_bytes = blocks_per_row * 20;
    device const uchar *row = W + row_idx * row_bytes;
    float sum = 0.0;
    for (uint b = simd_lid; b < blocks_per_row; b += 32) {
        device const uchar *blk = row + b * 20;
        float d = float(*((device const half *)(blk)));
        float m = float(*((device const half *)(blk + 2)));
        device const uchar *qs = blk + 4;
        uint base = b * 32;
        for (uint i = 0; i < 16; i++) {
            uchar qb = qs[i];
            sum += (d * float(qb & 0xF) + m) * x[base + i];
            sum += (d * float(qb >> 4) + m) * x[base + 16 + i];
        }
    }
    sum = simd_sum(sum);
    if (simd_lid == 0) res[row_idx] += sum;
}
"#;

/// Q4_K dequant matvec: y = W_q4k @ x.
/// Q4_K: 256 values/block, 144 bytes/block (2B d, 2B dmin, 12B scales, 128B quants)
/// Each simdgroup (32 threads) handles one row. Threadgroup processes num_sg rows.
const MATVEC_Q4K: &str = r#"
inline void get_scale_min_k4(int j, device const uchar *q, thread float &sc, thread float &mn) {
    if (j < 4) { sc = float(q[j] & 63); mn = float(q[j + 4] & 63); }
    else { sc = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); mn = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}
kernel void matvec_q4k(
    device const uchar *W    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device float       *y    [[buffer(2)]],
    constant uint      &cols [[buffer(3)]],
    constant uint      &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint num_sg [[simdgroups_per_threadgroup]]
) {
    uint row_idx = tgid * num_sg + simd_gid;
    if (row_idx >= rows) return;
    uint blocks_per_row = cols / 256;
    device const uchar *row = W + row_idx * blocks_per_row * 144;
    float sum = 0.0;
    for (uint b = simd_lid; b < blocks_per_row; b += 32) {
        device const uchar *blk = row + b * 144;
        float d    = float(*((device const half *)(blk)));
        float dmin = float(*((device const half *)(blk + 2)));
        device const uchar *scales = blk + 4;
        device const uchar *qs = blk + 16;
        uint base = b * 256;
        for (int pair = 0; pair < 4; pair++) {
            int is0 = pair * 2, is1 = pair * 2 + 1;
            float sc0, mn0, sc1, mn1;
            get_scale_min_k4(is0, scales, sc0, mn0);
            get_scale_min_k4(is1, scales, sc1, mn1);
            float ds0 = d * sc0, dm0 = dmin * mn0;
            float ds1 = d * sc1, dm1 = dmin * mn1;
            device const uchar *qp = qs + pair * 32;
            uint idx = base + pair * 64;
            for (uint l = 0; l < 32; l++) {
                uchar qb = qp[l];
                sum += (ds0 * float(qb & 0xF) - dm0) * x[idx + l];
                sum += (ds1 * float(qb >> 4)   - dm1) * x[idx + 32 + l];
            }
        }
    }
    sum = simd_sum(sum);
    if (simd_lid == 0) y[row_idx] = sum;
}
"#;

/// Fused Q4_K matvec + residual add. Simdgroup-per-row pattern.
const MATVEC_Q4K_ADD: &str = r#"
inline void get_scale_min_k4_add(int j, device const uchar *q, thread float &sc, thread float &mn) {
    if (j < 4) { sc = float(q[j] & 63); mn = float(q[j + 4] & 63); }
    else { sc = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); mn = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}
kernel void matvec_q4k_add(
    device const uchar *W    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device float       *res  [[buffer(2)]],
    constant uint      &cols [[buffer(3)]],
    constant uint      &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint num_sg [[simdgroups_per_threadgroup]]
) {
    uint row_idx = tgid * num_sg + simd_gid;
    if (row_idx >= rows) return;
    uint blocks_per_row = cols / 256;
    device const uchar *row = W + row_idx * blocks_per_row * 144;
    float sum = 0.0;
    for (uint b = simd_lid; b < blocks_per_row; b += 32) {
        device const uchar *blk = row + b * 144;
        float d    = float(*((device const half *)(blk)));
        float dmin = float(*((device const half *)(blk + 2)));
        device const uchar *scales = blk + 4;
        device const uchar *qs = blk + 16;
        uint base = b * 256;
        for (int pair = 0; pair < 4; pair++) {
            int is0 = pair * 2, is1 = pair * 2 + 1;
            float sc0, mn0, sc1, mn1;
            get_scale_min_k4_add(is0, scales, sc0, mn0);
            get_scale_min_k4_add(is1, scales, sc1, mn1);
            float ds0 = d * sc0, dm0 = dmin * mn0;
            float ds1 = d * sc1, dm1 = dmin * mn1;
            device const uchar *qp = qs + pair * 32;
            uint idx = base + pair * 64;
            for (uint l = 0; l < 32; l++) {
                uchar qb = qp[l];
                sum += (ds0 * float(qb & 0xF) - dm0) * x[idx + l];
                sum += (ds1 * float(qb >> 4)   - dm1) * x[idx + 32 + l];
            }
        }
    }
    sum = simd_sum(sum);
    if (simd_lid == 0) res[row_idx] += sum;
}
"#;

/// Fused Q6_K matvec + residual add. Simdgroup-per-row pattern.
const MATVEC_Q6K_ADD: &str = r#"
kernel void matvec_q6k_add(
    device const uchar *W    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device float       *res  [[buffer(2)]],
    constant uint      &cols [[buffer(3)]],
    constant uint      &rows [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint num_sg [[simdgroups_per_threadgroup]]
) {
    uint row_idx = tgid * num_sg + simd_gid;
    if (row_idx >= rows) return;
    uint blocks_per_row = cols / 256;
    device const uchar *row = W + row_idx * blocks_per_row * 210;
    float sum = 0.0;
    for (uint b = simd_lid; b < blocks_per_row; b += 32) {
        device const uchar *blk = row + b * 210;
        device const uchar *ql = blk;
        device const uchar *qh = blk + 128;
        device const char  *sc = (device const char *)(blk + 192);
        float d = float(*((device const half *)(blk + 208)));
        uint base = b * 256;
        for (uint n = 0; n < 2; n++) {
            uint sb = n * 8;
            device const uchar *qlp = ql + n * 64;
            device const uchar *qhp = qh + n * 32;
            uint idx = base + n * 128;
            for (uint h = 0; h < 2; h++) {
                float s0 = d * float(sc[sb+h+0]), s2 = d * float(sc[sb+h+2]);
                float s4 = d * float(sc[sb+h+4]), s6 = d * float(sc[sb+h+6]);
                for (uint l = h * 16; l < h * 16 + 16; l++) {
                    int q1 = (int(qlp[l]    & 0xF) | (int((qhp[l] >> 0) & 3) << 4)) - 32;
                    int q2 = (int(qlp[l+32] & 0xF) | (int((qhp[l] >> 2) & 3) << 4)) - 32;
                    int q3 = (int(qlp[l]     >> 4) | (int((qhp[l] >> 4) & 3) << 4)) - 32;
                    int q4 = (int(qlp[l+32]  >> 4) | (int((qhp[l] >> 6) & 3) << 4)) - 32;
                    sum += s0 * float(q1) * x[idx + l];
                    sum += s2 * float(q2) * x[idx + 32 + l];
                    sum += s4 * float(q3) * x[idx + 64 + l];
                    sum += s6 * float(q4) * x[idx + 96 + l];
                }
            }
        }
    }
    sum = simd_sum(sum);
    if (simd_lid == 0) res[row_idx] += sum;
}
"#;

/// Dequant single row from Q4_0 embedding table.
/// Launch with dim threads. Q4_0: 32 vals/block, 18 bytes/block.
const EMBED_Q4_0: &str = r#"
kernel void embed_q4_0(
    device const uchar *W        [[buffer(0)]],
    device float       *out      [[buffer(1)]],
    constant uint      &dim      [[buffer(2)]],
    constant uint      &token_id [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dim) return;
    uint bpr = dim / 32;
    device const uchar *row = W + token_id * bpr * 18;
    uint bi = gid / 32, vi = gid % 32;
    device const uchar *blk = row + bi * 18;
    float d = float(*((device const half *)(blk)));
    device const uchar *qs = blk + 2;
    uchar qb = qs[vi < 16 ? vi : vi - 16];
    float q = (vi < 16) ? float(qb & 0xF) - 8.0 : float(qb >> 4) - 8.0;
    out[gid] = d * q;
}
"#;

/// Dequant single row from Q6_K embedding table.
/// Launch with dim threads.
const EMBED_Q6K: &str = r#"
kernel void embed_q6k(
    device const uchar *W        [[buffer(0)]],
    device float       *out      [[buffer(1)]],
    constant uint      &dim      [[buffer(2)]],
    constant uint      &token_id [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dim) return;
    uint bpr = dim / 256;
    device const uchar *row = W + token_id * bpr * 210;
    uint bi = gid / 256, vi = gid % 256;
    device const uchar *blk = row + bi * 210;
    device const uchar *ql = blk, *qh = blk + 128;
    device const char *sc = (device const char *)(blk + 192);
    float d = float(*((device const half *)(blk + 208)));
    uint n = vi / 128, rem = vi % 128, l = rem % 32, sub = rem / 32;
    device const uchar *qlp = ql + n * 64, *qhp = qh + n * 32;
    uint sb = n * 8;
    uint is = l / 16;  // 0 for l=0..15, 1 for l=16..31
    int q;
    float s;
    switch (sub) {
        case 0: q = (int(qlp[l]    & 0xF) | (int((qhp[l] >> 0) & 3) << 4)) - 32; s = d * float(sc[sb+is+0]); break;
        case 1: q = (int(qlp[l+32] & 0xF) | (int((qhp[l] >> 2) & 3) << 4)) - 32; s = d * float(sc[sb+is+2]); break;
        case 2: q = (int(qlp[l]     >> 4) | (int((qhp[l] >> 4) & 3) << 4)) - 32; s = d * float(sc[sb+is+4]); break;
        case 3: q = (int(qlp[l+32]  >> 4) | (int((qhp[l] >> 6) & 3) << 4)) - 32; s = d * float(sc[sb+is+6]); break;
    }
    out[gid] = s * float(q);
}
"#;

/// RoPE: rotary position embeddings. Operates on pairs within each head.
/// Launch with n_heads * head_dim/2 threads.
const ROPE: &str = r#"
kernel void rope(
    device float       *vec       [[buffer(0)]],
    constant uint      &head_dim  [[buffer(1)]],
    constant uint      &position  [[buffer(2)]],
    constant float     &freq_base [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_hd = head_dim / 2;
    uint head = gid / half_hd;
    uint pair = gid % half_hd;
    float freq = 1.0 / pow(freq_base, float(2 * pair) / float(head_dim));
    float angle = float(position) * freq;
    float ca = cos(angle), sa = sin(angle);
    uint idx = head * head_dim + pair * 2;
    float x0 = vec[idx], x1 = vec[idx + 1];
    vec[idx]     = x0 * ca - x1 * sa;
    vec[idx + 1] = x0 * sa + x1 * ca;
}
"#;

/// Store K/V into cache at position. Launch with kv_dim threads.
const KV_STORE: &str = r#"
kernel void kv_store(
    device const float *k       [[buffer(0)]],
    device const float *v       [[buffer(1)]],
    device float       *k_cache [[buffer(2)]],
    device float       *v_cache [[buffer(3)]],
    constant uint      &kv_dim  [[buffer(4)]],
    constant uint      &pos     [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= kv_dim) return;
    k_cache[pos * kv_dim + gid] = k[gid];
    v_cache[pos * kv_dim + gid] = v[gid];
}
"#;

/// Multi-head attention with GQA. One threadgroup per query head.
/// Computes: softmax(Q·K^T / sqrt(d)) · V for each head.
const ATTENTION: &str = r#"
kernel void attention(
    device const float *Q       [[buffer(0)]],
    device const float *K_cache [[buffer(1)]],
    device const float *V_cache [[buffer(2)]],
    device float       *out     [[buffer(3)]],
    constant uint      &seq_len    [[buffer(4)]],
    constant uint      &head_dim   [[buffer(5)]],
    constant uint      &n_kv_heads [[buffer(6)]],
    constant uint      &gqa_ratio  [[buffer(7)]],
    constant uint      &kv_dim     [[buffer(8)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    uint q_head = gid;
    uint kv_head = q_head / gqa_ratio;
    device const float *q = Q + q_head * head_dim;
    float inv_sqrt = rsqrt(float(head_dim));

    threadgroup float scores[2048];
    // Dot products: Q · K_t for each cached position
    for (uint t = lid; t < seq_len; t += tgs) {
        device const float *kt = K_cache + t * kv_dim + kv_head * head_dim;
        float dot = 0.0;
        for (uint d = 0; d < head_dim; d++) dot += q[d] * kt[d];
        scores[t] = dot * inv_sqrt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax: find max
    threadgroup float sm[32];
    float mx = -INFINITY;
    for (uint t = lid; t < seq_len; t += tgs) mx = max(mx, scores[t]);
    mx = simd_max(mx);
    if (lid % 32 == 0) sm[lid/32] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32) { float v = (lid < (tgs+31)/32) ? sm[lid] : -INFINITY; v = simd_max(v); if (lid==0) sm[0]=v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxv = sm[0];

    // Softmax: exp and sum
    float s = 0.0;
    for (uint t = lid; t < seq_len; t += tgs) { scores[t] = exp(scores[t]-maxv); s += scores[t]; }
    s = simd_sum(s);
    if (lid % 32 == 0) sm[lid/32] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32) { float v = (lid < (tgs+31)/32) ? sm[lid] : 0.0; v = simd_sum(v); if (lid==0) sm[0]=v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = 1.0 / sm[0];
    for (uint t = lid; t < seq_len; t += tgs) scores[t] *= inv_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weighted sum of values
    device float *o = out + q_head * head_dim;
    for (uint d = lid; d < head_dim; d += tgs) {
        float val = 0.0;
        for (uint t = 0; t < seq_len; t++)
            val += scores[t] * V_cache[t * kv_dim + kv_head * head_dim + d];
        o[d] = val;
    }
}
"#;

/// SiLU(gate) * up — the SwiGLU activation. Launch with dim threads.
const SILU_MUL: &str = r#"
kernel void silu_mul(
    device const float *gate [[buffer(0)]],
    device const float *up   [[buffer(1)]],
    device float       *out  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float g = gate[gid];
    out[gid] = (g / (1.0 + exp(-g))) * up[gid];
}
"#;

/// GPU argmax: find index of maximum value in logits. Single threadgroup.
/// Eliminates CPU-side scan of 128K+ floats (~1.7ms → ~10µs).
const ARGMAX: &str = r#"
kernel void argmax(
    device const float *logits [[buffer(0)]],
    device uint        *result [[buffer(1)]],
    constant uint      &n      [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    // Phase 1: each thread finds local max
    float best_val = -INFINITY;
    uint best_id = 0;
    for (uint i = lid; i < n; i += tgs) {
        float v = logits[i];
        if (v > best_val) { best_val = v; best_id = i; }
    }
    // Phase 2: simdgroup reduction (32 threads)
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_down(best_val, offset);
        uint other_id = simd_shuffle_down(best_id, offset);
        if (other_val > best_val) { best_val = other_val; best_id = other_id; }
    }
    // Phase 3: threadgroup reduction
    threadgroup float tg_vals[32];
    threadgroup uint tg_ids[32];
    if (lid % 32 == 0) { tg_vals[lid / 32] = best_val; tg_ids[lid / 32] = best_id; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32) {
        float v = (lid < (tgs + 31) / 32) ? tg_vals[lid] : -INFINITY;
        uint id = (lid < (tgs + 31) / 32) ? tg_ids[lid] : 0;
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float ov = simd_shuffle_down(v, offset);
            uint oid = simd_shuffle_down(id, offset);
            if (ov > v) { v = ov; id = oid; }
        }
        if (lid == 0) result[0] = id;
    }
}
"#;

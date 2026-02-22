//! Metal shader source for Qwen3 transformer inference.
//! Q4_K/Q6_K dequant fused into matvec for bandwidth-optimal inference.
//! Element-wise ops (RMSNorm, RoPE, SiLU, attention) run on GPU via single command buffer.

pub fn all_kernels() -> String {
    format!("{HEADER}{RMSNORM}{MATVEC_Q4K}{MATVEC_Q6K}{EMBED_Q6K}{ROPE}{KV_STORE}{ATTENTION}{SILU_MUL}{RESIDUAL_ADD}")
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

/// Q4_K dequant matvec: y = W_q4k @ x. Fused dequant in dot-product loop.
/// Q4_K: 256 values/block, 144 bytes/block (2h d, 2h dmin, 12 scales, 128 qs)
const MATVEC_Q4K: &str = r#"
inline void get_scale_min_k4(int j, device const uchar *q, thread float &sc, thread float &mn) {
    if (j < 4) {
        sc = float(q[j] & 63);
        mn = float(q[j + 4] & 63);
    } else {
        sc = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        mn = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

kernel void matvec_q4k(
    device const uchar *W    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device float       *y    [[buffer(2)]],
    constant uint      &cols [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    uint blocks_per_row = cols / 256;
    device const uchar *row = W + gid * blocks_per_row * 144;
    float sum = 0.0;

    for (uint b = lid; b < blocks_per_row; b += tgs) {
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
    threadgroup float parts[32];
    if (lid % 32 == 0) parts[lid / 32] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32) {
        float v = (lid < (tgs + 31) / 32) ? parts[lid] : 0.0;
        v = simd_sum(v);
        if (lid == 0) y[gid] = v;
    }
}
"#;

/// Q6_K dequant matvec: y = W_q6k @ x.
/// Q6_K: 256 values/block, 210 bytes/block (128 ql, 64 qh, 16 scales, 2 d)
const MATVEC_Q6K: &str = r#"
kernel void matvec_q6k(
    device const uchar *W    [[buffer(0)]],
    device const float *x    [[buffer(1)]],
    device float       *y    [[buffer(2)]],
    constant uint      &cols [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]]
) {
    uint blocks_per_row = cols / 256;
    device const uchar *row = W + gid * blocks_per_row * 210;
    float sum = 0.0;

    for (uint b = lid; b < blocks_per_row; b += tgs) {
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
            float s0 = d * float(sc[sb+0]), s2 = d * float(sc[sb+2]);
            float s4 = d * float(sc[sb+4]), s6 = d * float(sc[sb+6]);
            uint idx = base + n * 128;
            for (uint l = 0; l < 32; l++) {
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

    sum = simd_sum(sum);
    threadgroup float parts[32];
    if (lid % 32 == 0) parts[lid / 32] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32) {
        float v = (lid < (tgs + 31) / 32) ? parts[lid] : 0.0;
        v = simd_sum(v);
        if (lid == 0) y[gid] = v;
    }
}
"#;

/// Dequant single row from Q6_K embedding table.
/// Launch with dim threads (1024 for Qwen3-0.6B).
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
    int q;
    float s;
    switch (sub) {
        case 0: q = (int(qlp[l]    & 0xF) | (int((qhp[l] >> 0) & 3) << 4)) - 32; s = d * float(sc[sb+0]); break;
        case 1: q = (int(qlp[l+32] & 0xF) | (int((qhp[l] >> 2) & 3) << 4)) - 32; s = d * float(sc[sb+2]); break;
        case 2: q = (int(qlp[l]     >> 4) | (int((qhp[l] >> 4) & 3) << 4)) - 32; s = d * float(sc[sb+4]); break;
        case 3: q = (int(qlp[l+32]  >> 4) | (int((qhp[l] >> 6) & 3) << 4)) - 32; s = d * float(sc[sb+6]); break;
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

/// Residual add: x += y. Launch with dim threads.
const RESIDUAL_ADD: &str = r#"
kernel void residual_add(
    device float       *x [[buffer(0)]],
    device const float *y [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    x[gid] += y[gid];
}
"#;

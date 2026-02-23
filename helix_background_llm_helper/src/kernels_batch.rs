//! Batch matmul kernels with explicit scalar accumulators.
//! Each kernel is generated with N named float accumulators (s0, s1, ..., sN-1)
//! guaranteed to stay in GPU registers (no arrays, no dynamic indexing, no spill).
//! W is read exactly ONCE per row. X reads from L1/L2 cache (coalesced).
//! Grid = (ceil(rows/8), 1), TG = 256 = 8 simdgroups.
//! Iteration 10: unconditional FMA, conditional writes only.

/// Default batch size: 80 = 10 MMA groups × 8.
/// Compute ceiling at ~0.44ms/token (39% of M3 Max FP16 MMA peak).
/// B=128 tested: same per-token time but worse attention cost.
pub const BATCH_SIZE: u32 = 80;

/// Padded batch size for MMA kernel (must be multiple of 8 for simdgroup_matrix).
/// Buffers allocated with this size; element-wise kernels still dispatch with BATCH_SIZE.
pub const BATCH_PADDED: u32 = (BATCH_SIZE + 7) / 8 * 8; // 56

/// Grid Y is always 1 — kernel handles full batch internally.
pub const BATCH_NB: u32 = 1024;

const HEADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

// Q6K dequant: extract one of 256 elements from a 210-byte Q6K block.
inline float q6k_dequant(device const uchar *blk, uint elem) {
    device const uchar *ql = blk, *qh = blk + 128;
    device const char *sc = (device const char *)(blk + 192);
    float d = float(*((device const half *)(blk + 208)));
    uint n = elem / 128, rem = elem % 128, l = rem % 32, sub = rem / 32;
    device const uchar *qlp = ql + n * 64, *qhp = qh + n * 32;
    uint sb = n * 8, is_off = l / 16;
    int q; float ds;
    switch (sub) {
        case 0: q = (int(qlp[l] & 0xF) | (int((qhp[l] >> 0) & 3) << 4)) - 32; ds = d * float(sc[sb+is_off+0]); break;
        case 1: q = (int(qlp[l+32] & 0xF) | (int((qhp[l] >> 2) & 3) << 4)) - 32; ds = d * float(sc[sb+is_off+2]); break;
        case 2: q = (int(qlp[l] >> 4) | (int((qhp[l] >> 4) & 3) << 4)) - 32; ds = d * float(sc[sb+is_off+4]); break;
        default: q = (int(qlp[l+32] >> 4) | (int((qhp[l] >> 6) & 3) << 4)) - 32; ds = d * float(sc[sb+is_off+6]); break;
    }
    return ds * float(q);
}
"#;

/// Generate Q4_0 batch matmul: Y = W @ X, N explicit accumulators.
/// Coalesced: all 32 lanes cooperate per Q4_0 block, lane k handles element k.
pub fn gen_matmul_q4_0(n: u32) -> String {
    let mut s = String::with_capacity(4096);
    s += "kernel void matmul_q4_0(\n";
    s += "    device const uchar *W     [[buffer(0)]],\n";
    s += "    device const float *X     [[buffer(1)]],\n";
    s += "    device float       *Y     [[buffer(2)]],\n";
    s += "    constant uint      &cols  [[buffer(3)]],\n";
    s += "    constant uint      &rows  [[buffer(4)]],\n";
    s += "    constant uint      &batch [[buffer(5)]],\n";
    s += "    uint3 tgid [[threadgroup_position_in_grid]],\n";
    s += "    uint3 tid  [[thread_position_in_threadgroup]]\n";
    s += ") {\n";
    s += "    uint lid = tid.x;\n";
    s += "    uint lane = lid % 32;\n";
    s += "    uint sgid = lid / 32;\n";
    s += "    uint row_base = tgid.x * 8;\n";
    s += "    uint my_row = row_base + sgid;\n";
    s += "    if (my_row >= rows) return;\n";
    s += "    uint blocks_per_row = cols / 32;\n";
    s += "    uint row_bytes = blocks_per_row * 18;\n";
    s += "    device const uchar *wp = W + my_row * row_bytes;\n\n";
    for i in 0..n {
        s += &format!("    float s{i} = 0.0f;\n");
    }
    s += "\n    for (uint bi = 0; bi < blocks_per_row; bi++) {\n";
    s += "        device const uchar *bp = wp + bi * 18;\n";
    s += "        float d = float(*((device const half *)bp));\n";
    s += "        uchar qb = bp[2 + (lane & 15)];\n";
    s += "        uint shift = (lane >> 4) * 4;\n";
    s += "        float w = d * (float((qb >> shift) & 0xF) - 8.0f);\n";
    s += "        uint c = bi * 32 + lane;\n\n";
    for i in 0..n {
        s += &format!("        s{i} += w * X[{i}u * cols + c];\n");
    }
    s += "    }\n\n";
    for i in 0..n {
        s += &format!("    s{i} = simd_sum(s{i});\n");
    }
    s += "\n    if (lane == 0) {\n";
    for i in 0..n {
        s += &format!("        if ({i}u < batch) Y[{i}u * rows + my_row] = s{i};\n");
    }
    s += "    }\n";
    s += "}\n\n";
    s
}

/// Generate Q4_0 batch matmul + residual: res += W @ X.
pub fn gen_matmul_q4_0_add(n: u32) -> String {
    let mut s = String::with_capacity(4096);
    s += "kernel void matmul_q4_0_add(\n";
    s += "    device const uchar *W     [[buffer(0)]],\n";
    s += "    device const float *X     [[buffer(1)]],\n";
    s += "    device float       *res   [[buffer(2)]],\n";
    s += "    constant uint      &cols  [[buffer(3)]],\n";
    s += "    constant uint      &rows  [[buffer(4)]],\n";
    s += "    constant uint      &batch [[buffer(5)]],\n";
    s += "    uint3 tgid [[threadgroup_position_in_grid]],\n";
    s += "    uint3 tid  [[thread_position_in_threadgroup]]\n";
    s += ") {\n";
    s += "    uint lid = tid.x;\n";
    s += "    uint lane = lid % 32;\n";
    s += "    uint sgid = lid / 32;\n";
    s += "    uint row_base = tgid.x * 8;\n";
    s += "    uint my_row = row_base + sgid;\n";
    s += "    if (my_row >= rows) return;\n";
    s += "    uint blocks_per_row = cols / 32;\n";
    s += "    uint row_bytes = blocks_per_row * 18;\n";
    s += "    device const uchar *wp = W + my_row * row_bytes;\n\n";
    for i in 0..n {
        s += &format!("    float s{i} = 0.0f;\n");
    }
    s += "\n    for (uint bi = 0; bi < blocks_per_row; bi++) {\n";
    s += "        device const uchar *bp = wp + bi * 18;\n";
    s += "        float d = float(*((device const half *)bp));\n";
    s += "        uchar qb = bp[2 + (lane & 15)];\n";
    s += "        uint shift = (lane >> 4) * 4;\n";
    s += "        float w = d * (float((qb >> shift) & 0xF) - 8.0f);\n";
    s += "        uint c = bi * 32 + lane;\n\n";
    for i in 0..n {
        s += &format!("        s{i} += w * X[{i}u * cols + c];\n");
    }
    s += "    }\n\n";
    for i in 0..n {
        s += &format!("    s{i} = simd_sum(s{i});\n");
    }
    s += "\n    if (lane == 0) {\n";
    for i in 0..n {
        s += &format!("        if ({i}u < batch) res[{i}u * rows + my_row] += s{i};\n");
    }
    s += "    }\n";
    s += "}\n\n";
    s
}

/// Generate Q6K batch matmul: Y = W @ X, N explicit accumulators.
/// Q6K: 256 elements/block, 210 bytes/block. 32 lanes × 8 elements = coalesced.
pub fn gen_matmul_q6k(n: u32) -> String {
    let mut s = String::with_capacity(4096);
    s += "kernel void matmul_q6k(\n";
    s += "    device const uchar *W     [[buffer(0)]],\n";
    s += "    device const float *X     [[buffer(1)]],\n";
    s += "    device float       *Y     [[buffer(2)]],\n";
    s += "    constant uint      &cols  [[buffer(3)]],\n";
    s += "    constant uint      &rows  [[buffer(4)]],\n";
    s += "    constant uint      &batch [[buffer(5)]],\n";
    s += "    uint3 tgid [[threadgroup_position_in_grid]],\n";
    s += "    uint3 tid  [[thread_position_in_threadgroup]]\n";
    s += ") {\n";
    s += "    uint lid = tid.x;\n";
    s += "    uint lane = lid % 32;\n";
    s += "    uint sgid = lid / 32;\n";
    s += "    uint row_base = tgid.x * 8;\n";
    s += "    uint my_row = row_base + sgid;\n";
    s += "    if (my_row >= rows) return;\n";
    s += "    uint blocks_per_row = cols / 256;\n";
    s += "    device const uchar *wp = W + my_row * blocks_per_row * 210;\n\n";
    for i in 0..n {
        s += &format!("    float s{i} = 0.0f;\n");
    }
    s += "\n    for (uint bi = 0; bi < blocks_per_row; bi++) {\n";
    s += "        device const uchar *blk = wp + bi * 210;\n";
    s += "        for (uint e = 0; e < 8; e++) {\n";
    s += "            uint elem = e * 32 + lane;\n";
    s += "            float w = q6k_dequant(blk, elem);\n";
    s += "            uint c = bi * 256 + elem;\n";
    for i in 0..n {
        s += &format!("            s{i} += w * X[{i}u * cols + c];\n");
    }
    s += "        }\n";
    s += "    }\n\n";
    for i in 0..n {
        s += &format!("    s{i} = simd_sum(s{i});\n");
    }
    s += "\n    if (lane == 0) {\n";
    for i in 0..n {
        s += &format!("        if ({i}u < batch) Y[{i}u * rows + my_row] = s{i};\n");
    }
    s += "    }\n";
    s += "}\n\n";
    s
}

/// Generate FP16 Q4_0 batch matmul: Y(half) = W @ X(half), half accumulators.
/// All compute in half precision: 2x FMA throughput, half register pressure.
pub fn gen_matmul_q4_0_f16(n: u32) -> String {
    let mut s = String::with_capacity(4096);
    s += "kernel void matmul_q4_0_f16(\n";
    s += "    device const uchar *W     [[buffer(0)]],\n";
    s += "    device const half  *X     [[buffer(1)]],\n";
    s += "    device half        *Y     [[buffer(2)]],\n";
    s += "    constant uint      &cols  [[buffer(3)]],\n";
    s += "    constant uint      &rows  [[buffer(4)]],\n";
    s += "    constant uint      &batch [[buffer(5)]],\n";
    s += "    uint3 tgid [[threadgroup_position_in_grid]],\n";
    s += "    uint3 tid  [[thread_position_in_threadgroup]]\n";
    s += ") {\n";
    s += "    uint lid = tid.x;\n";
    s += "    uint lane = lid % 32;\n";
    s += "    uint sgid = lid / 32;\n";
    s += "    uint row_base = tgid.x * 8;\n";
    s += "    uint my_row = row_base + sgid;\n";
    s += "    if (my_row >= rows) return;\n";
    s += "    uint blocks_per_row = cols / 32;\n";
    s += "    uint row_bytes = blocks_per_row * 18;\n";
    s += "    device const uchar *wp = W + my_row * row_bytes;\n\n";
    // Half-precision accumulators — half register pressure
    for i in 0..n {
        s += &format!("    half s{i} = 0.0h;\n");
    }
    s += "\n    for (uint bi = 0; bi < blocks_per_row; bi++) {\n";
    s += "        device const uchar *bp = wp + bi * 18;\n";
    // Read scale directly as half (Q4_0 stores scale as FP16)
    s += "        half d = *((device const half *)bp);\n";
    s += "        uchar qb = bp[2 + (lane & 15)];\n";
    s += "        uint shift = (lane >> 4) * 4;\n";
    s += "        half w = d * (half(int(qb >> shift) & 0xF) - 8.0h);\n";
    s += "        uint c = bi * 32 + lane;\n\n";
    for i in 0..n {
        s += &format!("        s{i} += w * X[{i}u * cols + c];\n");
    }
    s += "    }\n\n";
    for i in 0..n {
        s += &format!("    s{i} = simd_sum(s{i});\n");
    }
    s += "\n    if (lane == 0) {\n";
    for i in 0..n {
        s += &format!("        if ({i}u < batch) Y[{i}u * rows + my_row] = s{i};\n");
    }
    s += "    }\n";
    s += "}\n\n";
    s
}

/// Generate FP16 Q4_1 batch matmul: Y(half) = W @ X(half), half accumulators.
/// Q4_1: 20 bytes per block (2B scale d, 2B min m, 16B nibbles). Dequant: w = d*nibble + m.
pub fn gen_matmul_q4_1_f16(n: u32) -> String {
    let mut s = String::with_capacity(4096);
    s += "kernel void matmul_q4_1_f16(\n";
    s += "    device const uchar *W     [[buffer(0)]],\n";
    s += "    device const half  *X     [[buffer(1)]],\n";
    s += "    device half        *Y     [[buffer(2)]],\n";
    s += "    constant uint      &cols  [[buffer(3)]],\n";
    s += "    constant uint      &rows  [[buffer(4)]],\n";
    s += "    constant uint      &batch [[buffer(5)]],\n";
    s += "    uint3 tgid [[threadgroup_position_in_grid]],\n";
    s += "    uint3 tid  [[thread_position_in_threadgroup]]\n";
    s += ") {\n";
    s += "    uint lid = tid.x;\n";
    s += "    uint lane = lid % 32;\n";
    s += "    uint sgid = lid / 32;\n";
    s += "    uint row_base = tgid.x * 8;\n";
    s += "    uint my_row = row_base + sgid;\n";
    s += "    if (my_row >= rows) return;\n";
    s += "    uint blocks_per_row = cols / 32;\n";
    s += "    uint row_bytes = blocks_per_row * 20;\n"; // Q4_1: 20 bytes per block
    s += "    device const uchar *wp = W + my_row * row_bytes;\n\n";
    for i in 0..n {
        s += &format!("    half s{i} = 0.0h;\n");
    }
    s += "\n    for (uint bi = 0; bi < blocks_per_row; bi++) {\n";
    s += "        device const uchar *bp = wp + bi * 20;\n"; // 20 bytes per Q4_1 block
    s += "        half d = *((device const half *)bp);\n";
    s += "        half m = *((device const half *)(bp + 2));\n";
    s += "        uchar qb = bp[4 + (lane & 15)];\n"; // nibbles start at byte 4
    s += "        uint shift = (lane >> 4) * 4;\n";
    s += "        half w = d * half(int(qb >> shift) & 0xF) + m;\n";
    s += "        uint c = bi * 32 + lane;\n\n";
    for i in 0..n {
        s += &format!("        s{i} += w * X[{i}u * cols + c];\n");
    }
    s += "    }\n\n";
    for i in 0..n {
        s += &format!("    s{i} = simd_sum(s{i});\n");
    }
    s += "\n    if (lane == 0) {\n";
    for i in 0..n {
        s += &format!("        if ({i}u < batch) Y[{i}u * rows + my_row] = s{i};\n");
    }
    s += "    }\n";
    s += "}\n\n";
    s
}

/// Generate FP16 Q6K batch matmul: Y(half) = W @ X(half).
pub fn gen_matmul_q6k_f16(n: u32) -> String {
    let mut s = String::with_capacity(4096);
    s += "kernel void matmul_q6k_f16(\n";
    s += "    device const uchar *W     [[buffer(0)]],\n";
    s += "    device const half  *X     [[buffer(1)]],\n";
    s += "    device half        *Y     [[buffer(2)]],\n";
    s += "    constant uint      &cols  [[buffer(3)]],\n";
    s += "    constant uint      &rows  [[buffer(4)]],\n";
    s += "    constant uint      &batch [[buffer(5)]],\n";
    s += "    uint3 tgid [[threadgroup_position_in_grid]],\n";
    s += "    uint3 tid  [[thread_position_in_threadgroup]]\n";
    s += ") {\n";
    s += "    uint lid = tid.x;\n";
    s += "    uint lane = lid % 32;\n";
    s += "    uint sgid = lid / 32;\n";
    s += "    uint row_base = tgid.x * 8;\n";
    s += "    uint my_row = row_base + sgid;\n";
    s += "    if (my_row >= rows) return;\n";
    s += "    uint blocks_per_row = cols / 256;\n";
    s += "    device const uchar *wp = W + my_row * blocks_per_row * 210;\n\n";
    for i in 0..n {
        s += &format!("    half s{i} = 0.0h;\n");
    }
    // Q6K dequant produces half directly
    s += "\n    for (uint bi = 0; bi < blocks_per_row; bi++) {\n";
    s += "        device const uchar *blk = wp + bi * 210;\n";
    s += "        for (uint e = 0; e < 8; e++) {\n";
    s += "            uint elem = e * 32 + lane;\n";
    s += "            half w = half(q6k_dequant(blk, elem));\n";
    s += "            uint c = bi * 256 + elem;\n";
    for i in 0..n {
        s += &format!("            s{i} += w * X[{i}u * cols + c];\n");
    }
    s += "        }\n";
    s += "    }\n\n";
    for i in 0..n {
        s += &format!("    s{i} = simd_sum(s{i});\n");
    }
    s += "\n    if (lane == 0) {\n";
    for i in 0..n {
        s += &format!("        if ({i}u < batch) Y[{i}u * rows + my_row] = s{i};\n");
    }
    s += "    }\n";
    s += "}\n\n";
    s
}

/// Generate named FP16 Q4_0 kernel variant for benchmarking different N.
pub fn gen_matmul_q4_0_f16_named(name: &str, n: u32) -> String {
    let mut s = String::with_capacity(4096);
    s += &format!("kernel void {name}(\n");
    s += "    device const uchar *W     [[buffer(0)]],\n";
    s += "    device const half  *X     [[buffer(1)]],\n";
    s += "    device half        *Y     [[buffer(2)]],\n";
    s += "    constant uint      &cols  [[buffer(3)]],\n";
    s += "    constant uint      &rows  [[buffer(4)]],\n";
    s += "    constant uint      &batch [[buffer(5)]],\n";
    s += "    uint3 tgid [[threadgroup_position_in_grid]],\n";
    s += "    uint3 tid  [[thread_position_in_threadgroup]]\n";
    s += ") {\n";
    s += "    uint lid = tid.x;\n";
    s += "    uint lane = lid % 32;\n";
    s += "    uint sgid = lid / 32;\n";
    s += "    uint row_base = tgid.x * 8;\n";
    s += "    uint my_row = row_base + sgid;\n";
    s += "    if (my_row >= rows) return;\n";
    s += "    uint blocks_per_row = cols / 32;\n";
    s += "    uint row_bytes = blocks_per_row * 18;\n";
    s += "    device const uchar *wp = W + my_row * row_bytes;\n\n";
    for i in 0..n {
        s += &format!("    half s{i} = 0.0h;\n");
    }
    s += "\n    for (uint bi = 0; bi < blocks_per_row; bi++) {\n";
    s += "        device const uchar *bp = wp + bi * 18;\n";
    s += "        half d = *((device const half *)bp);\n";
    s += "        uchar qb = bp[2 + (lane & 15)];\n";
    s += "        uint shift = (lane >> 4) * 4;\n";
    s += "        half w = d * (half(int(qb >> shift) & 0xF) - 8.0h);\n";
    s += "        uint c = bi * 32 + lane;\n\n";
    for i in 0..n {
        s += &format!("        s{i} += w * X[{i}u * cols + c];\n");
    }
    s += "    }\n\n";
    for i in 0..n {
        s += &format!("    s{i} = simd_sum(s{i});\n");
    }
    s += "\n    if (lane == 0) {\n";
    for i in 0..n {
        s += &format!("        if ({i}u < batch) Y[{i}u * rows + my_row] = s{i};\n");
    }
    s += "    }\n";
    s += "}\n\n";
    s
}

/// Generate Q4_0 batch matmul v3b using simdgroup_matrix hardware MMA.
/// Grid: (ceil(rows/32), 1), TG: 128 = 4 simdgroups, 32 rows per TG.
/// Per-simdgroup tile staging with simdgroup_barrier (not threadgroup_barrier).
/// Each simdgroup independently stages its 8 rows × 32 K elements into a local tile,
/// then iterates 4 K steps loading B from the tile, 7 A loads + MMAs per step.
/// Note: 256-thread variant tested and was 5.4% slower (worse TG occupancy).
pub fn gen_matmul_q4_0_mma() -> String {
    let mut s = String::with_capacity(8192);
    s += "kernel void matmul_q4_0_mma(\n";
    s += "    device const uchar *W     [[buffer(0)]],\n";
    s += "    device const half  *X     [[buffer(1)]],\n";
    s += "    device half        *Y     [[buffer(2)]],\n";
    s += "    constant uint      &cols  [[buffer(3)]],\n";
    s += "    constant uint      &rows  [[buffer(4)]],\n";
    s += "    constant uint      &batch [[buffer(5)]],\n";
    s += "    uint3 tgid [[threadgroup_position_in_grid]],\n";
    s += "    uint3 tid  [[thread_position_in_threadgroup]]\n";
    s += ") {\n";
    s += "    uint lid = tid.x;\n";
    s += "    uint sgid = lid / 32;\n";
    s += "    uint lane = lid % 32;\n";
    s += "    uint row_base = tgid.x * 32;\n";
    s += "    if (row_base >= rows) return;\n";
    s += "\n";
    // Per-simdgroup tile: 32 K × 8 rows, stride=9 (bank conflict avoidance)
    // 4 simdgroups × 32 × 9 = 1152 halfs = 2304 bytes threadgroup memory
    s += "    threadgroup half tiles[4 * 32 * 9];\n";
    s += "    threadgroup half *tile = tiles + sgid * 32 * 9;\n";
    s += "\n";
    let n_groups = BATCH_SIZE / 8;
    // Declare accumulators (4 per line for readability)
    for g in 0..n_groups {
        if g % 4 == 0 {
            let end = std::cmp::min(g + 4, n_groups);
            s += "    simdgroup_half8x8 ";
            for i in g..end {
                if i > g { s += ", "; }
                s += &format!("C{i}(0.0h)");
            }
            s += ";\n";
        }
    }
    s += "\n";
    s += "    uint blocks_per_row = cols / 32;\n";
    s += "    uint row_bytes = blocks_per_row * 18;\n";
    s += "\n";
    s += "    for (uint bi = 0; bi < blocks_per_row; bi++) {\n";
    s += "\n";
    // Stage: 32 threads per SG fill 256 elements (32K × 8rows)
    s += "        for (uint t = lane; t < 256; t += 32) {\n";
    s += "            uint k = t >> 3;\n";
    s += "            uint r = t & 7;\n";
    s += "            uint global_row = row_base + sgid * 8 + r;\n";
    s += "            half w = 0.0h;\n";
    s += "            if (global_row < rows) {\n";
    s += "                device const uchar *bp = W + global_row * row_bytes + bi * 18;\n";
    s += "                half dd = *((device const half *)bp);\n";
    s += "                w = dd * (half(int(bp[2 + (k & 15)] >> ((k >> 4) * 4)) & 0xF) - 8.0h);\n";
    s += "            }\n";
    s += "            tile[k * 9 + r] = w;\n";
    s += "        }\n";
    s += "        simdgroup_barrier(mem_flags::mem_threadgroup);\n";
    s += "\n";
    // 4 K steps: load B from tile, N A loads + MMAs
    s += "        for (uint ks = 0; ks < 4; ks++) {\n";
    s += "            simdgroup_half8x8 B;\n";
    s += "            simdgroup_load(B, tile + ks * 8 * 9, 9);\n";
    s += "\n";
    for g in 0..n_groups {
        let off = g * 8;
        s += &format!("            {{ simdgroup_half8x8 A; simdgroup_load(A, X + {off}u * cols + bi * 32 + ks * 8, cols); simdgroup_multiply_accumulate(C{g}, A, B, C{g}); }}\n");
    }
    s += "        }\n";
    s += "\n";
    s += "        simdgroup_barrier(mem_flags::mem_threadgroup);\n";
    s += "    }\n";
    s += "\n";
    // Store all batch groups directly
    for g in 0..n_groups {
        let off = g * 8;
        s += &format!("    simdgroup_store(C{g}, Y + {off}u * rows + row_base + sgid * 8, ulong(rows));\n");
    }
    s += "}\n\n";
    s
}
// ── Batch element-wise kernels (FP16) ──
// Layout: column-major, X[batch_idx * dim + elem_idx].
// All operate on half-precision data matching the FP16 matmul path.

/// Embed B tokens from Q4_0 embedding table.
/// Grid: (dim, batch), TG: (256, 1).
const EMBED_BATCH_Q4_0: &str = r#"
kernel void embed_batch_q4_0(
    device const uchar *W        [[buffer(0)]],
    device half        *out      [[buffer(1)]],
    device const uint  *tokens   [[buffer(2)]],
    constant uint      &dim      [[buffer(3)]],
    constant uint      &batch    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint elem = gid.x;
    uint bi = gid.y;
    if (elem >= dim || bi >= batch) return;
    uint token_id = tokens[bi];
    uint bpr = dim / 32;
    device const uchar *row = W + token_id * bpr * 18;
    uint blk_idx = elem / 32, vi = elem % 32;
    device const uchar *blk = row + blk_idx * 18;
    half d = *((device const half *)(blk));
    device const uchar *qs = blk + 2;
    uchar qb = qs[vi < 16 ? vi : vi - 16];
    half q = (vi < 16) ? half(int(qb & 0xF)) - 8.0h : half(int(qb >> 4)) - 8.0h;
    out[bi * dim + elem] = d * q;
}
"#;

/// Embed B tokens from Q6K embedding table.
const EMBED_BATCH_Q6K: &str = r#"
kernel void embed_batch_q6k(
    device const uchar *W        [[buffer(0)]],
    device half        *out      [[buffer(1)]],
    device const uint  *tokens   [[buffer(2)]],
    constant uint      &dim      [[buffer(3)]],
    constant uint      &batch    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint elem = gid.x;
    uint bi = gid.y;
    if (elem >= dim || bi >= batch) return;
    uint token_id = tokens[bi];
    uint bpr = dim / 256;
    device const uchar *row = W + token_id * bpr * 210;
    uint blk_idx = elem / 256, vi = elem % 256;
    device const uchar *blk = row + blk_idx * 210;
    // Reuse q6k_dequant from header
    out[bi * dim + elem] = half(q6k_dequant(blk, vi));
}
"#;

/// RMSNorm for B vectors independently. Same weight vector shared.
/// One threadgroup per batch item. TG = 256.
const RMSNORM_BATCH: &str = r#"
kernel void rmsnorm_batch(
    device const half  *x      [[buffer(0)]],
    device half        *y      [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    constant uint      &dim    [[buffer(3)]],
    constant float     &eps    [[buffer(4)]],
    constant uint      &batch  [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    if (tgid >= batch) return;
    device const half *xi = x + tgid * dim;
    device half *yi = y + tgid * dim;
    // Accumulate sum of squares in float for precision
    float sum_sq = 0.0;
    for (uint i = lid; i < dim; i += tgs) {
        float v = float(xi[i]);
        sum_sq += v * v;
    }
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
    for (uint i = lid; i < dim; i += tgs)
        yi[i] = half(float(xi[i]) * rms * float(weight[i]));
}
"#;

/// Fused residual add + RMSNorm: res += add; norm = rmsnorm(res) * weight.
/// Eliminates one read of res vs separate residual_add + rmsnorm.
/// One threadgroup per batch item. TG = 256.
const RESIDUAL_RMSNORM_BATCH: &str = r#"
kernel void residual_rmsnorm_batch(
    device half        *res    [[buffer(0)]],
    device const half  *add    [[buffer(1)]],
    device half        *norm   [[buffer(2)]],
    device const float *weight [[buffer(3)]],
    constant uint      &dim    [[buffer(4)]],
    constant float     &eps    [[buffer(5)]],
    constant uint      &batch  [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    if (tgid >= batch) return;
    device half *ri = res + tgid * dim;
    device const half *ai = add + tgid * dim;
    device half *ni = norm + tgid * dim;
    // Pass 1: residual add + accumulate sum of squares
    float sum_sq = 0.0;
    for (uint i = lid; i < dim; i += tgs) {
        float v = float(ri[i]) + float(ai[i]);
        ri[i] = half(v);
        sum_sq += v * v;
    }
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
    // Pass 2: normalize (re-read updated res)
    for (uint i = lid; i < dim; i += tgs)
        ni[i] = half(float(ri[i]) * rms * float(weight[i]));
}
"#;

/// RoPE for B vectors with consecutive positions [start_pos, start_pos+1, ...].
/// Grid: (n_pairs, batch), TG: (64, 1).
const ROPE_BATCH: &str = r#"
kernel void rope_batch(
    device half        *vec       [[buffer(0)]],
    constant uint      &head_dim  [[buffer(1)]],
    constant uint      &start_pos [[buffer(2)]],
    constant float     &freq_base [[buffer(3)]],
    constant uint      &total_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint pair_idx = gid.x;
    uint bi = gid.y;
    uint half_hd = head_dim / 2;
    uint head = pair_idx / half_hd;
    uint pair = pair_idx % half_hd;
    float freq = 1.0 / pow(freq_base, float(2 * pair) / float(head_dim));
    uint pos = start_pos + bi;
    float angle = float(pos) * freq;
    float ca = cos(angle), sa = sin(angle);
    uint base = bi * total_dim + head * head_dim + pair * 2;
    float x0 = float(vec[base]), x1 = float(vec[base + 1]);
    vec[base]     = half(x0 * ca - x1 * sa);
    vec[base + 1] = half(x0 * sa + x1 * ca);
}
"#;

/// Store B K/V vectors at consecutive cache positions.
/// Grid: (kv_dim, batch), TG: (64, 1).
const KV_STORE_BATCH: &str = r#"
kernel void kv_store_batch(
    device const half  *k       [[buffer(0)]],
    device const half  *v       [[buffer(1)]],
    device half        *k_cache [[buffer(2)]],
    device half        *v_cache [[buffer(3)]],
    constant uint      &kv_dim  [[buffer(4)]],
    constant uint      &start_pos [[buffer(5)]],
    constant uint      &batch   [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint elem = gid.x;
    uint bi = gid.y;
    if (elem >= kv_dim || bi >= batch) return;
    uint pos = start_pos + bi;
    k_cache[pos * kv_dim + elem] = k[bi * kv_dim + elem];
    v_cache[pos * kv_dim + elem] = v[bi * kv_dim + elem];
}
"#;

/// SiLU(gate) * up for B vectors. Flat 1D dispatch over B * dim elements.
const SILU_MUL_BATCH: &str = r#"
kernel void silu_mul_batch(
    device const half *gate [[buffer(0)]],
    device const half *up   [[buffer(1)]],
    device half       *out  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float g = float(gate[gid]);
    out[gid] = half((g / (1.0 + exp(-g))) * float(up[gid]));
}
"#;

/// Residual add: res += add. Flat 1D dispatch over B * dim elements.
const RESIDUAL_ADD_BATCH: &str = r#"
kernel void residual_add_batch(
    device half       *res [[buffer(0)]],
    device const half *add [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    res[gid] = res[gid] + add[gid];
}
"#;

/// Causal attention for B query tokens. One threadgroup per (batch_item, head).
/// Query i at position start_pos+i attends to K/V cache [0..start_pos+i].
/// Grid: (n_heads, batch), TG: (128, 1).
const ATTENTION_BATCH_CAUSAL: &str = r#"
kernel void attention_batch_causal(
    device const half  *Q       [[buffer(0)]],
    device const half  *K_cache [[buffer(1)]],
    device const half  *V_cache [[buffer(2)]],
    device half        *out     [[buffer(3)]],
    constant uint      &start_pos  [[buffer(4)]],
    constant uint      &head_dim   [[buffer(5)]],
    constant uint      &n_kv_heads [[buffer(6)]],
    constant uint      &gqa_ratio  [[buffer(7)]],
    constant uint      &kv_dim     [[buffer(8)]],
    constant uint      &total_q_dim [[buffer(9)]],
    constant uint      &batch      [[buffer(10)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tpg [[threads_per_threadgroup]]
) {
    uint q_head = gid.x;
    uint bi = gid.y;
    if (bi >= batch) return;
    uint lid = tid.x;
    uint tgs = tpg.x;
    uint kv_head = q_head / gqa_ratio;
    uint pos = start_pos + bi;
    uint seq_len = pos + 1;

    device const half *q = Q + bi * total_q_dim + q_head * head_dim;
    float inv_sqrt = rsqrt(float(head_dim));

    // Dot products: Q · K_t for each cached position
    threadgroup float scores[2048];
    for (uint t = lid; t < seq_len; t += tgs) {
        device const half *kt = K_cache + t * kv_dim + kv_head * head_dim;
        float dot = 0.0;
        for (uint d = 0; d < head_dim; d++) dot += float(q[d]) * float(kt[d]);
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
    device half *o = out + bi * total_q_dim + q_head * head_dim;
    for (uint d = lid; d < head_dim; d += tgs) {
        float val = 0.0;
        for (uint t = 0; t < seq_len; t++)
            val += scores[t] * float(V_cache[t * kv_dim + kv_head * head_dim + d]);
        o[d] = half(val);
    }
}
"#;

/// Batch argmax: find argmax for B logit vectors.
/// Grid: (batch, 1), TG: (1024, 1). One TG per batch item.
const ARGMAX_BATCH: &str = r#"
kernel void argmax_batch(
    device const half *logits [[buffer(0)]],
    device uint       *result [[buffer(1)]],
    constant uint     &n      [[buffer(2)]],
    constant uint     &batch  [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    if (tgid >= batch) return;
    device const half *lg = logits + tgid * n;
    float best_val = -INFINITY;
    uint best_id = 0;
    for (uint i = lid; i < n; i += tgs) {
        float v = float(lg[i]);
        if (v > best_val) { best_val = v; best_id = i; }
    }
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other_val = simd_shuffle_down(best_val, offset);
        uint other_id = simd_shuffle_down(best_id, offset);
        if (other_val > best_val) { best_val = other_val; best_id = other_id; }
    }
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
        if (lid == 0) result[tgid] = id;
    }
}
"#;

pub fn all_batch_kernels() -> String {
    let mut s = HEADER.to_string();
    // Q4_0 MMA v3b: quantized dequant + simdgroup staging (bandwidth-optimal)
    s += &gen_matmul_q4_0_mma();
    // FP16 scalar matmul (fallback for Q4_1/Q6K when MMA not available)
    s += &gen_matmul_q4_0_f16(BATCH_SIZE);
    s += &gen_matmul_q4_1_f16(BATCH_SIZE);
    s += &gen_matmul_q6k_f16(BATCH_SIZE);
    // Pure FP16 MMA matmul (for Q4_1/Q6K via pre-dequant, 256-thread TG)
    s += &crate::kernels_fp16::gen_matmul_fp16_mma();
    // Batch element-wise kernels
    s += crate::kernels_fp16::EMBED_FP16;
    s += EMBED_BATCH_Q4_0;
    s += EMBED_BATCH_Q6K;
    s += RMSNORM_BATCH;
    s += RESIDUAL_RMSNORM_BATCH;
    s += ROPE_BATCH;
    s += KV_STORE_BATCH;
    s += SILU_MUL_BATCH;
    s += RESIDUAL_ADD_BATCH;
    s += ATTENTION_BATCH_CAUSAL;
    s += ARGMAX_BATCH;
    s
}

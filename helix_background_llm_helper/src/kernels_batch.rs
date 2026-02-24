//! 70B-optimized Metal batch kernels: Q5K/Q6K/Q8_0 MMA staging + element-wise.
//! Each MMA kernel dequants quantized weights in threadgroup memory, then uses
//! simdgroup_multiply_accumulate for hardware FP16 MMA throughput.

/// Tile size for grid-tiled MMA kernels.
pub const TILE_B: u32 = 80;

/// MSL header: includes, dequant helpers for Q6K and Q5K.
pub const HEADER_STR: &str = r#"
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

// Q5K dequant: extract one of 256 elements from a 176-byte Q5K block.
// Layout: d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176 bytes.
inline float q5k_dequant(device const uchar *blk, uint elem) {
    float d    = float(*((device const half *)(blk)));
    float dmin = float(*((device const half *)(blk + 2)));
    device const uchar *scales = blk + 4;
    device const uchar *qh = blk + 16;
    device const uchar *qs = blk + 48;
    uint pair = elem / 64;
    uint sub = (elem / 32) & 1;
    uint l = elem & 31;
    int is_val = int(pair * 2 + sub);
    float sc, mn;
    if (is_val < 4) {
        sc = float(scales[is_val] & 63);
        mn = float(scales[is_val + 4] & 63);
    } else {
        sc = float((scales[is_val + 4] & 0xF) | ((scales[is_val - 4] >> 6) << 4));
        mn = float((scales[is_val + 4] >> 4) | ((scales[is_val] >> 6) << 4));
    }
    uchar qb = qs[pair * 32 + l];
    uint nibble = (qb >> (sub * 4)) & 0xF;
    uint hb = (qh[l] >> (pair * 2 + sub)) & 1;
    return (d * sc) * float(nibble + hb * 16) - (dmin * mn);
}
"#;

// ── MMA staging grid kernels (batch matmul, b>1) ──

/// Grid-tiled Q5K MMA: optimized sub-blocked staging + simdgroup MMA.
/// Key optimization: within a sub-block, all elements in a lane share the same
/// row, hence the same d/dmin/sc/mn. Hoists shared dequant values to eliminate
/// ~60% of per-element ALU (from ~15 ops to ~5 ops per element).
/// Grid: (ceil(rows/32), ceil(b/tile_b)), TG: 128 = 4 SG.
pub fn gen_matmul_q5k_mma_grid(name: &str, tile_b: u32) -> String {
    assert!(tile_b % 8 == 0, "tile_b must be multiple of 8");
    let n_groups = tile_b / 8;
    let mut s = String::with_capacity(12000);
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
    s += "    uint sgid = lid / 32;\n";
    s += "    uint lane = lid % 32;\n";
    s += "    uint row_base = tgid.x * 32;\n";
    s += "    if (row_base >= rows) return;\n";
    s += &format!("    uint b_off = tgid.y * {tile_b}u;\n");
    s += "\n";
    s += "    threadgroup half tiles[4 * 32 * 9];\n";
    s += "    threadgroup half *tile = tiles + sgid * 32 * 9;\n";
    s += "\n";
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
    s += "    int blocks_per_row = int(cols / 256);\n";
    s += "    int row_bytes = blocks_per_row * 176;\n";
    // lane -> (r, k_base): r = lane & 7 is constant per lane (the row within SG tile)
    // k values = (lane/8) + {0,4,8,12,16,20,24,28} — 8 columns per lane
    s += "    int my_r = int(lane & 7);\n";
    s += "    int global_row = int(row_base) + int(sgid) * 8 + my_r;\n";
    s += "    bool valid = global_row < int(rows);\n";
    s += "\n";
    // Pre-compute the block pointer base for this row (constant across all blocks)
    s += "    device const uchar *row_w = W + global_row * row_bytes;\n";
    s += "\n";
    s += "    for (int bi = 0; bi < blocks_per_row; bi++) {\n";
    // Load block header ONCE per block per lane (shared across 8 sub-blocks)
    s += "        device const uchar *blk = row_w + bi * 176;\n";
    s += "        float d_val = valid ? float(*((device const half *)(blk))) : 0.0;\n";
    s += "        float dmin_val = valid ? float(*((device const half *)(blk + 2))) : 0.0;\n";
    s += "        device const uchar *scales = blk + 4;\n";
    s += "        device const uchar *qh = blk + 16;\n";
    s += "        device const uchar *qs = blk + 48;\n";
    s += "\n";
    s += "        for (int sb = 0; sb < 8; sb++) {\n";
    // Within a sub-block, all 8 elements per lane share pair/sub/sc/mn:
    // pair = sb/2, sub = sb&1. Hoist scale loads (shared across 8 cols per lane).
    s += "            int pair = sb >> 1;\n";
    s += "            int sub_idx = sb & 1;\n";
    s += "            float ds = 0.0, dm = 0.0;\n";
    s += "            if (valid) {\n";
    s += "                float sc, mn;\n";
    s += "                if (sb < 4) { sc = float(scales[sb] & 63); mn = float(scales[sb + 4] & 63); }\n";
    s += "                else { sc = float((scales[sb + 4] & 0xF) | ((scales[sb - 4] >> 6) << 4)); mn = float((scales[sb + 4] >> 4) | ((scales[sb] >> 6) << 4)); }\n";
    s += "                ds = d_val * sc; dm = dmin_val * mn;\n";
    s += "            }\n";
    s += "\n";
    // Stage 8 elements per lane: r = lane & 7 (constant), k varies.
    // Per-element work: 1 qb load + nibble extract + hb extract + multiply = ~5 ALU.
    s += "            for (uint t = lane; t < 256; t += 32) {\n";
    s += "                int k = int(t >> 3);\n";
    s += "                int r = int(t & 7);\n";
    s += "                half w = 0.0h;\n";
    s += "                if (valid) {\n";
    s += "                    uchar qb = qs[pair * 32 + k];\n";
    s += "                    uint nibble = (uint(qb) >> (sub_idx * 4)) & 0xF;\n";
    s += "                    uint hb = (uint(qh[k]) >> sb) & 1;\n";
    s += "                    w = half(ds * float(nibble + hb * 16) - dm);\n";
    s += "                }\n";
    s += "                tile[k * 9 + r] = w;\n";
    s += "            }\n";
    s += "            simdgroup_barrier(mem_flags::mem_threadgroup);\n";
    s += "\n";
    s += "            for (uint ks = 0; ks < 4; ks++) {\n";
    s += "                simdgroup_half8x8 B;\n";
    s += "                simdgroup_load(B, tile + ks * 8 * 9, 9);\n";
    for g in 0..n_groups {
        let off = g * 8;
        s += &format!("                {{ simdgroup_half8x8 A; simdgroup_load(A, X + (b_off + {off}u) * cols + bi * 256 + sb * 32 + ks * 8, cols); simdgroup_multiply_accumulate(C{g}, A, B, C{g}); }}\n");
    }
    s += "            }\n";
    s += "            simdgroup_barrier(mem_flags::mem_threadgroup);\n";
    s += "        }\n";
    s += "    }\n";
    s += "\n";
    for g in 0..n_groups {
        let off = g * 8;
        s += &format!("    simdgroup_store(C{g}, Y + (b_off + {off}u) * rows + row_base + sgid * 8, ulong(rows));\n");
    }
    s += "}\n\n";
    s
}

/// Grid-tiled Q6K MMA: optimized sub-blocked staging + simdgroup MMA.
/// Hoists shared Q6K dequant values per sub-block: d, switch(sub) scale lookup.
/// Q6K layout: ql[128] + qh[64] + sc[16] + d(2) = 210 bytes per 256-element block.
/// Grid: (ceil(rows/32), ceil(b/tile_b)), TG: 128 = 4 SG.
pub fn gen_matmul_q6k_mma_grid(name: &str, tile_b: u32) -> String {
    assert!(tile_b % 8 == 0, "tile_b must be multiple of 8");
    let n_groups = tile_b / 8;
    let mut s = String::with_capacity(12000);
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
    s += "    uint sgid = lid / 32;\n";
    s += "    uint lane = lid % 32;\n";
    s += "    uint row_base = tgid.x * 32;\n";
    s += "    if (row_base >= rows) return;\n";
    s += &format!("    uint b_off = tgid.y * {tile_b}u;\n");
    s += "\n";
    s += "    threadgroup half tiles[4 * 32 * 9];\n";
    s += "    threadgroup half *tile = tiles + sgid * 32 * 9;\n";
    s += "\n";
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
    s += "    int blocks_per_row = int(cols / 256);\n";
    s += "    int row_bytes = blocks_per_row * 210;\n";
    s += "    int my_r = int(lane & 7);\n";
    s += "    int global_row = int(row_base) + int(sgid) * 8 + my_r;\n";
    s += "    bool valid = global_row < int(rows);\n";
    s += "    device const uchar *row_w = W + global_row * row_bytes;\n";
    s += "\n";
    s += "    for (int bi = 0; bi < blocks_per_row; bi++) {\n";
    s += "        device const uchar *blk = row_w + bi * 210;\n";
    // Q6K layout: ql[128] at +0, qh[64] at +128, sc[16] at +192, d(2) at +208
    s += "        device const uchar *ql = blk;\n";
    s += "        device const uchar *qh_ptr = blk + 128;\n";
    s += "        device const char *sc = (device const char *)(blk + 192);\n";
    s += "        float d_val = valid ? float(*((device const half *)(blk + 208))) : 0.0;\n";
    s += "\n";
    // Q6K sub-blocks: 8 sub-blocks of 32 elements each.
    // elem = sb * 32 + k. n = elem / 128, rem = elem % 128, l = rem % 32, sub = rem / 32.
    // For sb in 0..7, k in 0..31:
    //   n = (sb*32+k)/128 = sb/4 (for k<32)
    //   rem = (sb*32+k)%128 = (sb%4)*32+k
    //   l = rem%32 = k
    //   sub = rem/32 = sb%4
    // So per sb: n = sb/4, sub = sb%4, l = k.
    // sb_base = n * 8, is_off = l / 16 = k / 16 (0 or 1)
    // Scale indices: sc[sb_base + is_off + sub*2] — depends on sub and k/16
    s += "        for (int sb = 0; sb < 8; sb++) {\n";
    s += "            int n = sb >> 2;\n";
    s += "            int sub = sb & 3;\n";
    s += "            int sb_base = n * 8;\n";
    // ql/qh pointers offset by n (constant per sub-block pair)
    s += "            device const uchar *qlp = ql + n * 64;\n";
    s += "            device const uchar *qhp = qh_ptr + n * 32;\n";
    s += "\n";
    s += "            for (uint t = lane; t < 256; t += 32) {\n";
    s += "                int k = int(t >> 3);\n";
    s += "                int r = int(t & 7);\n";
    s += "                half w = 0.0h;\n";
    s += "                if (valid) {\n";
    s += "                    int is_off = k >> 4;\n";  // 0 for k<16, 1 for k>=16
    // Dequant: switch on sub to extract q and ds
    s += "                    int q; float ds;\n";
    s += "                    switch (sub) {\n";
    s += "                        case 0: q = (int(qlp[k] & 0xF) | (int((qhp[k] >> 0) & 3) << 4)) - 32; ds = d_val * float(sc[sb_base+is_off+0]); break;\n";
    s += "                        case 1: q = (int(qlp[k+32] & 0xF) | (int((qhp[k] >> 2) & 3) << 4)) - 32; ds = d_val * float(sc[sb_base+is_off+2]); break;\n";
    s += "                        case 2: q = (int(qlp[k] >> 4) | (int((qhp[k] >> 4) & 3) << 4)) - 32; ds = d_val * float(sc[sb_base+is_off+4]); break;\n";
    s += "                        default: q = (int(qlp[k+32] >> 4) | (int((qhp[k] >> 6) & 3) << 4)) - 32; ds = d_val * float(sc[sb_base+is_off+6]); break;\n";
    s += "                    }\n";
    s += "                    w = half(ds * float(q));\n";
    s += "                }\n";
    s += "                tile[k * 9 + r] = w;\n";
    s += "            }\n";
    s += "            simdgroup_barrier(mem_flags::mem_threadgroup);\n";
    s += "\n";
    s += "            for (uint ks = 0; ks < 4; ks++) {\n";
    s += "                simdgroup_half8x8 B;\n";
    s += "                simdgroup_load(B, tile + ks * 8 * 9, 9);\n";
    for g in 0..n_groups {
        let off = g * 8;
        s += &format!("                {{ simdgroup_half8x8 A; simdgroup_load(A, X + (b_off + {off}u) * cols + bi * 256 + sb * 32 + ks * 8, cols); simdgroup_multiply_accumulate(C{g}, A, B, C{g}); }}\n");
    }
    s += "            }\n";
    s += "            simdgroup_barrier(mem_flags::mem_threadgroup);\n";
    s += "        }\n";
    s += "    }\n";
    s += "\n";
    for g in 0..n_groups {
        let off = g * 8;
        s += &format!("    simdgroup_store(C{g}, Y + (b_off + {off}u) * rows + row_base + sgid * 8, ulong(rows));\n");
    }
    s += "}\n\n";
    s
}

/// Grid-tiled Q8_0 MMA: staging + simdgroup MMA.
/// Q8_0 block = 32 elements, 34 bytes: half d + 32 × int8 quants.
pub fn gen_matmul_q8_0_mma_grid(name: &str, tile_b: u32) -> String {
    assert!(tile_b % 8 == 0, "tile_b must be multiple of 8");
    let n_groups = tile_b / 8;
    let mut s = String::with_capacity(8192);
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
    s += "    uint sgid = lid / 32;\n";
    s += "    uint lane = lid % 32;\n";
    s += "    uint row_base = tgid.x * 32;\n";
    s += "    if (row_base >= rows) return;\n";
    s += &format!("    uint b_off = tgid.y * {tile_b}u;\n");
    s += "\n";
    s += "    threadgroup half tiles[4 * 32 * 9];\n";
    s += "    threadgroup half *tile = tiles + sgid * 32 * 9;\n";
    s += "\n";
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
    s += "    uint row_bytes = blocks_per_row * 34;\n";
    s += "\n";
    s += "    for (uint bi = 0; bi < blocks_per_row; bi++) {\n";
    s += "        for (uint t = lane; t < 256; t += 32) {\n";
    s += "            uint k = t >> 3;\n";
    s += "            uint r = t & 7;\n";
    s += "            uint global_row = row_base + sgid * 8 + r;\n";
    s += "            half w = 0.0h;\n";
    s += "            if (global_row < rows) {\n";
    s += "                device const uchar *bp = W + global_row * row_bytes + bi * 34;\n";
    s += "                half dd = *((device const half *)bp);\n";
    s += "                w = dd * half(int(((device const char *)(bp + 2))[k]));\n";
    s += "            }\n";
    s += "            tile[k * 9 + r] = w;\n";
    s += "        }\n";
    s += "        simdgroup_barrier(mem_flags::mem_threadgroup);\n";
    s += "\n";
    s += "        for (uint ks = 0; ks < 4; ks++) {\n";
    s += "            simdgroup_half8x8 B;\n";
    s += "            simdgroup_load(B, tile + ks * 8 * 9, 9);\n";
    for g in 0..n_groups {
        let off = g * 8;
        s += &format!("            {{ simdgroup_half8x8 A; simdgroup_load(A, X + (b_off + {off}u) * cols + bi * 32 + ks * 8, cols); simdgroup_multiply_accumulate(C{g}, A, B, C{g}); }}\n");
    }
    s += "        }\n";
    s += "        simdgroup_barrier(mem_flags::mem_threadgroup);\n";
    s += "    }\n";
    s += "\n";
    for g in 0..n_groups {
        let off = g * 8;
        s += &format!("    simdgroup_store(C{g}, Y + (b_off + {off}u) * rows + row_base + sgid * 8, ulong(rows));\n");
    }
    s += "}\n\n";
    s
}

// ── Scalar matvec (n=1 decode) ──

/// Q5K scalar matvec for n=1 decode. 8 SGs per TG, 1 row per SG.
pub fn gen_matvec_q5k_f16(name: &str) -> String {
    let mut s = String::with_capacity(4096);
    s += &format!("kernel void {name}(\n");
    s += "    device const uchar *W    [[buffer(0)]],\n";
    s += "    device const half  *x    [[buffer(1)]],\n";
    s += "    device half        *y    [[buffer(2)]],\n";
    s += "    constant uint      &cols [[buffer(3)]],\n";
    s += "    constant uint      &rows [[buffer(4)]],\n";
    s += "    constant uint      &batch [[buffer(5)]],\n";
    s += "    uint tgid [[threadgroup_position_in_grid]],\n";
    s += "    uint simd_gid [[simdgroup_index_in_threadgroup]],\n";
    s += "    uint simd_lid [[thread_index_in_simdgroup]],\n";
    s += "    uint num_sg [[simdgroups_per_threadgroup]]\n";
    s += ") {\n";
    s += "    uint row_idx = tgid * num_sg + simd_gid;\n";
    s += "    if (row_idx >= rows) return;\n";
    s += "    uint blocks_per_row = cols / 256;\n";
    s += "    device const uchar *row = W + row_idx * blocks_per_row * 176;\n";
    s += "    float sum = 0.0;\n";
    s += "    for (uint b = simd_lid; b < blocks_per_row; b += 32) {\n";
    s += "        device const uchar *blk = row + b * 176;\n";
    s += "        float d    = float(*((device const half *)(blk)));\n";
    s += "        float dmin = float(*((device const half *)(blk + 2)));\n";
    s += "        device const uchar *scales = blk + 4;\n";
    s += "        device const uchar *qh = blk + 16;\n";
    s += "        device const uchar *qs = blk + 48;\n";
    s += "        uint base = b * 256;\n";
    s += "        for (int pair = 0; pair < 4; pair++) {\n";
    s += "            int is0 = pair * 2, is1 = pair * 2 + 1;\n";
    s += "            float sc0, mn0, sc1, mn1;\n";
    s += "            if (is0 < 4) { sc0 = float(scales[is0] & 63); mn0 = float(scales[is0 + 4] & 63); }\n";
    s += "            else { sc0 = float((scales[is0 + 4] & 0xF) | ((scales[is0 - 4] >> 6) << 4)); mn0 = float((scales[is0 + 4] >> 4) | ((scales[is0] >> 6) << 4)); }\n";
    s += "            if (is1 < 4) { sc1 = float(scales[is1] & 63); mn1 = float(scales[is1 + 4] & 63); }\n";
    s += "            else { sc1 = float((scales[is1 + 4] & 0xF) | ((scales[is1 - 4] >> 6) << 4)); mn1 = float((scales[is1 + 4] >> 4) | ((scales[is1] >> 6) << 4)); }\n";
    s += "            float ds0 = d * sc0, dm0 = dmin * mn0;\n";
    s += "            float ds1 = d * sc1, dm1 = dmin * mn1;\n";
    s += "            device const uchar *qp = qs + pair * 32;\n";
    s += "            uint idx = base + pair * 64;\n";
    s += "            uint shift0 = pair * 2, shift1 = pair * 2 + 1;\n";
    s += "            for (uint l = 0; l < 32; l++) {\n";
    s += "                uchar qb = qp[l];\n";
    s += "                uint hb0 = (qh[l] >> shift0) & 1;\n";
    s += "                uint hb1 = (qh[l] >> shift1) & 1;\n";
    s += "                sum += (ds0 * float((qb & 0xF) + hb0 * 16) - dm0) * float(x[idx + l]);\n";
    s += "                sum += (ds1 * float((qb >> 4)  + hb1 * 16) - dm1) * float(x[idx + 32 + l]);\n";
    s += "            }\n";
    s += "        }\n";
    s += "    }\n";
    s += "    sum = simd_sum(sum);\n";
    s += "    if (simd_lid == 0) y[row_idx] = half(sum);\n";
    s += "}\n\n";
    s
}

/// Q6K scalar matvec for n=1 decode. 8 SGs per TG, 1 row per SG.
fn gen_matvec_q6k_f16(name: &str) -> String {
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
    s += "    uint blocks_per_row = cols / 256;\n";
    s += "    device const uchar *wp = W + my_row * blocks_per_row * 210;\n\n";
    s += "    half s0 = 0.0h;\n";
    s += "\n    for (uint bi = 0; bi < blocks_per_row; bi++) {\n";
    s += "        device const uchar *blk = wp + bi * 210;\n";
    s += "        for (uint e = 0; e < 8; e++) {\n";
    s += "            uint elem = e * 32 + lane;\n";
    s += "            half w = half(q6k_dequant(blk, elem));\n";
    s += "            uint c = bi * 256 + elem;\n";
    s += "            s0 += w * X[0u * cols + c];\n";
    s += "        }\n";
    s += "    }\n\n";
    s += "    s0 = simd_sum(s0);\n";
    s += "\n    if (lane == 0) {\n";
    s += "        if (0u < batch) Y[0u * rows + my_row] = s0;\n";
    s += "    }\n";
    s += "}\n\n";
    s
}

/// Q8_0 scalar matvec for n=1 decode. 8 SGs per TG, 1 row per SG.
pub fn gen_matvec_q8_0_f16(name: &str) -> String {
    let mut s = String::with_capacity(2048);
    s += &format!("kernel void {name}(\n");
    s += "    device const uchar *W    [[buffer(0)]],\n";
    s += "    device const half  *x    [[buffer(1)]],\n";
    s += "    device half        *y    [[buffer(2)]],\n";
    s += "    constant uint      &cols [[buffer(3)]],\n";
    s += "    constant uint      &rows [[buffer(4)]],\n";
    s += "    constant uint      &batch [[buffer(5)]],\n";
    s += "    uint tgid [[threadgroup_position_in_grid]],\n";
    s += "    uint simd_gid [[simdgroup_index_in_threadgroup]],\n";
    s += "    uint simd_lid [[thread_index_in_simdgroup]],\n";
    s += "    uint num_sg [[simdgroups_per_threadgroup]]\n";
    s += ") {\n";
    s += "    uint row_idx = tgid * num_sg + simd_gid;\n";
    s += "    if (row_idx >= rows) return;\n";
    s += "    uint blocks_per_row = cols / 32;\n";
    s += "    device const uchar *row = W + row_idx * blocks_per_row * 34;\n";
    s += "    float sum = 0.0;\n";
    s += "    for (uint b = simd_lid; b < blocks_per_row; b += 32) {\n";
    s += "        device const uchar *blk = row + b * 34;\n";
    s += "        float d = float(*((device const half *)(blk)));\n";
    s += "        device const char *qs = (device const char *)(blk + 2);\n";
    s += "        uint base = b * 32;\n";
    s += "        for (uint i = 0; i < 32; i++) {\n";
    s += "            sum += d * float(qs[i]) * float(x[base + i]);\n";
    s += "        }\n";
    s += "    }\n";
    s += "    sum = simd_sum(sum);\n";
    s += "    if (simd_lid == 0) y[row_idx] = half(sum);\n";
    s += "}\n\n";
    s
}

// ── Embedding kernels ──

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
    out[bi * dim + elem] = half(q6k_dequant(blk, vi));
}
"#;

pub const EMBED_BATCH_Q5K: &str = r#"
inline void get_scale_min_k5_be(int j, device const uchar *q, thread float &sc, thread float &mn) {
    if (j < 4) { sc = float(q[j] & 63); mn = float(q[j + 4] & 63); }
    else { sc = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); mn = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}
kernel void embed_batch_q5k(
    device const uchar *W       [[buffer(0)]],
    device half        *out     [[buffer(1)]],
    device const uint  *tokens  [[buffer(2)]],
    constant uint      &dim     [[buffer(3)]],
    constant uint      &batch   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint elem = gid.x;
    uint bi = gid.y;
    if (elem >= dim || bi >= batch) return;
    uint bpr = dim / 256;
    device const uchar *blk = W + tokens[bi] * bpr * 176 + (elem / 256) * 176;
    float d    = float(*((device const half *)(blk)));
    float dmin = float(*((device const half *)(blk + 2)));
    device const uchar *scales = blk + 4;
    device const uchar *qh = blk + 16;
    device const uchar *qs = blk + 48;
    uint vi = elem % 256;
    uint pair = vi / 64;
    uint rem = vi % 64;
    uint l = rem % 32;
    uint half_idx = rem / 32;
    int is_idx = pair * 2 + half_idx;
    float sc, mn;
    get_scale_min_k5_be(is_idx, scales, sc, mn);
    float ds = d * sc, dm = dmin * mn;
    uchar qb = qs[pair * 32 + l];
    uint shift = pair * 2 + half_idx;
    uint hb = (qh[l] >> shift) & 1;
    uint q;
    if (half_idx == 0) { q = (qb & 0xF) + hb * 16; }
    else               { q = (qb >> 4)  + hb * 16; }
    out[bi * dim + elem] = half(ds * float(q) - dm);
}
"#;

pub const EMBED_BATCH_Q8_0: &str = r#"
kernel void embed_batch_q8_0(
    device const uchar *W       [[buffer(0)]],
    device half        *out     [[buffer(1)]],
    device const uint  *tokens  [[buffer(2)]],
    constant uint      &dim     [[buffer(3)]],
    constant uint      &batch   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint elem = gid.x;
    uint bi = gid.y;
    if (elem >= dim || bi >= batch) return;
    uint bpr = dim / 32;
    device const uchar *row = W + tokens[bi] * bpr * 34;
    uint block_idx = elem / 32, vi = elem % 32;
    device const uchar *blk = row + block_idx * 34;
    float d = float(*((device const half *)(blk)));
    device const char *qs = (device const char *)(blk + 2);
    out[bi * dim + elem] = half(d * float(qs[vi]));
}
"#;

// ── Element-wise batch kernels ──

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
    for (uint i = lid; i < dim; i += tgs)
        ni[i] = half(float(ri[i]) * rms * float(weight[i]));
}
"#;

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

const RESIDUAL_ADD_BATCH: &str = r#"
kernel void residual_add_batch(
    device half       *res [[buffer(0)]],
    device const half *add [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    res[gid] = res[gid] + add[gid];
}
"#;

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
    threadgroup float scores[2048];
    for (uint t = lid; t < seq_len; t += tgs) {
        device const half *kt = K_cache + t * kv_dim + kv_head * head_dim;
        float dot = 0.0;
        for (uint d = 0; d < head_dim; d++) dot += float(q[d]) * float(kt[d]);
        scores[t] = dot * inv_sqrt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float sm[32];
    float mx = -INFINITY;
    for (uint t = lid; t < seq_len; t += tgs) mx = max(mx, scores[t]);
    mx = simd_max(mx);
    if (lid % 32 == 0) sm[lid/32] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < 32) { float v = (lid < (tgs+31)/32) ? sm[lid] : -INFINITY; v = simd_max(v); if (lid==0) sm[0]=v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maxv = sm[0];
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
    device half *o = out + bi * total_q_dim + q_head * head_dim;
    for (uint d = lid; d < head_dim; d += tgs) {
        float val = 0.0;
        for (uint t = 0; t < seq_len; t++)
            val += scores[t] * float(V_cache[t * kv_dim + kv_head * head_dim + d]);
        o[d] = half(val);
    }
}
"#;

const ARGMAX_BATCH: &str = r#"
kernel void argmax_batch(
    device const half *logits [[buffer(0)]],
    device uint       *result [[buffer(1)]],
    constant uint     &vocab  [[buffer(2)]],
    constant uint     &batch  [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    if (tgid >= batch) return;
    device const half *l = logits + tgid * vocab;
    float best_val = -INFINITY;
    uint best_idx = 0;
    for (uint i = lid; i < vocab; i += tgs) {
        float v = float(l[i]);
        if (v > best_val) { best_val = v; best_idx = i; }
    }
    threadgroup float vals[1024];
    threadgroup uint idxs[1024];
    vals[lid] = best_val;
    idxs[lid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgs / 2; s > 0; s >>= 1) {
        if (lid < s && vals[lid + s] > vals[lid]) {
            vals[lid] = vals[lid + s];
            idxs[lid] = idxs[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) result[tgid] = idxs[0];
}
"#;

// ── Kernel assembly ──

/// Build all batch Metal source: Q5K/Q6K/Q8_0 MMA + matvec + embed + element-wise.
pub fn all_batch_kernels() -> String {
    let mut s = HEADER_STR.to_string();
    // MMA staging grid kernels (batch matmul, tile_b=80)
    s += &gen_matmul_q5k_mma_grid("matmul_q5k_grid", 80);
    s += &gen_matmul_q6k_mma_grid("matmul_q6k_grid", 80);
    s += &gen_matmul_q8_0_mma_grid("matmul_q8_0_grid", 80);
    // Scalar matvec (n=1 decode)
    s += &gen_matvec_q5k_f16("matvec_q5k_f16");
    s += &gen_matvec_q6k_f16("matvec_q6k_f16");
    s += &gen_matvec_q8_0_f16("matvec_q8_0_f16");
    // Embedding
    s += EMBED_BATCH_Q5K;
    s += EMBED_BATCH_Q6K;
    s += EMBED_BATCH_Q8_0;
    // Element-wise
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

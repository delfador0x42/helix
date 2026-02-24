//! FP16 weight pre-dequantization and pure FP16 MMA matmul kernels.
//!
//! Pre-dequant: GPU kernels expand Q4_0/Q4_1/Q6K → FP16 row-major.
//! Run once at model load time. ~2.4GB for Llama 3.2 1B.
//!
//! Matmul: Pure FP16 MMA — simdgroup_load from device memory with transpose flag.
//! No dequant, no staging, no barriers. 256-thread TG = 8 SG, 64 rows per TG.

/// MSL source for dequant kernels (compiled separately at model load).
pub fn dequant_source() -> String {
    let mut s = String::with_capacity(4096);
    s += "#include <metal_stdlib>\nusing namespace metal;\n\n";
    s += Q6K_DEQUANT_FN;
    s += DEQUANT_Q4_0;
    s += DEQUANT_Q4_1;
    s += DEQUANT_Q6K;
    s
}

const Q6K_DEQUANT_FN: &str = r#"
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

const DEQUANT_Q4_0: &str = r#"
kernel void dequant_q4_0(
    device const uchar *W  [[buffer(0)]],
    device half        *Y  [[buffer(1)]],
    constant uint      &cols [[buffer(2)]],
    constant uint      &rows [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x, row = gid.y;
    if (col >= cols || row >= rows) return;
    uint bpr = cols / 32;
    device const uchar *bp = W + row * bpr * 18 + (col / 32) * 18;
    half d = *((device const half *)bp);
    uint vi = col % 32;
    uchar qb = bp[2 + (vi & 15)];
    uint shift = (vi >> 4) * 4;
    Y[row * cols + col] = d * (half(int(qb >> shift) & 0xF) - 8.0h);
}
"#;

const DEQUANT_Q4_1: &str = r#"
kernel void dequant_q4_1(
    device const uchar *W  [[buffer(0)]],
    device half        *Y  [[buffer(1)]],
    constant uint      &cols [[buffer(2)]],
    constant uint      &rows [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x, row = gid.y;
    if (col >= cols || row >= rows) return;
    uint bpr = cols / 32;
    device const uchar *bp = W + row * bpr * 20 + (col / 32) * 20;
    half d = *((device const half *)bp);
    half m = *((device const half *)(bp + 2));
    uint vi = col % 32;
    uchar qb = bp[4 + (vi & 15)];
    uint shift = (vi >> 4) * 4;
    Y[row * cols + col] = d * half(int(qb >> shift) & 0xF) + m;
}
"#;

const DEQUANT_Q6K: &str = r#"
kernel void dequant_q6k(
    device const uchar *W  [[buffer(0)]],
    device half        *Y  [[buffer(1)]],
    constant uint      &cols [[buffer(2)]],
    constant uint      &rows [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x, row = gid.y;
    if (col >= cols || row >= rows) return;
    uint bpr = cols / 256;
    device const uchar *blk = W + row * bpr * 210 + (col / 256) * 210;
    Y[row * cols + col] = half(q6k_dequant(blk, col % 256));
}
"#;

/// Pure FP16 MMA matmul: Y = W @ X, all FP16, direct device memory loads.
/// Uses simdgroup_load transpose flag for weight matrix.
/// N batch groups of 8 = BATCH_SIZE items.
/// Parameterized by n_sg: 4 SG = 128 threads (32 rows), 8 SG = 256 threads (64 rows).
fn gen_matmul_fp16_mma_impl(name: &str, n_sg: u32) -> String {
    let n_groups = crate::kernels_batch::BATCH_SIZE / 8;
    let rows_per_tg = n_sg * 8;
    let mut s = String::with_capacity(4096);
    s += &format!("kernel void {name}(\n");
    s += "    device const half *W     [[buffer(0)]],\n";
    s += "    device const half *X     [[buffer(1)]],\n";
    s += "    device half       *Y     [[buffer(2)]],\n";
    s += "    constant uint     &cols  [[buffer(3)]],\n";
    s += "    constant uint     &rows  [[buffer(4)]],\n";
    s += "    constant uint     &batch [[buffer(5)]],\n";
    s += "    uint3 tgid [[threadgroup_position_in_grid]],\n";
    s += "    uint3 tid  [[thread_position_in_threadgroup]]\n";
    s += ") {\n";
    s += "    uint lid = tid.x;\n";
    s += "    uint sgid = lid / 32;\n";
    s += &format!("    uint row_base = tgid.x * {rows_per_tg} + sgid * 8;\n");
    s += "    if (row_base >= rows) return;\n";
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
    s += "    for (uint k = 0; k < cols; k += 8) {\n";
    s += "        simdgroup_half8x8 B;\n";
    s += "        simdgroup_load(B, W + row_base * cols + k, cols, ulong2(0,0), true);\n";
    s += "\n";
    for g in 0..n_groups {
        let off = g * 8;
        s += &format!("        {{ simdgroup_half8x8 A; simdgroup_load(A, X + {off}u*cols + k, cols); simdgroup_multiply_accumulate(C{g}, A, B, C{g}); }}\n");
    }
    s += "    }\n";
    s += "\n";
    for g in 0..n_groups {
        let off = g * 8;
        s += &format!("    simdgroup_store(C{g}, Y + {off}u*rows + row_base, ulong(rows));\n");
    }
    s += "}\n\n";
    s
}

/// 256-thread FP16 MMA: 8 SG, 64 rows per TG. Original variant.
pub fn gen_matmul_fp16_mma() -> String {
    gen_matmul_fp16_mma_impl("matmul_fp16_mma", 8)
}

/// 128-thread FP16 MMA: 4 SG, 32 rows per TG. Better occupancy.
pub fn gen_matmul_fp16_mma_128() -> String {
    gen_matmul_fp16_mma_impl("matmul_fp16_mma_128", 4)
}

/// Grid-tiled FP16 MMA: batch via grid_y, constant register pressure.
/// Grid: (ceil(rows/32), ceil(B/tile_b)), TG: 128 = 4 SG, 32 rows per TG.
/// Each TG handles tile_b batch items for its 32 rows.
pub fn gen_matmul_fp16_mma_grid(name: &str, tile_b: u32) -> String {
    assert!(tile_b % 8 == 0, "tile_b must be multiple of 8");
    let n_groups = tile_b / 8;
    let mut s = String::with_capacity(4096);
    s += &format!("kernel void {name}(\n");
    s += "    device const half *W     [[buffer(0)]],\n";
    s += "    device const half *X     [[buffer(1)]],\n";
    s += "    device half       *Y     [[buffer(2)]],\n";
    s += "    constant uint     &cols  [[buffer(3)]],\n";
    s += "    constant uint     &rows  [[buffer(4)]],\n";
    s += "    constant uint     &batch [[buffer(5)]],\n";
    s += "    uint3 tgid [[threadgroup_position_in_grid]],\n";
    s += "    uint3 tid  [[thread_position_in_threadgroup]]\n";
    s += ") {\n";
    s += "    uint lid = tid.x;\n";
    s += "    uint sgid = lid / 32;\n";
    s += "    uint row_base = tgid.x * 32 + sgid * 8;\n";
    s += "    if (row_base >= rows) return;\n";
    s += &format!("    uint b_off = tgid.y * {tile_b}u;\n");
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
    s += "    for (uint k = 0; k < cols; k += 8) {\n";
    s += "        simdgroup_half8x8 B;\n";
    s += "        simdgroup_load(B, W + row_base * cols + k, cols, ulong2(0,0), true);\n";
    s += "\n";
    for g in 0..n_groups {
        let off = g * 8;
        s += &format!("        {{ simdgroup_half8x8 A; simdgroup_load(A, X + (b_off + {off}u)*cols + k, cols); simdgroup_multiply_accumulate(C{g}, A, B, C{g}); }}\n");
    }
    s += "    }\n";
    s += "\n";
    for g in 0..n_groups {
        let off = g * 8;
        s += &format!("    simdgroup_store(C{g}, Y + (b_off + {off}u)*rows + row_base, ulong(rows));\n");
    }
    s += "}\n\n";
    s
}

/// FP16 batch embedding: simple copy, no dequant.
pub const EMBED_FP16: &str = r#"
kernel void embed_fp16(
    device const half *W       [[buffer(0)]],
    device half       *out     [[buffer(1)]],
    device const uint *tokens  [[buffer(2)]],
    constant uint     &dim     [[buffer(3)]],
    constant uint     &batch   [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint elem = gid.x;
    uint bi = gid.y;
    if (elem >= dim || bi >= batch) return;
    out[bi * dim + elem] = W[tokens[bi] * dim + elem];
}
"#;

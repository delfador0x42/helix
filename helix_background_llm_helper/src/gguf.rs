//! Zero-dependency GGUF file parser. mmap-based, zero-copy where possible.
//! Follows amaranthine pattern: repr(C, packed) structs, unchecked reads after
//! upfront validation, no intermediate allocations.

use std::collections::HashMap;
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as LE u32
const DEFAULT_ALIGNMENT: usize = 32;

// ── ggml_type enum ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
}

impl GGMLType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),   1 => Some(Self::F16),
            2 => Some(Self::Q4_0),  3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),  7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),  9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),  11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),  13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),  15 => Some(Self::Q8K),
            16 => Some(Self::IQ2XXS), 17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS), 19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),  21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),   23 => Some(Self::IQ4XS),
            24 => Some(Self::I8),   25 => Some(Self::I16),
            26 => Some(Self::I32),  27 => Some(Self::I64),
            28 => Some(Self::F64),  29 => Some(Self::IQ1M),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Block size for this quantization type.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 |
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 |
            Self::Q8_0 | Self::Q8_1 | Self::IQ4NL => 32,
            _ => 256, // K-quants and IQ types
        }
    }

    /// Bytes per block for this quantization type.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,    Self::F16 => 2,   Self::BF16 => 2,  Self::F64 => 8,
            Self::I8 => 1,     Self::I16 => 2,   Self::I32 => 4,   Self::I64 => 8,
            Self::Q4_0 => 18,  Self::Q4_1 => 20, Self::Q5_0 => 22, Self::Q5_1 => 24,
            Self::Q8_0 => 34,  Self::Q8_1 => 36,
            Self::Q2K => 84,   Self::Q3K => 110,  Self::Q4K => 144,
            Self::Q5K => 176,  Self::Q6K => 210,  Self::Q8K => 292,
            Self::IQ2XXS => 80, Self::IQ2XS => 136, Self::IQ3XXS => 104,
            Self::IQ1S => 144,  Self::IQ4NL => 18,  Self::IQ3S => 152,
            Self::IQ2S => 180,  Self::IQ4XS => 155, Self::IQ1M => 144,
        }
    }

    /// Calculate total bytes for a tensor with this type and element count.
    pub fn tensor_bytes(self, n_elements: u64) -> u64 {
        let bs = self.block_size() as u64;
        let bb = self.block_bytes() as u64;
        (n_elements / bs) * bb
    }
}

// ── Metadata value types ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8), I8(i8), U16(u16), I16(i16),
    U32(u32), I32(i32), F32(f32), Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
    U64(u64), I64(i64), F64(f64),
}

impl MetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self { Self::U32(v) => Some(*v), _ => None }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self { Self::U64(v) => Some(*v), Self::U32(v) => Some(*v as u64), _ => None }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self { Self::F32(v) => Some(*v), _ => None }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self { Self::Str(s) => Some(s), _ => None }
    }
    pub fn as_array(&self) -> Option<&[MetaValue]> {
        match self { Self::Array(a) => Some(a), _ => None }
    }
}

// ── Tensor info ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,       // dimensions (innermost first in GGML convention)
    pub dtype: GGMLType,
    pub offset: u64,          // relative to tensor data section start
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product::<u64>().max(1)
    }

    pub fn byte_size(&self) -> u64 {
        self.dtype.tensor_bytes(self.n_elements())
    }
}

// ── Model config (extracted from metadata) ──────────────────────────

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: String,
    pub name: String,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub hidden_dim: u32,
    pub ffn_dim: u32,
    pub vocab_size: u32,
    pub context_len: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f32,
    pub bos_token: u32,
    pub eos_token: u32,
}

// ── GGUF file (parsed, mmap'd) ─────────────────────────────────────

pub struct GGUFFile {
    _mmap: Mmap,                          // owns the mapping
    pub data: &'static [u8],             // view into mmap (lifetime tied to struct)
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_start: usize,        // absolute file offset where tensor data begins
    pub alignment: usize,
    pub config: ModelConfig,
}

// ── mmap wrapper ────────────────────────────────────────────────────

struct Mmap {
    ptr: *mut u8,
    len: usize,
}

impl Mmap {
    fn open(path: &Path) -> Result<Self, String> {
        use std::os::unix::io::AsRawFd;
        let f = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
        let len = f.metadata().map_err(|e| format!("metadata: {e}"))?.len() as usize;
        if len < 24 { return Err("file too small for GGUF header".into()); }

        extern "C" {
            fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
        }
        let ptr = unsafe {
            mmap(std::ptr::null_mut(), len, 1 /* PROT_READ */, 2 /* MAP_PRIVATE */, f.as_raw_fd(), 0)
        };
        drop(f);
        if ptr as usize == usize::MAX { // MAP_FAILED = (void*)-1
            return Err("mmap failed".into());
        }
        Ok(Mmap { ptr, len })
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        extern "C" { fn munmap(addr: *mut u8, len: usize) -> i32; }
        unsafe { munmap(self.ptr, self.len); }
    }
}

// ── Cursor for sequential binary reads ──────────────────────────────

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self { Cursor { data, pos: 0 } }

    fn remaining(&self) -> usize { self.data.len().saturating_sub(self.pos) }

    fn read_u8(&mut self) -> Result<u8, String> {
        if self.remaining() < 1 { return Err("truncated u8".into()); }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        if self.remaining() < 2 { return Err("truncated u16".into()); }
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        if self.remaining() < 4 { return Err("truncated u32".into()); }
        let b = &self.data[self.pos..self.pos + 4];
        let v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        if self.remaining() < 8 { return Err("truncated u64".into()); }
        let b = &self.data[self.pos..self.pos + 8];
        let v = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        self.pos += 8;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8, String> { Ok(self.read_u8()? as i8) }
    fn read_i16(&mut self) -> Result<i16, String> { Ok(self.read_u16()? as i16) }
    fn read_i32(&mut self) -> Result<i32, String> { Ok(self.read_u32()? as i32) }
    fn read_i64(&mut self) -> Result<i64, String> { Ok(self.read_u64()? as i64) }

    fn read_f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u64()? as usize;
        if self.remaining() < len { return Err(format!("truncated string len={len}")); }
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|e| format!("invalid utf8: {e}"))?
            .to_string();
        self.pos += len;
        Ok(s)
    }

    fn read_meta_value(&mut self) -> Result<MetaValue, String> {
        let type_id = self.read_u32()?;
        self.read_typed_value(type_id)
    }

    fn read_typed_value(&mut self, type_id: u32) -> Result<MetaValue, String> {
        match type_id {
            0  => Ok(MetaValue::U8(self.read_u8()?)),
            1  => Ok(MetaValue::I8(self.read_i8()?)),
            2  => Ok(MetaValue::U16(self.read_u16()?)),
            3  => Ok(MetaValue::I16(self.read_i16()?)),
            4  => Ok(MetaValue::U32(self.read_u32()?)),
            5  => Ok(MetaValue::I32(self.read_i32()?)),
            6  => Ok(MetaValue::F32(self.read_f32()?)),
            7  => { let v = self.read_u8()?; Ok(MetaValue::Bool(v != 0)) }
            8  => Ok(MetaValue::Str(self.read_string()?)),
            9  => {
                let elem_type = self.read_u32()?;
                let count = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(count.min(1_000_000));
                for _ in 0..count {
                    arr.push(self.read_typed_value(elem_type)?);
                }
                Ok(MetaValue::Array(arr))
            }
            10 => Ok(MetaValue::U64(self.read_u64()?)),
            11 => Ok(MetaValue::I64(self.read_i64()?)),
            12 => Ok(MetaValue::F64(self.read_f64()?)),
            _ => Err(format!("unknown metadata type {type_id}")),
        }
    }
}

// ── Parsing ─────────────────────────────────────────────────────────

impl GGUFFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        let mmap = Mmap::open(path)?;
        let data = mmap.as_slice();

        // Safety: we need the data slice to outlive the struct.
        // Mmap is stored in the struct so it stays alive.
        let data: &'static [u8] = unsafe { std::mem::transmute(data) };

        let mut cur = Cursor::new(data);

        // Header
        let magic = cur.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(format!("bad magic: {magic:#x} (expected {GGUF_MAGIC:#x})"));
        }
        let version = cur.read_u32()?;
        if version < 2 || version > 3 {
            return Err(format!("unsupported GGUF version {version}"));
        }
        let tensor_count = cur.read_u64()? as usize;
        let metadata_kv_count = cur.read_u64()? as usize;

        eprintln!("gguf: version={version}, tensors={tensor_count}, metadata_kv={metadata_kv_count}");

        // Metadata
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = cur.read_string()?;
            let val = cur.read_meta_value()?;
            metadata.insert(key, val);
        }

        // Alignment
        let alignment = metadata.get("general.alignment")
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_ALIGNMENT);

        // Tensor info
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let ndim = cur.read_u32()? as usize;
            let mut dims = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                dims.push(cur.read_u64()?);
            }
            let type_id = cur.read_u32()?;
            let dtype = GGMLType::from_u32(type_id)
                .ok_or_else(|| format!("unknown ggml_type {type_id} for tensor '{name}'"))?;
            let offset = cur.read_u64()?;
            tensors.push(TensorInfo { name, dims, dtype, offset });
        }

        // Tensor data start (aligned after all headers)
        let tensor_data_start = align(cur.pos, alignment);

        // Extract model config
        let config = extract_config(&metadata)?;

        eprintln!("gguf: arch={}, layers={}, hidden={}, heads={}, kv_heads={}, ffn={}, vocab={}",
            config.arch, config.n_layers, config.hidden_dim, config.n_heads,
            config.n_kv_heads, config.ffn_dim, config.vocab_size);
        eprintln!("gguf: tensor_data_start={tensor_data_start:#x}, alignment={alignment}");

        // Summary of tensor types
        let mut type_counts: HashMap<GGMLType, (usize, u64)> = HashMap::new();
        for t in &tensors {
            let e = type_counts.entry(t.dtype).or_insert((0, 0));
            e.0 += 1;
            e.1 += t.byte_size();
        }
        for (dtype, (count, bytes)) in &type_counts {
            eprintln!("gguf:   {:?}: {} tensors, {:.1}MB", dtype, count, *bytes as f64 / 1e6);
        }

        Ok(GGUFFile {
            _mmap: mmap,
            data,
            metadata,
            tensors,
            tensor_data_start,
            alignment,
            config,
        })
    }

    /// Find tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

fn extract_config(meta: &HashMap<String, MetaValue>) -> Result<ModelConfig, String> {
    let arch = meta.get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let name = meta.get("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let get_u32 = |key: &str| -> u32 {
        meta.get(key)
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32
    };
    let get_f32 = |key: &str, default: f32| -> f32 {
        meta.get(key).and_then(|v| v.as_f32()).unwrap_or(default)
    };

    let n_layers = get_u32(&format!("{arch}.block_count"));
    let n_heads = get_u32(&format!("{arch}.attention.head_count"));
    let n_kv_heads = get_u32(&format!("{arch}.attention.head_count_kv"));
    let hidden_dim = get_u32(&format!("{arch}.embedding_length"));
    let ffn_dim = get_u32(&format!("{arch}.feed_forward_length"));
    let vocab_size = get_u32(&format!("{arch}.vocab_size"))
        .max(meta.get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| a.len() as u32)
            .unwrap_or(0));
    let context_len = get_u32(&format!("{arch}.context_length"));
    let head_dim = if n_heads > 0 { hidden_dim / n_heads } else { 0 };
    let rope_dim = get_u32(&format!("{arch}.rope.dimension_count"));
    let rope_freq_base = get_f32(&format!("{arch}.rope.freq_base"), 10000.0);
    eprintln!("  CONFIG: n_layers={n_layers} n_heads={n_heads} n_kv_heads={n_kv_heads} hidden={hidden_dim} ffn={ffn_dim} vocab={vocab_size} ctx={context_len} head_dim={head_dim} rope_dim={rope_dim} rope_freq_base={rope_freq_base} arch={arch}");
    let rms_norm_eps = get_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"), 1e-6);
    let bos_token = get_u32("tokenizer.ggml.bos_token_id");
    let eos_token = get_u32("tokenizer.ggml.eos_token_id");

    Ok(ModelConfig {
        arch, name, n_layers, n_heads, n_kv_heads, hidden_dim, ffn_dim,
        vocab_size, context_len, head_dim, rope_dim, rope_freq_base,
        rms_norm_eps, bos_token, eos_token,
    })
}

fn align(offset: usize, alignment: usize) -> usize {
    offset + (alignment - (offset % alignment)) % alignment
}

// ── f16 conversion ──────────────────────────────────────────────────

/// Convert IEEE 754 binary16 (half-precision) to f32.
/// Zero-dep implementation, handles denorms/inf/nan.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    let result = if exp == 0 {
        if mant == 0 {
            sign << 31 // +-0
        } else {
            // Denormalized: renormalize
            let mut m = mant;
            let mut e = 0u32;
            while (m & 0x400) == 0 { m <<= 1; e += 1; }
            m &= 0x3FF;
            (sign << 31) | ((127 - 15 + 1 - e) << 23) | (m << 13)
        }
    } else if exp == 31 {
        // Inf or NaN
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        // Normal
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };

    f32::from_bits(result)
}

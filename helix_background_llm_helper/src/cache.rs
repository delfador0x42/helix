//! Embedding cache: binary file mapping data.log offsets → embedding vectors.
//! Format: "HXEC" magic + version(u32) + count(u32) + dim(u32) + entries.
//! Each entry: offset(u32) + f32×dim.

use std::path::Path;

use crate::embed::EMBED_DIM;

const MAGIC: &[u8; 4] = b"HXEC";
const ENTRY_SIZE: usize = 4 + EMBED_DIM * 4; // 1540 bytes

pub struct EmbeddingCache {
    pub entries: Vec<(u32, Vec<f32>)>, // (data.log offset, embedding)
}

impl EmbeddingCache {
    pub fn load_or_new(dir: &Path) -> Self {
        let path = dir.join("embeddings.bin");
        if let Ok(data) = std::fs::read(&path) {
            if let Some(cache) = Self::parse(&data) { return cache; }
        }
        EmbeddingCache { entries: Vec::new() }
    }

    pub fn has(&self, offset: u32) -> bool {
        self.entries.iter().any(|(o, _)| *o == offset)
    }

    pub fn add(&mut self, offset: u32, embedding: Vec<f32>) {
        self.entries.push((offset, embedding));
    }

    pub fn get(&self, offset: u32) -> Option<&[f32]> {
        self.entries.iter().find(|(o, _)| *o == offset).map(|(_, e)| e.as_slice())
    }

    pub fn save(&self, dir: &Path) {
        let path = dir.join("embeddings.bin");
        let count = self.entries.len() as u32;
        let mut buf = Vec::with_capacity(16 + self.entries.len() * ENTRY_SIZE);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&(EMBED_DIM as u32).to_le_bytes());
        for (offset, emb) in &self.entries {
            buf.extend_from_slice(&offset.to_le_bytes());
            for &v in emb { buf.extend_from_slice(&v.to_le_bytes()); }
        }
        let tmp = dir.join("embeddings.bin.tmp");
        if std::fs::write(&tmp, &buf).is_ok() {
            std::fs::rename(&tmp, &path).ok();
        }
    }

    fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 16 || &data[..4] != MAGIC { return None; }
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != 1 { return None; }
        let count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let dim = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        if dim != EMBED_DIM || data.len() < 16 + count * ENTRY_SIZE { return None; }
        let mut entries = Vec::with_capacity(count);
        let mut pos = 16;
        for _ in 0..count {
            let off = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]);
            let emb: Vec<f32> = (0..dim).map(|i| {
                let s = pos + 4 + i * 4;
                f32::from_le_bytes([data[s], data[s+1], data[s+2], data[s+3]])
            }).collect();
            entries.push((off, emb));
            pos += ENTRY_SIZE;
        }
        Some(EmbeddingCache { entries })
    }
}

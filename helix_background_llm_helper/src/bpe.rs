//! GPT-2 byte-level BPE tokenizer for Llama 3.3.
//! Loads vocab + merges from GGUF metadata. Encodes text, decodes tokens,
//! formats Llama 3.3 chat template.

use std::collections::HashMap;
use crate::gguf::{GGUFFile, MetaValue};

pub struct BpeTokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merge_rank: HashMap<(String, String), usize>,
    byte_to_unicode: [char; 256],
    unicode_to_byte: HashMap<char, u8>,
    pub bos: u32,
    pub eos: u32,
    pub eot: u32,
    pub start_header: u32,
    pub end_header: u32,
}

impl BpeTokenizer {
    pub fn from_gguf(gguf: &GGUFFile) -> Result<Self, String> {
        let vocab: Vec<String> = gguf.metadata.get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| {
                if let MetaValue::Str(s) = v { Some(s.clone()) } else { None }
            }).collect())
            .ok_or("missing tokenizer.ggml.tokens")?;

        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (i, t) in vocab.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }

        let merges: Vec<(String, String)> = gguf.metadata.get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| {
                if let MetaValue::Str(s) = v {
                    let mut parts = s.splitn(2, ' ');
                    let a = parts.next()?.to_string();
                    let b = parts.next()?.to_string();
                    Some((a, b))
                } else { None }
            }).collect())
            .ok_or("missing tokenizer.ggml.merges")?;

        let merge_rank: HashMap<(String, String), usize> = merges.into_iter()
            .enumerate().map(|(i, pair)| (pair, i)).collect();

        // GPT-2 byte-to-unicode table
        let mut byte_to_unicode = ['\0'; 256];
        let mut bs: Vec<u8> = (b'!'..=b'~').collect();
        bs.extend(161u8..=172);
        bs.extend(174u8..=255);
        let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
        let mut n = 0u32;
        for b in 0u8..=255 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        for (&b, &c) in bs.iter().zip(cs.iter()) {
            byte_to_unicode[b as usize] = char::from_u32(c).unwrap_or('?');
        }

        let unicode_to_byte: HashMap<char, u8> = byte_to_unicode.iter()
            .enumerate().map(|(b, &c)| (c, b as u8)).collect();

        let find = |name: &str| -> u32 { *token_to_id.get(name).unwrap_or(&0) };
        let bos = gguf.config.bos_token;
        let eos = gguf.config.eos_token;
        let eot = find("<|eot_id|>");
        let start_header = find("<|start_header_id|>");
        let end_header = find("<|end_header_id|>");

        Ok(Self { vocab, token_to_id, merge_rank, byte_to_unicode,
                  unicode_to_byte, bos, eos, eot, start_header, end_header })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<String> = text.bytes()
            .map(|b| { let mut s = String::new(); s.push(self.byte_to_unicode[b as usize]); s })
            .collect();
        loop {
            let mut best_rank = usize::MAX;
            let mut best_idx = 0;
            for i in 0..tokens.len().saturating_sub(1) {
                if let Some(&rank) = self.merge_rank.get(&(tokens[i].clone(), tokens[i+1].clone())) {
                    if rank < best_rank { best_rank = rank; best_idx = i; }
                }
            }
            if best_rank == usize::MAX { break; }
            let merged = format!("{}{}", tokens[best_idx], tokens[best_idx + 1]);
            tokens[best_idx] = merged;
            tokens.remove(best_idx + 1);
        }
        tokens.iter().map(|t| *self.token_to_id.get(t).unwrap_or(&0)).collect()
    }

    pub fn decode_token(&self, t: u32) -> String {
        if (t as usize) >= self.vocab.len() { return format!("[{t}]"); }
        let tok_str = &self.vocab[t as usize];
        if tok_str.starts_with("<|") { return String::new(); }
        let bytes: Vec<u8> = tok_str.chars()
            .map(|c| *self.unicode_to_byte.get(&c).unwrap_or(&b'?'))
            .collect();
        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|&t| self.decode_token(t)).collect()
    }

    /// Format a Llama 3.3 chat prompt: system + user message.
    pub fn format_chat(&self, system: &str, user: &str) -> Vec<u32> {
        let mut toks = Vec::with_capacity(256);
        toks.push(self.bos);
        // system header
        toks.push(self.start_header);
        toks.extend(self.encode("system"));
        toks.push(self.end_header);
        toks.extend(self.encode(&format!("\n\n{system}")));
        toks.push(self.eot);
        // user header
        toks.push(self.start_header);
        toks.extend(self.encode("user"));
        toks.push(self.end_header);
        toks.extend(self.encode(&format!("\n\n{user}")));
        toks.push(self.eot);
        // assistant header (model generates from here)
        toks.push(self.start_header);
        toks.extend(self.encode("assistant"));
        toks.push(self.end_header);
        toks.extend(self.encode("\n\n"));
        toks
    }
}

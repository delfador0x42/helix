//! WordPiece tokenizer for all-MiniLM-L6-v2 (uncased BERT vocab).
//! Loads vocab.txt, tokenizes text into input_ids + attention_mask.

use std::collections::HashMap;
use std::path::Path;

const CLS: i64 = 101;
const SEP: i64 = 102;
const UNK: i64 = 100;
const MAX_SEQ: usize = 256;

pub struct Tokenizer {
    vocab: HashMap<String, i64>,
}

pub struct TokenizedInput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
}

impl Tokenizer {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let path = model_dir.join("vocab.txt");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("vocab.txt at {}: {e}", path.display()))?;
        let vocab: HashMap<String, i64> = text.lines()
            .enumerate()
            .map(|(i, line)| (line.to_string(), i as i64))
            .collect();
        if vocab.len() < 1000 { return Err("vocab.txt too small".into()); }
        Ok(Tokenizer { vocab })
    }

    pub fn encode(&self, text: &str) -> TokenizedInput {
        let mut ids = vec![CLS];
        let lower = text.to_lowercase();
        for word in basic_tokenize(&lower) {
            for id in self.wordpiece(&word) {
                ids.push(id);
                if ids.len() >= MAX_SEQ - 1 { break; }
            }
            if ids.len() >= MAX_SEQ - 1 { break; }
        }
        ids.push(SEP);
        let len = ids.len();
        TokenizedInput { input_ids: ids, attention_mask: vec![1i64; len] }
    }

    fn wordpiece(&self, word: &str) -> Vec<i64> {
        let chars: Vec<char> = word.chars().collect();
        if chars.is_empty() { return vec![]; }
        let mut tokens = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = false;
            while start < end {
                let sub: String = if start == 0 {
                    chars[start..end].iter().collect()
                } else {
                    let mut s = String::with_capacity(2 + (end - start) * 4);
                    s.push_str("##");
                    s.extend(&chars[start..end]);
                    s
                };
                if let Some(&id) = self.vocab.get(&sub) {
                    tokens.push(id);
                    start = end;
                    found = true;
                    break;
                }
                end -= 1;
            }
            if !found { tokens.push(UNK); break; }
        }
        tokens
    }
}

/// Split text into tokens: whitespace-separated, punctuation as own tokens.
fn basic_tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() { tokens.push(std::mem::take(&mut current)); }
        } else if ch.is_ascii_punctuation() {
            if !current.is_empty() { tokens.push(std::mem::take(&mut current)); }
            tokens.push(ch.to_string());
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() { tokens.push(current); }
    tokens
}

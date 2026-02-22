//! ONNX inference for all-MiniLM-L6-v2 via ort with CoreML acceleration.
//! Loads model once, embeds text into 384-dim L2-normalized vectors.

use std::path::Path;
use ort::execution_providers::coreml::{CoreMLExecutionProvider, CoreMLComputeUnits};

pub const EMBED_DIM: usize = 384;

pub struct Embedder {
    session: ort::session::Session,
}

impl Embedder {
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let model_path = model_dir.join("model.onnx");
        if !model_path.exists() {
            return Err(format!(
                "Model not found: {}\n\
                 Download: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx\n\
                 Place at: {}", model_path.display(), model_path.display()
            ));
        }
        // CoreML + static shapes for ANE. Falls back to CPU for unsupported ops.
        let session = ort::session::Session::builder()
            .map_err(|e| format!("ort builder: {e}"))?
            .with_execution_providers([
                CoreMLExecutionProvider::default()
                    .with_subgraphs(true)
                    .with_static_input_shapes(true)
                    .with_compute_units(CoreMLComputeUnits::CPUAndNeuralEngine)
                    .build()
            ])
            .map_err(|e| format!("ort EP: {e}"))?
            .commit_from_file(&model_path)
            .map_err(|e| format!("ort load: {e}"))?;
        eprintln!("helix-bg: model loaded from {}", model_path.display());
        Ok(Embedder { session })
    }

    /// Embed tokenized input → 384-dim L2-normalized vector.
    pub fn embed(&mut self, input: &crate::tokenize::TokenizedInput) -> Result<Vec<f32>, String> {
        let seq_len = input.input_ids.len();
        let ids = ort::value::Tensor::from_array(
            ([1usize, seq_len], input.input_ids.clone().into_boxed_slice())
        ).map_err(|e| format!("tensor: {e}"))?;
        let mask = ort::value::Tensor::from_array(
            ([1usize, seq_len], input.attention_mask.clone().into_boxed_slice())
        ).map_err(|e| format!("tensor: {e}"))?;
        let type_ids = ort::value::Tensor::from_array(
            ([1usize, seq_len], vec![0i64; seq_len].into_boxed_slice())
        ).map_err(|e| format!("tensor: {e}"))?;

        let outputs = self.session.run(
            ort::inputs!["input_ids" => ids, "attention_mask" => mask, "token_type_ids" => type_ids]
        ).map_err(|e| format!("inference: {e}"))?;

        // Try known output names: pooled → token embeddings → last hidden state
        for name in ["sentence_embedding", "token_embeddings", "last_hidden_state"] {
            if let Some(val) = outputs.get(name) {
                let (_shape, data) = val.try_extract_tensor::<f32>()
                    .map_err(|e| format!("extract {name}: {e}"))?;
                if name == "sentence_embedding" {
                    return Ok(data[..EMBED_DIM].to_vec());
                }
                return Ok(mean_pool_normalize(data, &input.attention_mask, seq_len));
            }
        }
        Err("no embedding output found in model".into())
    }
}

/// Mean pooling over token embeddings + L2 normalization.
fn mean_pool_normalize(token_emb: &[f32], mask: &[i64], seq_len: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; EMBED_DIM];
    let mut count = 0.0f32;
    for i in 0..seq_len {
        if mask[i] == 1 {
            for d in 0..EMBED_DIM { result[d] += token_emb[i * EMBED_DIM + d]; }
            count += 1.0;
        }
    }
    let c = count.max(1e-9);
    for d in 0..EMBED_DIM { result[d] /= c; }
    let norm = result.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for d in 0..EMBED_DIM { result[d] /= norm; }
    result
}

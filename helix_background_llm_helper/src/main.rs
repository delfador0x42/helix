//! helix-bg: background intelligence daemon for helix knowledge base.
//! Embeds entries with all-MiniLM-L6-v2 (CoreML/ANE), finds patterns.

mod datalog;
mod tokenize;
mod embed;
mod cache;
mod similarity;
mod insight;
mod gpu;
mod metal_test;
mod gguf;
mod kernels;
mod model;
mod infer;

use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { usage(); }
    match args[1].as_str() {
        "run" => daemon(args.get(2)),
        "analyze" => one_shot(args.get(2)),
        "embed-test" => embed_test(),
        "metal-test" => metal_test::run_bandwidth_test(),
        "load-model" => load_model(args.get(2)),
        "generate" => run_generate(args.get(2)),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("helix-bg — background intelligence for helix knowledge base\n");
    eprintln!("  helix-bg run [kb-dir]       Start background daemon (watches data.log)");
    eprintln!("  helix-bg analyze [kb-dir]   One-shot analysis (embed + find patterns)");
    eprintln!("  helix-bg embed-test         Test embedding pipeline");
    eprintln!("  helix-bg metal-test         Benchmark Metal GPU bandwidth + matmul");
    eprintln!("  helix-bg load-model [path]  Load and inspect a GGUF model file");
    eprintln!("  helix-bg generate [path]    Generate tokens from GGUF model");
    std::process::exit(1);
}

fn daemon(dir_arg: Option<&String>) {
    let dir = resolve_kb_dir(dir_arg);
    let model_dir = models_dir();
    let tokenizer = tokenize::Tokenizer::load(&model_dir).unwrap_or_else(|e| fatal(&e));
    let mut embedder = embed::Embedder::load(&model_dir).unwrap_or_else(|e| fatal(&e));
    let log_path = dir.join("data.log");
    let mut last_mtime = mtime(&log_path);
    let mut ecache = cache::EmbeddingCache::load_or_new(&dir);
    eprintln!("helix-bg: watching {} ({} cached embeddings)", dir.display(), ecache.entries.len());

    loop {
        let current = mtime(&log_path);
        if current != last_mtime {
            last_mtime = current;
            if let Ok(entries) = datalog::read_entries(&log_path) {
                let new_count = embed_new(&entries, &tokenizer, &mut embedder, &mut ecache);
                if new_count > 0 {
                    eprintln!("helix-bg: embedded {} new entries (total: {})", new_count, ecache.entries.len());
                    ecache.save(&dir);
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_secs(30));
    }
}

fn one_shot(dir_arg: Option<&String>) {
    let dir = resolve_kb_dir(dir_arg);
    let model_dir = models_dir();
    let tokenizer = tokenize::Tokenizer::load(&model_dir).unwrap_or_else(|e| fatal(&e));
    let mut embedder = embed::Embedder::load(&model_dir).unwrap_or_else(|e| fatal(&e));
    let log_path = dir.join("data.log");
    let entries = datalog::read_entries(&log_path).unwrap_or_else(|e| fatal(&e));
    eprintln!("helix-bg: {} entries loaded", entries.len());

    let mut ecache = cache::EmbeddingCache::load_or_new(&dir);
    let new_count = embed_new(&entries, &tokenizer, &mut embedder, &mut ecache);
    if new_count > 0 {
        eprintln!("helix-bg: embedded {} new entries", new_count);
        ecache.save(&dir);
    }
    eprintln!("helix-bg: {} total embeddings cached", ecache.entries.len());

    let findings = insight::analyze(&entries, &ecache);
    if findings.is_empty() {
        println!("No findings.");
    } else {
        println!("--- {} findings ---", findings.len());
        for f in &findings { println!("{f}"); }
    }
}

fn embed_test() {
    let model_dir = models_dir();
    let tok = tokenize::Tokenizer::load(&model_dir).unwrap_or_else(|e| fatal(&e));
    let mut emb = embed::Embedder::load(&model_dir).unwrap_or_else(|e| fatal(&e));

    let texts = ["hello world", "rust programming", "machine learning embeddings"];
    for text in &texts {
        let tokens = tok.encode(text);
        let vec = emb.embed(&tokens).unwrap_or_else(|e| fatal(&e));
        println!("{}: [{:.4}, {:.4}, {:.4}, ...] dim={}", text, vec[0], vec[1], vec[2], vec.len());
    }
    println!("\nSimilarity tests:");
    let pairs = [
        ("rust programming language", "cargo build system for rust"),
        ("rust programming language", "chocolate cake recipe"),
        ("helix knowledge base", "amaranthine knowledge store"),
    ];
    for (a, b) in &pairs {
        let ea = emb.embed(&tok.encode(a)).unwrap();
        let eb = emb.embed(&tok.encode(b)).unwrap();
        println!("  {:.4}  '{}' vs '{}'", similarity::cosine(&ea, &eb), a, b);
    }
}

/// Embed all entries not yet in cache. Returns count of newly embedded.
fn embed_new(
    entries: &[datalog::Entry], tok: &tokenize::Tokenizer,
    emb: &mut embed::Embedder, ecache: &mut cache::EmbeddingCache,
) -> usize {
    let mut count = 0;
    for entry in entries {
        if ecache.has(entry.offset) { continue; }
        let text = format!("{} {}", entry.topic, entry.content());
        let tokens = tok.encode(&text);
        match emb.embed(&tokens) {
            Ok(vec) => { ecache.add(entry.offset, vec); count += 1; }
            Err(e) => eprintln!("helix-bg: embed failed for offset {}: {e}", entry.offset),
        }
    }
    count
}

fn resolve_kb_dir(arg: Option<&String>) -> PathBuf {
    if let Some(d) = arg { return PathBuf::from(d); }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let p = PathBuf::from(&home).join(".helix");
    if p.join("data.log").exists() { return p; }
    PathBuf::from(".")
}

fn models_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".helix/models")
}

fn mtime(path: &Path) -> u64 {
    std::fs::metadata(path).ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs()).unwrap_or(0)
}

fn load_model(path_arg: Option<&String>) {
    let path = path_arg.map(|s| std::path::PathBuf::from(s)).unwrap_or_else(|| {
        // Default: look in models/ dir
        let mut p = std::env::current_dir().unwrap_or_default();
        p.push("models");
        // Find first .gguf file
        if let Ok(entries) = std::fs::read_dir(&p) {
            for e in entries.flatten() {
                if e.path().extension().map(|x| x == "gguf").unwrap_or(false) {
                    return e.path();
                }
            }
        }
        eprintln!("No .gguf file found in models/. Pass path as argument.");
        std::process::exit(1);
    });
    eprintln!("Loading GGUF: {}", path.display());
    let model = gguf::GGUFFile::open(&path).unwrap_or_else(|e| fatal(&e));
    eprintln!("\n--- Tensors ({}) ---", model.tensors.len());
    let mut total_bytes: u64 = 0;
    for t in &model.tensors {
        let bytes = t.byte_size();
        total_bytes += bytes;
        eprintln!("  {:.<50} {:?} {:>10} {:>8.2}MB  {:?}",
            t.name, t.dims, t.n_elements(), bytes as f64 / 1e6, t.dtype);
    }
    eprintln!("\nTotal weight data: {:.1}MB", total_bytes as f64 / 1e6);
}

/// CPU-side Q6K dequant for verification
fn cpu_dequant_q6k_row(data: &[u8], row: usize, dim: usize) -> Vec<f32> {
    let bpr = dim / 256;
    let row_bytes = bpr * 210;
    let row_data = &data[row * row_bytes..];
    let mut out = vec![0.0f32; dim];

    for b in 0..bpr {
        let blk = &row_data[b * 210..];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208]; // int8_t
        let d = gguf::f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
        let base = b * 256;

        for n in 0..2u32 {
            let sb = (n * 8) as usize;
            let ql_off = (n * 64) as usize;
            let qh_off = (n * 32) as usize;
            let s0 = d * (scales[sb] as i8 as f32);
            let s2 = d * (scales[sb + 2] as i8 as f32);
            let s4 = d * (scales[sb + 4] as i8 as f32);
            let s6 = d * (scales[sb + 6] as i8 as f32);
            for l in 0..32usize {
                let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;
                let idx = base + (n as usize) * 128;
                out[idx + l] = s0 * q1 as f32;
                out[idx + 32 + l] = s2 * q2 as f32;
                out[idx + 64 + l] = s4 * q3 as f32;
                out[idx + 96 + l] = s6 * q4 as f32;
            }
        }
    }
    out
}

fn run_generate(path_arg: Option<&String>) {
    let path = path_arg.map(|s| std::path::PathBuf::from(s)).unwrap_or_else(|| {
        let mut p = std::env::current_dir().unwrap_or_default();
        p.push("models");
        if let Ok(entries) = std::fs::read_dir(&p) {
            for e in entries.flatten() {
                if e.path().extension().map(|x| x == "gguf").unwrap_or(false) {
                    return e.path();
                }
            }
        }
        eprintln!("No .gguf file found. Pass path as argument.");
        std::process::exit(1);
    });
    eprintln!("Loading GGUF: {}", path.display());
    let gguf = gguf::GGUFFile::open(&path).unwrap_or_else(|e| fatal(&e));
    // Print tokenizer metadata
    for key in gguf.metadata.keys() {
        if key.contains("token") && !key.contains("tokens") {
            eprintln!("  {key}: {:?}", gguf.metadata[key]);
        }
    }

    // Extract vocab for decoding
    let vocab: Vec<String> = gguf.metadata.get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| {
            if let gguf::MetaValue::Str(s) = v { Some(s.clone()) } else { None }
        }).collect())
        .unwrap_or_default();
    eprintln!("  vocab size: {}", vocab.len());

    // Verify Q6K dequant: CPU vs GPU for embedding lookup
    let embd_tensor = gguf.tensor("token_embd.weight").unwrap();
    let embd_data = gguf.tensor_data(embd_tensor);
    let test_token = 151644u32; // <|im_start|>
    let cpu_embed = cpu_dequant_q6k_row(embd_data, test_token as usize, 1024);
    eprintln!("  CPU embed[0..8]: {:?}", &cpu_embed[..8]);

    let mdl = model::Model::load(&gguf).unwrap_or_else(|e| fatal(&e));

    // GPU embed: run just the embedding step
    {
        let cmd = mdl.queue.new_command_buffer();
        infer::forward(&mdl, test_token, 0);
        let ptr = mdl.x.contents() as *const f32;
        let gpu_embed: Vec<f32> = (0..8).map(|i| unsafe { *ptr.add(i) }).collect();
        eprintln!("  GPU x[0..8] (after full fwd): {:?}", gpu_embed);
    }

    // Isolated embedding test: just embed, no layers
    {
        use std::ffi::c_void;
        let cmd = mdl.queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        enc.set_pipeline(&mdl.p_embed);
        enc.set_buffer(0, &mdl.weights, mdl.embd_off);
        enc.set_buffer(1, &mdl.norm_out, 0); // use norm_out as temp
        let dim = mdl.cfg.hidden_dim;
        enc.set_bytes(2, &dim as *const u32 as *const c_void, 4);
        enc.set_bytes(3, &test_token as *const u32 as *const c_void, 4);
        enc.dispatch_threads(
            gpu::MTLSize::new(dim as u64, 1, 1),
            gpu::MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let ptr = mdl.norm_out.contents() as *const f32;
        let gpu_only: Vec<f32> = (0..8).map(|i| unsafe { *ptr.add(i) }).collect();
        eprintln!("  GPU embed only[0..8]: {:?}", gpu_only);

        // Compare CPU vs GPU
        let mut max_diff: f32 = 0.0;
        for i in 0..1024 {
            let diff = (cpu_embed[i] - unsafe { *ptr.add(i) }).abs();
            if diff > max_diff { max_diff = diff; }
        }
        eprintln!("  Max CPU-GPU diff: {max_diff:.6}");
    }

    // Qwen3 has no explicit BOS; use <|im_start|> = 151644
    let start_token = if mdl.cfg.bos_token == 0 { 151644u32 } else { mdl.cfg.bos_token };
    let n_tokens = 32;

    let start = std::time::Instant::now();
    let tokens = infer::generate(&mdl, start_token, n_tokens);
    let elapsed = start.elapsed();

    let tok_per_sec = n_tokens as f64 / elapsed.as_secs_f64();
    eprintln!("\n--- Generated {} tokens in {:.2}ms ({:.1} tok/s) ---",
        tokens.len(), elapsed.as_secs_f64() * 1000.0, tok_per_sec);
    eprintln!("Token IDs: {:?}", &tokens[..tokens.len().min(20)]);

    // Decode tokens to text
    let text: String = tokens.iter().map(|&t| {
        if (t as usize) < vocab.len() {
            vocab[t as usize].replace("Ġ", " ").replace("Ċ", "\n")
        } else {
            format!("[{t}]")
        }
    }).collect();
    eprintln!("\nDecoded:\n{text}");
}

fn fatal(msg: &str) -> ! {
    eprintln!("helix-bg: {msg}");
    std::process::exit(1);
}

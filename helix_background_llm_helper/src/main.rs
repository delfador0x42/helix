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
mod bench_dispatch;
mod bench_kernels;
mod kernels_batch;

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
        "bench-dispatch" => bench_dispatch::run(),
        "bench-kernels" => bench_kernels::run(),
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
    // Print key tokenizer metadata (skip large arrays)
    for key in gguf.metadata.keys() {
        if key.contains("token") && !key.contains("tokens") && !key.contains("merges") {
            let val = &gguf.metadata[key];
            let is_large = matches!(val, gguf::MetaValue::Array(_));
            if !is_large { eprintln!("  {key}: {:?}", val); }
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

    // Use BOS token from GGUF metadata (not hardcoded)
    let test_token = gguf.config.bos_token;
    eprintln!("  BOS token: {test_token}");

    let mdl = model::Model::load(&gguf).unwrap_or_else(|e| fatal(&e));

    // Isolated embedding test: just embed, no layers
    {
        use std::ffi::c_void;
        let cmd = mdl.queue.new_command_buffer();
        let enc = cmd.new_compute_encoder();
        enc.set_pipeline(mdl.embed_pipeline(mdl.embd_type));
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
        eprintln!("  GPU embed[0..8]: {:?}", gpu_only);
    }

    let start_token = test_token;
    let n_tokens = 128;

    // Debug: print logits after first forward pass
    {
        infer::forward(&mdl, start_token, 0);
        let ptr = mdl.logits.contents() as *const f32;
        let n = mdl.cfg.vocab_size as usize;
        let logits: Vec<f32> = (0..n).map(|i| unsafe { *ptr.add(i) }).collect();
        let mut best_id = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut count_nan = 0;
        let mut count_inf = 0;
        for (i, &v) in logits.iter().enumerate() {
            if v.is_nan() { count_nan += 1; }
            if v.is_infinite() { count_inf += 1; }
            sum += v as f64;
            if v > best_val { best_val = v; best_id = i; }
        }
        let mean = sum / n as f64;
        // Find top 5
        let mut top5: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("  Logits: mean={mean:.4}, best={best_id}({best_val:.4}), nan={count_nan}, inf={count_inf}");
        eprintln!("  Top 5 tokens:");
        for (id, val) in top5.iter().take(5) {
            let tok = if (*id as usize) < vocab.len() { vocab[*id].clone() } else { format!("[{id}]") };
            eprintln!("    {id}: {val:.4} = {:?}", tok);
        }
        eprintln!("  Bottom 5:");
        for (id, val) in top5.iter().rev().take(5) {
            let tok = if (*id as usize) < vocab.len() { vocab[*id].clone() } else { format!("[{id}]") };
            eprintln!("    {id}: {val:.4} = {:?}", tok);
        }
        // Also print x after first layer
        let xptr = mdl.x.contents() as *const f32;
        let xvals: Vec<f32> = (0..8).map(|i| unsafe { *xptr.add(i) }).collect();
        eprintln!("  x[0..8] after full fwd: {:?}", xvals);
    }

    // Per-token timing breakdown
    {
        infer::forward(&mdl, start_token, 0);
        let t0 = std::time::Instant::now();
        let _tok = infer::argmax(&mdl);
        let argmax_us = t0.elapsed().as_nanos() as f64 / 1000.0;

        let t1 = std::time::Instant::now();
        infer::forward(&mdl, _tok, 1);
        let fwd_us = t1.elapsed().as_micros();
        eprintln!("  TIMING: argmax={:.1}µs, forward={fwd_us}µs", argmax_us);
    }

    // Per-operation timing breakdown (separate cmd bufs, overcounts total)
    {
        eprintln!("\n  Per-operation breakdown (pos=0):");
        infer::forward(&mdl, start_token, 0); // warmup
        let t = infer::forward_timed(&mdl, start_token, 0);
        t.print();
    }

    eprintln!("\n  Per-token timing:");
    let start = std::time::Instant::now();
    let tokens = infer::generate_timed(&mdl, start_token, n_tokens);
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

//! helix-bg: background intelligence daemon for helix knowledge base.
//! Embeds entries with all-MiniLM-L6-v2 (CoreML/ANE), finds patterns.
//! GPU inference: 70B-only engine (Q5K/Q6K/Q8_0, MMA staging).

mod datalog;
mod tokenize;
mod embed;
mod cache;
mod similarity;
mod insight;
mod gpu;
mod gguf;
mod model;
mod kernels_batch;
mod infer_batch;

use std::path::{Path, PathBuf};

extern "C" { fn setpriority(which: i32, who: u32, prio: i32) -> i32; }

fn main() {
    // PRIO_DARWIN_GPU(5) = UI_FOCAL(7): max GPU frequency, highest scheduling priority.
    // PRIO_DARWIN_GAME_MODE(7) = ON(1): dedicated CPU/GPU scheduling.
    unsafe {
        setpriority(5, 0, 7); // GPU priority
        setpriority(7, 0, 1); // Game mode
    }

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { usage(); }
    match args[1].as_str() {
        "run" => daemon(args.get(2)),
        "analyze" => one_shot(args.get(2)),
        "embed-test" => embed_test(),
        "load-model" => load_model(args.get(2)),
        "bench-batch" => run_bench_batch(args.get(2)),
        "bench-profile" => run_bench_profile(args.get(2)),
        "generate-batch" => run_generate_batch(args.get(2)),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("helix-bg — background intelligence for helix knowledge base\n");
    eprintln!("  helix-bg run [kb-dir]          Start background daemon (watches data.log)");
    eprintln!("  helix-bg analyze [kb-dir]      One-shot analysis (embed + find patterns)");
    eprintln!("  helix-bg embed-test            Test embedding pipeline");
    eprintln!("  helix-bg load-model [path]     Load and inspect a GGUF model file");
    eprintln!("  helix-bg bench-batch [path]    Benchmark batch forward pass (70B)");
    eprintln!("  helix-bg bench-profile [path]  Per-operation GPU profiling (70B)");
    eprintln!("  helix-bg generate-batch [path] Generate text using batch prefill + decode");
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

fn run_bench_batch(path_arg: Option<&String>) {
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
    let mdl = model::Model::load(&gguf).unwrap_or_else(|e| fatal(&e));
    let batch = infer_batch::BatchState::new(&mdl).unwrap_or_else(|e| fatal(&e));
    // Use B=640 for benchmark: optimal throughput without KV cache overflow
    let bs: u32 = 640;

    eprintln!("\n=== Batch Forward Benchmark (B={}, FP16 grid-tiled) ===\n", bs);

    let bos = gguf.config.bos_token;
    let tokens: Vec<u32> = vec![bos; bs as usize];

    // Warmup
    for _ in 0..3 {
        infer_batch::forward_batch(&mdl, &batch, &tokens, 0);
    }

    let iters = 20u32;

    // Benchmark: fixed pos=0 (pure prefill throughput, no attention growth)
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        infer_batch::forward_batch(&mdl, &batch, &tokens, 0);
    }
    let fixed_per = t0.elapsed().as_secs_f64() / iters as f64;
    let fixed_tps = bs as f64 / fixed_per;
    eprintln!("  Batch (pos=0 fixed):   {:.2}ms = {:.0} tok/s", fixed_per * 1000.0, fixed_tps);

    // Benchmark: advancing positions (realistic — attention grows, limited to KV cache)
    let max_pos_iters = std::cmp::min(iters, (2048 - bs) / bs);
    if max_pos_iters > 0 {
        let t0 = std::time::Instant::now();
        for i in 0..max_pos_iters {
            infer_batch::forward_batch(&mdl, &batch, &tokens, i * bs);
        }
        let per_iter = t0.elapsed().as_secs_f64() / max_pos_iters as f64;
        let tok_per_sec = bs as f64 / per_iter;
        eprintln!("  Batch (advancing pos): {:.2}ms = {:.0} tok/s ({max_pos_iters} iters)", per_iter * 1000.0, tok_per_sec);
    }

    // Single-token decode (B=1 through batch path)
    let single_tokens: Vec<u32> = vec![bos; 1];
    for _ in 0..3 { infer_batch::forward_batch(&mdl, &batch, &single_tokens, 0); }
    let t0 = std::time::Instant::now();
    for i in 0..iters { infer_batch::forward_batch(&mdl, &batch, &single_tokens, i); }
    let single_per = t0.elapsed().as_secs_f64() / iters as f64;
    eprintln!("  Single-token (B=1):    {:.2}ms = {:.0} tok/s", single_per * 1000.0, 1.0 / single_per);
    eprintln!("  Batch speedup: {:.1}x", fixed_tps * single_per);

    eprintln!("\n=== Done ===");
}

fn run_bench_profile(path_arg: Option<&String>) {
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
    let mdl = model::Model::load(&gguf).unwrap_or_else(|e| fatal(&e));
    let batch = infer_batch::BatchState::new(&mdl).unwrap_or_else(|e| fatal(&e));
    let bs: u32 = 640; // optimal for profiling
    let bos = gguf.config.bos_token;
    let tokens: Vec<u32> = vec![bos; bs as usize];

    eprintln!("\n=== Per-Operation GPU Profiling (B={}, pos=0) ===\n", bs);

    // Warmup
    for _ in 0..3 {
        infer_batch::forward_batch(&mdl, &batch, &tokens, 0);
    }

    // Profile at pos=0 (minimal attention)
    let iters = 5;
    let mut t_total = 0.0; let mut t_mm = 0.0; let mut t_attn = 0.0;
    let mut t_elem = 0.0; let mut t_out = 0.0;
    for _ in 0..iters {
        let (total, mm, attn, elem, out) =
            infer_batch::forward_profiled(&mdl, &batch, &tokens, 0, bs);
        t_total += total; t_mm += mm; t_attn += attn; t_elem += elem; t_out += out;
    }
    let d = iters as f64;
    t_total /= d; t_mm /= d; t_attn /= d; t_elem /= d; t_out /= d;

    let tps = bs as f64 / (t_total / 1000.0);
    eprintln!("  Total:       {:.2}ms = {:.0} tok/s (serialized, overcounts due to cmd buf overhead)", t_total, tps);
    eprintln!("  Matmul:      {:.2}ms ({:.1}%)", t_mm, t_mm / t_total * 100.0);
    eprintln!("  Attention:   {:.2}ms ({:.1}%)", t_attn, t_attn / t_total * 100.0);
    eprintln!("  Element-wise:{:.2}ms ({:.1}%)", t_elem, t_elem / t_total * 100.0);
    eprintln!("  Output proj: {:.2}ms ({:.1}%)", t_out, t_out / t_total * 100.0);

    // Compare with non-profiled (single cmd buf)
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        infer_batch::forward_batch(&mdl, &batch, &tokens, 0);
    }
    let real = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;
    let real_tps = bs as f64 / (real / 1000.0);
    eprintln!("\n  Non-profiled: {:.2}ms = {:.0} tok/s", real, real_tps);
    eprintln!("  Profiling overhead: {:.2}ms ({:.1}x)", t_total - real, t_total / real);

    // Also profile at pos=200 (heavier attention)
    eprintln!("\n=== Profiling at pos=200 ===\n");
    // Fill KV cache to pos=200
    for i in 0..3 { infer_batch::forward_batch(&mdl, &batch, &tokens, i * bs); }

    t_total = 0.0; t_mm = 0.0; t_attn = 0.0; t_elem = 0.0; t_out = 0.0;
    for _ in 0..iters {
        let (total, mm, attn, elem, out) =
            infer_batch::forward_profiled(&mdl, &batch, &tokens, 200, bs);
        t_total += total; t_mm += mm; t_attn += attn; t_elem += elem; t_out += out;
    }
    t_total /= d; t_mm /= d; t_attn /= d; t_elem /= d; t_out /= d;
    let tps200 = bs as f64 / (t_total / 1000.0);
    eprintln!("  Total:       {:.2}ms = {:.0} tok/s", t_total, tps200);
    eprintln!("  Matmul:      {:.2}ms ({:.1}%)", t_mm, t_mm / t_total * 100.0);
    eprintln!("  Attention:   {:.2}ms ({:.1}%)", t_attn, t_attn / t_total * 100.0);
    eprintln!("  Element-wise:{:.2}ms ({:.1}%)", t_elem, t_elem / t_total * 100.0);
    eprintln!("  Output proj: {:.2}ms ({:.1}%)", t_out, t_out / t_total * 100.0);

    eprintln!("\n=== FLOP Analysis ===\n");
    let flop_per_token = 2.474e9_f64;
    let total_flop = flop_per_token * bs as f64;
    let achieved_tflops = total_flop / (real / 1000.0) / 1e12;
    eprintln!("  {:.2} GFLOP per token × {} tokens = {:.2} GFLOP per batch",
        flop_per_token / 1e9, bs, total_flop / 1e9);
    eprintln!("  At {:.2}ms: {:.2} TFLOP/s effective ({:.1}% of 14.2 TFLOP/s FP16 peak)",
        real, achieved_tflops, achieved_tflops / 14.2 * 100.0);

    eprintln!("\n=== Done ===");
}

fn run_generate_batch(path_arg: Option<&String>) {
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

    // Extract vocab for decoding
    let vocab: Vec<String> = gguf.metadata.get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| {
            if let gguf::MetaValue::Str(s) = v { Some(s.clone()) } else { None }
        }).collect())
        .unwrap_or_default();

    let mdl = model::Model::load(&gguf).unwrap_or_else(|e| fatal(&e));
    let batch = infer_batch::BatchState::new(&mdl).unwrap_or_else(|e| fatal(&e));

    let bos = gguf.config.bos_token;
    let eos = gguf.config.eos_token;

    // Simulate a prompt: BOS + repeated tokens (for prefill benchmark)
    let prompt_len = 200u32;
    let prompt: Vec<u32> = std::iter::once(bos)
        .chain(std::iter::repeat(bos).take(prompt_len as usize - 1))
        .collect();
    let n_generate = 128u32;

    eprintln!("\n=== Batch Generate (prompt={}, generate={}, B={}) ===\n",
        prompt_len, n_generate, batch.batch_size);

    // ── Prefill benchmark ──
    let t0 = std::time::Instant::now();
    let pos = infer_batch::prefill(&mdl, &batch, &prompt);
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let prefill_tps = prompt_len as f64 / t0.elapsed().as_secs_f64();
    eprintln!("  Prefill: {} tokens in {:.2}ms = {:.0} tok/s",
        prompt_len, prefill_ms, prefill_tps);

    // ── Decode benchmark ──
    let first_token = infer_batch::argmax_one(&batch);
    let t1 = std::time::Instant::now();
    let mut generated = vec![first_token];
    let mut next = first_token;
    for i in 0..n_generate - 1 {
        next = infer_batch::decode_step(&mdl, &batch, next, pos + i);
        generated.push(next);
        if next == eos { break; }
    }
    let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let decode_tps = generated.len() as f64 / t1.elapsed().as_secs_f64();
    eprintln!("  Decode:  {} tokens in {:.2}ms = {:.0} tok/s",
        generated.len(), decode_ms, decode_tps);

    // Decode tokens to text
    let text: String = generated.iter().map(|&t| {
        if (t as usize) < vocab.len() {
            vocab[t as usize].replace("Ġ", " ").replace("Ċ", "\n")
        } else {
            format!("[{t}]")
        }
    }).collect();
    eprintln!("\n  Generated text:\n{text}");

    eprintln!("\n=== Done ===");
}

fn fatal(msg: &str) -> ! {
    eprintln!("helix-bg: {msg}");
    std::process::exit(1);
}

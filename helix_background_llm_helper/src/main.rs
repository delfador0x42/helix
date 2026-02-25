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
        "dump-vocab" => dump_vocab(args.get(2)),
        "chat" => run_chat(args.get(2), args.get(3).map(|s| s.as_str())),
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
    // Use random tokens per iteration to prevent Metal from caching/eliding computation
    let vocab = gguf.config.vocab_size;
    let max_pos_iters = std::cmp::min(iters, (2048 - bs) / bs);
    if max_pos_iters > 0 {
        let mut rng_state: u64 = 0xdeadbeef;
        let t0 = std::time::Instant::now();
        for i in 0..max_pos_iters {
            let tok: Vec<u32> = (0..bs).map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng_state >> 33) as u32 % vocab
            }).collect();
            infer_batch::forward_batch(&mdl, &batch, &tok, i * bs);
            let _ = infer_batch::argmax_batch(&batch, bs);
        }
        let per_iter = t0.elapsed().as_secs_f64() / max_pos_iters as f64;
        let tok_per_sec = bs as f64 / per_iter;
        eprintln!("  Batch (advancing pos): {:.2}ms = {:.0} tok/s ({max_pos_iters} iters)", per_iter * 1000.0, tok_per_sec);
    }

    // Single-token decode (B=1 through batch path)
    // Use varying tokens + argmax readback to prevent Metal elision
    let single_tokens: Vec<u32> = vec![bos; 1];
    for _ in 0..3 { infer_batch::forward_batch(&mdl, &batch, &single_tokens, 0); }
    let mut rng_s: u64 = 0xcafebabe;
    let t0 = std::time::Instant::now();
    for i in 0..iters {
        rng_s = rng_s.wrapping_mul(6364136223846793005).wrapping_add(1);
        let tok = vec![(rng_s >> 33) as u32 % vocab; 1];
        infer_batch::forward_batch(&mdl, &batch, &tok, i);
        let _ = infer_batch::argmax_batch(&batch, 1);
    }
    let single_per = t0.elapsed().as_secs_f64() / iters as f64;
    eprintln!("  Single-token (B=1):    {:.2}ms = {:.0} tok/s", single_per * 1000.0, 1.0 / single_per);
    eprintln!("  Batch speedup: {:.1}x", fixed_tps * single_per);

    // 70B: 80 layers × 7 matmuls × avg(8192×28672) × 2 FLOPs ≈ 139.2 GFLOP/token
    let flop_per_token = 139.2e9_f64;

    // Scaling curve: test multiple batch sizes
    // Random tokens + argmax readback per iteration to defeat Metal caching
    eprintln!("\n=== Batch Scaling Curve (pos=0) ===\n");
    let mut rng_c: u64 = 0xf00dface;
    for &test_b in &[80u32, 160, 320, 480, 640, 800, 960, 1024] {
        // Warmup with random tokens
        rng_c = rng_c.wrapping_add(test_b as u64);
        let warmup_tok: Vec<u32> = (0..test_b).map(|_| {
            rng_c = rng_c.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_c >> 33) as u32 % vocab
        }).collect();
        infer_batch::forward_batch(&mdl, &batch, &warmup_tok, 0);
        let iters_s = 5u32;
        let t = std::time::Instant::now();
        for _ in 0..iters_s {
            let tok: Vec<u32> = (0..test_b).map(|_| {
                rng_c = rng_c.wrapping_mul(6364136223846793005).wrapping_add(1);
                (rng_c >> 33) as u32 % vocab
            }).collect();
            infer_batch::forward_batch(&mdl, &batch, &tok, 0);
            let _ = infer_batch::argmax_batch(&batch, test_b);
        }
        let per = t.elapsed().as_secs_f64() / iters_s as f64;
        let tps = test_b as f64 / per;
        let tflops = tps * flop_per_token / 1e12;
        let pct = tflops / 14.2 * 100.0;
        eprintln!("  B={:>4}: {:>8.2}ms = {:>4.0} tok/s | {:.2} TFLOP/s ({:.1}% peak)",
            test_b, per * 1000.0, tps, tflops, pct);
    }

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
    let vocab = gguf.config.vocab_size;
    let mut rng: u64 = 0xb10f1e;

    // Test both B=160 (peak efficiency) and B=640
    for &bs in &[160u32, 640] {
        let rand_tok = |rng: &mut u64| -> Vec<u32> {
            (0..bs).map(|_| {
                *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                (*rng >> 33) as u32 % vocab
            }).collect()
        };

        eprintln!("\n=== Per-Operation GPU Profiling (B={}, pos=0) ===\n", bs);

        // Warmup
        for _ in 0..3 {
            infer_batch::forward_batch(&mdl, &batch, &rand_tok(&mut rng), 0);
        }

        let iters = 3;
        let mut t_total = 0.0; let mut t_mm = 0.0; let mut t_attn = 0.0;
        let mut t_elem = 0.0; let mut t_out = 0.0;
        for _ in 0..iters {
            let tok = rand_tok(&mut rng);
            let (total, mm, attn, elem, out) =
                infer_batch::forward_profiled(&mdl, &batch, &tok, 0, bs);
            t_total += total; t_mm += mm; t_attn += attn; t_elem += elem; t_out += out;
        }
        let d = iters as f64;
        t_total /= d; t_mm /= d; t_attn /= d; t_elem /= d; t_out /= d;

        let tps = bs as f64 / (t_total / 1000.0);
        eprintln!("  Total:       {:.2}ms = {:.0} tok/s (serialized — overcounts due to per-group cmd bufs)", t_total, tps);
        eprintln!("  Matmul:      {:.2}ms ({:.1}%)", t_mm, t_mm / t_total * 100.0);
        eprintln!("  Attention:   {:.2}ms ({:.1}%)", t_attn, t_attn / t_total * 100.0);
        eprintln!("  Element-wise:{:.2}ms ({:.1}%)", t_elem, t_elem / t_total * 100.0);
        eprintln!("  Output proj: {:.2}ms ({:.1}%)", t_out, t_out / t_total * 100.0);

        // Compare with non-profiled (single cmd buf)
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let tok = rand_tok(&mut rng);
            infer_batch::forward_batch(&mdl, &batch, &tok, 0);
            let _ = infer_batch::argmax_batch(&batch, bs);
        }
        let real = t0.elapsed().as_secs_f64() / iters as f64 * 1000.0;
        let real_tps = bs as f64 / (real / 1000.0);
        eprintln!("\n  Non-profiled: {:.2}ms = {:.0} tok/s", real, real_tps);
        eprintln!("  Profiling overhead: {:.2}ms ({:.1}x)", t_total - real, t_total / real);

        let flop_per_token = 139.2e9_f64;
        let total_flop = flop_per_token * bs as f64;
        let achieved_tflops = total_flop / (real / 1000.0) / 1e12;
        eprintln!("  {:.2} GFLOP/token × {} = {:.1} TFLOP | {:.2} TFLOP/s ({:.1}% of 14.2 peak)",
            flop_per_token / 1e9, bs, total_flop / 1e12, achieved_tflops, achieved_tflops / 14.2 * 100.0);
    }

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

fn dump_vocab(path_arg: Option<&String>) {
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
        eprintln!("No .gguf file found."); std::process::exit(1);
    });
    let gguf = gguf::GGUFFile::open(&path).unwrap_or_else(|e| fatal(&e));

    // Dump tokenizer metadata keys
    eprintln!("\n--- Tokenizer metadata keys ---");
    for (k, v) in &gguf.metadata {
        if k.contains("tokenizer") {
            let desc = match v {
                gguf::MetaValue::Str(s) => format!("str: {}", &s[..s.len().min(80)]),
                gguf::MetaValue::U32(n) => format!("u32: {n}"),
                gguf::MetaValue::Array(a) => format!("array[{}]", a.len()),
                _ => format!("{:?}", v),
            };
            eprintln!("  {k}: {desc}");
        }
    }

    // Dump first 260 vocab entries + special tokens at end
    let vocab: Vec<String> = gguf.metadata.get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| {
            if let gguf::MetaValue::Str(s) = v { Some(s.clone()) } else { None }
        }).collect())
        .unwrap_or_default();

    eprintln!("\n--- Vocab (first 260 / last 300) total={} ---", vocab.len());
    for (i, t) in vocab.iter().enumerate().take(260) {
        eprintln!("  {:>6}: {:?}", i, t);
    }
    if vocab.len() > 127900 {
        eprintln!("\n  ... (skipping middle) ...\n");
        for (i, t) in vocab.iter().enumerate().skip(vocab.len() - 300) {
            eprintln!("  {:>6}: {:?}", i, t);
        }
    }

    // Dump merges count
    if let Some(gguf::MetaValue::Array(merges)) = gguf.metadata.get("tokenizer.ggml.merges") {
        eprintln!("\n--- Merges: {} total ---", merges.len());
        for (i, m) in merges.iter().take(20).enumerate() {
            if let gguf::MetaValue::Str(s) = m {
                eprintln!("  {:>4}: {:?}", i, s);
            }
        }
    } else {
        eprintln!("\nNo merges found in GGUF metadata");
    }
}

fn run_chat(path_arg: Option<&String>, prompt: Option<&str>) {
    let prompt = prompt.unwrap_or("What is 2+2? Answer in one sentence.");
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
        eprintln!("No .gguf file found."); std::process::exit(1);
    });
    eprintln!("Loading GGUF: {}", path.display());
    let gguf = gguf::GGUFFile::open(&path).unwrap_or_else(|e| fatal(&e));

    // Build vocab lookup
    let vocab: Vec<String> = gguf.metadata.get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| {
            if let gguf::MetaValue::Str(s) = v { Some(s.clone()) } else { None }
        }).collect())
        .unwrap_or_default();

    // Build reverse lookup: token_string -> id
    let mut token_to_id: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for (i, t) in vocab.iter().enumerate() {
        token_to_id.insert(t.clone(), i as u32);
    }

    // Load BPE merges
    let merges: Vec<(String, String)> = gguf.metadata.get("tokenizer.ggml.merges")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| {
            if let gguf::MetaValue::Str(s) = v {
                let parts: Vec<&str> = s.splitn(2, ' ').collect();
                if parts.len() == 2 { Some((parts[0].to_string(), parts[1].to_string())) }
                else { None }
            } else { None }
        }).collect())
        .unwrap_or_default();

    // Build merge rank lookup
    let merge_rank: std::collections::HashMap<(String, String), usize> = merges.iter()
        .enumerate().map(|(i, (a, b))| ((a.clone(), b.clone()), i)).collect();

    // GPT-2 byte-to-unicode mapping (Llama 3 uses this)
    let byte_to_unicode: Vec<char> = {
        let mut table = vec!['\0'; 256];
        // Printable ASCII (33-126): maps to themselves
        // Plus some Latin-1: 161-172, 174-255
        let mut bs: Vec<u8> = (b'!'..=b'~').collect();
        bs.extend(161u8..=172);
        bs.extend(174u8..=255);
        let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
        // All other bytes get mapped starting at 256
        let mut n = 0u32;
        for b in 0u8..=255 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        for (&b, &c) in bs.iter().zip(cs.iter()) {
            table[b as usize] = char::from_u32(c).unwrap_or('?');
        }
        table
    };

    // BPE encode a string
    let bpe_encode = |text: &str| -> Vec<u32> {
        // Convert bytes to GPT-2 unicode characters
        let mut tokens: Vec<String> = text.bytes().map(|b| {
            let c = byte_to_unicode[b as usize];
            let mut s = String::new();
            s.push(c);
            s
        }).collect();

        // Apply BPE merges greedily
        loop {
            let mut best_rank = usize::MAX;
            let mut best_idx = 0;
            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(&rank) = merge_rank.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }
            if best_rank == usize::MAX { break; }
            let merged = format!("{}{}", tokens[best_idx], tokens[best_idx + 1]);
            tokens[best_idx] = merged;
            tokens.remove(best_idx + 1);
        }

        tokens.iter().map(|t| {
            *token_to_id.get(t).unwrap_or(&0)
        }).collect()
    };

    // Find special token IDs
    let find_special = |name: &str| -> u32 {
        *token_to_id.get(name).unwrap_or(&0)
    };

    let bos = gguf.config.bos_token;
    let start_header = find_special("<|start_header_id|>");
    let end_header = find_special("<|end_header_id|>");
    let eot = find_special("<|eot_id|>");
    let eos = gguf.config.eos_token;

    eprintln!("Special tokens: BOS={bos} start_header={start_header} end_header={end_header} eot={eot} eos={eos}");

    // Build Llama 3.3 chat format:
    // <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>
    // <|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>
    // <|start_header_id|>assistant<|end_header_id|>\n\n
    let mut tokens: Vec<u32> = Vec::new();
    tokens.push(bos);
    tokens.push(start_header);
    tokens.extend(bpe_encode("system"));
    tokens.push(end_header);
    tokens.extend(bpe_encode("\n\nYou are a helpful assistant."));
    tokens.push(eot);
    tokens.push(start_header);
    tokens.extend(bpe_encode("user"));
    tokens.push(end_header);
    tokens.extend(bpe_encode(&format!("\n\n{prompt}")));
    tokens.push(eot);
    tokens.push(start_header);
    tokens.extend(bpe_encode("assistant"));
    tokens.push(end_header);
    tokens.extend(bpe_encode("\n\n"));

    eprintln!("Prompt tokens ({}):", tokens.len());
    for (i, &t) in tokens.iter().enumerate() {
        let name = if (t as usize) < vocab.len() { &vocab[t as usize] } else { "?" };
        eprintln!("  [{i:>3}] {t:>6} = {:?}", name);
    }

    // Load model and run inference
    let mdl = model::Model::load(&gguf).unwrap_or_else(|e| fatal(&e));
    let batch = infer_batch::BatchState::new(&mdl).unwrap_or_else(|e| fatal(&e));

    eprintln!("\n--- Prefill ({} tokens) ---", tokens.len());
    let t0 = std::time::Instant::now();
    let pos = infer_batch::prefill(&mdl, &batch, &tokens);
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Prefill: {:.0}ms ({:.0} tok/s)", prefill_ms, tokens.len() as f64 / t0.elapsed().as_secs_f64());

    // Decode
    eprintln!("\n--- Generating ---\n");
    let first = infer_batch::argmax_one(&batch);
    let mut generated = vec![first];
    let mut next = first;
    let max_gen = 256u32;
    let t1 = std::time::Instant::now();
    for i in 0..max_gen - 1 {
        next = infer_batch::decode_step(&mdl, &batch, next, pos + i);
        if next == eos || next == eot { break; }
        generated.push(next);
    }
    let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let decode_tps = generated.len() as f64 / t1.elapsed().as_secs_f64();

    // Decode to text using GPT-2 unicode-to-byte mapping
    let unicode_to_byte: std::collections::HashMap<char, u8> = byte_to_unicode.iter()
        .enumerate().map(|(b, &c)| (c, b as u8)).collect();
    let decode_token = |t: u32| -> String {
        if (t as usize) >= vocab.len() { return format!("[{t}]"); }
        let tok_str = &vocab[t as usize];
        // Special tokens
        if tok_str.starts_with("<|") { return String::new(); }
        // Convert GPT-2 unicode chars back to bytes
        let bytes: Vec<u8> = tok_str.chars().map(|c| {
            *unicode_to_byte.get(&c).unwrap_or(&b'?')
        }).collect();
        String::from_utf8_lossy(&bytes).to_string()
    };
    let text: String = generated.iter().map(|&t| decode_token(t)).collect();
    println!("{text}");

    eprintln!("\n--- Stats ---");
    eprintln!("  Generated: {} tokens in {:.0}ms = {:.1} tok/s", generated.len(), decode_ms, decode_tps);
}

fn fatal(msg: &str) -> ! {
    eprintln!("helix-bg: {msg}");
    std::process::exit(1);
}

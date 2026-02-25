//! helix-bg: background intelligence daemon for helix knowledge base.
//! Keeps 70B Llama GPU-resident. Watches KB for changes.
//! Core functions: session synthesis, terminology drift, stale knowledge.

mod datalog;
mod bpe;
mod analyze;
mod daemon;
mod gpu;
mod gguf;
mod model;
mod kernels_batch;
mod infer_batch;

use std::path::PathBuf;

extern "C" { fn setpriority(which: i32, who: u32, prio: i32) -> i32; }

fn main() {
    unsafe {
        setpriority(5, 0, 7); // PRIO_DARWIN_GPU = UI_FOCAL
        setpriority(7, 0, 1); // PRIO_DARWIN_GAME_MODE = ON
    }
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { usage(); }
    match args[1].as_str() {
        "serve" => daemon::serve(args.get(2)),
        "query" => {
            let cmd = args.get(2).map(|s| s.as_str()).unwrap_or("STATUS");
            let rest: Vec<&str> = args[3..].iter().map(|s| s.as_str()).collect();
            daemon::query(cmd, &rest);
        }
        "chat" => run_chat(args.get(2), args.get(3).map(|s| s.as_str())),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("helix-bg — background intelligence for helix knowledge base\n");
    eprintln!("  helix-bg serve [model-path]       Start daemon (70B GPU-resident)");
    eprintln!("  helix-bg query <cmd> [args...]     Query daemon");
    eprintln!("    STATUS                           Check daemon status");
    eprintln!("    CHAT <system> <user> [max_tok]   Chat with model");
    eprintln!("    SYNTHESIZE [topic]               Session continuity synthesis");
    eprintln!("    DRIFT                            Terminology drift detection");
    eprintln!("    STALE [topic]                    Stale knowledge flagging");
    eprintln!("  helix-bg chat [model] [prompt]     Direct chat (loads model)");
    std::process::exit(1);
}

fn run_chat(path_arg: Option<&String>, prompt: Option<&str>) {
    let prompt = prompt.unwrap_or("What is 2+2? Answer in one sentence.");
    let path = resolve_model_path(path_arg);
    eprintln!("Loading GGUF: {}", path.display());
    let gguf = gguf::GGUFFile::open(&path).unwrap_or_else(|e| fatal(&e));
    let tok = bpe::BpeTokenizer::from_gguf(&gguf).unwrap_or_else(|e| fatal(&e));
    let tokens = tok.format_chat("You are a helpful assistant.", prompt);
    eprintln!("Prompt: {} tokens", tokens.len());
    let mdl = model::Model::load(&gguf).unwrap_or_else(|e| fatal(&e));
    let batch = infer_batch::BatchState::new(&mdl).unwrap_or_else(|e| fatal(&e));
    let t0 = std::time::Instant::now();
    let pos = infer_batch::prefill(&mdl, &batch, &tokens);
    eprintln!("Prefill: {:.0}ms ({:.0} tok/s)", t0.elapsed().as_secs_f64() * 1000.0,
        tokens.len() as f64 / t0.elapsed().as_secs_f64());
    let first = infer_batch::argmax_one(&batch);
    let mut generated = vec![first];
    let mut next = first;
    let t1 = std::time::Instant::now();
    for i in 0..255 {
        next = infer_batch::decode_step(&mdl, &batch, next, pos + i);
        if next == tok.eos || next == tok.eot { break; }
        generated.push(next);
    }
    let decode_tps = generated.len() as f64 / t1.elapsed().as_secs_f64();
    println!("{}", tok.decode(&generated));
    eprintln!("\n--- {} tokens, {:.1} tok/s ---", generated.len(), decode_tps);
}

pub fn resolve_model_path(arg: Option<&String>) -> PathBuf {
    if let Some(s) = arg { return PathBuf::from(s); }
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
}

pub fn resolve_kb_dir(arg: Option<&String>) -> PathBuf {
    if let Some(d) = arg { return PathBuf::from(d); }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    for sub in &[".helix-kb", ".helix", ".local/share/helix"] {
        let p = PathBuf::from(&home).join(sub);
        if p.join("data.log").exists() { return p; }
    }
    PathBuf::from(".")
}

pub fn mtime(path: &std::path::Path) -> u64 {
    std::fs::metadata(path).ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs()).unwrap_or(0)
}

pub fn fatal(msg: &str) -> ! {
    eprintln!("helix-bg: {msg}");
    std::process::exit(1);
}

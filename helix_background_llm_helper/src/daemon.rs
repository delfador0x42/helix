//! Persistent daemon: loads 70B model once, keeps it GPU-resident.
//! Watches data.log, auto-runs analysis on session boundaries.
//! Auto-schedules drift detection (hourly) and stale flagging (6h).
//! Protocol: client sends request line, server streams text + stats.
//!
//! Request format (tab-delimited):
//!   STATUS
//!   CHAT\t<system>\t<user>\t<max_tokens>
//!   SYNTHESIZE\t[topic_filter]\t[max_tokens]
//!   DRIFT\t[max_tokens]
//!   STALE\t[topic_filter]\t[max_tokens]
//! Response: streamed UTF-8 text, then \0 byte, then stats line.

use std::io::{Read, Write, BufRead, BufReader};
use std::os::unix::net::{UnixListener, UnixStream};
use std::time::{Duration, Instant};
use crate::{bpe, datalog, gguf, model, infer_batch, analyze};

const SOCK_NAME: &str = "helix-bg.sock";
const SESSION_GAP_SECS: u64 = 300;     // 5min idle = session boundary
const AUTO_SYNTH_THRESHOLD: usize = 20; // auto-synthesize after 20 new entries
const DRIFT_INTERVAL_SECS: u64 = 3600;  // auto-drift every hour
const STALE_INTERVAL_SECS: u64 = 21600; // auto-stale every 6 hours
const REP_PENALTY: f32 = 1.3;           // repetition penalty base (^freq)
const TEMPERATURE: f32 = 0.7;           // sampling temperature (lower = more focused)
const TOP_P: f32 = 0.9;                 // nucleus sampling threshold

pub fn sock_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(home).join(".helix").join(SOCK_NAME)
}

pub fn serve(model_arg: Option<&String>) {
    let model_path = crate::resolve_model_path(model_arg);
    let kb_dir = crate::resolve_kb_dir(None);
    let log_path = kb_dir.join("data.log");
    let vocab_size = {
        let gguf = gguf::GGUFFile::open(&model_path).unwrap_or_else(|e| crate::fatal(&e));
        gguf.config.vocab_size
    };

    eprintln!("daemon: loading model...");
    let gguf = gguf::GGUFFile::open(&model_path).unwrap_or_else(|e| crate::fatal(&e));
    let tok = bpe::BpeTokenizer::from_gguf(&gguf).unwrap_or_else(|e| crate::fatal(&e));
    let mdl = model::Model::load(&gguf).unwrap_or_else(|e| crate::fatal(&e));
    let batch = infer_batch::BatchState::new(&mdl).unwrap_or_else(|e| crate::fatal(&e));
    eprintln!("daemon: model loaded, 70B GPU-resident, vocab={}", vocab_size);

    let entry_count = datalog::read_entries(&log_path).map(|e| e.len()).unwrap_or(0);
    let mut kb_mtime = crate::mtime(&log_path);
    let mut last_offset = max_offset(&log_path);
    let mut last_change = Instant::now();
    let mut new_since_synth: usize = 0;
    let mut session_active = false;
    let mut last_drift = Instant::now();
    let mut last_stale = Instant::now();

    // Bind unix socket
    let sp = sock_path();
    let _ = std::fs::remove_file(&sp);
    let _ = std::fs::create_dir_all(sp.parent().unwrap());
    let listener = UnixListener::bind(&sp).unwrap_or_else(|e| {
        crate::fatal(&format!("bind {}: {e}", sp.display()))
    });
    listener.set_nonblocking(true).ok();
    eprintln!("daemon: listening on {}", sp.display());
    eprintln!("daemon: {} KB entries, watching for changes", entry_count);

    loop {
        // Handle incoming connections (drain all pending)
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false).ok();
                    handle_connection(&mdl, &batch, &tok, &log_path, vocab_size, stream);
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) => { eprintln!("daemon: accept error: {e}"); break; }
            }
        }

        // Watch for KB changes — stat() is ~1μs on APFS
        let now_mtime = crate::mtime(&log_path);
        if now_mtime != kb_mtime {
            kb_mtime = now_mtime;
            let cur_offset = max_offset(&log_path);
            if cur_offset > last_offset {
                let n = count_new_entries(&log_path, last_offset);
                new_since_synth += n;
                last_offset = cur_offset;
                last_change = Instant::now();
                session_active = true;
                eprintln!("daemon: +{} entries ({} new since last synthesis)", n, new_since_synth);
            }
        }

        // Session boundary: auto-synthesize
        if session_active && last_change.elapsed() > Duration::from_secs(SESSION_GAP_SECS) {
            session_active = false;
            if new_since_synth >= AUTO_SYNTH_THRESHOLD {
                eprintln!("daemon: session end, auto-synthesizing {} new entries", new_since_synth);
                auto_run(&mdl, &batch, &tok, &log_path, vocab_size, "synthesis", last_offset);
                new_since_synth = 0;
            }
        }

        // Periodic auto-drift (hourly)
        if last_drift.elapsed() > Duration::from_secs(DRIFT_INTERVAL_SECS) {
            last_drift = Instant::now();
            eprintln!("daemon: scheduled drift detection");
            auto_run(&mdl, &batch, &tok, &log_path, vocab_size, "drift", 0);
        }

        // Periodic auto-stale (6h)
        if last_stale.elapsed() > Duration::from_secs(STALE_INTERVAL_SECS) {
            last_stale = Instant::now();
            eprintln!("daemon: scheduled stale detection");
            auto_run(&mdl, &batch, &tok, &log_path, vocab_size, "stale", 0);
        }

        std::thread::sleep(Duration::from_secs(1));
    }
}

fn handle_connection(
    mdl: &model::Model, batch: &infer_batch::BatchState,
    tok: &bpe::BpeTokenizer, log_path: &std::path::Path,
    vocab_size: u32, stream: UnixStream,
) {
    let mut reader = BufReader::new(&stream);
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() { return; }
    let line = line.trim_end();
    let parts: Vec<&str> = line.split('\t').collect();
    if parts.is_empty() { return; }

    match parts[0] {
        "STATUS" => {
            let n = datalog::read_entries(log_path).map(|e| e.len()).unwrap_or(0);
            let _ = (&stream).write_all(
                format!("ready\t{} entries\t70B GPU-resident\n", n).as_bytes()
            );
        }
        "CHAT" => {
            let system = parts.get(1).unwrap_or(&"You are a helpful assistant.");
            let user = parts.get(2).unwrap_or(&"Hello");
            let max_tok: u32 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
            let tokens = tok.format_chat(system, user);
            generate_and_stream(mdl, batch, tok, &tokens, max_tok, vocab_size, &stream);
        }
        "SYNTHESIZE" => {
            let topic = parts.get(1).filter(|s| !s.is_empty());
            let max_tok: u32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
            run_analysis(mdl, batch, tok, log_path, "synthesize", topic.copied(), max_tok, vocab_size, &stream);
        }
        "DRIFT" => {
            let max_tok: u32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
            run_analysis(mdl, batch, tok, log_path, "drift", None, max_tok, vocab_size, &stream);
        }
        "STALE" => {
            let topic = parts.get(1).filter(|s| !s.is_empty());
            let max_tok: u32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);
            run_analysis(mdl, batch, tok, log_path, "stale", topic.copied(), max_tok, vocab_size, &stream);
        }
        _ => { let _ = (&stream).write_all(b"error\tunknown command\n"); }
    }
}

fn run_analysis(
    mdl: &model::Model, batch: &infer_batch::BatchState,
    tok: &bpe::BpeTokenizer, log_path: &std::path::Path,
    mode: &str, topic: Option<&str>, max_tok: u32,
    vocab_size: u32, mut stream: &UnixStream,
) {
    let entries = datalog::read_entries(log_path).unwrap_or_default();
    let (system, prompt) = match mode {
        "synthesize" => {
            let last = max_offset(log_path).saturating_sub(1);
            match analyze::session_synthesis_prompt(&entries, last) {
                Some(p) => (analyze::SYNTHESIS_SYSTEM, p),
                None => { let _ = (stream).write_all(b"no new entries\n"); return; }
            }
        }
        "drift" => match analyze::drift_prompt(&entries) {
            Some(p) => (analyze::DRIFT_SYSTEM, p),
            None => { let _ = (stream).write_all(b"not enough topics\n"); return; }
        }
        "stale" => match analyze::stale_prompt(&entries, topic) {
            Some(p) => (analyze::STALE_SYSTEM, p),
            None => { let _ = (stream).write_all(b"not enough entries\n"); return; }
        }
        _ => unreachable!(),
    };
    let tokens = tok.format_chat(system, &prompt);
    eprintln!("daemon: {} analysis, {} prompt tokens", mode, tokens.len());
    generate_and_stream(mdl, batch, tok, &tokens, max_tok, vocab_size, stream);
}

/// Auto-run analysis and write result to insights.log.
fn auto_run(
    mdl: &model::Model, batch: &infer_batch::BatchState,
    tok: &bpe::BpeTokenizer, log_path: &std::path::Path,
    vocab_size: u32, mode: &str, synth_offset: u32,
) {
    let entries = datalog::read_entries(log_path).unwrap_or_default();
    let (system, prompt) = match mode {
        "synthesis" => match analyze::session_synthesis_prompt(&entries, synth_offset) {
            Some(p) => (analyze::SYNTHESIS_SYSTEM, p),
            None => return,
        },
        "drift" => match analyze::drift_prompt(&entries) {
            Some(p) => (analyze::DRIFT_SYSTEM, p),
            None => return,
        },
        "stale" => match analyze::stale_prompt(&entries, None) {
            Some(p) => (analyze::STALE_SYSTEM, p),
            None => return,
        },
        _ => return,
    };
    let tokens = tok.format_chat(system, &prompt);
    eprintln!("daemon: auto-{}, {} prompt tokens", mode, tokens.len());
    let text = generate_to_string(mdl, batch, tok, &tokens, 1024, vocab_size);
    // Append to insights.log
    let kb_dir = crate::resolve_kb_dir(None);
    let path = kb_dir.join("insights.log");
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs()).unwrap_or(0);
    let entry = format!("\n--- {} [ts={}] ---\n{}\n", mode, ts, text);
    let _ = std::fs::OpenOptions::new().create(true).append(true)
        .open(&path).and_then(|mut f| f.write_all(entry.as_bytes()));
    eprintln!("daemon: auto-{} written to insights.log", mode);
}

/// Generate text to a String (for auto-analysis). Top-p sampling + freq penalty + loop detect.
fn generate_to_string(
    mdl: &model::Model, batch: &infer_batch::BatchState,
    tok: &bpe::BpeTokenizer, tokens: &[u32], max_tok: u32, vocab_size: u32,
) -> String {
    let t0 = Instant::now();
    let pos = infer_batch::prefill(mdl, batch, tokens);
    let first = infer_batch::argmax_one(batch);
    let mut generated = vec![first];
    let mut next = first;
    let mut freq = vec![0u16; vocab_size as usize];
    infer_batch::freq_inc(&mut freq, first);
    let mut rng = t0.elapsed().as_nanos() as u64 | 1; // seed from clock, ensure odd
    let mut tail = [0u8; 16];
    let mut tail_len = 0usize;
    for i in 0..max_tok.saturating_sub(1) {
        next = infer_batch::decode_step_sampled(
            mdl, batch, next, pos + i, &freq, vocab_size,
            REP_PENALTY, TEMPERATURE, TOP_P, &mut rng,
        );
        if next == tok.eos || next == tok.eot { break; }
        let w = tok.decode_token(next);
        for &b in w.as_bytes() {
            if tail_len < 16 { tail[tail_len] = b; tail_len += 1; }
            else { tail.rotate_left(1); tail[15] = b; }
        }
        if tail_len == 16 && tail.iter().all(|&b| b == tail[0]) {
            eprintln!("daemon: char loop detected, stopping");
            break;
        }
        generated.push(next);
        infer_batch::freq_inc(&mut freq, next);
    }
    let tps = generated.len() as f64 / t0.elapsed().as_secs_f64();
    eprintln!("daemon: {} tokens, {:.1} tok/s", generated.len(), tps);
    tok.decode(&generated)
}

/// Generate text and stream over socket. Top-p sampling + freq penalty + loop detect.
fn generate_and_stream(
    mdl: &model::Model, batch: &infer_batch::BatchState,
    tok: &bpe::BpeTokenizer, tokens: &[u32], max_tok: u32,
    vocab_size: u32, mut stream: &UnixStream,
) {
    let t0 = Instant::now();
    let pos = infer_batch::prefill(mdl, batch, tokens);
    eprintln!("daemon: prefill {:.0}ms ({:.0} tok/s)",
        t0.elapsed().as_secs_f64() * 1000.0,
        tokens.len() as f64 / t0.elapsed().as_secs_f64());
    let first = infer_batch::argmax_one(batch);
    let mut generated = vec![first];
    let mut next = first;
    let mut freq = vec![0u16; vocab_size as usize];
    infer_batch::freq_inc(&mut freq, first);
    let mut rng = t0.elapsed().as_nanos() as u64 | 1;
    let mut tail = [0u8; 16];
    let mut tail_len = 0usize;
    let t1 = Instant::now();
    let w = tok.decode_token(first);
    for &b in w.as_bytes() {
        if tail_len < 16 { tail[tail_len] = b; tail_len += 1; }
        else { tail.rotate_left(1); tail[15] = b; }
    }
    if !w.is_empty() { let _ = stream.write_all(w.as_bytes()); }
    for i in 0..max_tok.saturating_sub(1) {
        next = infer_batch::decode_step_sampled(
            mdl, batch, next, pos + i, &freq, vocab_size,
            REP_PENALTY, TEMPERATURE, TOP_P, &mut rng,
        );
        if next == tok.eos || next == tok.eot { break; }
        let w = tok.decode_token(next);
        for &b in w.as_bytes() {
            if tail_len < 16 { tail[tail_len] = b; tail_len += 1; }
            else { tail.rotate_left(1); tail[15] = b; }
        }
        if tail_len == 16 && tail.iter().all(|&b| b == tail[0]) {
            eprintln!("daemon: char loop detected, stopping");
            break;
        }
        generated.push(next);
        infer_batch::freq_inc(&mut freq, next);
        if !w.is_empty() { let _ = stream.write_all(w.as_bytes()); }
    }
    let decode_tps = generated.len() as f64 / t1.elapsed().as_secs_f64();
    let _ = stream.write_all(b"\0");
    let _ = stream.write_all(
        format!("{}\t{:.1}\n", generated.len(), decode_tps).as_bytes()
    );
    eprintln!("daemon: {} tokens, {:.1} tok/s", generated.len(), decode_tps);
}

fn max_offset(log_path: &std::path::Path) -> u32 {
    datalog::read_entries(log_path).ok()
        .and_then(|e| e.last().map(|e| e.offset))
        .unwrap_or(0)
}

fn count_new_entries(log_path: &std::path::Path, after: u32) -> usize {
    datalog::read_entries(log_path).ok()
        .map(|e| e.iter().filter(|e| e.offset > after).count())
        .unwrap_or(0)
}

/// Send a query to the running daemon.
pub fn query(cmd: &str, args: &[&str]) {
    let sp = sock_path();
    let mut stream = UnixStream::connect(&sp).unwrap_or_else(|e| {
        eprintln!("helix-bg: cannot connect to daemon at {}: {e}", sp.display());
        eprintln!("helix-bg: start daemon with: helix-bg serve [model-path]");
        std::process::exit(1);
    });
    let mut req = String::from(cmd);
    for a in args { req.push('\t'); req.push_str(a); }
    req.push('\n');
    stream.write_all(req.as_bytes()).unwrap();
    let mut buf = [0u8; 4096];
    let mut got_null = false;
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let chunk = &buf[..n];
                if let Some(pos) = chunk.iter().position(|&b| b == 0) {
                    print!("{}", String::from_utf8_lossy(&chunk[..pos]));
                    let stats = String::from_utf8_lossy(&chunk[pos+1..]);
                    if !stats.trim().is_empty() {
                        let parts: Vec<&str> = stats.trim().split('\t').collect();
                        if parts.len() >= 2 {
                            eprintln!("\n--- {} tokens, {} tok/s ---", parts[0], parts[1]);
                        }
                    }
                    got_null = true;
                } else if !got_null {
                    print!("{}", String::from_utf8_lossy(chunk));
                } else {
                    let stats = String::from_utf8_lossy(chunk);
                    let parts: Vec<&str> = stats.trim().split('\t').collect();
                    if parts.len() >= 2 {
                        eprintln!("\n--- {} tokens, {} tok/s ---", parts[0], parts[1]);
                    }
                }
            }
            Err(_) => break,
        }
    }
    println!();
}

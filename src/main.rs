mod fxhash;
mod lock;
mod time;
mod json;
mod config;
mod text;
mod format;
mod datalog;
mod cache;
mod index;
mod search;
mod brief;
mod write;
mod mcp;
mod hook;
mod sock;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { usage(); }
    match args[1].as_str() {
        "serve" => {
            let dir = config::resolve_dir(args.get(2).cloned());
            let _sock = sock::start_listener(&dir);
            if let Err(e) = mcp::run(&dir) { eprintln!("helix: {e}"); std::process::exit(1); }
        }
        "hook" => {
            let htype = match args.get(2) { Some(s) => s.as_str(), None => usage() };
            let dir = config::resolve_dir(args.get(3).cloned());
            match hook::run(htype, &dir) {
                Ok(output) => { if !output.is_empty() { print!("{output}"); } }
                Err(e) => eprintln!("helix hook: {e}"),
            }
        }
        "hooks" => {
            let action = args.get(2).map(|s| s.as_str()).unwrap_or("status");
            let result = match action {
                "install" => hook::install_hooks(),
                "uninstall" => hook::uninstall_hooks(),
                "status" | _ => hook::hooks_status(),
            };
            match result {
                Ok(msg) => println!("{msg}"),
                Err(e) => { eprintln!("helix hooks: {e}"); std::process::exit(1); }
            }
        }
        "index" => {
            let dir = config::resolve_dir(args.get(2).cloned());
            match index::rebuild(&dir, true) {
                Ok((msg, _)) => println!("{msg}"),
                Err(e) => { eprintln!("helix index: {e}"); std::process::exit(1); }
            }
        }
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("helix — knowledge store for LLM agents\n");
    eprintln!("  helix serve [dir]           Start MCP server (stdio JSON-RPC)");
    eprintln!("  helix hook <type> [dir]     Run hook handler");
    eprintln!("  helix hooks install         Install Claude Code hooks");
    eprintln!("  helix hooks uninstall       Remove Claude Code hooks");
    eprintln!("  helix hooks status          Show hook configuration");
    eprintln!("  helix index [dir]           Rebuild search index");
    std::process::exit(1);
}

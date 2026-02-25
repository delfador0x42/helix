//! Hook management — install, uninstall, status for Claude Code hooks.

/// Generate hooks JSON config for all 9 hooks.
pub fn hooks_config(binary: &str) -> String {
    let mut b = String::with_capacity(binary.len() + 10);
    crate::json::escape_into(binary, &mut b);
    format!(concat!(
        "{{",
        "\"SessionStart\":[{{\"matcher\":\"\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook session\",\"timeout\":5}}]}}],",
        "\"UserPromptSubmit\":[{{\"matcher\":\"\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook prompt\",\"timeout\":5}}]}}],",
        "\"PreToolUse\":[{{\"matcher\":\"Read|Edit|Write|Glob|Grep|NotebookEdit\",",
            "\"hooks\":[{{\"type\":\"command\",\"command\":\"{b} hook ambient\",\"timeout\":5}}]}}],",
        "\"PostToolUse\":[{{\"matcher\":\"Bash\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook post-build\",\"async\":true,\"timeout\":5}}]}}],",
        "\"PostToolUseFailure\":[{{\"matcher\":\"Bash\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook error-context\",\"timeout\":5}}]}}],",
        "\"PreCompact\":[{{\"matcher\":\"\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook pre-compact\",\"timeout\":5}}]}}],",
        "\"Stop\":[{{\"matcher\":\"\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook stop\",\"timeout\":5}}]}}],",
        "\"SubagentStart\":[{{\"matcher\":\"\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook subagent\",\"timeout\":5}}]}}],",
        "\"PermissionRequest\":[{{\"matcher\":\"mcp__helix__.*\",\"hooks\":[{{\"type\":\"command\",",
            "\"command\":\"{b} hook approve-mcp\"}}]}}]",
        "}}"),
        b = b)
}

/// Install helix hooks into ~/.claude/settings.json.
pub fn install_hooks() -> Result<String, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let settings_dir = format!("{home}/.claude");
    let settings_path = format!("{settings_dir}/settings.json");
    let binary = std::env::current_exe().map_err(|e| format!("exe path: {e}"))?
        .to_string_lossy().to_string();
    std::fs::create_dir_all(&settings_dir).map_err(|e| e.to_string())?;
    let mut settings = if std::path::Path::new(&settings_path).exists() {
        let content = std::fs::read_to_string(&settings_path).map_err(|e| e.to_string())?;
        crate::json::parse(&content)?
    } else {
        crate::json::Value::Obj(vec![])
    };
    let hooks_json = hooks_config(&binary);
    let hooks_value = crate::json::parse(&hooks_json)?;
    match &mut settings {
        crate::json::Value::Obj(entries) => {
            if let Some(pos) = entries.iter().position(|(k, _)| k == "hooks") {
                entries[pos].1 = hooks_value;
            } else {
                entries.push(("hooks".to_string(), hooks_value));
            }
        }
        _ => return Err("settings.json root is not an object".into()),
    }
    std::fs::write(&settings_path, format!("{settings}")).map_err(|e| e.to_string())?;
    Ok(format!("Installed 9 helix hooks to {settings_path}\n\
        Hooks: session, prompt, ambient, post-build, error-context, pre-compact, stop, subagent, approve-mcp\n\
        Restart Claude Code or open /hooks to activate."))
}

/// Remove helix hooks from ~/.claude/settings.json.
pub fn uninstall_hooks() -> Result<String, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let settings_path = format!("{home}/.claude/settings.json");
    if !std::path::Path::new(&settings_path).exists() { return Ok("No settings.json found".into()); }
    let content = std::fs::read_to_string(&settings_path).map_err(|e| e.to_string())?;
    let mut settings = crate::json::parse(&content)?;
    match &mut settings {
        crate::json::Value::Obj(entries) => { entries.retain(|(k, _)| k != "hooks"); }
        _ => {}
    }
    std::fs::write(&settings_path, format!("{settings}")).map_err(|e| e.to_string())?;
    Ok(format!("Removed hooks from {settings_path}"))
}

/// Show current hooks configuration.
pub fn hooks_status() -> Result<String, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let settings_path = format!("{home}/.claude/settings.json");
    if !std::path::Path::new(&settings_path).exists() { return Ok("No settings.json found".into()); }
    let content = std::fs::read_to_string(&settings_path).map_err(|e| e.to_string())?;
    let settings = crate::json::parse(&content)?;
    let hooks = match settings.get("hooks") {
        Some(h) => h, None => return Ok("No hooks configured".into()),
    };
    let events = match hooks {
        crate::json::Value::Obj(pairs) => pairs,
        _ => return Ok("Invalid hooks section".into()),
    };
    let mut out = String::with_capacity(256);
    crate::text::itoa_push(&mut out, events.len() as u32);
    out.push_str(" hook events configured:\n");
    for (event, _) in events { out.push_str("  "); out.push_str(event); out.push('\n'); }
    Ok(out)
}

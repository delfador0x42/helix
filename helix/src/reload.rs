//! Binary reload: build, deploy, verify KB, re-exec.

use std::path::Path;

/// Build, deploy binaries, verify KB, return report. Called before re-exec.
pub(crate) fn reload_verify(dir: &Path) -> String {
    let mut report = Vec::new();
    let mut ok_count = 0u32;
    let mut fail_count = 0u32;
    let home = std::env::var("HOME").unwrap_or_default();
    let src = std::path::PathBuf::from(&home)
        .join("wudan/dojo/crash3/llm_double_helix/helix/target/release/helix");

    // 1. Update MCP server binary (self)
    let exe = std::env::current_exe().unwrap_or_default();
    if src.exists() {
        let tmp = exe.with_extension("tmp");
        if std::fs::copy(&src, &tmp).is_ok() {
            if std::fs::rename(&tmp, &exe).is_ok() {
                let _ = std::process::Command::new("codesign")
                    .args(["-s", "-", "-f"]).arg(&exe).output();
                report.push("OK  MCP server binary updated + codesigned".to_string());
                ok_count += 1;
            } else {
                let _ = std::fs::remove_file(&tmp);
                report.push("FAIL  MCP server binary rename failed".to_string());
                fail_count += 1;
            }
        } else {
            report.push("FAIL  MCP server binary copy failed".to_string());
            fail_count += 1;
        }
    } else {
        report.push(format!("FAIL  release binary not found: {}", src.display()));
        fail_count += 1;
    }

    // 2. Update hooks binary (~/.local/bin/helix)
    let hooks_bin = std::path::PathBuf::from(&home).join(".local/bin/helix");
    if src.exists() {
        let hooks_tmp = hooks_bin.with_extension("tmp");
        if std::fs::copy(&src, &hooks_tmp).is_ok() {
            if std::fs::rename(&hooks_tmp, &hooks_bin).is_ok() {
                let _ = std::process::Command::new("codesign")
                    .args(["-s", "-", "-f"]).arg(&hooks_bin).output();
                report.push("OK  hooks binary updated + codesigned".to_string());
                ok_count += 1;
            } else {
                let _ = std::fs::remove_file(&hooks_tmp);
                report.push("FAIL  hooks binary rename failed".to_string());
                fail_count += 1;
            }
        } else {
            report.push("FAIL  hooks binary copy failed".to_string());
            fail_count += 1;
        }
    }

    // 3. Fix MCP config
    fix_mcp_config(&home, &src);

    // 4. Verify hooks binary runs
    match std::process::Command::new(&hooks_bin).arg("--help").output() {
        Ok(out) => {
            let combined = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
            if combined.contains("helix") {
                report.push("OK  hooks binary executes".to_string()); ok_count += 1;
            } else {
                report.push(format!("FAIL  hooks binary unexpected output: {}", &combined[..combined.len().min(80)])); fail_count += 1;
            }
        }
        Err(e) => { report.push(format!("FAIL  hooks binary: {e}")); fail_count += 1; }
    }

    // 5. Verify data.log readable
    let log_path = crate::config::log_path(dir);
    match crate::datalog::iter_live(&log_path) {
        Ok(entries_vec) => {
            let mut topics = std::collections::HashSet::new();
            for e in &entries_vec { topics.insert(e.topic.as_str()); }
            report.push(format!("OK  data.log: {} entries, {} topics", entries_vec.len(), topics.len()));
            ok_count += 1;
        }
        Err(e) => { report.push(format!("FAIL  data.log: {e}")); fail_count += 1; }
    }

    // 6. Verify index builds
    match crate::index::rebuild(dir, true) {
        Ok((count, bytes)) => {
            report.push(format!("OK  index: {count} entries, {} bytes", bytes.len()));
            crate::mcp::store_index(bytes);
            ok_count += 1;
        }
        Err(e) => { report.push(format!("FAIL  index rebuild: {e}")); fail_count += 1; }
    }

    // 7. Verify search works
    let filter = crate::index::Filter::none();
    let terms = vec!["helix".to_string()];
    match crate::index::search_scored(dir, &terms, &filter, Some(3), None, false) {
        Ok((hits, _)) => { report.push(format!("OK  search: {} hits for 'helix'", hits.len())); ok_count += 1; }
        Err(e) => { report.push(format!("FAIL  search: {e}")); fail_count += 1; }
    }

    // 8. Verify hooks config
    let settings_path = format!("{home}/.claude/settings.json");
    match std::fs::read_to_string(&settings_path) {
        Ok(content) => {
            let hook_count = content.matches("helix hook").count();
            if hook_count >= 8 { report.push(format!("OK  settings.json: {hook_count} hook commands")); ok_count += 1; }
            else { report.push(format!("WARN  settings.json: only {hook_count} hook commands")); fail_count += 1; }
        }
        Err(e) => { report.push(format!("FAIL  settings.json: {e}")); fail_count += 1; }
    }

    // 9. Verify MCP config
    let claude_config = std::path::PathBuf::from(&home).join(".claude.json");
    match std::fs::read_to_string(&claude_config) {
        Ok(content) => {
            let expected = src.to_string_lossy();
            if content.contains(expected.as_ref()) { report.push(format!("OK  .claude.json: helix MCP → {}", expected)); ok_count += 1; }
            else if content.contains("helix") { report.push("WARN  .claude.json: helix entry exists but wrong binary path".into()); fail_count += 1; }
            else { report.push("FAIL  .claude.json: no helix MCP server configured".into()); fail_count += 1; }
        }
        Err(e) => { report.push(format!("FAIL  .claude.json: {e}")); fail_count += 1; }
    }

    // 10. Verify hooks binary path in settings
    let hooks_str = hooks_bin.to_string_lossy();
    if let Ok(content) = std::fs::read_to_string(&settings_path) {
        if content.contains(hooks_str.as_ref()) { report.push(format!("OK  hooks point to {}", hooks_str)); ok_count += 1; }
        else { report.push(format!("WARN  hooks binary path mismatch")); fail_count += 1; }
    }

    // 11. Verify brief works
    match crate::brief::run(dir, "helix", "scan", None, None) {
        Ok(output) if !output.is_empty() => {
            report.push(format!("OK  brief('helix'): {} lines", output.lines().count())); ok_count += 1;
        }
        Ok(_) => { report.push("WARN  brief('helix'): empty output".into()); fail_count += 1; }
        Err(e) => { report.push(format!("FAIL  brief: {e}")); fail_count += 1; }
    }

    // 12. Verify binary sizes
    let hooks_size = std::fs::metadata(&hooks_bin).map(|m| m.len()).unwrap_or(0);
    let exe_size = std::fs::metadata(&exe).map(|m| m.len()).unwrap_or(0);
    if hooks_size > 100_000 && exe_size > 100_000 {
        report.push(format!("OK  binaries: hooks={hooks_size}b server={exe_size}b")); ok_count += 1;
    } else {
        report.push(format!("FAIL  binary too small: hooks={hooks_size}b server={exe_size}b")); fail_count += 1;
    }

    let status = if fail_count == 0 { "ALL CHECKS PASSED" } else { "SOME CHECKS FAILED" };
    format!("_reload: {ok_count} ok, {fail_count} failed — {status}\n\n{}\n\nre-execing...", report.join("\n"))
}

/// Re-exec the server binary (call after reload_verify).
pub(crate) fn do_reexec() {
    use std::os::unix::process::CommandExt;
    let exe = match std::env::current_exe() { Ok(p) => p, Err(_) => return };
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::env::set_var("HELIX_REEXEC", "1");
    let _err = std::process::Command::new(&exe).args(&args).exec();
    eprintln!("helix reload failed: {_err}");
}

/// Ensure ~/.claude.json mcpServers.helix.command points to the release binary.
fn fix_mcp_config(home: &str, release_bin: &std::path::Path) {
    let config_path = std::path::PathBuf::from(home).join(".claude.json");
    let data = match std::fs::read_to_string(&config_path) { Ok(d) => d, Err(_) => return };
    let expected = release_bin.to_string_lossy();
    if data.contains(expected.as_ref()) { return; }
    let needle = "\"command\":";
    let helix_key = "\"helix\"";
    let helix_pos = match data.find(helix_key) { Some(p) => p, None => return };
    let after_helix = &data[helix_pos..];
    let cmd_offset = match after_helix.find(needle) { Some(p) => p, None => return };
    let abs_cmd = helix_pos + cmd_offset + needle.len();
    let rest = &data[abs_cmd..];
    let quote_start = match rest.find('"') { Some(p) => p, None => return };
    let val_start = abs_cmd + quote_start + 1;
    let val_rest = &data[val_start..];
    let quote_end = match val_rest.find('"') { Some(p) => p, None => return };
    let old_cmd = &data[val_start..val_start + quote_end];
    if old_cmd == expected.as_ref() { return; }
    let mut patched = String::with_capacity(data.len() + 32);
    patched.push_str(&data[..val_start]);
    patched.push_str(&expected);
    patched.push_str(&data[val_start + quote_end..]);
    let _ = std::fs::write(&config_path, patched);
}

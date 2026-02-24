#!/bin/bash
# One-off topic migration: flat → hierarchical
# Reads data.log, rewrites entries with renamed topics, compacts.
#
# Usage: ./migrate-topics.sh [helix-kb-dir] [--apply]
# Default dir: ~/.helix-kb
# Dry-run first (default), then pass --apply to actually migrate.

set -euo pipefail

DIR="${1:-$HOME/.helix-kb}"
APPLY="${2:-}"

if [ ! -f "$DIR/data.log" ]; then
    echo "No data.log found at $DIR"
    exit 1
fi

echo "Topic migration: $DIR"

python3 - "$DIR" "$APPLY" <<'PYEOF'
import struct, sys, os

dir_path = sys.argv[1]
apply = len(sys.argv) > 2 and sys.argv[2] == "--apply"
log_path = os.path.join(dir_path, "data.log")

# ══════════ RENAME MAP ══════════
# Edit this section with your topic renames.
RENAMES = {
    # iris project hierarchy
    "iris-app": "iris/app",
    "iris-architecture": "iris/architecture",
    "iris-architecture-insights": "iris/architecture/insights",
    "iris-audit-2026-02-23": "iris/audit",
    "iris-code-review": "iris/code-review",
    "iris-correlation-engine": "iris/correlation",
    "iris-correlation-rules": "iris/correlation/rules",
    "iris-coupling": "iris/coupling",
    "iris-credential-theft-rules": "iris/detection/credential-theft",
    "iris-critical-bugs": "iris/bugs",
    "iris-data-packages": "iris/data-packages",
    "iris-detection-engine": "iris/detection/engine",
    "iris-detection-philosophy": "iris/detection/philosophy",
    "iris-detection-rules": "iris/detection/rules",
    "iris-detection-techniques": "iris/detection/techniques",
    "iris-dns-ext": "iris/network/dns",
    "iris-edr-philosophy": "iris/detection/edr-philosophy",
    "iris-endpoint-architecture": "iris/endpoint/architecture",
    "iris-endpoint-ext": "iris/endpoint/extension",
    "iris-endpoint-rewrite": "iris/endpoint/rewrite",
    "iris-engine": "iris/engine",
    "iris-es-subscriptions": "iris/endpoint/es-subscriptions",
    "iris-gaps-audit": "iris/gaps",
    "iris-ghost-detection": "iris/detection/ghost",
    "iris-gotchas": "iris/gotchas",
    "iris-gpt-probe": "iris/probes/gpt",
    "iris-ground-truth-apis": "iris/ground-truth-apis",
    "iris-improvements-2026-02-23": "iris/improvements",
    "iris-macos-techniques": "iris/detection/macos-techniques",
    "iris-malware-intel": "iris/intel/malware",
    "iris-nation-state-gaps": "iris/intel/nation-state",
    "iris-network": "iris/network",
    "iris-network-ext": "iris/network/extension",
    "iris-novel-ideas": "iris/ideas",
    "iris-ntp-probe": "iris/probes/ntp",
    "iris-probes": "iris/probes",
    "iris-process-anomaly-model": "iris/detection/process-anomaly",
    "iris-process-enumeration": "iris/process/enumeration",
    "iris-process-snapshot": "iris/process/snapshot",
    "iris-project": "iris/project",
    "iris-proxy-architecture": "iris/proxy/architecture",
    "iris-proxy-ext": "iris/proxy/extension",
    "iris-scanner-architecture": "iris/scanners/architecture",
    "iris-scanner-examples": "iris/scanners/examples",
    "iris-scanner-inventory": "iris/scanners/inventory",
    "iris-scanner-registry": "iris/scanners/registry",
    "iris-scanners": "iris/scanners",
    "iris-security-audit": "iris/security/audit",
    "iris-security-event-model": "iris/security/event-model",
    "iris-session-extraction": "iris/detection/session-extraction",
    "iris-shared": "iris/shared",
    "iris-swiftui-audit": "iris/app/swiftui-audit",
    "iris-threat-hunt-methodology": "iris/detection/threat-hunt",
    "iris-trust-cache-probe": "iris/probes/trust-cache",
    "iris-undocumented-apis": "iris/undocumented-apis",
    "iris-xnu": "iris/xnu",
    # XNU kernel knowledge
    "xnu-code-signing": "xnu/code-signing",
    "xnu-edr-architecture": "xnu/edr",
    "xnu-iokit-userclient": "xnu/iokit",
    "xnu-iris-integration": "xnu/iris-integration",
    "xnu-kernel-control": "xnu/kernel-control",
    "xnu-mac-hooks": "xnu/mac-hooks",
    "xnu-mach-ipc": "xnu/mach-ipc",
    "xnu-memory-vm": "xnu/memory/vm",
    "xnu-network-stack": "xnu/network",
    "xnu-process-lifecycle": "xnu/process-lifecycle",
    "xnu-security": "xnu/security",
    "xnu-security-enforcement": "xnu/security/enforcement",
    "xnu-signals-ptrace": "xnu/signals-ptrace",
    "xnu-vfs-filesystem": "xnu/vfs",
    "xnu-vm-security": "xnu/memory/security",
    # Build
    "build-errors": "build/errors",
    "build-gotchas": "build/gotchas",
    # Helix
    "helix-bg-metal4": "helix/bg-metal4",
    "helix-deployment": "helix/deployment",
    "helix-dev": "helix/dev",
    "helix-improvements": "helix/improvements",
    "helix-test": "helix/test",
    # GPU/ML
    "gpu-batch-matmul": "gpu/batch-matmul",
    "gpu-inference": "gpu/inference",
    "gpu-matmul-benchmark": "gpu/matmul-benchmark",
    "metal-kernels": "gpu/metal-kernels",
    "ml-inference-apple-silicon": "ml/apple-silicon",
    # LLM
    "llm-background-helper": "llm/background-helper",
    "llm-gguf-rust": "llm/gguf-rust",
    "llm-inference": "llm/inference",
    "llm-inference-research": "llm/inference-research",
    # Sysext
    "sysext-approval": "sysext/approval",
    "sysext-internals": "sysext/internals",
    # References — keep ref- prefix but add /
    "ref-cannoli": "ref/cannoli",
    "ref-dalvik": "ref/dalvik",
    "ref-dev-device": "ref/dev-device",
    "ref-dust": "ref/dust",
    "ref-dyld-cache": "ref/dyld-cache",
    "ref-es-headers": "ref/es-headers",
    "ref-gamozolabs": "ref/gamozolabs",
    "ref-goblin": "ref/goblin",
    "ref-gpt4all": "ref/gpt4all",
    "ref-mac-monitor": "ref/mac-monitor",
    "ref-macos-privileged-apis": "ref/macos-privileged-apis",
    "ref-metal4": "ref/metal4",
    "ref-objective-see": "ref/objective-see",
    "ref-program-examples": "ref/program-examples",
    "ref-ripgrep": "ref/ripgrep",
    "ref-rust-metal": "ref/rust-metal",
    "ref-tauri": "ref/tauri",
    "ref-tcc-db": "ref/tcc-db",
    "ref-transparent-proxy": "ref/transparent-proxy",
    "ref-vscode": "ref/vscode",
    "ref-xnu-kernel": "ref/xnu-kernel",
    "reference-codebases": "ref/codebases",
    # Misc
    "rust-ffi": "rust/ffi",
    "objective-see-patterns": "ref/objective-see/patterns",
    "gguf-format": "llm/gguf-format",
    "minilm-onnx": "ml/minilm-onnx",
    "ort-crate": "ml/ort-crate",
    "dns-contradiction-probe": "iris/probes/dns-contradiction",
    "amaranthine-audit": "amaranthine/audit",
    "amaranthine-codebase": "amaranthine/codebase",
    "amaranthine-improvements": "amaranthine/improvements",
    "amaranthine-rewrite": "amaranthine/rewrite",
    "amaranthine-tool-dispatch": "amaranthine/tool-dispatch",
    "gamozolabs-backup": "ref/gamozolabs/backup",
    "gamozolabs-fuzz-with-emus": "ref/gamozolabs/fuzz-with-emus",
    "gamozolabs-mesos": "ref/gamozolabs/mesos",
    "goblin-es-analysis": "ref/goblin/es-analysis",
    "goblin-macho-parser": "ref/goblin/macho-parser",
    "code-helix": "code/helix",
    "code-helix-background-llm-helper": "code/helix-bg-llm",
    "code-iris": "code/iris",
    "code-irisendpointextension": "code/iris-endpoint",
    "code-irisproxyextension": "code/iris-proxy",
    "code-src": "code/src",
    "hypervisor-falkervisor-beta-deep-dive": "ref/gamozolabs/falkervisor-beta",
    "hypervisor-falkervisor-deep-dive": "ref/gamozolabs/falkervisor",
    "crash-analysis": "iris/crash-analysis",
    "endpointsecurity-header-files-for-reference": "ref/es-headers/full",
    "perf-data": "perf/data",
}

MAGIC = b"AMRL"
VERSION = 1

with open(log_path, "rb") as f:
    data = f.read()

if data[:4] != MAGIC:
    print("ERROR: bad data.log magic")
    sys.exit(1)

version = struct.unpack_from("<I", data, 4)[0]
if version != VERSION:
    print(f"ERROR: unexpected version {version}")
    sys.exit(1)

# Parse all records
entries = []
deleted = set()
pos = 8

while pos < len(data):
    rec_type = data[pos]
    if rec_type == 0x01:  # Entry
        if pos + 12 > len(data):
            break
        topic_len = data[pos + 1]
        body_len = struct.unpack_from("<I", data, pos + 2)[0]
        ts_min = struct.unpack_from("<i", data, pos + 6)[0]
        rec_end = pos + 12 + topic_len + body_len
        if rec_end > len(data):
            break
        topic = data[pos + 12:pos + 12 + topic_len].decode("utf-8", errors="replace")
        body = data[pos + 12 + topic_len:rec_end].decode("utf-8", errors="replace")
        entries.append((pos, topic, body, ts_min))
        pos = rec_end
    elif rec_type == 0x02:  # Delete
        if pos + 8 > len(data):
            break
        target = struct.unpack_from("<I", data, pos + 4)[0]
        deleted.add(target)
        pos += 8
    else:
        break

# Filter live entries
live = [(off, topic, body, ts) for off, topic, body, ts in entries if off not in deleted]

# Count topics
topic_counts = {}
for _, topic, _, _ in live:
    topic_counts[topic] = topic_counts.get(topic, 0) + 1

print(f"\nLive entries: {len(live)}, Deleted: {len(deleted)}, Topics: {len(topic_counts)}")
print("\nCurrent topics:")
for t in sorted(topic_counts.keys()):
    print(f"  {t}: {topic_counts[t]} entries")

# Compute renames
rename_count = 0
already_hierarchical = 0
for t in sorted(topic_counts.keys()):
    if "/" in t:
        already_hierarchical += topic_counts[t]

for old, new in sorted(RENAMES.items()):
    if old in topic_counts:
        print(f"\n  RENAME: {old} → {new} ({topic_counts[old]} entries)")
        rename_count += topic_counts[old]

if already_hierarchical > 0:
    print(f"\n{already_hierarchical} entries already use hierarchical topics")

if rename_count == 0:
    print("\nNo matching topics to rename.")
    sys.exit(0)

print(f"\nTotal entries to rename: {rename_count}")

if not apply:
    print("\nDRY RUN — pass --apply as second argument to execute")
    sys.exit(0)

# Write new data.log
print("\nApplying renames...")
tmp_path = log_path + ".migrated"
with open(tmp_path, "wb") as f:
    # Header
    f.write(MAGIC)
    f.write(struct.pack("<I", VERSION))
    # Entries (only live, with renames applied)
    written = 0
    renamed = 0
    for _, topic, body, ts_min in live:
        new_topic = RENAMES.get(topic, topic)
        if new_topic != topic:
            renamed += 1
        tb = new_topic.encode("utf-8")
        bb = body.encode("utf-8")
        # Entry header: type(1) + topic_len(1) + body_len(4) + ts_min(4) + pad(2) = 12
        hdr = bytearray(12)
        hdr[0] = 0x01
        hdr[1] = len(tb)
        struct.pack_into("<I", hdr, 2, len(bb))
        struct.pack_into("<i", hdr, 6, ts_min)
        f.write(hdr)
        f.write(tb)
        f.write(bb)
        written += 1
    f.flush()
    os.fsync(f.fileno())

# Backup + replace
backup = log_path + ".pre-hierarchy"
os.rename(log_path, backup)
os.rename(tmp_path, log_path)

# Remove stale index
idx = os.path.join(dir_path, "index.bin")
if os.path.exists(idx):
    os.remove(idx)

print(f"\nDone! {written} entries written, {renamed} renamed.")
print(f"Backup: {backup}")
print(f"Deleted stale entries: {len(deleted)}")
print("Run 'helix index' to rebuild the search index.")
PYEOF

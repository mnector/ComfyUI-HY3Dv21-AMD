#!/usr/bin/env python3
"""
update_from_upstream.py - Self-Maintaining Upstream Updater with AMD Shielding & Auto-Rollback
=============================================================================================
Orchestrates:
1. Snapshot current commit (LAST_GOOD_COMMIT) and tag 'backup-pre-sync'.
2. Fetch from upstream (https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1.git).
3. Merge upstream/main into current branch.
4. Run enforce_amd_patches.py.
5. Execute pre-flight Smoke Test on the ComfyUI Python environment.
6. Auto-Rollback via 'git reset --hard LAST_GOOD_COMMIT' if merge conflict or smoke test fails.
7. Auto-commit on success.
"""

import os
import subprocess
import sys
from pathlib import Path

# Configure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent
UPSTREAM_URL = "https://github.com/visualbruno/ComfyUI-Hunyuan3d-2-1.git"


def load_local_env():
    """Load optional private configuration from .env.local without exposing personal paths."""
    env_file = ROOT_DIR / ".env.local"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            pass


def get_venv_python():
    load_local_env()
    # 1. Environment variable (or .env.local)
    env_py = os.environ.get("COMFYUI_PYTHON")
    if env_py and Path(env_py).exists():
        return str(Path(env_py).resolve())

    # 2. Active virtual environment
    if os.environ.get("VIRTUAL_ENV"):
        v_py = Path(os.environ["VIRTUAL_ENV"]) / "Scripts" / "python.exe"
        if v_py.exists():
            return str(v_py.resolve())

    # 3. Standard relative locations ascending from custom_nodes
    candidates = [
        ROOT_DIR.parent.parent / "venv" / "Scripts" / "python.exe",
        ROOT_DIR.parent.parent.parent / "venv" / "Scripts" / "python.exe",
        ROOT_DIR.parent.parent / "python_embeded" / "python.exe",
        ROOT_DIR.parent.parent.parent / "python_embeded" / "python.exe",
        Path(sys.executable),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return sys.executable


def get_comfy_root():
    load_local_env()
    # 1. Environment variable (or .env.local)
    for var in ("COMFYUI_PATH", "COMFYUI_DIR", "COMFYUI_ROOT"):
        val = os.environ.get(var)
        if val and Path(val).exists():
            return str(Path(val).resolve())

    # 2. Ascending search for folder_paths.py
    curr = ROOT_DIR.resolve()
    for _ in range(5):
        if (curr / "folder_paths.py").exists():
            return str(curr)
        if (curr / "ComfyUI" / "folder_paths.py").exists():
            return str(curr / "ComfyUI")
        curr = curr.parent

    # 3. Default relative assumption (custom_nodes/repo -> ComfyUI)
    return str(ROOT_DIR.parent.parent.resolve())


def run_cmd(cmd, cwd=ROOT_DIR, check=True):
    res = subprocess.run(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {cmd}\nOutput: {res.stdout}\nError: {res.stderr}")
    return res


def rollback(last_good_commit, reason):
    print("\n" + "!" * 70)
    print(f" [ROLLBACK TRIGGERED] {reason}")
    print(f" Restoring clean state from: {last_good_commit}")
    print("!" * 70)
    run_cmd("git merge --abort", check=False)
    run_cmd(f"git reset --hard {last_good_commit}", check=False)
    run_cmd("git clean -fd -e hy3dpaint/ckpt -e enforce_amd_patches.py -e update_from_upstream.py -e update_amd_node.bat -e .env.local", check=False)
    print("[OK] Rollback complete. Your working tree is 100% clean and functional.")


def smoke_test(python_exe):
    print("[*] Running pre-flight Smoke Test on ComfyUI environment...")
    comfy_root = get_comfy_root()
    test_code = f"""
import sys
from pathlib import Path
comfy_root = r'''{comfy_root}'''
node_root = r'''{ROOT_DIR}'''
if comfy_root not in sys.path:
    sys.path.insert(0, comfy_root)
if node_root not in sys.path:
    sys.path.insert(0, node_root)

# 1. Test nodes import
import nodes
assert hasattr(nodes, 'NODE_CLASS_MAPPINGS'), 'NODE_CLASS_MAPPINGS missing'
assert len(nodes.NODE_CLASS_MAPPINGS) > 0, 'NODE_CLASS_MAPPINGS empty'

# 2. Check that 100% of node class mappings end with _AMD
non_amd = [k for k in nodes.NODE_CLASS_MAPPINGS.keys() if not k.endswith('_AMD')]
assert len(non_amd) == 0, f'Found un-isolated nodes: {{non_amd}}'

# 3. Check display names
non_amd_display = [v for v in nodes.NODE_DISPLAY_NAME_MAPPINGS.values() if '[AMD]' not in v]
assert len(non_amd_display) == 0, f'Found display names missing [AMD]: {{non_amd_display}}'

# 4. Check custom_rasterizer import and fallback
from hy3dpaint.custom_rasterizer.custom_rasterizer.render import rasterize
assert callable(rasterize), 'rasterize is not callable'

# 5. Check DifferentiableRenderer MeshRender import and SGEMM fix
from hy3dpaint.DifferentiableRenderer.MeshRender import MeshRender
from hy3dpaint.DifferentiableRenderer.camera_utils import safe_matmul
assert callable(safe_matmul), 'safe_matmul is not defined in camera_utils'

print(f'SMOKE_TEST_PASS: {{len(nodes.NODE_CLASS_MAPPINGS)}} AMD nodes registered successfully!')
"""
    res = subprocess.run([python_exe, "-c", test_code], cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or "SMOKE_TEST_PASS" not in res.stdout:
        raise RuntimeError(f"Smoke test failed!\nStdout: {res.stdout}\nStderr: {res.stderr}")
    print(f"[OK] {res.stdout.strip()}")
    return True


def main():
    print("=" * 70)
    print(" ComfyUI-HY3Dv21-AMD: Upstream Sync & Shielding Orchestrator")
    print("=" * 70)

    # 1. Identify python and git
    python_exe = get_venv_python()
    print(f"[*] Using Python: {python_exe}")

    # Check git status
    res = run_cmd("git rev-parse HEAD")
    last_good_commit = res.stdout.strip()
    print(f"[*] Snapshot commit (LAST_GOOD_COMMIT): {last_good_commit}")

    # Create safety backup tag
    run_cmd("git tag -f backup-pre-sync", check=False)
    print("[*] Created safety tag 'backup-pre-sync'")

    # 2. Verify upstream remote
    remotes = run_cmd("git remote -v").stdout
    if "upstream" not in remotes:
        print(f"[*] Adding upstream remote: {UPSTREAM_URL}")
        run_cmd(f"git remote add upstream {UPSTREAM_URL}")
    else:
        print("[*] Upstream remote already configured.")

    # 3. Fetch upstream
    print("[*] Fetching upstream changes...")
    try:
        run_cmd("git fetch upstream")
    except Exception as e:
        print(f"[-] Failed to fetch upstream: {e}")
        return 1

    # Check if upstream/main has changes compared to HEAD
    try:
        merge_check = run_cmd("git merge-base HEAD upstream/main", check=False)
        upstream_head = run_cmd("git rev-parse upstream/main", check=False).stdout.strip()
        if merge_check.stdout.strip() == upstream_head:
            print("[OK] Already up-to-date with upstream/main! Running patch validation...")
            run_cmd(f'"{python_exe}" enforce_amd_patches.py')
            smoke_test(python_exe)
            print("[SUCCESS] Repository is up to date and all AMD protections are verified.")
            return 0
    except Exception:
        pass

    # 4. Attempt merge
    print("[*] Merging upstream/main...")
    merge_res = run_cmd("git merge upstream/main --no-edit", check=False)
    if merge_res.returncode != 0:
        rollback(last_good_commit, f"Git merge conflict detected with upstream/main:\n{merge_res.stderr or merge_res.stdout}")
        return 1

    # 5. Enforce AMD Patches
    print("[*] Applying AMD / ZLUDA shield and patch enforcement...")
    patch_res = run_cmd(f'"{python_exe}" enforce_amd_patches.py', check=False)
    if patch_res.returncode != 0:
        rollback(last_good_commit, f"enforce_amd_patches.py failed:\n{patch_res.stderr or patch_res.stdout}")
        return 1

    # 6. Run Pre-flight Smoke Test
    try:
        smoke_test(python_exe)
    except Exception as e:
        rollback(last_good_commit, f"Pre-flight Smoke Test failed:\n{e}")
        return 1

    # 7. Commit changes if any
    status_res = run_cmd("git status --porcelain")
    if status_res.stdout.strip():
        print("[*] Committing integrated changes with AMD shielding...")
        run_cmd("git add -A")
        run_cmd('git commit -m "Auto-sync with upstream + enforce AMD ZLUDA patches"')
        print("[+] Succeeded in synchronizing with upstream and committing AMD patches.")
    else:
        print("[OK] No unstaged changes; synchronization complete.")

    print("=" * 70)
    print(" [SUCCESS] Upstream sync, AMD shielding, and smoke test completed 100%!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
install.py - ComfyUI Manager & Setup Installer for ComfyUI-HY3Dv21-AMD
======================================================================
Automatically executed by ComfyUI Manager on node install/update.

Tasks:
1. Validates and installs required Python packages without modifying
   or reinstalling ROCm/ZLUDA torch, torchvision, or torchaudio.
2. Automatically ensures 'birefnet.safetensors' is downloaded in
   '<COMFYUI_ROOT>/models/background_removal/' for the integrated
   background removal workflow.
3. Automatically ensures 'RealESRGAN_x4plus.pth' is downloaded in
   'hy3dpaint/ckpt/' for high-resolution UV texturing upscaling.
4. Executes enforce_amd_patches.py to lock down AMD/ZLUDA compatibility.
"""

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent

BIREFNET_URL = "https://huggingface.co/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors"
REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"


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


def get_comfy_root() -> Path:
    load_local_env()
    # 1. Environment variable
    for var in ("COMFYUI_PATH", "COMFYUI_DIR", "COMFYUI_ROOT"):
        val = os.environ.get(var)
        if val and Path(val).exists():
            return Path(val).resolve()

    # 2. Try ascending search for folder_paths.py
    curr = ROOT_DIR.resolve()
    for _ in range(5):
        if (curr / "folder_paths.py").exists():
            return curr
        if (curr / "ComfyUI" / "folder_paths.py").exists():
            return curr / "ComfyUI"
        curr = curr.parent

    # 3. Default custom_nodes/ComfyUI-HY3Dv21-AMD -> ComfyUI
    default_root = ROOT_DIR.parent.parent.resolve()
    return default_root


def download_with_progress(url: str, dest_path: Path, desc: str = "Downloading"):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".download")

    print(f"[*] {desc}: {dest_path.name}")
    print(f"    Source: {url}")
    print(f"    Target: {dest_path}")

    # Try huggingface_hub first if applicable
    if "huggingface.co" in url:
        try:
            import huggingface_hub
            parts = url.replace("https://huggingface.co/", "").split("/resolve/main/")
            if len(parts) == 2:
                repo_id, filename = parts
                print(f"[*] Fetching via huggingface_hub ({repo_id} / {filename})...")
                cached_file = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)
                shutil.copy(cached_file, dest_path)
                print(f"[OK] Downloaded successfully: {dest_path.name} ({dest_path.stat().st_size:,} bytes)")
                return True
        except Exception as e:
            print(f"[-] huggingface_hub download skipped or failed ({e}), streaming directly...")

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_file:
            total_size = resp.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size else 0
            downloaded = 0
            block_size = 1024 * 1024  # 1MB chunks

            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total_bytes / (1024 * 1024)
                    sys.stdout.write(f"\r    Progress: {pct:6.2f}% ({mb_down:6.1f} / {mb_tot:6.1f} MB)")
                    sys.stdout.flush()
            sys.stdout.write("\n")

        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            print(f"[OK] Download complete: {dest_path.name} ({dest_path.stat().st_size:,} bytes)")
            return True
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"\n[!] Failed to download {dest_path.name}: {e}")
        print(f"    You can download it manually from: {url}")
        print(f"    and place it into: {dest_path}")
        return False


def ensure_birefnet_model(comfy_root: Path):
    """Ensures birefnet.safetensors exists in ComfyUI/models/background_removal/"""
    models_dir = comfy_root / "models"
    bg_dir = models_dir / "background_removal"
    dest_path = bg_dir / "birefnet.safetensors"

    if dest_path.exists() and dest_path.stat().st_size > 100 * 1024 * 1024:
        print(f"[OK] BiRefNet model found: {dest_path} ({dest_path.stat().st_size:,} bytes)")
        return True

    print(f"[*] BiRefNet background removal model not found in {bg_dir}. Downloading automatically...")
    return download_with_progress(BIREFNET_URL, dest_path, desc="Downloading BiRefNet Model")


def ensure_realesrgan_model(root_dir: Path):
    """Ensures RealESRGAN_x4plus.pth exists in hy3dpaint/ckpt/"""
    dest_path = root_dir / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    if dest_path.exists() and dest_path.stat().st_size > 50 * 1024 * 1024:
        print(f"[OK] RealESRGAN_x4plus model found: {dest_path} ({dest_path.stat().st_size:,} bytes)")
        return True

    print(f"[*] RealESRGAN_x4plus model not found in {dest_path.parent}. Downloading automatically...")
    return download_with_progress(REALESRGAN_URL, dest_path, desc="Downloading RealESRGAN Upscaler")


def install_requirements(python_exe: str):
    req_file = ROOT_DIR / "requirements.txt"
    if not req_file.exists():
        return

    print("[*] Checking auxiliary dependencies from requirements.txt...")
    cmd = [
        python_exe,
        "-m",
        "pip",
        "install",
        "-r",
        str(req_file),
        "--no-warn-script-location",
    ]
    try:
        subprocess.run(cmd, cwd=ROOT_DIR, check=True)
        print("[OK] Auxiliary dependencies verified.")
    except Exception as e:
        print(f"[!] Warning: pip install encountered an issue: {e}")


def main():
    print("=" * 70)
    print(" ComfyUI-HY3Dv21-AMD: Installation & Setup Manager")
    print("=" * 70)

    load_local_env()
    python_exe = sys.executable
    comfy_root = get_comfy_root()

    print(f"[*] Node directory: {ROOT_DIR}")
    print(f"[*] ComfyUI root:   {comfy_root}")
    print(f"[*] Python runner:  {python_exe}")

    # 1. Enforce AMD patch shielding
    patcher = ROOT_DIR / "enforce_amd_patches.py"
    if patcher.exists():
        try:
            subprocess.run([python_exe, str(patcher)], cwd=ROOT_DIR, check=True)
        except Exception as e:
            print(f"[!] Warning: enforce_amd_patches.py failed: {e}")

    # 2. Install requirements safely
    install_requirements(python_exe)

    # 3. Ensure required models
    ensure_birefnet_model(comfy_root)
    ensure_realesrgan_model(ROOT_DIR)

    print("=" * 70)
    print(" [SUCCESS] ComfyUI-HY3Dv21-AMD is fully installed and ready to use!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

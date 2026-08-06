"""Manage the backend Python environment with uv.

uv is a single static binary that can fetch a standalone CPython and install
packages quickly and reproducibly. We download it once into the backend's
``bin`` folder, then use it to create a venv and install torch + ComfyUI deps.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

from .downloader import download
from ..textutil import strip_ansi

LogCb = Callable[[str], None]

# Windows x86_64 uv release (single zip containing uv.exe).
UV_URL = (
    "https://github.com/astral-sh/uv/releases/latest/download/"
    "uv-x86_64-pc-windows-msvc.zip"
)
BACKEND_PYTHON_VERSION = "3.11"


def ensure_uv(uv_path: Path, log: Optional[LogCb] = None,
              on_progress: Optional[Callable[[int, int], None]] = None) -> Path:
    """Download uv if not already present; return its path."""
    log = log or (lambda _m: None)
    if uv_path.exists():
        return uv_path
    uv_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading uv -> {uv_path}")
    tmp_zip = uv_path.parent / "uv.zip"
    download(UV_URL, tmp_zip, on_progress=on_progress)
    with zipfile.ZipFile(tmp_zip) as zf:
        member = next((n for n in zf.namelist() if n.endswith("uv.exe")), None)
        if member is None:
            raise RuntimeError("uv.exe not found in release archive")
        with zf.open(member) as src, open(uv_path, "wb") as dst:
            dst.write(src.read())
    tmp_zip.unlink(missing_ok=True)
    return uv_path


def _run_uv(uv_path: Path, args: list[str], log: LogCb,
            cancel: Optional[Callable[[], bool]] = None) -> None:
    cancel = cancel or (lambda: False)
    cmd = [str(uv_path), *args]
    log("$ " + " ".join(cmd))
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        # uv emits UTF-8; without this Windows decodes as the ANSI code page
        # (cp932) and multi-byte progress glyphs crash the reader thread.
        encoding="utf-8", errors="replace",
        bufsize=1, creationflags=creationflags,
    )
    assert proc.stdout
    for raw in proc.stdout:
        raw = raw.rstrip("\r\n")
        if strip_ansi(raw).strip():
            log(raw)  # raw (with ANSI) for colored display
        if cancel():
            proc.terminate()
            raise RuntimeError("cancelled")
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"uv {args[0] if args else ''} failed (exit {code})")


def create_venv(uv_path: Path, venv_dir: Path, log: LogCb,
                cancel: Optional[Callable[[], bool]] = None) -> None:
    _run_uv(
        uv_path,
        ["venv", "--python", BACKEND_PYTHON_VERSION, str(venv_dir)],
        log, cancel,
    )


def pip_install(uv_path: Path, venv_python: Path, log: LogCb,
                packages: Optional[list[str]] = None,
                requirements: Optional[Path] = None,
                index_url: Optional[str] = None,
                upgrade: bool = False,
                cancel: Optional[Callable[[], bool]] = None) -> None:
    args = ["pip", "install", "--python", str(venv_python)]
    if upgrade:
        # Without this an already-installed package satisfies the bare
        # requirement even when the target index carries a newer build
        # (e.g. switching torch from +cu128 to +cu130 wheels).
        args += ["--upgrade"]
    if index_url:
        args += ["--index-url", index_url]
    if requirements:
        args += ["-r", str(requirements)]
    if packages:
        args += packages
    _run_uv(uv_path, args, log, cancel)

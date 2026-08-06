"""Runtime configuration: locating directories and scanning model files.

The app is designed to run both as a PyInstaller-built exe and from source
during development. In both cases the model files live next to the executable
(or project root in dev) under ``models/{diffusion,vae,te}``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Application identity. Embedded into saved-image metadata (Software / Version
# tags) so files can be recognized as produced by this app.
APP_NAME = "scom"
APP_VERSION = "1.0.0"
APP_SIGNATURE = f"{APP_NAME} {APP_VERSION}"

# Model file extensions we consider selectable.
MODEL_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pt", ".gguf", ".bin")

# Model component subdirectories under ``models/``. Names match both the
# ComfyUI model categories (used in extra_model_paths.yaml) and the upstream
# Hugging Face ``split_files`` layout, so the mapping is 1:1.
MODEL_DIRS = ("diffusion_models", "vae", "text_encoders", "loras")


def base_dir() -> Path:
    """Directory that anchors models/, settings.toml and bundled resources.

    For a frozen exe this is the folder containing the .exe so that an end user
    can drop a ``models`` folder right next to it. In dev it is the repo root.
    ``SCOM_BASE_DIR`` overrides both — useful for tests so they never touch the
    real settings.toml/output of a dev checkout.
    """
    env = os.environ.get("SCOM_BASE_DIR")
    if env:
        return Path(env).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # app/config.py -> app -> repo root
    return Path(__file__).resolve().parent.parent


def models_root() -> Path:
    return base_dir() / "models"


def ensure_model_dirs() -> dict[str, Path]:
    """Create models/{diffusion,vae,te} if missing and return their paths."""
    root = models_root()
    paths: dict[str, Path] = {}
    for name in MODEL_DIRS:
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        paths[name] = p
    return paths


def scan_models(kind: str) -> list[str]:
    """List model filenames (relative to the kind dir) for a given component.

    ``kind`` is one of MODEL_DIRS. Returns names sorted case-insensitively.
    Subdirectories are walked so users can organize files.
    """
    if kind not in MODEL_DIRS:
        raise ValueError(f"unknown model kind: {kind}")
    root = models_root() / kind
    if not root.exists():
        return []
    found: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS:
            # OS-native separators: ComfyUI lists files via os.path.relpath
            # and its prompt validation requires an exact string match, so a
            # subfolder file must be sent as "sub\\file.safetensors" on
            # Windows, not "sub/file.safetensors".
            found.append(str(path.relative_to(root)))
    return sorted(found, key=str.lower)


def comfyui_dir() -> Path | None:
    """Locate the ComfyUI source tree used as the generation backend.

    Resolution order:
      1. SCOM_COMFYUI_DIR environment variable
      2. ``<base>/vendor/ComfyUI``  (bundled location)
      3. ``<base>/ComfyUI``         (sibling clone for dev)
    Returns None if none exist.
    """
    env = os.environ.get("SCOM_COMFYUI_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(base_dir() / "vendor" / "ComfyUI")
    candidates.append(base_dir() / "ComfyUI")
    for c in candidates:
        if c and (c / "main.py").exists():
            return c.resolve()
    return None


@dataclass
class AppPaths:
    base: Path = field(default_factory=base_dir)
    models: Path = field(default_factory=models_root)

    @property
    def user_data(self) -> Path:
        """Where we write generated config, outputs cache, logs, etc."""
        p = self.base / "userdata"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ----- managed backend environment (created on first run) --------------
    @property
    def backend_root(self) -> Path:
        return self.user_data / "backend"

    @property
    def venv_dir(self) -> Path:
        return self.backend_root / "venv"

    @property
    def backend_python(self) -> Path:
        """Interpreter used to run the ComfyUI subprocess.

        Falls back to the current interpreter (dev) when the managed venv has
        not been created yet.
        """
        if sys.platform == "win32":
            py = self.venv_dir / "Scripts" / "python.exe"
        else:
            py = self.venv_dir / "bin" / "python"
        return py if py.exists() else Path(sys.executable)

    @property
    def uv_path(self) -> Path:
        name = "uv.exe" if sys.platform == "win32" else "uv"
        return self.backend_root / "bin" / name

    @property
    def managed_comfyui(self) -> Path:
        return self.backend_root / "ComfyUI"

    @property
    def comfyui(self) -> Path | None:
        """Resolve the ComfyUI source tree: managed location first, then the
        dev fallbacks (env var / vendor / sibling clone)."""
        if (self.managed_comfyui / "main.py").exists():
            return self.managed_comfyui
        return comfyui_dir()

    @property
    def manifest_path(self) -> Path:
        return self.user_data / "setup_manifest.json"

    @property
    def settings_path(self) -> Path:
        """User-editable settings file, kept next to the executable."""
        return self.base / "settings.toml"

    @property
    def prompts_path(self) -> Path:
        """User-editable prompt-preset CSV, kept next to the executable."""
        return self.base / "prompts.csv"

    @property
    def output_dir(self) -> Path:
        """Where generated images are saved, next to the executable."""
        p = self.base / "output"
        p.mkdir(parents=True, exist_ok=True)
        return p

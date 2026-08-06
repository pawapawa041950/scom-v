"""MiniMax H3 model manifest + downloader.

Files live in the Comfy-Org repackage ``Comfy-Org/MiniMax-H3`` under
``{diffusion_models,text_encoders,vae}/``. The manifest is written to
``userdata/models.json`` on first run so users can edit URLs / add variants
without touching code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from .. import config
from .downloader import download

# Comfy-Org repackage of MiniMax H3 (folder layout matches models/ 1:1).
H3_BASE = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"


@dataclass
class ModelFile:
    kind: str         # one of config.MODEL_DIRS
    filename: str
    url: str
    size: int = 0     # bytes, for progress + verification (0 = unknown)
    required: bool = True


_FIELDS = ("kind", "filename", "url", "size", "required")


def _h3(kind: str, filename: str, size: int, required: bool = False) -> "ModelFile":
    return ModelFile(kind, filename, f"{H3_BASE}/{kind}/{filename}",
                     size, required=required)


# サイズは HF API の実測値（検証に使うため正確に保つこと）。
# required=True は「これが無いと何も生成できない」共通ファイル（TE/VAE）。
# diffusion は量子化・パイプラインの選択制なので required にしない。
DEFAULT_MODELS: list[ModelFile] = [
    # --- diffusion (fl2va = t2v/i2v, ref2va = r2v) -------------------------
    _h3("diffusion_models", "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        20_970_379_616),
    _h3("diffusion_models", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        20_958_205_608),
    _h3("diffusion_models", "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        20_970_379_616),
    _h3("diffusion_models", "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
        20_958_205_608),
    # --- text encoder (Qwen3-VL-32B) ---------------------------------------
    _h3("text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        27_141_342_152, required=True),
    # --- VAE（動画 + 音声）--------------------------------------------------
    _h3("vae", "minimax_h3_video_vae_fp16.safetensors",
        5_207_808_496, required=True),
    _h3("vae", "minimax_h3_audio_vae_fp32.safetensors",
        605_254_808, required=True),
]


def _write_manifest(mf: Path, manifest: list[ModelFile]) -> None:
    mf.write_text(json.dumps([asdict(m) for m in manifest], indent=2),
                  encoding="utf-8")


def load_manifest(paths: config.AppPaths) -> list[ModelFile]:
    """Read userdata/models.json, creating it from DEFAULT_MODELS if absent.

    New default entries shipped in later versions are merged into an existing
    models.json, keyed by (kind, filename), so users who already ran the app
    see them without losing any hand-edited URLs/entries.
    """
    mf = paths.user_data / "models.json"
    if not mf.exists():
        manifest = list(DEFAULT_MODELS)
        _write_manifest(mf, manifest)
        return manifest
    data = json.loads(mf.read_text(encoding="utf-8"))
    manifest = [ModelFile(**{k: d[k] for k in _FIELDS if k in d}) for d in data]
    have = {(m.kind, m.filename) for m in manifest}
    added = [m for m in DEFAULT_MODELS if (m.kind, m.filename) not in have]
    if added:
        manifest.extend(added)
        _write_manifest(mf, manifest)
    return manifest


def target_path(paths: config.AppPaths, m: ModelFile) -> Path:
    return paths.models / m.kind / m.filename


def missing_required(paths: config.AppPaths,
                     manifest: Optional[list[ModelFile]] = None) -> list[ModelFile]:
    manifest = manifest or load_manifest(paths)
    out = []
    for m in manifest:
        if not m.required:
            continue
        p = target_path(paths, m)
        if not p.exists() or (m.size and p.stat().st_size != m.size):
            out.append(m)
    return out


def download_model(paths: config.AppPaths, m: ModelFile,
                   on_progress: Optional[Callable[[int, int], None]] = None,
                   cancel: Optional[Callable[[], bool]] = None) -> Path:
    dest = target_path(paths, m)
    return download(
        m.url, dest, on_progress=on_progress,
        expected_size=m.size or None, cancel=cancel,
    )

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
    # --- text encoder nvfp4 版（15.7GB。Blackwell 不要と公式 README に明記。
    #     公式テンプレートの既定。int8 版より 11GB 軽い）--------------------
    _h3("text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        15_687_142_551),
    # --- Turbo (PDD) LoRA: 4〜8 ステップの蒸留 LoRA（各 1.96GB）-------------
    #     fl2v = t2v/i2v 用、ref2v = r2v 用。公式テンプレは fl2v 8step を
    #     4 ステップで使う。
    _h3("loras", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        1_956_193_000),
    _h3("loras", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        1_956_192_992),
    _h3("loras", "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        1_956_193_000),
    # --- プレビュー用 tiny デコーダ（コミュニティ製、9.8MB）------------------
    # サンプリング中プレビューを Latent2RGB より大幅に高品質化する。
    # 選択UIには出さず、セットアップ時と通常起動時に自動取得する。
    # 無くても生成は動く（プレビューが Latent2RGB になるだけ）ため必須扱い
    # にはしない。
    ModelFile("vae_approx", "taeh3.safetensors",
              "https://huggingface.co/Kijai/MiniMax-H3-TAE/resolve/main/"
              "vae_approx/taeh3.safetensors",
              9_791_388, required=False),
]


# Turbo LoRA のファイル名（チェックポイント系統 × 種類）。
TURBO_LORAS = {
    ("fl2va", "8step"): "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    ("fl2va", "4step_768p"): "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    ("ref2va", "4step"): "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
}
TE_NVFP4 = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"


def turbo_lora_for(ckpt_kind: str, variant: str) -> str:
    """チェックポイント系統（fl2va / ref2va）と版から Turbo LoRA 名を返す。
    ref2va は 4step 版しか無いので variant は無視する。"""
    if ckpt_kind == "ref2va":
        return TURBO_LORAS[("ref2va", "4step")]
    return TURBO_LORAS.get(("fl2va", variant), TURBO_LORAS[("fl2va", "8step")])


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

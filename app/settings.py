"""Persisted UI settings (TOML).

All settings are auto-saved whenever the user changes a control. The startup
prompt/negative are NOT stored here — they come from the first entry of
prompts.csv. The file lives next to the executable so it is easy to edit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # Python 3.10
    import tomli as _toml  # type: ignore

DEFAULTS: dict[str, Any] = {
    # models
    "preset": "",      # anima | krea2 | sdxl (空/旧 "all" は起動時に移行)
    # 表示モデルごとの Models + 設定カテゴリの記憶 (JSON {preset: {...}})
    "preset_conf": "{}",
    "diffusion": "",
    "vae": "",
    "te1": "",
    "te2": "",
    "dual_te": False,
    "clip_type": "stable_diffusion",
    # Model merges: JSON list of entries, each
    #   {"id": int, "name": str, "models": [["file", weight], ...],
    #    "quant": ""|"fp8"|"int8_convrot", "low_memory": bool}
    # merge_seq is the last id handed out (numbering never reuses ids).
    "merges": "[]",
    "merge_seq": 0,
    # NOTE: 適用中の LoRA は意図的に永続化しない（毎回まっさらで起動）。
    # SageAttention（量子化attentionによる高速化）を使うか。ONでもパッケージ
    # 未導入なら起動フラグは付けない（バックエンドが起動不能になるため）。
    "sage_attention": False,
    # Hires fix (latent): 2段目サンプリングによる高解像度化
    "hires_enabled": False,
    "hires_scale": 1.5,
    "hires_denoise": 0.55,
    "hires_steps": 0,          # 0 = メインの steps と同じ
    "hires_method": "bislerp",
    # generation settings
    "width": 1024,
    "height": 1024,
    "steps": 30,
    "cfg": 4.0,
    "batch": 1,
    "sampler": "er_sde",
    "scheduler": "simple",
    # seed 欄の値（"-1" = 毎回ランダム）。生成後の書き戻しは無いので、
    # 保存されるのはユーザーが欄に入力した値そのもの。
    "seed": "-1",
    "dtype": "default",
    # image output
    "image_format": "png",   # png | jpg | webp
    "png_compress": 6,       # 0..9
    "jpg_quality": 92,       # 1..100
    "webp_quality": 90,      # 1..100
    "embed_metadata": True,  # 画像にメタ情報（parameters/EXIF）を埋め込む
    # XYZ プロットウィンドウの入力状態 (JSON)
    "xyz": "{}",
}


def load(path: Path) -> tuple[dict[str, Any], str | None]:
    """Return (settings, error).

    ``settings`` is DEFAULTS overlaid with the file's values. ``error`` is a
    human-readable message if the file exists but could not be parsed (in which
    case defaults are returned) — callers should surface it rather than let a
    broken file silently reset everything.
    """
    data = dict(DEFAULTS)
    if not path.exists():
        return data, None
    try:
        with open(path, "rb") as f:
            data.update(_toml.load(f))
        return data, None
    except (OSError, ValueError) as e:
        return data, str(e)


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    s = (str(value)
         .replace("\\", "\\\\").replace('"', '\\"')
         .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
    return f'"{s}"'


def save(path: Path, data: dict[str, Any]) -> None:
    """Write all known keys (stable order) as a flat TOML table.

    注意: DEFAULTS に無いキーは保存されない（allowlist 方式）。設定キーを
    増やしたら必ず DEFAULTS にも追加すること — 忘れると UI 上は動くのに
    再起動で消える、という分かりにくい不具合になる。
    """
    lines = [
        "# scom 設定ファイル（変更すると自動保存されます）。",
        "# 起動時のプロンプト/ネガティブは prompts.csv の1個目の設定から読み込まれます。",
        "",
    ]
    for key in DEFAULTS:
        if key in data:
            lines.append(f"{key} = {_fmt(data[key])}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

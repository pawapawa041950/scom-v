"""Persisted UI settings (TOML).

All settings are auto-saved whenever the user changes a control. The startup
prompt is NOT stored here — it comes from the first entry of prompts.csv. The
file lives next to the executable so it is easy to edit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # Python 3.10
    import tomli as _toml  # type: ignore

DEFAULTS: dict[str, Any] = {
    # mode: t2v | i2v | r2v
    "mode": "t2v",
    # models（ファイル名。空 = 再スキャン時に自動選択）
    "diffusion_fl2va": "",
    "diffusion_ref2va": "",
    "te": "",
    "vae_video": "",
    "vae_audio": "",
    # generation settings
    "aspect": "16:9",
    "quality_mp": 1.0,       # 目標メガピクセル（1.0 ≒ 768p級）
    "length_sec": 5.0,       # 動画の長さ（秒。17n+5 フレームへスナップ）
    "steps": 20,
    "sampler": "res_multistep",
    "scheduler": "simple",
    # seed 欄の値（"-1" = 毎回ランダム）。生成後の書き戻しは無い。
    "seed": "-1",
    "dtype": "default",
    # Sigma shift（OFF ならモデル既定）
    "shift_enabled": False,
    "shift_video": 12.0,
    "shift_audio": 3.0,
    # r2v
    "ref_image_size": "match",
    # SageAttention（量子化attentionによる高速化）。ONでもパッケージ未導入なら
    # 起動フラグは付けない（バックエンドが起動不能になるため）。
    "sage_attention": False,
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
        "# scom-v 設定ファイル（変更すると自動保存されます）。",
        "# 起動時のプロンプトは prompts.csv の1個目の設定から読み込まれます。",
        "",
    ]
    for key in DEFAULTS:
        if key in data:
            lines.append(f"{key} = {_fmt(data[key])}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

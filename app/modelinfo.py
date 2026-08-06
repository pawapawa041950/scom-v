"""Identify a model file's family (anima / krea2) from its safetensors header.

Files stay where ComfyUI already keeps them (``models/<kind>/``). To decide
which preset a file belongs to we read only the *safetensors header* — the
tensor names and shapes, a few KB–MB at the start of the file — never the
weights. This makes detection independent of the filename, so a model the user
downloaded under any name is still sorted correctly.

Signatures (confirmed against the real published files):
  * text encoder: a vision tower (``visual``/``vision`` keys) or hidden dim
    2560 -> krea2 (Qwen3-VL 4B); hidden dim 1024 -> anima (Qwen3 0.6B).
  * diffusion:    top-level keys ``txtfusion``/``tmlp``/``tproj`` -> krea2
    (~12B DiT); top-level ``net`` -> anima. Param count is the fallback.
  * vae:          the Qwen-Image VAE is shared by both families.

Results are cached per (path, size, mtime) so repeated preset switches don't
re-read headers.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional

ANIMA = "anima"
KREA2 = "krea2"
SDXL = "sdxl"
SHARED = "shared"   # anima/krea2 共通（Qwen-Image VAE）。sdxl には含まれない。
OTHER = "other"     # アーキテクチャは特定できたが、どのプリセットでも使えない
                    # （SD1.x/SD2.x・Flux・Qwen-Image 20B などの LoRA）
UNKNOWN = "unknown"

# Refuse to allocate for an absurd/garbage header length.
_HEADER_MAX = 100_000_000

# path -> (size, mtime, family)
_cache: dict[str, tuple[int, float, str]] = {}


def _read_header(path: Path) -> Optional[dict]:
    """Return the parsed safetensors header dict, or None if unreadable."""
    try:
        with open(path, "rb") as f:
            raw_n = f.read(8)
            if len(raw_n) < 8:
                return None
            n = struct.unpack("<Q", raw_n)[0]
            if n <= 0 or n > _HEADER_MAX:
                return None
            data = f.read(n)
        if len(data) < n:
            return None
        return json.loads(data)
    except (OSError, ValueError):
        return None


def _param_count(h: dict, keys: list[str]) -> int:
    total = 0
    for k in keys:
        shape = h[k].get("shape") or []
        p = 1
        for d in shape:
            p *= d
        total += p
    return total


def _lora_in_features(h: dict, keys: list[str], *substrs: str) -> Optional[int]:
    """in_features of the first LoRA down-projection matching all substrings.

    kohya は ``<module>.lora_down.weight``、diffusers/PEFT は
    ``<module>.lora_A.weight``。どちらも形状 [rank, in_features] なので、
    接続先モジュールの入力次元がヘッダだけで分かる。キーは '.'→'_' 正規化
    して照合する（kohya はモジュール名の '.' を '_' に潰すため）。
    """
    for k in keys:
        norm = k.replace(".", "_")
        if not (norm.endswith("_lora_down_weight")
                or norm.endswith("_lora_A_weight")):
            continue
        if all(s in norm for s in substrs):
            shape = h[k].get("shape") or []
            if len(shape) == 2:
                return shape[1]
    return None


def _classify_lora(h: dict, keys: list[str]) -> str:
    """LoRA ファイルの対応系統をキー名+形状から判定する。

    LoRA のキーには学習対象モジュールのパスがそのまま残る（kohya:
    lora_unet_<path>_..., diffusers: transformer.<path>.lora_A.weight,
    comfy: diffusion_model.<path>）ので、ベースモデルのアーキテクチャが
    特定できる。判定できたが対応プリセットが無いものは OTHER（フィルタで
    非表示）、構造を認識できないものは UNKNOWN（常に表示 = 安全側）。
    """
    norm_all = " ".join(k.replace(".", "_") for k in keys)

    # krea2: txtfusion（ネイティブ名）/ text_fusion（diffusers 名）は固有。
    if "txtfusion" in norm_all or "text_fusion" in norm_all:
        return KREA2

    # anima: blocks.N.cross_attn.q_proj（hidden 2048）と llm_adapter が固有。
    if "cross_attn_q_proj" in norm_all or "llm_adapter" in norm_all:
        d = _lora_in_features(h, keys, "cross_attn_q_proj")
        if d is not None and d != 2048:
            return OTHER  # cross_attn.q_proj を持つ別アーキテクチャ
        return ANIMA

    # SD-UNet（kohya の LDM 名 input_blocks / diffusers の down_blocks）。
    # attn2（cross-attention）の入力次元が context dim: 2048=SDXL,
    # 768=SD1.x, 1024=SD2.x。
    if (any(m in norm_all for m in ("input_blocks", "output_blocks",
                                    "middle_block"))
            or ("down_blocks" in norm_all and "attentions" in norm_all)):
        d = (_lora_in_features(h, keys, "attn2_to_k")
             or _lora_in_features(h, keys, "attn2_to_v"))
        if d == 2048:
            return SDXL
        if d is not None:
            return OTHER  # SD1.x / SD2.x
        # UNet 側に手掛かりなし（attn2 を学習していない LoRA）。
        # SDXL だけが text encoder を2つ持つ。
        return SDXL if "lora_te2" in norm_all else UNKNOWN

    if "transformer_blocks" in norm_all:
        # Qwen-Image (20B) / Flux(diffusers名): joint-attention の
        # add_*_proj や img_mod/img_mlp。anima とは名前も次元も別物。
        if any(m in norm_all for m in ("add_q_proj", "add_k_proj",
                                       "img_mod", "img_mlp", "txt_mod")):
            return OTHER
        # krea2 (diffusers名) 固有の gate 付き attention/FF。
        if "to_gate" in norm_all or "ff_gate" in norm_all:
            return KREA2
        return UNKNOWN

    if "double_blocks" in norm_all or "single_blocks" in norm_all:
        return OTHER  # Flux / HunyuanVideo 系（kohya 名）

    return UNKNOWN


def _classify_header(kind: str, h: dict) -> str:
    keys = [k for k in h if k != "__metadata__"]
    if not keys:
        return UNKNOWN

    if kind == "loras":
        return _classify_lora(h, keys)

    if kind == "vae":
        # kl-f8 (SD/SDXL) VAE: quant_conv / decoder.conv_in が特徴。
        if "post_quant_conv.weight" in h or "decoder.conv_in.weight" in h:
            return SDXL
        return SHARED  # Qwen-Image VAE (anima/krea2 共通) ほか

    if kind == "text_encoders":
        if any("visual" in k or "vision" in k for k in keys):
            return KREA2
        # CLIP (SDXL の clip_l/clip_g)。transformers 形式は text_model.*、
        # チェックポイントから取り出した clip_g は open_clip 形式
        # (transformer.resblocks.*)。
        if any(k.startswith("text_model.")
               or "transformer.resblocks." in k for k in keys):
            return SDXL
        hidden = None
        for k in keys:
            if k.endswith("embed_tokens.weight"):
                shape = h[k].get("shape") or []
                if len(shape) == 2:
                    hidden = shape[1]
                break
        if hidden == 2560:
            return KREA2
        if hidden == 1024:
            return ANIMA
        p = _param_count(h, keys)
        if p >= 2_500_000_000:
            return KREA2
        if p <= 1_500_000_000:
            return ANIMA
        return UNKNOWN

    if kind == "diffusion_models":
        prefixes = {k.split(".")[0] for k in keys}
        if {"txtfusion", "tmlp", "tproj"} & prefixes:
            return KREA2
        if "net" in prefixes:
            return ANIMA
        # SD/SDXL U-Net（単体ファイル / フルチェックポイントの両形式）。
        # パラメタ数フォールバックより先に判定しないと SDXL (~2.6-3.5B) が
        # anima に誤分類される。
        if any(k.startswith(("input_blocks.",
                             "model.diffusion_model.input_blocks."))
               for k in keys):
            return SDXL
        p = _param_count(h, keys)
        if p >= 8_000_000_000:
            return KREA2
        if p <= 4_000_000_000:
            return ANIMA
        return UNKNOWN

    return UNKNOWN


def _filename_family(kind: str, name: str) -> str:
    """Last-resort guess from the filename (non-safetensors / unreadable file)."""
    n = name.lower()
    if kind == "vae":
        if "sdxl" in n or "sd_xl" in n:
            return SDXL
        return SHARED if "qwen_image" in n else UNKNOWN
    if "sdxl" in n or "sd_xl" in n:
        return SDXL
    if kind == "text_encoders" and ("clip_l" in n or "clip_g" in n):
        return SDXL
    if "krea2" in n or "krea-2" in n:
        return KREA2
    if kind == "text_encoders" and "qwen3vl" in n:
        return KREA2
    if "anima" in n:
        return ANIMA
    if kind == "text_encoders" and ("qwen_3" in n or "qwen3" in n):
        return ANIMA
    return UNKNOWN


def sdxl_te_kind(path: Path) -> Optional[str]:
    """SDXL 系 TE の "clip_l" / "clip_g" 判別（それ以外は None）。

    token embedding の隠れ次元で判定する: 768 -> clip_l、1280 -> clip_g
    （open_clip 形式のスタンドアロン clip_g は ``token_embedding.weight``）。
    ヘッダが読めない場合はファイル名で推測する。
    """
    h = _read_header(path) if path.suffix.lower() == ".safetensors" else None
    if h:
        v = (h.get("text_model.embeddings.token_embedding.weight")
             or h.get("token_embedding.weight"))
        shape = (v or {}).get("shape") or []
        if len(shape) == 2:
            if shape[1] == 768:
                return "clip_l"
            if shape[1] == 1280:
                return "clip_g"
    n = path.name.lower()
    if "clip_l" in n:
        return "clip_l"
    if "clip_g" in n:
        return "clip_g"
    return None


# path -> (size, mtime, has_vae, has_clip)
_builtin_cache: dict[str, tuple[int, float, bool, bool]] = {}


def builtin_components(path: Path) -> tuple[bool, bool]:
    """(has_builtin_vae, has_builtin_clip) for a diffusion file.

    Full SD/SDXL checkpoints bundle the VAE (``first_stage_model.*`` keys) and
    the CLIP text encoders (``conditioner.*`` for SDXL / ``cond_stage_model.*``
    for SD1.x). UNet-only diffusion files (anima/krea2 の Qwen-Image、split
    SDXL の UNet 単体) carry neither. Header-only, cached by size+mtime."""
    try:
        st = path.stat()
    except OSError:
        return (False, False)
    key = str(path)
    c = _builtin_cache.get(key)
    if c and c[0] == st.st_size and c[1] == st.st_mtime:
        return (c[2], c[3])
    has_vae = has_clip = False
    if path.suffix.lower() == ".safetensors":
        h = _read_header(path)
        if h:
            has_vae = any(k.startswith("first_stage_model.") for k in h)
            has_clip = any(k.startswith(("conditioner.", "cond_stage_model."))
                           for k in h)
    _builtin_cache[key] = (st.st_size, st.st_mtime, has_vae, has_clip)
    return (has_vae, has_clip)


def is_checkpoint(path: Path) -> bool:
    """True if ``path`` is a full checkpoint bundling both VAE and CLIP, so it
    can be loaded via CheckpointLoaderSimple without separate VAE/TE files."""
    has_vae, has_clip = builtin_components(path)
    return has_vae and has_clip


def family(kind: str, path: Path) -> str:
    """Return 'anima' | 'krea2' | 'shared' | 'unknown' for the model file.

    Reads the safetensors header (cached by size+mtime); falls back to the
    filename when the file isn't safetensors or the header can't be parsed.
    """
    try:
        st = path.stat()
    except OSError:
        return UNKNOWN
    key = str(path)
    cached = _cache.get(key)
    if cached and cached[0] == st.st_size and cached[1] == st.st_mtime:
        return cached[2]

    fam = UNKNOWN
    if path.suffix.lower() == ".safetensors":
        h = _read_header(path)
        if h is not None:
            fam = _classify_header(kind, h)
    if fam == UNKNOWN:
        fam = _filename_family(kind, path.name)

    _cache[key] = (st.st_size, st.st_mtime, fam)
    return fam

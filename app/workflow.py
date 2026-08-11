"""MiniMax H3 video generation graphs (ComfyUI API format).

公式テンプレート (Comfy-Org/workflow_templates video_minimax_h3_*.json) の
配線をそのまま API 形式で構築する:

  UNETLoader ─┬─(任意 MiniMaxH3SigmaShift)─┬→ BasicScheduler → sigmas
              │                            └→ BasicGuider ← conditioning
  CLIPLoader(type="minimax") ─┐
  VAELoader(video) ───────────┼→ MiniMaxH3ImageToVideo / ReferenceToVideo
  VAELoader(audio) ───────────┘        → (positive, LATENT[video+audio])
  RandomNoise + KSamplerSelect + BasicGuider + sigmas
      → SamplerCustomAdvanced → VAEDecode(video) → frames ┐
                              → VAEDecodeAudio(audio) ────┼→ CreateVideo(24fps)
                                                          └→ SaveVideo

特徴: CFG 無し（BasicGuider）・ネガティブプロンプト無し。長さは 24fps の
17n+5 フレームグリッドにスナップされる。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

FPS = 24

# 公式テンプレの既定は res_multistep / simple / 20 steps。
SAMPLERS = [
    "res_multistep", "euler", "euler_ancestral", "dpmpp_2m", "lms", "heun",
    "ddim", "uni_pc", "er_sde",
]
SCHEDULERS = ["simple", "normal", "sgm_uniform", "beta", "karras",
              "exponential"]

# モデルが公式対応するアスペクト比（幅:高さ）。
ASPECT_PRESETS = [
    ("21:9", 21, 9),
    ("16:9", 16, 9),
    ("4:3", 4, 3),
    ("1:1", 1, 1),
    ("3:4", 3, 4),
    ("9:16", 9, 16),
]

# 解像度の目安（メガピクセル）。1.0 ≒ 768p 標準、0.4 は軽量・高速。
# 2.0736 = 1920x1080 相当（実サイズは32の倍数丸めで 16:9 → 1920x1088）。
# 学習中心の約1MPを超えるため品質が崩れる可能性あり・VRAM/時間も増える。
QUALITY_PRESETS = [
    ("標準 (768p級)", 1.0),
    ("HD級 (1080p級)", 2.0736),
    ("軽量 (0.4MP)", 0.4),
]

_SIZE_MULTIPLE = 32


def frames_for_seconds(seconds: float) -> int:
    """秒数を 24fps・17n+5 グリッドのフレーム数へスナップ（公式テンプレの式）。"""
    f = max(5, round(seconds * FPS))
    return f + (5 - (f % 17)) % 17


def size_for_aspect(aw: int, ah: int, megapixels: float) -> tuple[int, int]:
    """アスペクト比と目標画素数から 32 の倍数の (width, height) を求める。

    公式テンプレの ResolutionSelector と同じく幅を先に丸め、高さは丸めた幅
    から従属して丸める（16:9 1.0MP → 1344x768、0.2MP → 608x352 に一致）。
    """
    ratio = aw / ah
    w = math.sqrt(megapixels * 1_000_000 * ratio)
    w = max(_SIZE_MULTIPLE, round(w / _SIZE_MULTIPLE) * _SIZE_MULTIPLE)
    h = max(_SIZE_MULTIPLE, round(w / ratio / _SIZE_MULTIPLE) * _SIZE_MULTIPLE)
    return int(w), int(h)


def size_for_image(img_w: int, img_h: int, megapixels: float) -> tuple[int, int]:
    """入力画像のアスペクト比を保ったまま目標画素数へ（i2v 用）。"""
    return size_for_aspect(img_w, img_h, megapixels)


@dataclass
class GenParams:
    """1回の動画生成のスナップショット。

    画像/動画/音声の参照ファイルは、事前に ComfyUI の input ディレクトリへ
    アップロード済みの名前（LoadImage/LoadVideo/LoadAudio が読める形）で持つ。
    """
    mode: str = "t2v"            # t2v | i2v | r2v
    diffusion: str = ""          # fl2va (t2v/i2v) / ref2va (r2v) のファイル名
    te: str = ""
    vae_video: str = ""
    vae_audio: str = ""
    prompt: str = ""
    width: int = 1344
    height: int = 768
    frames: int = 124            # 24fps・17n+5 グリッド（124 ≒ 5秒）
    steps: int = 20
    sampler: str = "res_multistep"
    scheduler: str = "simple"
    seed: int = 0
    weight_dtype: str = "default"  # UNETLoader の読み込み精度
    # Sigma shift（OFF ならノード自体を入れずモデル既定に任せる）
    shift_enabled: bool = False
    shift_video: float = 12.0
    shift_audio: float = 3.0
    # Applied LoRAs: [(models/loras 内のファイル名, 強度), ...] を順に連結。
    # 強度は model / clip 共通。
    loras: list[tuple[str, float]] = field(default_factory=list)
    # EasyCache: 変化の小さいサンプリングステップをスキップする公式高速化。
    # threshold を上げるほど速いが品質が落ちる（既定 0.2）。v0.31 で H3 の
    # 音声破損が修正され安全に併用できる。
    easycache_enabled: bool = False
    easycache_threshold: float = 0.2
    # i2v: 開始/終端フレーム（アップロード済み画像名。空 = 未指定）
    first_frame: str = ""
    last_frame: str = ""
    # r2v: 参照（アップロード済みファイル名）
    ref_image_size: str = "match"
    ref_images: list[str] = field(default_factory=list)   # 最大9
    # [{"name": <video file>, "use_audio": bool}] 最大3
    ref_videos: list[dict] = field(default_factory=list)
    ref_audios: list[str] = field(default_factory=list)   # 最大3
    filename_prefix: str = "video/scomv"


def build_graph(p: GenParams) -> dict:
    """Return a ComfyUI API-format prompt graph for the given parameters."""
    if not p.diffusion:
        raise ValueError("diffusion モデルを選択してください")
    if not p.te:
        raise ValueError("text encoder を選択してください")
    if not p.vae_video:
        raise ValueError("動画 VAE を選択してください")
    if not p.vae_audio:
        raise ValueError("音声 VAE を選択してください")
    if p.mode not in ("t2v", "i2v", "r2v"):
        raise ValueError(f"不明なモードです: {p.mode}")
    if p.mode == "i2v" and not p.first_frame and not p.last_frame:
        raise ValueError("i2v には開始フレーム（または終端フレーム）画像が必要です")
    if p.mode == "r2v" and not (p.ref_images or p.ref_videos or p.ref_audios):
        raise ValueError("r2v には参照（画像/動画/音声）が1つ以上必要です")

    g: dict[str, dict] = {}
    g["1"] = {"class_type": "UNETLoader",
              "inputs": {"unet_name": p.diffusion,
                         "weight_dtype": p.weight_dtype}}
    g["2"] = {"class_type": "CLIPLoader",
              "inputs": {"clip_name": p.te, "type": "minimax",
                         "device": "default"}}
    g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": p.vae_video}}
    g["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": p.vae_audio}}

    # LoRA chain: model と clip を LoraLoader に順に通す（node id 60〜。
    # 20〜は画像/参照ローダが使うため衝突しない）。
    model_src: list = ["1", 0]
    clip_src: list = ["2", 0]
    for i, (lora_name, strength) in enumerate(p.loras):
        if not lora_name:
            raise ValueError("LoRA のファイル名が空です")
        nid = str(60 + i)
        g[nid] = {"class_type": "LoraLoader",
                  "inputs": {"lora_name": lora_name,
                             "strength_model": float(strength),
                             "strength_clip": float(strength),
                             "model": model_src, "clip": clip_src}}
        model_src = [nid, 0]
        clip_src = [nid, 1]

    if p.shift_enabled:
        g["6"] = {"class_type": "MiniMaxH3SigmaShift",
                  "inputs": {"model": model_src,
                             "shift_video": float(p.shift_video),
                             "shift_audio": float(p.shift_audio)}}
        model_src = ["6", 0]

    if p.easycache_enabled:
        g["17"] = {"class_type": "EasyCache",
                   "inputs": {"model": model_src,
                              "reuse_threshold": float(p.easycache_threshold),
                              "start_percent": 0.15, "end_percent": 0.95,
                              "verbose": False}}
        model_src = ["17", 0]

    # ----- conditioning + AV latent ---------------------------------------
    if p.mode in ("t2v", "i2v"):
        inputs = {
            "clip": clip_src,
            "vae": ["3", 0],
            "prompt": p.prompt,
            "width": int(p.width),
            "height": int(p.height),
            "length": int(p.frames),
        }
        nid = 20
        if p.first_frame:
            g[str(nid)] = {"class_type": "LoadImage",
                           "inputs": {"image": p.first_frame}}
            inputs["first_frame"] = [str(nid), 0]
            nid += 1
        if p.last_frame:
            g[str(nid)] = {"class_type": "LoadImage",
                           "inputs": {"image": p.last_frame}}
            inputs["last_frame"] = [str(nid), 0]
            nid += 1
        g["5"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": inputs}
    else:  # r2v
        if len(p.ref_images) > 9:
            raise ValueError("参照画像は最大9枚です")
        if len(p.ref_videos) > 3:
            raise ValueError("参照動画は最大3本です")
        if len(p.ref_audios) > 3:
            raise ValueError("参照音声は最大3本です")
        inputs = {
            "clip": clip_src,
            "vae": ["3", 0],
            "audio_vae": ["4", 0],
            "prompt": p.prompt,
            "width": int(p.width),
            "height": int(p.height),
            "length": int(p.frames),
            "ref_image_size": p.ref_image_size,
        }
        # Autogrow 入力の API 形式は「<親ID>.<プレフィックス><0始まり連番>」
        # （例: ref_images.ref_image_0）。プロンプト内の <Picture i> タグは
        # 1始まりだが、入力キーは 0 始まりである点に注意。
        nid = 20
        for i, name in enumerate(p.ref_images):
            g[str(nid)] = {"class_type": "LoadImage",
                           "inputs": {"image": name}}
            inputs[f"ref_images.ref_image_{i}"] = [str(nid), 0]
            nid += 1
        for i, rv in enumerate(p.ref_videos):
            load_id = str(nid); nid += 1
            comp_id = str(nid); nid += 1
            g[load_id] = {"class_type": "LoadVideo",
                          "inputs": {"file": rv["name"]}}
            g[comp_id] = {"class_type": "GetVideoComponents",
                          "inputs": {"video": [load_id, 0]}}
            inputs[f"ref_videos.ref_video_{i}"] = [comp_id, 0]
            if rv.get("use_audio"):
                inputs[f"ref_video_audios.ref_video_audio_{i}"] = [comp_id, 1]
        for i, name in enumerate(p.ref_audios):
            g[str(nid)] = {"class_type": "LoadAudio",
                           "inputs": {"audio": name}}
            inputs[f"ref_audios.ref_audio_{i}"] = [str(nid), 0]
            nid += 1
        g["5"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": inputs}

    # ----- sampling (CFG 無し: BasicGuider) --------------------------------
    g["7"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": p.seed}}
    g["8"] = {"class_type": "KSamplerSelect",
              "inputs": {"sampler_name": p.sampler}}
    g["9"] = {"class_type": "BasicScheduler",
              "inputs": {"model": model_src, "scheduler": p.scheduler,
                         "steps": int(p.steps), "denoise": 1.0}}
    g["10"] = {"class_type": "BasicGuider",
               "inputs": {"model": model_src, "conditioning": ["5", 0]}}
    g["11"] = {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["7", 0], "guider": ["10", 0],
                          "sampler": ["8", 0], "sigmas": ["9", 0],
                          "latent_image": ["5", 1]}}

    # ----- decode + mux + save --------------------------------------------
    g["12"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["11", 0], "vae": ["3", 0]}}
    g["13"] = {"class_type": "VAEDecodeAudio",
               "inputs": {"samples": ["11", 0], "vae": ["4", 0]}}
    g["14"] = {"class_type": "CreateVideo",
               "inputs": {"images": ["12", 0], "fps": float(FPS),
                          "audio": ["13", 0], "bit_depth": 8}}
    # codec は DynamicCombo: API 形式ではオプションキーの文字列を渡す
    # （実行側が {"codec": "auto"} に組み立てて execute に渡す）。
    g["15"] = {"class_type": "SaveVideo",
               "inputs": {"video": ["14", 0],
                          "filename_prefix": p.filename_prefix,
                          "format": "auto", "codec": "auto"}}
    return g

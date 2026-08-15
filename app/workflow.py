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

import json
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
    # ContexLoop（長尺チェーン）の設定。mode == "chain" のとき必須。
    # ChainDialog.plan() の辞書に、参照/開始フレームをアップロード済み
    # ファイル名へ差し替えたものを入れる。
    chain: dict | None = None


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
    if p.mode == "chain":
        return build_chain_graph(p)
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


# ----- ContexLoop（長尺チェーン）------------------------------------------
# サードパーティのカスタムノードパック ComfyUI-MiniMaxH3-Contex-Loop を使う。
# 1回のプロンプト実行でグラフが再帰展開され、全シーンが順に生成される:
#
#   Plan → LoopStart ─flow──────────────────────────────────────┐
#            └→ Current ─┬→ (FirstSceneImage) → 条件付けノード  │
#                        ├→ RandomNoise / BasicScheduler        │
#                        └→ Context → BasicGuider → Sampler     │
#                             → VAEDecode(+Audio) → LoopTrim    │
#                             → SegmentSave → LoopEnd ──────────┘
#                                                └→ Assemble（結合mp4）
CHAIN_ENCODE_MODE = "video"
CHAIN_ANCHOR_MODE = "head"
CHAIN_CROP = "disabled"
CHAIN_MAX_SHOTS = 128


def chain_plan_json(chain: dict) -> str:
    """ChainDialog の設定を Plan ノードの plan_json 文字列にする。"""
    shots = []
    for s in chain.get("shots", []):
        shot: dict = {
            "id": str(s.get("id") or "").strip(),
            "prompt": str(s.get("prompt") or ""),
            "duration_seconds": float(s.get("duration_seconds") or 5.0),
        }
        if int(s.get("steps") or 0) > 0:
            shot["steps"] = int(s["steps"])
        seed = str(s.get("seed") or "").strip()
        if seed:
            shot["seed"] = seed
        shots.append(shot)
    doc: dict = {"shots": shots}
    prefix = str(chain.get("prompt_prefix") or "")
    if prefix.strip():
        doc["prompt_prefix"] = prefix
    return json.dumps(doc, ensure_ascii=False)


def chain_frames(chain: dict) -> tuple[int, int]:
    """(生成フレーム合計, 実尺フレーム) を返す（引き継ぎ分を差し引く）。"""
    raw = [frames_for_seconds(float(s.get("duration_seconds") or 5.0))
           for s in chain.get("shots", [])]
    ctx = int(chain.get("context_length") or 22)
    return sum(raw), sum(raw) - ctx * max(0, len(raw) - 1)


def validate_chain(chain: dict) -> None:
    """生成前チェック（UI からも呼べるよう公開）。"""
    shots = chain.get("shots") or []
    if not shots:
        raise ValueError("シーンが1つもありません")
    if len(shots) > CHAIN_MAX_SHOTS:
        raise ValueError(f"シーンは最大 {CHAIN_MAX_SHOTS} 個です")
    prefix = str(chain.get("prompt_prefix") or "").strip()
    for i, s in enumerate(shots, 1):
        if not str(s.get("prompt") or "").strip() and not prefix:
            raise ValueError(f"シーン {i} のプロンプトが空です")
    ids = [str(s.get("id") or "").strip() for s in shots]
    if "" in ids:
        raise ValueError("シーン ID が空のものがあります")
    if len(set(ids)) != len(ids):
        raise ValueError("シーン ID が重複しています")
    ctx = int(chain.get("context_length") or 22)
    raw = [frames_for_seconds(float(s.get("duration_seconds") or 5.0))
           for s in shots]
    short = [i for i, f in enumerate(raw[:-1], 1) if f <= ctx]
    if short:
        raise ValueError(
            "シーン " + ", ".join(map(str, short))
            + f" は引き継ぎフレーム数（{ctx}）以下のため生成できません")
    kind = chain.get("chain_type")
    if kind == "i2v" and not str(shots[0].get("first_frame") or "").strip():
        raise ValueError("i2v チェーンにはシーン1の開始フレーム画像が必要です")
    if kind == "r2v" and not chain.get("references"):
        raise ValueError("r2v チェーンには参照素材が1つ以上必要です")
    if (chain.get("audio_mode") in ("source_track", "source_plus_timeline")
            and not str(chain.get("audio_file") or "").strip()):
        raise ValueError("この音声モードには外部音源ファイルが必要です")


def build_chain_graph(p: GenParams) -> dict:
    """ContexLoop の長尺チェーン用 API グラフを組み立てる。"""
    chain = p.chain or {}
    validate_chain(chain)
    kind = chain.get("chain_type") or "t2v"

    g: dict[str, dict] = {}
    g["1"] = {"class_type": "UNETLoader",
              "inputs": {"unet_name": p.diffusion,
                         "weight_dtype": p.weight_dtype}}
    g["2"] = {"class_type": "CLIPLoader",
              "inputs": {"clip_name": p.te, "type": "minimax",
                         "device": "default"}}
    g["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": p.vae_video}}
    g["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": p.vae_audio}}

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

    # ----- チェーン制御 ----------------------------------------------------
    g["100"] = {"class_type": "MiniMaxH3ChainPlan", "inputs": {
        "plan_json": chain_plan_json(chain),
        "run_name": str(chain.get("run_name") or "h3_chain"),
        "generation_fingerprint": "",
        "width": int(p.width),
        "height": int(p.height),
        "context_length": int(chain.get("context_length") or 22),
        "encode_mode": CHAIN_ENCODE_MODE,
        "anchor_mode": CHAIN_ANCHOR_MODE,
        "crop": CHAIN_CROP,
        "audio_mode": str(chain.get("audio_mode") or "generated_audio"),
        "audio_context_length": int(chain.get("audio_context_length") or 22),
        "default_duration_seconds": 15.0,
        "default_steps": int(chain.get("default_steps") or p.steps),
        "base_seed": int(str(chain.get("base_seed") or "0") or 0),
        "segment_crf": int(chain.get("segment_crf") or 18),
    }}

    start_inputs: dict = {"plan": ["100", 0], "start_clip": 1,
                          "scene_range": str(chain.get("scene_range") or "")}
    audio_file = str(chain.get("audio_file") or "").strip()
    if audio_file:
        g["150"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_file}}
        start_inputs["source_audio"] = ["150", 0]
    g["101"] = {"class_type": "MiniMaxH3ChainLoopStart", "inputs": start_inputs}

    cur_inputs: dict = {"state": ["101", 1]}
    if audio_file:
        cur_inputs["source_audio"] = ["150", 0]
    g["102"] = {"class_type": "MiniMaxH3ChainCurrent", "inputs": cur_inputs}

    # ----- 条件付け（シーンごとの値は Current から流れる）------------------
    if kind == "r2v":
        # 参照スケジュール: 各素材を previous で数珠つなぎにする。
        prev: list | None = None
        nid = 140
        for ref in chain.get("references", []):
            name = str(ref.get("name") or "")
            if not name:
                continue
            tag = str(ref.get("tag") or "")
            scenes = str(ref.get("scenes") or "") or "all"
            rkind = ref.get("kind")
            if rkind == "image":
                load = str(nid); nid += 1
                node = str(nid); nid += 1
                g[load] = {"class_type": "LoadImage",
                           "inputs": {"image": name}}
                ins = {"image": [load, 0], "tag": tag, "scenes": scenes}
                if prev:
                    ins["previous"] = prev
                g[node] = {"class_type": "MiniMaxH3ScheduledPictureReference",
                           "inputs": ins}
            elif rkind == "video":
                load = str(nid); nid += 1
                prep = str(nid); nid += 1
                node = str(nid); nid += 1
                g[load] = {"class_type": "LoadVideo",
                           "inputs": {"file": name}}
                # 参照動画は H3 有効長（17n+5）へ整える。既定 209f ≒ 8.7秒。
                g[prep] = {"class_type": "MiniMaxH3ReferenceVideoPrepare",
                           "inputs": {"source_video": [load, 0],
                                      "length": int(ref.get("length") or 209),
                                      "source_fps": float(FPS)}}
                ins = {"video": [prep, 0], "audio": [prep, 1],
                       "tag": tag, "scenes": scenes,
                       "audio_tag": tag + "_audio"}
                if prev:
                    ins["previous"] = prev
                g[node] = {"class_type": "MiniMaxH3ScheduledVideoReference",
                           "inputs": ins}
            else:
                load = str(nid); nid += 1
                node = str(nid); nid += 1
                g[load] = {"class_type": "LoadAudio",
                           "inputs": {"audio": name}}
                ins = {"audio": [load, 0], "tag": tag, "scenes": scenes}
                if prev:
                    ins["previous"] = prev
                g[node] = {"class_type": "MiniMaxH3ScheduledAudioReference",
                           "inputs": ins}
            prev = [node, 0]
        if prev is None:
            raise ValueError("r2v チェーンには参照素材が1つ以上必要です")
        g["110"] = {"class_type": "MiniMaxH3ScheduledReferenceToVideo",
                    "inputs": {
                        "clip": clip_src, "vae": ["3", 0],
                        "audio_vae": ["4", 0],
                        "reference_schedule": prev,
                        "clip_index": ["102", 1],
                        "clip_count": ["102", 2],
                        "prompt": ["102", 4],
                        "width": ["102", 8],
                        "height": ["102", 9],
                        "length": ["102", 6],
                        "ref_image_size": str(p.ref_image_size or "match"),
                    }}
    else:
        inputs: dict = {
            "clip": clip_src, "vae": ["3", 0],
            "prompt": ["102", 4],
            "width": ["102", 8],
            "height": ["102", 9],
            "length": ["102", 6],
        }
        if kind == "i2v":
            first = str(chain["shots"][0].get("first_frame") or "").strip()
            g["108"] = {"class_type": "LoadImage", "inputs": {"image": first}}
            # シーン1にだけ開始フレームを通す（2シーン目以降は渡らない）。
            g["107"] = {"class_type": "MiniMaxH3ChainFirstSceneImage",
                        "inputs": {"state": ["102", 0], "image": ["108", 0]}}
            inputs["first_frame"] = ["107", 0]
        g["110"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": inputs}

    # ----- 文脈の適用 → サンプリング ---------------------------------------
    g["103"] = {"class_type": "MiniMaxH3ChainContext",
                "inputs": {"state": ["102", 0], "conditioning": ["110", 0],
                           "vae": ["3", 0], "latent": ["110", 1],
                           "audio_vae": ["4", 0]}}
    g["120"] = {"class_type": "RandomNoise",
                "inputs": {"noise_seed": ["102", 5]}}
    g["121"] = {"class_type": "BasicGuider",
                "inputs": {"model": model_src, "conditioning": ["103", 0]}}
    g["122"] = {"class_type": "KSamplerSelect",
                "inputs": {"sampler_name": p.sampler}}
    g["123"] = {"class_type": "BasicScheduler",
                "inputs": {"model": model_src, "scheduler": p.scheduler,
                           "steps": ["102", 7], "denoise": 1.0}}
    g["124"] = {"class_type": "SamplerCustomAdvanced",
                "inputs": {"noise": ["120", 0], "guider": ["121", 0],
                           "sampler": ["122", 0], "sigmas": ["123", 0],
                           "latent_image": ["110", 1]}}
    g["130"] = {"class_type": "VAEDecode",
                "inputs": {"samples": ["124", 0], "vae": ["3", 0]}}
    g["131"] = {"class_type": "VAEDecodeAudio",
                "inputs": {"samples": ["124", 0], "vae": ["4", 0]}}
    g["132"] = {"class_type": "MiniMaxH3LoopTrim",
                "inputs": {"images": ["130", 0], "audio": ["131", 0],
                           "trim_frames": ["103", 1], "fps": float(FPS),
                           "match_tail": True}}

    # ----- 保存 → 次シーンへ再帰 → 結合 -----------------------------------
    g["104"] = {"class_type": "MiniMaxH3ChainSegmentSave",
                "inputs": {"state": ["102", 0], "images": ["132", 0],
                           "sampled_latent": ["124", 0],
                           "audio": ["132", 1]}}
    # レビュー（人手承認）は使わず SegmentSave を LoopEnd に直結する。
    g["105"] = {"class_type": "MiniMaxH3ChainLoopEnd",
                "inputs": {"flow": ["101", 0], "state": ["102", 0],
                           "images": ["132", 0],
                           "sampled_latent": ["124", 0],
                           "segment": ["104", 0]}}
    asm: dict = {"manifest": ["105", 0],
                 "audio_source": "plan",
                 "filename": str(chain.get("final_name") or "final"),
                 "audio_bitrate": int(chain.get("audio_bitrate") or 256)}
    if audio_file:
        asm["source_audio"] = ["150", 0]
    g["106"] = {"class_type": "MiniMaxH3ChainAssemble", "inputs": asm}
    return g

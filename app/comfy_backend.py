"""Embedded ComfyUI backend: launch as a local subprocess and drive via API.

Responsibilities:
  * Generate an ``extra_model_paths.yaml`` mapping our models/{diffusion,vae,te}
    folders onto ComfyUI's expected categories.
  * Start ``main.py`` as a subprocess bound to 127.0.0.1 on a chosen port.
  * Wait until the HTTP server is reachable.
  * Queue prompts, stream progress over the websocket, and return the decoded
    output image bytes.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import websocket  # websocket-client

from . import config
from .comfy_custom_nodes import ensure_custom_nodes
from .textutil import strip_ansi


def _free_port(preferred: int = 8199) -> int:
    """Return a usable localhost port, preferring ``preferred``."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("could not allocate a localhost port")


def write_extra_model_paths(paths: config.AppPaths) -> Path:
    """Write extra_model_paths.yaml pointing ComfyUI at our models folder.

    Also installs the scom-v custom node (taeh3 preview) and registers its
    directory — an absolute path passes through ComfyUI's os.path.join with
    base_path unchanged, so it can live outside the models tree.
    """
    models = paths.models
    nodes_dir = ensure_custom_nodes(paths)
    yaml_text = (
        "scomv:\n"
        f"  base_path: {models.as_posix()}\n"
        "  is_default: true\n"
        "  diffusion_models: diffusion_models/\n"
        "  vae: vae/\n"
        "  text_encoders: text_encoders/\n"
        "  loras: loras/\n"
        "  vae_approx: vae_approx/\n"
        f"  custom_nodes: {nodes_dir.as_posix()}\n"
    )
    out = paths.user_data / "extra_model_paths.yaml"
    out.write_text(yaml_text, encoding="utf-8")
    return out


@dataclass
class Progress:
    """A progress update streamed from the backend during generation."""
    value: int = 0
    maximum: int = 0
    note: str = ""


class BackendError(RuntimeError):
    pass


class ComfyBackend:
    """Manages the ComfyUI subprocess and exposes a simple generate() API."""

    def __init__(self, paths: Optional[config.AppPaths] = None, port: int = 8199):
        self.paths = paths or config.AppPaths()
        self.port = port
        self.host = "127.0.0.1"
        self.client_id = uuid.uuid4().hex
        # 起動前にメインウィンドウが設定から反映する（次回起動時に有効）。
        self.use_sage_attention = False
        self._proc: Optional[subprocess.Popen] = None
        self._log_thread: Optional[threading.Thread] = None
        self._log_tail: deque[str] = deque(maxlen=40)

    # ----- lifecycle -------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, log: Optional[Callable[[str], None]] = None,
              timeout: float = 120.0) -> None:
        """Launch ComfyUI and block until it responds, or raise BackendError."""
        log = log or (lambda _m: None)
        if self.is_running():
            return

        comfy = self.paths.comfyui
        if comfy is None:
            raise BackendError(
                "ComfyUI が見つかりません。SCOM_COMFYUI_DIR を設定するか "
                "vendor/ComfyUI に配置してください（README 参照）。"
            )

        config.ensure_model_dirs()
        # プレビュー用 tiny デコーダが欠けていれば取得を試みる（9.8MB。
        # 失敗しても続行 — 無い場合は Latent2RGB プレビューで動く）。
        from .bootstrap.setup import download_aux_models
        download_aux_models(self.paths, log)
        extra_paths = write_extra_model_paths(self.paths)
        self.port = _free_port(self.port)

        cmd = [
            str(self.paths.backend_python),
            str(comfy / "main.py"),
            "--listen", self.host,
            "--port", str(self.port),
            "--extra-model-paths-config", str(extra_paths),
            "--output-directory", str(self.paths.output_dir),
            "--preview-method", "auto",  # stream latent previews over the ws
            "--preview-size", "768",     # taeh3 デコードは16倍なので粗くしない
            "--disable-auto-launch",
        ]
        if self.use_sage_attention:
            # パッケージが実在するときだけフラグを付ける。無いのに付けると
            # ComfyUI は起動時に exit(-1) するため（attention.py 参照）。
            from .bootstrap.setup import sage_installed
            if sage_installed(self.paths):
                cmd.append("--use-sage-attention")
                log("SageAttention を有効化して起動します")
            else:
                log("SageAttention が未インストールのため無効で起動します"
                    "（「設定…」から導入できます）")
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        log(f"ComfyUI を起動中: {' '.join(cmd)}")

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(comfy),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
        # Drain stdout on a daemon thread so a quiet subprocess never blocks the
        # readiness poll below.
        self._start_log_reader(log)

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                time.sleep(0.2)  # let the reader flush the final lines
                raise BackendError(
                    f"ComfyUI が早期終了しました (code {self._proc.returncode})。\n"
                    + "\n".join(self._log_tail)
                )
            if self._ping():
                log(f"ComfyUI 準備完了: {self.base_url}")
                return
            time.sleep(0.4)
        self.stop()
        raise BackendError("制限時間内に ComfyUI が起動しませんでした")

    def _start_log_reader(self, log: Callable[[str], None]) -> None:
        def reader() -> None:
            assert self._proc and self._proc.stdout
            for raw in self._proc.stdout:
                raw = raw.rstrip("\r\n")
                clean = strip_ansi(raw).rstrip()
                if clean:
                    self._log_tail.append(clean)  # plain text for error messages
                    log(raw)                       # raw (with ANSI) for colored display

        self._log_thread = threading.Thread(target=reader, daemon=True)
        self._log_thread.start()

    def _ping(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/system_stats", timeout=2):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            if sys.platform == "win32":
                # Kill the whole process TREE. uv-managed venvs use a
                # trampoline python.exe that launches the real interpreter as
                # a child; terminate() alone kills only the trampoline and
                # orphans the actual ComfyUI process (which then keeps the
                # SQLite DB locked for every later launch).
                subprocess.run(
                    ["taskkill", "/PID", str(self._proc.pid), "/T", "/F"],
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    # ----- generation ------------------------------------------------------
    def _post_prompt(self, graph: dict,
                     extra_pnginfo: Optional[dict] = None) -> str:
        body: dict = {"prompt": graph, "client_id": self.client_id}
        if extra_pnginfo:
            # SaveVideo の hidden.extra_pnginfo に渡り、mp4 のコンテナタグ
            # として埋め込まれる（文字列はそのまま、辞書は JSON 化）。
            body["extra_data"] = {"extra_pnginfo": extra_pnginfo}
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            self.base_url + "/prompt", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise BackendError(f"prompt が拒否されました: {detail}") from e
        return data["prompt_id"]

    def _history_files(self, prompt_id: str) -> list[Path]:
        """Saved output files of a finished prompt, as absolute paths.

        SaveVideo 等の出力は history に {filename, subfolder, type} で載る
        （キー名は images/videos などノードにより異なるため全キーを走査）。
        バックエンドは --output-directory でアプリの output/ を指しているので、
        /view でダウンロードし直さずローカルパスをそのまま解決できる。
        """
        with urllib.request.urlopen(
            self.base_url + f"/history/{prompt_id}", timeout=30
        ) as resp:
            history = json.loads(resp.read())
        entry = history.get(prompt_id, {})
        out_dir = self.paths.output_dir
        files: list[Path] = []
        for node_out in entry.get("outputs", {}).values():
            for items in node_out.values():
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not (isinstance(it, dict) and it.get("filename")):
                        continue
                    if it.get("type", "output") != "output":
                        continue  # temp プレビュー等は成果物ではない
                    p = out_dir / it.get("subfolder", "") / it["filename"]
                    if p.exists() and p not in files:
                        files.append(p)
        return files

    def upload_input_file(self, path: Path) -> str:
        """Upload a local file into the backend's input directory.

        LoadImage / LoadVideo / LoadAudio が参照できる名前を返す（ComfyUI の
        /upload/image は画像以外のファイルもそのまま input へ保存する）。
        同名ファイルは ComfyUI 側がリネームで衝突回避し、その名前が返る。
        """
        path = Path(path)
        data = path.read_bytes()
        boundary = "----scomv" + uuid.uuid4().hex
        fname = path.name.replace('"', "_")
        body = (
            (f"--{boundary}\r\n"
             f'Content-Disposition: form-data; name="image"; '
             f'filename="{fname}"\r\n'
             "Content-Type: application/octet-stream\r\n\r\n").encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )
        req = urllib.request.Request(
            self.base_url + "/upload/image", data=body,
            headers={"Content-Type":
                     f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                info = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise BackendError(f"ファイルのアップロードに失敗: {e}") from e
        name = info.get("name", fname)
        sub = info.get("subfolder", "")
        return f"{sub}/{name}" if sub else name

    def generate(self, graph: dict,
                 on_progress: Optional[Callable[[Progress], None]] = None,
                 on_preview: Optional[Callable[[bytes], None]] = None,
                 cancel: Optional[Callable[[], bool]] = None,
                 on_cached: Optional[Callable[[list], None]] = None,
                 on_timing: Optional[Callable[[float], None]] = None,
                 extra_pnginfo: Optional[dict] = None) -> list[Path]:
        """Run a graph to completion and return the saved output file paths
        (SaveVideo が userdata/output に書いた動画等の絶対パス)。

        ``on_progress`` receives Progress updates; ``on_preview`` receives raw
        JPEG/PNG bytes of intermediate latent previews; ``cancel`` is polled
        and, if it returns True, the run is interrupted. ``on_cached`` receives
        the node ids served from the backend's output cache (sent once at the
        start of execution). ``on_timing`` receives, at completion, the pure
        inference time in seconds — the wall time the backend spent executing
        sampler nodes only, excluding model loading / text encode / VAE decode.
        """
        if not self.is_running():
            raise BackendError("バックエンドが起動していません")
        on_progress = on_progress or (lambda _p: None)
        on_preview = on_preview or (lambda _b: None)
        cancel = cancel or (lambda: False)
        on_cached = on_cached or (lambda _n: None)

        # 推論時間 = サンプラーノードが「実行中」だった時間の合計。
        # executing イベントはノード開始時に飛ぶので、サンプラーに入った時刻
        # から次のノードへ移った時刻までを積算する。
        sampler_nodes = {nid for nid, node in graph.items()
                         if node.get("class_type") in ("KSampler",
                                                       "SamplerCustomAdvanced")}
        sample_secs = 0.0
        sample_enter: Optional[float] = None

        ws = websocket.WebSocket()
        ws.connect(
            f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}", timeout=10
        )
        ws.settimeout(1.0)

        prompt_id = self._post_prompt(graph, extra_pnginfo)
        try:
            while True:
                if cancel():
                    self.interrupt()
                    raise BackendError("生成をキャンセルしました")
                try:
                    msg = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if isinstance(msg, (bytes, bytearray)):
                    # Binary frame: [4B event][4B image format][image bytes].
                    # event 1 == PREVIEW_IMAGE.
                    if len(msg) > 8 and int.from_bytes(msg[:4], "big") == 1:
                        on_preview(bytes(msg[8:]))
                    continue
                event = json.loads(msg)
                etype = event.get("type")
                data = event.get("data", {})
                if etype == "progress":
                    on_progress(Progress(
                        value=data.get("value", 0),
                        maximum=data.get("max", 0),
                        note="サンプリング",
                    ))
                elif etype == "execution_cached":
                    if data.get("prompt_id") == prompt_id:
                        on_cached(list(data.get("nodes", [])))
                elif etype == "executing":
                    if data.get("prompt_id") != prompt_id:
                        continue
                    node = data.get("node")
                    now = time.monotonic()
                    if sample_enter is not None and node not in sampler_nodes:
                        sample_secs += now - sample_enter
                        sample_enter = None
                    elif sample_enter is None and node in sampler_nodes:
                        sample_enter = now
                    if node is None:
                        break  # finished
                elif etype == "execution_error" and data.get("prompt_id") == prompt_id:
                    raise BackendError(
                        f"実行エラー: {data.get('exception_message', data)}"
                    )
        finally:
            try:
                ws.close()
            except Exception:
                pass

        if on_timing is not None and sample_secs > 0:
            on_timing(sample_secs)
        return self._history_files(prompt_id)

    def interrupt(self) -> None:
        try:
            req = urllib.request.Request(self.base_url + "/interrupt", data=b"")
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass

    def free_memory(self) -> None:
        """Ask the backend to drop ALL cached node outputs and loaded models.

        ComfyUI has no per-entry cache eviction, so this is all-or-nothing;
        anything still needed is rebuilt/reloaded on next use.
        """
        payload = json.dumps({"unload_models": True, "free_memory": True}).encode()
        req = urllib.request.Request(
            self.base_url + "/free", data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)

    def object_info(self, class_type: str) -> dict:
        """Fetch node metadata (used to discover valid sampler/clip options)."""
        with urllib.request.urlopen(
            self.base_url + f"/object_info/{class_type}", timeout=10
        ) as resp:
            return json.loads(resp.read())

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
import os
import re
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


# ----- 残留バックエンド対策 ---------------------------------------------------
# scom-v が異常終了（強制終了・クラッシュ・シャットダウン）すると ComfyUI の
# 子プロセスが取り残され、モデルを RAM/VRAM に抱えたまま、ポートと
# ComfyUI の SQLite DB（comfyui.db.lock）を掴み続ける。次回起動は別ポートに
# 逃げるので動きはするが、「Database is locked」が出て遅くなる。
#   1. Windows のジョブオブジェクト（KILL_ON_JOB_CLOSE）で scom-v が死ねば
#      バックエンドの木ごと OS に消させる。
#   2. 起動時に前回の PID ファイルを見て、持ち主の scom-v が居なければ掃除
#      する（1. が効かなかった場合や旧版の取り残し用）。
#   3. ComfyUI の DB はメモリ上にする（scom-v はアセット DB を使わない）ので、
#      多重起動しても DB ロックで待たされることが無い。
_PID_FILE = "comfyui.pid.json"
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


def _pid_alive(pid: int) -> bool:
    """Windows: その PID のプロセスがまだ動いているか（終了済み/不在は False）。"""
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def _listener_pid(port: int) -> Optional[int]:
    """127.0.0.1:port で LISTEN しているプロセスの PID（netstat 経由）。"""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "tcp"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=subprocess.CREATE_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    needle = f":{port}"
    for line in out.splitlines():
        cols = line.split()
        if (len(cols) >= 5 and cols[0].upper() == "TCP"
                and cols[1].endswith(needle) and cols[3] == "LISTENING"):
            try:
                return int(cols[4])
            except ValueError:
                return None
    return None


def _attach_kill_on_close_job(proc: subprocess.Popen):
    """子プロセスをジョブに入れ、ジョブハンドルが閉じたら（= scom-v が
    どんな死に方をしても）木ごと強制終了させる。戻り値はハンドル（保持
    し続けること）。失敗時は None（既存ジョブの制約下など）。"""
    if sys.platform != "win32" or os.environ.get("SCOMV_NO_JOB"):
        return None
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC_LIMIT),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.windll.kernel32
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                            ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
    ok = (k32.SetInformationJobObject(job, 9, ctypes.byref(info),
                                      ctypes.sizeof(info))   # ExtendedLimitInformation
          and k32.AssignProcessToJobObject(job, wintypes.HANDLE(proc._handle)))
    if not ok:
        k32.CloseHandle(job)
        return None
    return job


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


# ノードのテキスト出力に現れる動画の絶対パス（ContexLoop の Assemble 用）。
_VIDEO_PATH_RE = re.compile(
    r'[A-Za-z]:\\[^\r\n"<>|?*]+?\.(?:mp4|mkv|webm|mov)')


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
        self._job = None   # Windows ジョブオブジェクト（_attach_kill_on_close_job）
        self._log_thread: Optional[threading.Thread] = None
        self._log_tail: deque[str] = deque(maxlen=40)

    # ----- lifecycle -------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ----- 残留バックエンドの掃除 ------------------------------------------
    @property
    def _pid_file(self) -> Path:
        return self.paths.backend_root / _PID_FILE

    def _write_pid_file(self) -> None:
        try:
            self._pid_file.parent.mkdir(parents=True, exist_ok=True)
            self._pid_file.write_text(json.dumps({
                "owner_pid": os.getpid(), "pid": self._proc.pid,
                "port": self.port}), encoding="utf-8")
        except OSError:
            pass

    def _reap_stale_backend(self, log: Callable[[str], None]) -> None:
        """前回の scom-v が後始末できずに残した ComfyUI を終了させる。

        PID ファイルの持ち主（scom-v）がまだ生きていれば別インスタンスなので
        触らない。死んでいれば、記録したポートで応答している ComfyUI と
        記録した PID の木を止める。"""
        try:
            info = json.loads(self._pid_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        try:
            owner, pid, port = (int(info.get("owner_pid", 0)),
                                int(info.get("pid", 0)), int(info.get("port", 0)))
        except (TypeError, ValueError):
            self._pid_file.unlink(missing_ok=True)
            return
        if owner == os.getpid() or (owner and _pid_alive(owner)):
            return  # 自分、または起動中の別インスタンスのもの
        victims = []
        if port:
            try:
                with urllib.request.urlopen(
                        f"http://{self.host}:{port}/system_stats", timeout=2):
                    lp = _listener_pid(port)
                    if lp:
                        victims.append(lp)
            except (urllib.error.URLError, OSError):
                pass
        if pid and _pid_alive(pid) and pid not in victims:
            victims.append(pid)
        for v in victims:
            _kill_tree(v)
        if victims:
            log(f"前回の残留 ComfyUI プロセスを終了しました (PID {victims})")
            time.sleep(0.5)  # ポート解放を待つ
        self._pid_file.unlink(missing_ok=True)

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
        self._reap_stale_backend(log)
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
            # v0.37 以降は高速ディスクを自動検出してディスク経由の動的読込に
            # 切り替わるが、scom ではこれでプロセスごと落ちる事例があった
            # （anima / krea2 の TE で再現）。従来どおり RAM 経由で読む。
            "--disable-fast-disk",
            # ComfyUI のアセット DB は使わない。メモリ上にすると comfyui.db の
            # ファイルロックが無くなり、残留/多重起動時の「Database is
            # locked」とその待ちが起きない。
            "--database-url", "sqlite:///:memory:",
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
        # scom-v が死んだら OS にバックエンドを道連れにさせる + 次回起動時の
        # 掃除用に PID を記録する。
        self._job = _attach_kill_on_close_job(self._proc)
        self._write_pid_file()
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
        if self._job is not None and sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.CloseHandle(self._job)  # 残党も道連れ
            self._job = None
        try:
            self._pid_file.unlink(missing_ok=True)
        except OSError:
            pass

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
        # ContexLoop の Assemble は {filename} ではなく完成パスを含む文字列を
        # 返すため、テキスト出力からも動画パスを拾う（新しい順 = 最終結合物）。
        from_text: list[Path] = []
        for node_out in entry.get("outputs", {}).values():
            for items in node_out.values():
                if not isinstance(items, list):
                    continue
                for it in items:
                    if isinstance(it, str):
                        for m in _VIDEO_PATH_RE.finditer(it):
                            q = Path(m.group(0))
                            if q.exists() and q not in from_text:
                                from_text.append(q)
                        continue
                    if not (isinstance(it, dict) and it.get("filename")):
                        continue
                    if it.get("type", "output") != "output":
                        continue  # temp プレビュー等は成果物ではない
                    p = out_dir / it.get("subfolder", "") / it["filename"]
                    if p.exists() and p not in files:
                        files.append(p)
        for q in sorted(from_text, key=lambda x: x.stat().st_mtime,
                        reverse=True):
            if q not in files:
                files.append(q)
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

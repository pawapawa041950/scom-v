"""First-run setup orchestrator.

Runs the ordered steps needed to make the app usable on a fresh machine:
detect GPU -> fetch uv -> create venv -> install torch (tailored) -> fetch
ComfyUI -> install its deps -> download the selected anima models. Each step is
idempotent and records completion in ``setup_manifest.json`` so interrupted
runs resume. Progress is reported per component via ``StepUpdate`` events.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .. import config
from . import environment, models, uv_manager
from .downloader import download

LogCb = Callable[[str], None]

# Pin ComfyUI to a specific release tag for reproducible installs. The short
# /archive/<ref>.zip form resolves tags, branches, and commit SHAs alike, so
# SCOM_COMFYUI_REF can be overridden with any of them (e.g. "master").
# v0.28.0 (2026-07-15, commit 700821e): 旧ピン 1377a2f（v0.27 直後の
# 暫定 master、int4 ConvRot 対応目的）を含む正式リリース。
COMFYUI_REF = os.environ.get("SCOM_COMFYUI_REF", "v0.28.0")
COMFYUI_ZIP = f"https://github.com/comfyanonymous/ComfyUI/archive/{COMFYUI_REF}.zip"

# Bump when a newer ComfyUI is required (e.g. for a new model architecture such
# as Krea-2). Changing this re-fetches ComfyUI and reinstalls its deps on
# machines that were provisioned with an older copy. Tied to the pinned ref so
# the provisioned version is self-documenting.
COMFYUI_MARKER = os.environ.get("SCOM_COMFYUI_MARKER", "v0.28.0")

# Fixed (non-model) steps, in order: (step_id, title).
# Technical names (uv / PyTorch / ComfyUI) are kept in English by request.
FIXED_STEPS = [
    ("uv", "uv（バックエンドツール）"),
    ("venv", "Python 環境"),
    ("torch", "PyTorch"),
    ("comfyui", "ComfyUI"),
    ("deps", "ComfyUI 依存パッケージ"),
]


@dataclass
class StepUpdate:
    step_id: str                      # 'uv','venv','torch','comfyui','deps','model:<file>'
    title: str
    status: str                       # 'running' | 'done' | 'skipped' | 'error'
    fraction: Optional[float] = None  # 0..1 within the step; None = indeterminate
    detail: str = ""
    downloaded: int = 0
    total: int = 0


StepCb = Callable[[StepUpdate], None]


class SetupError(RuntimeError):
    pass


class _Manifest:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return {}
        return {}

    def get(self, key: str, default=None):
        return self.load().get(key, default)

    def set(self, **kw) -> None:
        data = self.load()
        data.update(kw)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class FirstRunSetup:
    def __init__(self, paths: Optional[config.AppPaths] = None):
        self.paths = paths or config.AppPaths()
        self.manifest = _Manifest(self.paths.manifest_path)
        self._plan: Optional[environment.TorchPlan] = None

    # ----- completion check ------------------------------------------------
    def backend_ready(self) -> bool:
        """True when the runnable backend (env + torch + ComfyUI) is in place.

        Models are intentionally NOT part of this gate — they are a user choice
        and can be downloaded later from the main window.
        """
        if not self.paths.backend_python.exists():
            return False
        if self.paths.backend_python == Path(sys.executable):
            return False  # venv not created -> fell back to host interpreter
        if not (self.paths.managed_comfyui / "main.py").exists():
            return False
        if self.manifest.get("comfyui_marker") != COMFYUI_MARKER:
            return False  # ComfyUI requirement bumped -> re-fetch + reinstall deps
        if self.manifest.get("torch_tag") is None:
            return False
        if self.manifest.get("torch_pkgset") != environment.TORCH_PKGSET:
            return False  # package set changed (e.g. torchaudio added) -> reinstall
        if self.manifest.get("deps_installed") is not True:
            return False
        return True

    # Backwards-compatible alias.
    def is_complete(self) -> bool:
        return self.backend_ready()

    def torch_plan(self) -> environment.TorchPlan:
        """Detect (cheaply) and cache the torch plan for display before run()."""
        if self._plan is None:
            self._plan = environment.choose_torch(environment.detect_gpu())
        return self._plan

    # ----- run -------------------------------------------------------------
    def run(self, on_step: StepCb, on_log: LogCb,
            cancel: Optional[Callable[[], bool]] = None,
            selected_models: Optional[list[str]] = None,
            install_sage: bool = False) -> None:
        cancel = cancel or (lambda: False)
        config.ensure_model_dirs()
        self.paths.backend_root.mkdir(parents=True, exist_ok=True)

        def emit(step_id, title, status, fraction=None, detail="",
                 downloaded=0, total=0):
            on_step(StepUpdate(step_id, title, status, fraction, detail,
                               downloaded, total))

        try:
            self._detect(on_log)
            self._step_uv(emit, on_log, cancel)
            self._step_venv(emit, on_log, cancel)
            self._step_torch(emit, on_log, cancel)
            self._step_comfyui(emit, on_log, cancel)
            self._step_deps(emit, on_log, cancel)
            if install_sage:
                self._step_sage(emit, on_log, cancel)
            self._step_models(emit, on_log, cancel, selected_models)
        except SetupError:
            raise
        except Exception as e:  # noqa: BLE001 - surface anything as SetupError
            raise SetupError(str(e)) from e

    # ----- steps -----------------------------------------------------------
    def _detect(self, log: LogCb) -> None:
        gpu = environment.detect_gpu()
        plan = environment.choose_torch(gpu)
        dtype = environment.recommended_weight_dtype(gpu, plan)
        log(f"GPU: {gpu.name or 'none'} (driver={gpu.driver_version}, "
            f"vram={gpu.vram_mb}MB)")
        log(f"torch 構成: {plan.reason}")
        self.manifest.set(
            gpu_name=gpu.name, driver=gpu.driver_version, vram_mb=gpu.vram_mb,
            torch_index=plan.index_url, torch_tag_target=plan.tag,
            torch_is_cuda=plan.is_cuda, recommended_dtype=dtype,
        )
        self._plan = plan

    def _step_uv(self, emit, log, cancel) -> None:
        title = "uv（バックエンドツール）"
        if self.paths.uv_path.exists():
            emit("uv", title, "skipped", 1.0, "導入済み")
            return
        emit("uv", title, "running", 0.0, "uv をダウンロード中…")
        uv_manager.ensure_uv(
            self.paths.uv_path, log,
            on_progress=lambda d, t: emit(
                "uv", title, "running", (d / t if t else None),
                _fmt_bytes(d, t), d, t),
        )
        emit("uv", title, "done", 1.0, "準備完了")

    def _step_venv(self, emit, log, cancel) -> None:
        title = "Python 環境"
        py = self.paths.backend_python
        if py.exists() and py != Path(sys.executable):
            emit("venv", title, "skipped", 1.0, "導入済み")
            return
        emit("venv", title, "running", None, "venv を作成中…")
        uv_manager.create_venv(self.paths.uv_path, self.paths.venv_dir, log, cancel)
        emit("venv", title, "done", 1.0, "作成完了")

    def _step_torch(self, emit, log, cancel) -> None:
        plan = self.torch_plan()
        title = f"PyTorch ({plan.tag})"
        if self.manifest.get("torch_pkgset") == environment.TORCH_PKGSET:
            emit("torch", title, "skipped", 1.0, "インストール済み")
            return
        emit("torch", title, "running", None,
             f"{', '.join(environment.TORCH_PACKAGES)} をインストール中（大容量）…")
        uv_manager.pip_install(
            self.paths.uv_path, self.paths.backend_python, log,
            packages=list(environment.TORCH_PACKAGES),
            index_url=plan.index_url, upgrade=True, cancel=cancel,
        )
        self.manifest.set(torch_tag=plan.tag, torch_pkgset=environment.TORCH_PKGSET)
        emit("torch", title, "done", 1.0, "インストール完了")

    def _step_comfyui(self, emit, log, cancel) -> None:
        title = "ComfyUI"
        have = (self.paths.managed_comfyui / "main.py").exists()
        if have and self.manifest.get("comfyui_marker") == COMFYUI_MARKER:
            emit("comfyui", title, "skipped", 1.0, "導入済み")
            return
        if have:
            # Upgrading an existing copy: replace the tree and force a deps
            # reinstall (a newer ComfyUI may pin newer frontend/template pkgs).
            log(f"ComfyUI を更新します（{COMFYUI_MARKER}）…")
            self.manifest.set(deps_installed=False)
        emit("comfyui", title, "running", 0.0, "ダウンロード中…")
        zip_path = self.paths.backend_root / "comfyui.zip"
        download(COMFYUI_ZIP, zip_path,
                 on_progress=lambda d, t: emit(
                     "comfyui", title, "running",
                     (0.9 * d / t) if t else None, _fmt_bytes(d, t), d, t),
                 cancel=cancel)
        emit("comfyui", title, "running", 0.92, "展開中…")
        with zipfile.ZipFile(zip_path) as zf:
            # GitHub archives nest everything under a single top-level folder
            # whose name varies by ref (e.g. tag "v0.26.0" -> "ComfyUI-0.26.0",
            # branch "master" -> "ComfyUI-master"). Read it rather than guess.
            top = zf.namelist()[0].split("/")[0]
            zf.extractall(self.paths.backend_root)
        extracted = self.paths.backend_root / top
        if extracted.exists():
            if self.paths.managed_comfyui.exists():
                shutil.rmtree(self.paths.managed_comfyui)
            extracted.rename(self.paths.managed_comfyui)
        zip_path.unlink(missing_ok=True)
        self.manifest.set(comfyui_marker=COMFYUI_MARKER)
        emit("comfyui", title, "done", 1.0, "準備完了")

    def _step_deps(self, emit, log, cancel) -> None:
        title = "ComfyUI 依存パッケージ"
        if self.manifest.get("deps_installed") is True:
            emit("deps", title, "skipped", 1.0, "インストール済み")
            return
        emit("deps", title, "running", None, "依存パッケージをインストール中…")
        req = self.paths.managed_comfyui / "requirements.txt"
        filtered = self._filter_requirements(req)
        uv_manager.pip_install(
            self.paths.uv_path, self.paths.backend_python, log,
            requirements=filtered, cancel=cancel,
        )
        self.manifest.set(deps_installed=True)
        emit("deps", title, "done", 1.0, "インストール完了")

    @staticmethod
    def _filter_requirements(req: Path) -> Path:
        """Drop torch/torchvision/torchaudio so the CUDA build isn't replaced
        by the PyPI default (CPU) wheels."""
        skip = {"torch", "torchvision", "torchaudio"}
        out = req.with_name("requirements.scom.txt")
        lines = []
        for line in req.read_text(encoding="utf-8").splitlines():
            name = re.split(r"[<>=!~ \[]", line.strip(), maxsplit=1)[0].lower()
            if name in skip:
                lines.append("# (scom) skipped to keep CUDA torch: " + line)
            else:
                lines.append(line)
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _step_sage(self, emit, log, cancel) -> None:
        title = "SageAttention"
        if sage_installed(self.paths):
            emit("sage", title, "skipped", 1.0, "インストール済み")
            self._enable_sage_setting(log)
            return
        emit("sage", title, "running", None, "インストール中…")
        install_sage_attention(self.paths, log, cancel)
        self._enable_sage_setting(log)
        emit("sage", title, "done", 1.0, "インストール完了")

    def _enable_sage_setting(self, log: LogCb) -> None:
        """settings.toml に sage_attention=true を書き、初回起動から有効化する
        （メインウィンドウはこのあと起動して設定を読む）。"""
        from .. import settings as app_settings
        data, err = app_settings.load(self.paths.settings_path)
        if err is not None:
            log(f"settings.toml が壊れているため SageAttention 設定を"
                f"書き込めません: {err}")
            return
        data["sage_attention"] = True
        app_settings.save(self.paths.settings_path, data)

    def _step_models(self, emit, log, cancel,
                     selected_models: Optional[list[str]]) -> None:
        manifest = models.load_manifest(self.paths)
        if selected_models is None:
            wanted = [m for m in manifest if m.required]
        else:
            sel = set(selected_models)
            wanted = [m for m in manifest if m.filename in sel]

        for m in wanted:
            step_id = "model:" + m.filename
            dest = models.target_path(self.paths, m)
            if dest.exists() and (not m.size or dest.stat().st_size == m.size):
                emit(step_id, m.filename, "skipped", 1.0, "ダウンロード済み")
                continue
            log(f"{m.filename} をダウンロード中 ({m.size / 1e9:.2f} GB)…")
            emit(step_id, m.filename, "running", 0.0, "開始中…")
            models.download_model(
                self.paths, m,
                on_progress=lambda d, t, _id=step_id, _fn=m.filename: emit(
                    _id, _fn, "running", (d / t if t else None),
                    _fmt_bytes(d, t), d, t),
                cancel=cancel,
            )
            emit(step_id, m.filename, "done", 1.0, "ダウンロード完了")


def _fmt_bytes(done: int, total: int) -> str:
    def g(n: int) -> str:
        return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"
    return f"{g(done)} / {g(total)}" if total else g(done)


# ----- SageAttention (optional speedup) --------------------------------------
def sage_installed(paths: config.AppPaths) -> bool:
    """バックエンド venv に sageattention が入っているか（ディレクトリ存在
    チェックのみ = 起動パスで毎回呼べる軽さ）。"""
    site = paths.venv_dir / "Lib" / "site-packages" / "sageattention"
    return site.exists()


def install_sage_attention(paths: config.AppPaths, log: LogCb,
                           cancel: Optional[Callable[[], bool]] = None) -> None:
    """triton-windows + SageAttention ホイールをバックエンド venv へ入れる。

    ホイールはインストール済み torch の CUDA タグ (cu128/cu130) に合わせて
    選ぶ。非対応環境（GPU無し・ドライバ不足・対応ホイール無し）は SetupError。
    """
    manifest = _Manifest(paths.manifest_path)
    torch_tag = str(manifest.get("torch_tag") or "")
    gpu = environment.detect_gpu()
    ok, reason = environment.sage_supported(gpu, torch_tag)
    if not ok:
        raise SetupError(reason)
    wheel = environment.sage_wheel_for(torch_tag)
    log(f"SageAttention をインストール中（torch {torch_tag} 用）…")
    # triton-windows は SageAttention の前提パッケージ（PyPI から取得）。
    uv_manager.pip_install(
        paths.uv_path, paths.backend_python, log,
        packages=["triton-windows", wheel], cancel=cancel,
    )

"""PySide6 main window: model selection, generation settings, preview."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QRegularExpression, QTimer, QEvent, QPoint,
)
from PySide6.QtCore import QUrl
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QDesktopServices, QImage, QPixmap,
    QRegularExpressionValidator, QTextCharFormat, QTextCursor, QTextFormat,
)
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy,
    QSpinBox, QSplitter, QStackedWidget, QVBoxLayout, QWidget, QCheckBox,
)

from .. import config, settings, metadata, modelinfo, prompt_presets, xyz
from .. import lora as lora_meta
from . import ansi_log
from .widgets import FlowLayout, GrowingTextEdit, WideComboBox
from ..comfy_backend import ComfyBackend, BackendError, Progress
from ..workflow import (
    GenParams, build_graph, build_merge_graph, merge_pin_key, merge_recipe,
    SAMPLERS, SCHEDULERS, CLIP_TYPES_SINGLE, CLIP_TYPES_DUAL, DEFAULT_NEGATIVE,
)

MAX_SEED = 2**63 - 1

# LoRA トリガーワードをプロンプト欄に挿入したときの区別用マーキング。
# 挿入した文字範囲に専用の文字書式（背景色 + token プロパティ）を付け、
# 見た目（背景ハイライト）とロジック（token による区間追跡）の両方で
# 元のプロンプトと区別する。token を持つ区間はユーザーが編集しても書式が
# 残るので、編集後のワードごとまとめて削除できる。
_LORA_TOKEN_PROP = QTextFormat.UserProperty + 17
# 挿入部分の色（LoRA ウィンドウのトリガーワード表示に合わせた明るい配色）。
# 暗色テーマだと本文の文字色が明るいので、明背景に合わせて文字色も濃色に。
_LORA_INSERT_BG = QColor("#e7edf5")   # 明るい背景
_LORA_INSERT_FG = QColor("#22456e")   # 濃い文字色

# XYZ 軸 id -> その軸が支配する GenParams フィールド。モデル軸のプリセット
# 適用時、軸で振っている値をプリセット設定で上書きしないための対応表。
_XYZ_AXIS_FIELDS = {
    "seed": {"seed"}, "steps": {"steps"}, "cfg": {"cfg"},
    "sampler": {"sampler"}, "scheduler": {"scheduler"},
    "size": {"width", "height"}, "dtype": {"weight_dtype"},
}

# Image format option that disables saving generated images to disk.
NO_SAVE = "保存しない"

# Merge entries in the diffusion combo: shown at the top of the list with a
# colored prefix; the item data is "__merge__:<id>".
MERGE_PREFIX = "マージモデル："
MERGE_TOKEN = "__merge__"
MERGE_COLOR = "#1a7f37"  # green tint for merge entries


class _StartWorker(QObject):
    """Starts the ComfyUI backend off the UI thread."""
    log = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(self, backend: ComfyBackend):
        super().__init__()
        self.backend = backend

    def run(self) -> None:
        try:
            self.backend.start(log=self.log.emit)
            self.done.emit()
        except BackendError as e:
            self.failed.emit(str(e))
        except Exception as e:  # pragma: no cover - defensive
            self.failed.emit(f"unexpected error: {e}")


class _GenWorker(QObject):
    progress = Signal(Progress)
    preview = Signal(bytes)
    cached = Signal(list)  # node ids served from the backend output cache
    timing = Signal(float)  # 純粋な推論(サンプリング)時間 [秒]
    done = Signal(list)    # list[bytes]
    failed = Signal(str)

    def __init__(self, backend: ComfyBackend, graph: dict):
        super().__init__()
        self.backend = backend
        self.graph = graph
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            images = self.backend.generate(
                self.graph,
                on_progress=self.progress.emit,
                on_preview=self.preview.emit,
                cancel=lambda: self._cancel,
                on_cached=self.cached.emit,
                on_timing=self.timing.emit,
            )
            self.done.emit(images)
        except BackendError as e:
            self.failed.emit(str(e))
        except Exception as e:  # pragma: no cover - defensive
            self.failed.emit(f"unexpected error: {e}")


class _XyzWorker(QObject):
    """Runs the XYZ plot cells sequentially off the UI thread.

    セル単位のエラーは placeholder (None) にして続行するが、最初のセルで
    失敗した場合は設定不備とみなして全体を中止する。
    """
    progress = Signal(Progress)
    preview = Signal(bytes)
    timing = Signal(float)          # セルごとの純粋な推論時間 [秒]
    cell_done = Signal(int, int)    # finished count, total
    cell_image = Signal(int, bytes)  # grid index, png (直後の逐次保存用)
    cell_error = Signal(int, str)   # grid index, message
    done = Signal(object)           # {grid_index: png bytes | None}
    failed = Signal(str)

    def __init__(self, backend: ComfyBackend, jobs: list):
        super().__init__()
        self.backend = backend
        self.jobs = jobs  # list[(grid_index, graph)]
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        results: dict[int, Optional[bytes]] = {}
        try:
            for n, (idx, graph) in enumerate(self.jobs):
                if self._cancel:
                    raise BackendError("生成をキャンセルしました")
                try:
                    images = self.backend.generate(
                        graph,
                        on_progress=self.progress.emit,
                        on_preview=self.preview.emit,
                        cancel=lambda: self._cancel,
                        on_timing=self.timing.emit,
                    )
                except BackendError as e:
                    if self._cancel or n == 0:
                        raise
                    results[idx] = None
                    self.cell_error.emit(idx, str(e))
                else:
                    results[idx] = images[0] if images else None
                    if results[idx]:
                        self.cell_image.emit(idx, results[idx])
                self.cell_done.emit(n + 1, len(self.jobs))
            self.done.emit(results)
        except BackendError as e:
            self.failed.emit(str(e))
        except Exception as e:  # pragma: no cover - defensive
            self.failed.emit(f"unexpected error: {e}")


class _XyzComposeWorker(QObject):
    """XYZ の比較グリッド画像を UI スレッド外で合成・保存する。

    セルの PNG デコード → グリッド描画（QPainter on QImage は非GUIスレッド
    可）→ PNG エンコード → （指定があれば）メタデータ付き保存、のすべてが
    大きなグリッドでは数秒かかり、UI スレッドで行うとフリーズして見える。
    """
    done = Signal(object)   # {"png": bytes, "w": int, "h": int, "log": [str]}

    def __init__(self, cells: list, nx: int, ny: int, nz: int,
                 labels: list, legend: bool, margin: int,
                 save_path, params_text: str, extra: dict, embed: bool):
        super().__init__()
        self._cells = cells          # list[bytes | None] (grid order)
        self._nx, self._ny, self._nz = nx, ny, nz
        self._labels = labels
        self._legend = legend
        self._margin = margin
        self._save_path = save_path  # Path | None (None = 保存しない)
        self._params_text = params_text
        self._extra = extra
        self._embed = embed

    def run(self) -> None:
        log: list[str] = []
        images = [QImage.fromData(b) if b else None for b in self._cells]
        grid = xyz.compose_grid(
            images, self._nx, self._ny, self._nz,
            self._labels[0], self._labels[1], self._labels[2],
            draw_legend=self._legend, margin=self._margin)
        png = xyz.qimage_png_bytes(grid)
        if self._save_path is not None:
            try:
                metadata.save_with_metadata(
                    png, self._save_path, "png", 6, self._params_text,
                    extra=self._extra, embed=self._embed)
                note = "メタデータ付き" if self._embed else "メタデータなし"
                log.append(f"{self._save_path} に保存しました（{note}）")
            except Exception as e:  # noqa: BLE001
                log.append(f"グリッド画像の保存に失敗: {e}")
        self.done.emit({"png": png, "w": grid.width(), "h": grid.height(),
                        "log": log})


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("scom - 画像生成")
        self.resize(1180, 760)

        self.paths = config.AppPaths()
        self.backend = ComfyBackend(self.paths)
        self._start_thread: Optional[QThread] = None
        self._gen_thread: Optional[QThread] = None
        self._gen_worker: Optional[_GenWorker] = None
        self._last_images: list[bytes] = []
        self._last_seed: int = 0
        self._last_params: Optional[GenParams] = None
        self._last_graph: Optional[dict] = None
        self._last_gen_ok: bool = False
        # スキップ（連続 ON 中のキャンセル）: 現在の生成だけ中断して
        # ループは続行するためのフラグ。cleanup で消費される。
        self._gen_skip: bool = False
        self._xyz_skip: bool = False
        # 生成中に積まれた待機タスク（押した時点の GenParams スナップショット）。
        self._gen_queue: list[GenParams] = []

        # Persisted settings. prompt/negative are read as initial values only.
        self.settings, settings_error = settings.load(self.paths.settings_path)
        # Named merge entries: {"id", "name", "models", "quant", "low_memory"}.
        self._merges: list[dict] = self._load_merges()
        self._merge_seq = int(self.settings.get("merge_seq", 0) or 0)
        self._merge_seq = max([self._merge_seq]
                              + [int(e["id"]) for e in self._merges])
        # Entries known to be built in backend RAM this session (best effort;
        # corrected via execution_cached events whenever an entry is used).
        self._merge_built_ids: set[int] = set()
        self._last_merge_id: Optional[int] = None   # entry used by last gen
        self._merge_was_cached = False              # node "4" was a cache hit
        self._merge_dlg = None  # non-modal MergeDialog (at most one)
        self._merge_running = False  # a merge-only run is in flight
        # Applied LoRAs: [{"name": str, "strength": float}, ...] in chain order.
        self._loras: list[dict] = []
        self._lora_dlg = None  # non-modal LoraDialog (at most one)
        # LoRA カードのホバーで出すトリガーワードのポップアップ（遅延生成）。
        self._lora_popup = None
        self._lora_pop_timer = QTimer(self)
        self._lora_pop_timer.setSingleShot(True)
        self._lora_pop_timer.setInterval(220)   # 離脱後この時間で閉じる
        self._lora_pop_timer.timeout.connect(self._hide_lora_popup)
        # Per-preset (anima/krea2/sdxl) memory of the Models + 設定 values;
        # filled from settings in _apply_settings, kept fresh in _do_save.
        self._preset_conf: dict = {}
        self._active_preset: str = "anima"
        # XYZ plot: non-modal dialog + sequential-cell worker state.
        self._xyz_dlg = None
        self._xyz_thread: Optional[QThread] = None
        self._xyz_worker: Optional[_XyzWorker] = None
        self._xyz_ctx: Optional[dict] = None
        # 比較グリッド合成スレッド（完了後に非同期で走る; 複数保持可）。
        self._xyz_compose_jobs: list = []
        self._loading = True
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(250)
        self._save_timer.timeout.connect(self._do_save)

        self._build_ui()
        self.refresh_models()       # fill combos before applying saved selection
        self._reload_prompt_presets(quiet=True)
        self._apply_settings()
        self._sync_builtin_for_diffusion()  # 復元した diffusion に内蔵チェックを追従
        self._loading = False
        self._connect_autosave()

        if settings_error:
            # Don't silently reset (and don't overwrite) a hand-broken file.
            self.append_log(f"settings.toml の読み込みエラー: {settings_error}")
            QMessageBox.warning(
                self, "settings.toml の読み込みに失敗",
                "settings.toml に文法エラーがあるため、既定値で起動します。\n"
                "ファイルを修正するまで自動保存は行いません。\n\n"
                f"エラー: {settings_error}",
            )
        else:
            # Ensure a settings.toml exists from the start (incl. initial prompts).
            self._do_save()
        self._settings_broken = bool(settings_error)

        self.start_backend()

    # ----- UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # Left block: an untitled outer group bundles Models + 設定 (with the
        # 表示モデル selector one level above them); the prompt box spans the
        # full width underneath.
        left = QWidget()
        grid = QGridLayout(left)
        grid.addWidget(self._build_top_group(), 0, 0)
        grid.addWidget(self._build_prompt_box(), 1, 0)
        grid.setRowStretch(1, 1)  # prompt area takes the remaining height
        left.setMinimumWidth(660)

        # Right column, top to bottom: log, preview, one-line image box,
        # action buttons.
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setPlaceholderText("バックエンドログ…")
        ansi_log.style_log(self.log_view)
        right_layout.addWidget(self.log_view, stretch=1)

        self.preview = QLabel("プレビュー")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(512, 512)
        self.preview.setStyleSheet(
            "QLabel { background:#1e1e1e; color:#888; border:1px solid #333; }"
        )
        right_layout.addWidget(self.preview, stretch=3)
        right_layout.addWidget(self._build_image_box())
        right_layout.addWidget(self._build_action_box())

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)  # preview/log column expands
        self.setCentralWidget(splitter)

        self.status = self.statusBar()
        self.status.addPermanentWidget(self.lbl_gen_time)  # プログレスバーの左
        self.status.addPermanentWidget(self.progress)
        self.status.showMessage("バックエンドを起動中…")

    def _build_top_group(self) -> QWidget:
        """Untitled outer group bundling the Models and 設定 categories.

        表示モデル is a cross-category setting (it filters Models AND drives
        the defaults), so it is drawn like the group's TITLE: a widget row
        overlaid on the frame's top border (QGroupBox titles are text-only,
        so the row is a floating child centered on the border line, with an
        opaque background that interrupts the line the way a title does)."""
        # Preset: picking anima/krea2/sdxl filters the model dropdowns to
        # that family. The Models + 設定 values are remembered per preset
        # (settings.toml "preset_conf") and restored on switch.
        self.cb_preset = WideComboBox()
        self.cb_preset.addItem("anima", "anima")
        self.cb_preset.addItem("krea2", "krea2")
        self.cb_preset.addItem("sdxl", "sdxl")
        self.cb_preset.setToolTip(
            "モデル・VAE・Text encoder をその系統に絞り込みます。"
            "Models と設定カテゴリの内容は系統ごとに記憶され、"
            "切り替えると前回の状態が復元されます"
        )
        self.cb_preset.currentIndexChanged.connect(self._on_preset_changed)

        title = QWidget()
        th = QHBoxLayout(title)
        th.setContentsMargins(6, 0, 6, 0)
        th.addWidget(QLabel("表示モデル:"))
        th.addWidget(self.cb_preset)
        title.setAutoFillBackground(True)  # hide the border line behind it
        title_h = title.sizeHint().height()

        wrap = QWidget()
        outer = QVBoxLayout(wrap)
        # The title row overhangs the group box by half its height.
        outer.setContentsMargins(0, title_h // 2, 0, 0)
        box = QGroupBox()
        v = QVBoxLayout(box)
        m = v.contentsMargins()
        v.setContentsMargins(m.left(), title_h // 2 + 4, m.right(), m.bottom())
        boxes = QHBoxLayout()
        boxes.addWidget(self._build_model_box(), stretch=1)
        settings_col = QVBoxLayout()
        settings_col.addWidget(self._build_settings_box())
        settings_col.addWidget(self._build_hires_box())
        settings_col.addStretch(1)
        boxes.addLayout(settings_col, stretch=1)
        v.addLayout(boxes)
        outer.addWidget(box)

        title.setParent(wrap)
        title.adjustSize()
        title.move(14, 0)  # same left offset feel as the "Models" title
        title.raise_()
        return wrap

    def _build_model_box(self) -> QGroupBox:
        box = QGroupBox("Models")
        form = QFormLayout(box)

        self.cb_diffusion = WideComboBox()
        self.cb_vae = WideComboBox()
        self.cb_te1 = WideComboBox()
        self.cb_te2 = WideComboBox()
        self.chk_dual_te = QCheckBox("2つ目の text encoder を使用 (DualCLIPLoader)")
        self.chk_dual_te.toggled.connect(self._on_dual_toggled)

        self.cb_clip_type = WideComboBox()
        self.cb_clip_type.addItems(CLIP_TYPES_SINGLE)

        # フルチェックポイント（VAE/CLIP 内蔵）を選んだときだけ有効。ON で内蔵の
        # VAE/CLIP を使い、下の VAE / Text encoder 指定は無視する。
        self.chk_builtin = QCheckBox("VAE / CLIP はモデル内蔵を使用（フルモデル）")
        self.chk_builtin.setToolTip(
            "SDXL のフルチェックポイントなど VAE/CLIP を内蔵するモデルで有効。"
            "ON にすると内蔵の VAE/CLIP を使い、下の VAE / Text encoder / "
            "CLIP type の指定は無視されます。")
        self.chk_builtin.setEnabled(False)
        self.chk_builtin.toggled.connect(self._on_builtin_toggled)

        # Picking a model auto-aligns the CLIP type / text encoder to its family
        # (e.g. a krea2 diffusion needs CLIP type 'krea2' + a Qwen3-VL encoder).
        self.cb_diffusion.currentTextChanged.connect(self._on_diffusion_changed)

        merge_btn = QPushButton("マージ設定…")
        merge_btn.clicked.connect(self._open_merge_dialog)
        refresh = QPushButton("再スキャン")
        refresh.clicked.connect(self.refresh_models)
        manage = QPushButton("設定…")
        manage.clicked.connect(self.open_models_dialog)
        btn_row = QHBoxLayout()
        btn_row.addWidget(merge_btn, stretch=1)
        btn_row.addWidget(refresh, stretch=1)
        btn_row.addWidget(manage, stretch=1)
        btns = QWidget(); btns.setLayout(btn_row)
        btn_row.setContentsMargins(0, 0, 0, 0)

        form.addRow("Diffusion:", self.cb_diffusion)
        form.addRow("", self.chk_builtin)
        form.addRow("VAE:", self.cb_vae)
        form.addRow("Text encoder 1:", self.cb_te1)
        form.addRow("", self.chk_dual_te)
        form.addRow("Text encoder 2:", self.cb_te2)
        form.addRow("CLIP type:", self.cb_clip_type)
        # Single-widget row: spans the label column too, so the three buttons
        # get the group box's full width (their labels were getting clipped).
        form.addRow(btns)
        self.cb_te2.setEnabled(False)
        return box

    def _build_prompt_box(self) -> QGroupBox:
        box = QGroupBox("Prompt")
        layout = QVBoxLayout(box)
        self.txt_prompt = GrowingTextEdit(min_lines=4)
        self.txt_prompt.setPlaceholderText("positive prompt…")
        self.txt_negative = GrowingTextEdit(min_lines=3)
        self.txt_negative.setPlaceholderText("negative prompt…")
        self.txt_negative.setPlainText(DEFAULT_NEGATIVE)
        layout.addWidget(QLabel("Positive"))
        layout.addWidget(self.txt_prompt)
        # LoRA（選択ボタン + 適用中チップ）。旧「LoRA」カテゴリをここへ移設。
        self.btn_lora = QPushButton("LoRA選択…")
        self.btn_lora.setToolTip(
            "LoRA の一覧（サムネイル・トリガーワード付き）を開いて"
            "適用する LoRA を選びます")
        self.btn_lora.clicked.connect(self._open_lora_dialog)
        lora_row = QWidget()
        self._lora_flow = FlowLayout(lora_row, hspacing=6, vspacing=4)
        self._lora_flow.setContentsMargins(0, 0, 0, 0)
        self._lora_flow.addWidget(self.btn_lora)
        layout.addWidget(lora_row)
        layout.addWidget(QLabel("Negative"))
        layout.addWidget(self.txt_negative)

        # Prompt presets: pick a named entry from prompts.csv and append its
        # prompt/negative to the fields with the pen button.
        self.cb_prompt_preset = WideComboBox()
        self.cb_prompt_preset.setToolTip(
            "prompts.csv のプリセット（1列目: 設定名、2列目: プロンプト、"
            "3列目: ネガティブプロンプト）")
        btn_apply = QPushButton("書込み")
        btn_apply.setToolTip("選択中のプリセットをプロンプト欄・ネガティブ欄に追記")
        btn_apply.clicked.connect(self._apply_prompt_preset)
        btn_edit = QPushButton("編集")
        btn_edit.setToolTip("設定ファイル (prompts.csv) を開いて編集")
        btn_edit.clicked.connect(self._open_prompt_csv)
        btn_reload = QPushButton("再読込み")
        btn_reload.setToolTip("設定ファイルを再読み込み")
        btn_reload.clicked.connect(self._reload_prompt_presets)
        layout.addWidget(QLabel("プロンプト入力 (prompts.csvの1個目の設定が起動時↑に設定されます)"))
        preset_row = QHBoxLayout()
        preset_row.addWidget(self.cb_prompt_preset, stretch=1)
        preset_row.addWidget(btn_apply)
        preset_row.addWidget(btn_edit)
        preset_row.addWidget(btn_reload)
        layout.addLayout(preset_row)

        layout.addStretch(1)
        # Shift+Enter in either prompt field starts generation.
        self.txt_prompt.installEventFilter(self)
        self.txt_negative.installEventFilter(self)
        # ユーザーが挿入済みハイライトを手で消したときも LoRA 窓の表示を追従。
        self.txt_prompt.textChanged.connect(self._push_lora_inserted)
        self.txt_negative.textChanged.connect(self._push_lora_inserted)
        return box

    def _build_settings_box(self) -> QGroupBox:
        box = QGroupBox("設定")
        grid = QGridLayout(box)

        self.sp_width = QSpinBox(); self.sp_width.setRange(64, 4096)
        self.sp_width.setSingleStep(64); self.sp_width.setValue(1024)
        self.sp_height = QSpinBox(); self.sp_height.setRange(64, 4096)
        self.sp_height.setSingleStep(64); self.sp_height.setValue(1024)
        self.sp_steps = QSpinBox(); self.sp_steps.setRange(1, 200)
        self.sp_steps.setValue(30)
        self.sp_cfg = QDoubleSpinBox(); self.sp_cfg.setRange(0.0, 30.0)
        self.sp_cfg.setSingleStep(0.5); self.sp_cfg.setValue(4.0)
        self.sp_batch = QSpinBox(); self.sp_batch.setRange(1, 16)
        self.sp_batch.setValue(1)

        self.cb_sampler = WideComboBox(); self.cb_sampler.addItems(SAMPLERS)
        self.cb_sampler.setCurrentText("er_sde")
        self.cb_scheduler = WideComboBox(); self.cb_scheduler.addItems(SCHEDULERS)
        self.cb_scheduler.setCurrentText("simple")

        # Seeds can exceed 32-bit (QSpinBox limit), so use a text field.
        # Seed は保存も生成後の書き戻しもしない。-1（起動時の初期値）なら
        # 生成のたびに内部でランダム値を引く。実際に使った値はログと画像
        # メタデータに残る。
        self.ed_seed = QLineEdit("-1")
        self.ed_seed.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"-1|\d{1,19}"))
        )
        self.ed_seed.setToolTip(
            "-1 = 生成ごとにランダム（欄は書き換わりません。使われた値は"
            "ログ・画像メタデータで確認できます）")

        self.cb_dtype = WideComboBox()
        self.cb_dtype.addItems(["default", "fp8_e4m3fn", "fp8_e5m2"])

        # サイズ行は他の行と違い列をまたいだ一体レイアウトにする:
        # 「サイズ [Width] ⇄ [Height]」。入れ替えボタンが両者の間に来て
        # 直感的になり、専用の列を作らないので他の設定行の幅にも影響しない
        # （スパン配置は各列の幅要求を変えない）。
        self.sp_width.setToolTip("Width（横）")
        self.sp_height.setToolTip("Height（縦）")
        self.btn_swap_size = QPushButton("⇄")
        self.btn_swap_size.setFlat(True)
        self.btn_swap_size.setFixedSize(22, 22)
        self.btn_swap_size.setStyleSheet(
            "QPushButton { border: 1px solid #666; border-radius: 3px; }"
            "QPushButton:hover { background: rgba(128,128,128,0.25); }")
        self.btn_swap_size.setToolTip("Width と Height を入れ替え")
        self.btn_swap_size.clicked.connect(self._swap_size)
        size_cell = QHBoxLayout()
        size_cell.setContentsMargins(0, 0, 0, 0)
        size_cell.setSpacing(4)
        size_cell.addWidget(QLabel("Width"))
        size_cell.addWidget(self.sp_width, stretch=1)
        size_cell.addWidget(self.btn_swap_size)
        size_cell.addWidget(self.sp_height, stretch=1)
        size_cell.addWidget(QLabel("Height"))

        r = 0
        grid.addLayout(size_cell, r, 0, 1, 4)   # 行全体（列0〜3）をスパン
        r += 1
        grid.addWidget(QLabel("Steps"), r, 0); grid.addWidget(self.sp_steps, r, 1)
        grid.addWidget(QLabel("CFG"), r, 2); grid.addWidget(self.sp_cfg, r, 3)
        r += 1
        grid.addWidget(QLabel("Sampler"), r, 0); grid.addWidget(self.cb_sampler, r, 1)
        grid.addWidget(QLabel("Scheduler"), r, 2); grid.addWidget(self.cb_scheduler, r, 3)
        r += 1
        grid.addWidget(QLabel("Seed"), r, 0); grid.addWidget(self.ed_seed, r, 1)
        grid.addWidget(QLabel("Batch"), r, 2); grid.addWidget(self.sp_batch, r, 3)
        r += 1
        grid.addWidget(QLabel("UNet dtype"), r, 0); grid.addWidget(self.cb_dtype, r, 1)
        return box

    def _build_hires_box(self) -> QGroupBox:
        """Hires fix (latent): チェック付きグループ。OFF なら中身ごと無効。
        1段目の latent を補間拡大し、2段目の KSampler で細部を描き直す。"""
        box = QGroupBox("Hires fix (latent)")
        box.setCheckable(True)
        box.setChecked(False)
        box.setToolTip(
            "生成した latent を拡大して同じプロンプトで再サンプリングし、"
            "高解像度化します（生成時間は約2倍になります）")
        self.grp_hires = box
        grid = QGridLayout(box)

        self.sp_hires_scale = QDoubleSpinBox()
        self.sp_hires_scale.setRange(1.05, 4.0)
        self.sp_hires_scale.setSingleStep(0.05)
        self.sp_hires_scale.setDecimals(2)
        self.sp_hires_scale.setValue(1.5)
        self.sp_hires_scale.setToolTip("最終解像度 = 生成サイズ × この倍率")

        self.sp_hires_denoise = QDoubleSpinBox()
        self.sp_hires_denoise.setRange(0.0, 1.0)
        self.sp_hires_denoise.setSingleStep(0.05)
        self.sp_hires_denoise.setDecimals(2)
        self.sp_hires_denoise.setValue(0.55)
        self.sp_hires_denoise.setToolTip(
            "2段目の描き直し量。latent 拡大では 0.4〜0.7 が目安"
            "（低いとボケ、高いと構図が変わる）")

        self.sp_hires_steps = QSpinBox()
        self.sp_hires_steps.setRange(0, 200)
        self.sp_hires_steps.setValue(0)
        self.sp_hires_steps.setSpecialValueText("メインと同じ")
        self.sp_hires_steps.setToolTip("2段目のステップ数（0 = メインと同じ）")

        self.cb_hires_method = WideComboBox()
        self.cb_hires_method.addItems(
            ["bislerp", "nearest-exact", "bilinear", "bicubic"])
        self.cb_hires_method.setToolTip("latent の補間方法")

        grid.addWidget(QLabel("倍率"), 0, 0)
        grid.addWidget(self.sp_hires_scale, 0, 1)
        grid.addWidget(QLabel("Denoise"), 0, 2)
        grid.addWidget(self.sp_hires_denoise, 0, 3)
        grid.addWidget(QLabel("Steps"), 1, 0)
        grid.addWidget(self.sp_hires_steps, 1, 1)
        grid.addWidget(QLabel("補間"), 1, 2)
        grid.addWidget(self.cb_hires_method, 1, 3)

        # タイトルに現在の倍率での出力解像度を表示（サイズ/倍率の変更に追従）。
        self.sp_hires_scale.valueChanged.connect(self._update_hires_title)
        self.sp_width.valueChanged.connect(self._update_hires_title)
        self.sp_height.valueChanged.connect(self._update_hires_title)
        self._update_hires_title()
        return box

    def _hires_out_size(self) -> tuple[int, int]:
        """現在の倍率での最終解像度。ComfyUI の LatentUpscaleBy は latent
        単位（1/8px）で丸めるため、それと同じ計算で表示する。"""
        scale = float(self.sp_hires_scale.value())
        w = round(self.sp_width.value() / 8 * scale) * 8
        h = round(self.sp_height.value() / 8 * scale) * 8
        return int(w), int(h)

    def _update_hires_title(self, *_a) -> None:
        w, h = self._hires_out_size()
        self.grp_hires.setTitle(f"Hires fix (latent)  →  {w}x{h}")

    def _build_action_box(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        # 連続 is a checkbox: two states with an unmistakable ON/OFF look
        # (a pressed QPushButton reads poorly). Same isChecked/setChecked API.
        self.btn_xyz = QPushButton("XYZ")
        self.btn_xyz.setToolTip(
            "XYZ プロット: 複数パラメータの値の組み合わせで生成し、"
            "1枚のグリッド画像にまとめます")
        self.btn_xyz.setMinimumHeight(40)
        self.btn_xyz.setMaximumWidth(56)
        self.btn_xyz.clicked.connect(self._open_xyz_dialog)
        self.btn_continuous = QCheckBox("連続")
        self.btn_continuous.setToolTip(
            "ONの間、生成が終わるたびに自動で次を生成します"
            "（ON中のキャンセルボタンは「スキップ」= 現在の生成だけ中断）")
        self.btn_continuous.setMinimumHeight(40)
        self.btn_generate = QPushButton("生成")
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_cancel = QPushButton("キャンセル")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.on_cancel)
        self.btn_continuous.toggled.connect(self._update_cancel_button)
        # 生成 stays about twice キャンセル; Ignored size policy makes the
        # layout use the stretch ratios alone instead of the text size hints.
        for b in (self.btn_generate, self.btn_cancel):
            b.setMinimumHeight(40)
            sp = b.sizePolicy()
            sp.setHorizontalPolicy(QSizePolicy.Ignored)
            b.setSizePolicy(sp)
        row.addWidget(self.btn_xyz)
        row.addWidget(self.btn_continuous)
        row.addWidget(self.btn_generate, stretch=2)
        row.addWidget(self.btn_cancel, stretch=1)
        # Progress moved to the status bar to save vertical space here.
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setMaximumWidth(220)
        # 直近の純粋な推論時間（プログレスバーの左隣に常駐表示）。
        self.lbl_gen_time = QLabel("")
        self.lbl_gen_time.setToolTip(
            "直近の生成の推論時間（サンプリングのみ。モデル読み込み・"
            "テキストエンコード・VAEデコード等は含みません）")
        return w

    # ----- image output settings (below the preview) ----------------------
    def _build_image_box(self) -> QGroupBox:
        """One-line Image box (sits between the preview and the log)."""
        box = QGroupBox("Image")
        row = QHBoxLayout(box)

        self.cb_img_format = WideComboBox()
        self.cb_img_format.addItems(["png", "jpg", "webp", NO_SAVE])  # index 0/1/2/3

        # One compact quality control per format, swapped on format change.
        self.stack_quality = QStackedWidget()

        self.sp_png_compress = QSpinBox()
        self.sp_png_compress.setRange(0, 9); self.sp_png_compress.setValue(6)
        self.sp_png_compress.setToolTip("0 = 低圧縮/大きい, 9 = 高圧縮/小さい")
        self.sp_jpg_quality = QSpinBox()
        self.sp_jpg_quality.setRange(1, 100); self.sp_jpg_quality.setValue(92)
        self.sp_webp_quality = QSpinBox()
        self.sp_webp_quality.setRange(1, 100); self.sp_webp_quality.setValue(90)
        self.sp_webp_quality.setToolTip("100 = lossless（ロスレス）")

        for label, spin in (("Compression (0-9)", self.sp_png_compress),
                             ("Quality (1-100)", self.sp_jpg_quality),
                             ("Quality (1-100)", self.sp_webp_quality)):
            page = QWidget()
            pr = QHBoxLayout(page)
            pr.setContentsMargins(0, 0, 0, 0)
            pr.addWidget(QLabel(label))
            pr.addWidget(spin)
            self.stack_quality.addWidget(page)
        # "保存しない" page (index 3): no quality control.
        nosave_page = QWidget()
        nr = QHBoxLayout(nosave_page); nr.setContentsMargins(0, 0, 0, 0)
        nr.addWidget(QLabel("（output へ自動保存しません）"))
        self.stack_quality.addWidget(nosave_page)
        self.cb_img_format.currentIndexChanged.connect(self.stack_quality.setCurrentIndex)

        self.chk_embed_meta = QCheckBox("画像にメタ情報を埋め込む")
        self.chk_embed_meta.setChecked(True)
        self.chk_embed_meta.setToolTip(
            "ONで生成設定（プロンプト・seed 等）を画像に埋め込みます"
            "（PNG: parameters チャンク / JPEG・WEBP: EXIF）。"
            "OFFにするとメタ情報を一切書き込みません")

        row.addWidget(QLabel("Format"))
        row.addWidget(self.cb_img_format)
        row.addSpacing(16)
        row.addWidget(self.stack_quality)
        row.addSpacing(16)
        row.addWidget(self.chk_embed_meta)
        row.addStretch(1)
        return box

    # ----- model scanning --------------------------------------------------
    def refresh_models(self) -> None:
        config.ensure_model_dirs()
        # Keep the unfiltered lists; the preset decides what the combos show.
        self._all_models = {
            "diffusion_models": config.scan_models("diffusion_models"),
            "vae": config.scan_models("vae"),
            "text_encoders": config.scan_models("text_encoders"),
            "loras": config.scan_models("loras"),
        }
        self.append_log(
            f"モデル: diffusion_models={len(self._all_models['diffusion_models'])} "
            f"vae={len(self._all_models['vae'])} "
            f"text_encoders={len(self._all_models['text_encoders'])} "
            f"loras={len(self._all_models['loras'])}"
        )
        self._apply_preset_filter()

    @staticmethod
    def _fill_combo(combo: QComboBox, items: list[str], allow_empty: bool = False) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        if allow_empty:
            combo.addItem("")
        combo.addItems(items)
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def open_models_dialog(self) -> None:
        from .models_dialog import ModelsDialog
        dlg = ModelsDialog(
            self.paths, sage_enabled=bool(self.settings.get("sage_attention",
                                                            False)),
            parent=self)
        dlg.sage_toggled.connect(self._on_sage_setting_toggled)
        dlg.exec()
        self.refresh_models()

    def _on_sage_setting_toggled(self, enabled: bool) -> None:
        self.settings["sage_attention"] = bool(enabled)
        self._schedule_save()
        self.append_log(
            "SageAttention を{}にしました（アプリ再起動後に反映）".format(
                "有効" if enabled else "無効"))

    def _on_dual_toggled(self, checked: bool) -> None:
        self.cb_te2.setEnabled(checked)
        current = self.cb_clip_type.currentText()
        self.cb_clip_type.clear()
        self.cb_clip_type.addItems(CLIP_TYPES_DUAL if checked else CLIP_TYPES_SINGLE)
        idx = self.cb_clip_type.findText(current)
        if idx >= 0:
            self.cb_clip_type.setCurrentIndex(idx)

    # ----- model family helpers (content-based, see app/modelinfo.py) ------
    def _diffusion_family(self, relname: str) -> str:
        if not relname:
            return modelinfo.UNKNOWN
        return modelinfo.family(
            "diffusion_models",
            config.models_root() / "diffusion_models" / relname)

    def _te_family(self, relname: str) -> str:
        if not relname:
            return modelinfo.UNKNOWN
        return modelinfo.family(
            "text_encoders", config.models_root() / "text_encoders" / relname)

    def _vae_family(self, relname: str) -> str:
        if not relname:
            return modelinfo.UNKNOWN
        return modelinfo.family("vae", config.models_root() / "vae" / relname)

    def _first_vae(self, families: tuple[str, ...]) -> Optional[str]:
        """First scanned VAE whose content family is in ``families``."""
        for name in self._all_models.get("vae", []):
            if self._vae_family(name) in families:
                return name
        return None

    def _sdxl_te_pair(self) -> tuple[Optional[str], Optional[str]]:
        """(clip_l, clip_g) filenames from the scanned text encoders."""
        root = config.models_root() / "text_encoders"
        clip_l = clip_g = None
        for name in self._all_models.get("text_encoders", []):
            if self._te_family(name) != "sdxl":
                continue
            kind = modelinfo.sdxl_te_kind(root / name)
            if kind == "clip_l" and clip_l is None:
                clip_l = name
            elif kind == "clip_g" and clip_g is None:
                clip_g = name
        return clip_l, clip_g

    def _select_te_for_family(self, fam: str) -> bool:
        """Select the first text encoder whose content matches ``fam``."""
        for i in range(self.cb_te1.count()):
            name = self.cb_te1.itemText(i)
            if name and self._te_family(name) == fam:
                self.cb_te1.setCurrentIndex(i)
                return True
        return False

    def _diffusion_is_checkpoint(self) -> bool:
        """選択中の diffusion が VAE/CLIP 内蔵のフルチェックポイントか。"""
        if self._merge_selected():
            return False
        name = self.cb_diffusion.currentText().strip()
        if not name:
            return False
        return modelinfo.is_checkpoint(
            config.models_root() / "diffusion_models" / name)

    def _use_builtin(self) -> bool:
        return self.chk_builtin.isEnabled() and self.chk_builtin.isChecked()

    def _apply_builtin_enabled(self) -> None:
        """内蔵利用中は VAE/TE/CLIP 指定を触れなくする（無視されるため）。"""
        builtin = self._use_builtin()
        for w in (self.cb_vae, self.cb_te1, self.cb_te2,
                  self.cb_clip_type, self.chk_dual_te):
            w.setEnabled(not builtin)
        if not builtin:
            self.cb_te2.setEnabled(self.chk_dual_te.isChecked())

    def _sync_builtin_for_diffusion(self) -> None:
        """diffusion 選択に合わせて内蔵チェックの有効/既定を更新する。フル
        チェックポイントなら有効化して既定ON（＝内蔵を使う）。"""
        is_ckpt = self._diffusion_is_checkpoint()
        self.chk_builtin.blockSignals(True)
        self.chk_builtin.setEnabled(is_ckpt)
        self.chk_builtin.setChecked(is_ckpt)
        self.chk_builtin.blockSignals(False)
        self._apply_builtin_enabled()

    def _on_builtin_toggled(self, _checked: bool) -> None:
        self._apply_builtin_enabled()
        self._schedule_save()

    def _on_diffusion_changed(self, name: str) -> None:
        """Auto-align CLIP type / text encoder to the selected model's family."""
        if self._loading:
            return  # honor saved settings on startup; only react to user picks
        # マージ/チェックポイントを含めて内蔵チェックの状態を先に反映する。
        self._sync_builtin_for_diffusion()
        if self._merge_selected():
            return  # the merge recipe decides family; nothing to auto-align
        if self._diffusion_is_checkpoint():
            return  # 内蔵 VAE/CLIP を使うので系統整列(clip_l/clip_g等)は不要
        fam = self._diffusion_family(name)
        if fam == "sdxl":
            # SDXL requires dual TE (clip_l + clip_g) + CLIP type 'sdxl'
            # + kl-f8 VAE.
            if (self.cb_clip_type.currentText() == "sdxl"
                    and self.chk_dual_te.isChecked()):
                return
            self.chk_dual_te.setChecked(True)  # switches the CLIP list to DUAL
            idx = self.cb_clip_type.findText("sdxl")
            if idx >= 0:
                self.cb_clip_type.setCurrentIndex(idx)
            clip_l, clip_g = self._sdxl_te_pair()
            if clip_l:
                self.cb_te1.setCurrentText(clip_l)
            if clip_g:
                self.cb_te2.setCurrentText(clip_g)
            if self._vae_family(self.cb_vae.currentText()) != "sdxl":
                v = self._first_vae(("sdxl",))
                if v:
                    self.cb_vae.setCurrentText(v)
            return
        # Coming back from an SDXL setup also needs re-aligning (dual TE and
        # CLIP type 'sdxl' would break anima/krea2 generations).
        from_sdxl = self.cb_clip_type.currentText() == "sdxl"
        if fam == "krea2" and (from_sdxl or not self.chk_dual_te.isChecked()):
            # Krea-2 requires CLIP type 'krea2' and a Qwen3-VL text encoder.
            self.chk_dual_te.setChecked(False)
            idx = self.cb_clip_type.findText("krea2")
            if idx >= 0:
                self.cb_clip_type.setCurrentIndex(idx)
            self._select_te_for_family("krea2")
        elif fam == "anima" and from_sdxl:
            self.chk_dual_te.setChecked(False)
            idx = self.cb_clip_type.findText("stable_diffusion")
            if idx >= 0:
                self.cb_clip_type.setCurrentIndex(idx)
            self._select_te_for_family("anima")
        else:
            return
        if self._vae_family(self.cb_vae.currentText()) == "sdxl":
            v = self._first_vae(("shared", fam))
            if v:
                self.cb_vae.setCurrentText(v)

    def _krea2_config_warning(self, params: GenParams) -> Optional[str]:
        """Pre-flight check: translate the cryptic backend mismatch into a
        clear, actionable message before a Krea-2 run is even submitted."""
        # For a merge, the family is judged from the first source model.
        name = params.merge_models[0][0] if params.merge_models else params.diffusion
        if self._diffusion_family(name) != "krea2":
            return None
        msgs = []
        if params.clip_type != "krea2":
            msgs.append(f"・CLIP type を 'krea2' にしてください（現在: {params.clip_type}）")
        if not any(self._te_family(t) == "krea2" for t in params.te):
            msgs.append("・Text encoder に Krea-2 用（Qwen3-VL）を選択してください"
                        "（未取得なら「設定…」からダウンロード）")
        if not msgs:
            return None
        return ("選択中のモデルは Krea-2 です。次を直してください:\n\n"
                + "\n".join(msgs))

    def _sdxl_config_warning(self, params: GenParams) -> Optional[str]:
        """Pre-flight check for SDXL: dual CLIP (clip_l+clip_g) / CLIP type
        'sdxl' / kl-f8 VAE — どれが欠けても生成は確実に失敗する。"""
        name = params.merge_models[0][0] if params.merge_models else params.diffusion
        if self._diffusion_family(name) != "sdxl":
            return None
        msgs = []
        if params.clip_type != "sdxl":
            msgs.append(f"・CLIP type を 'sdxl' にしてください（現在: {params.clip_type}）")
        sdxl_tes = [t for t in params.te if self._te_family(t) == "sdxl"]
        if len(params.te) < 2 or len(sdxl_tes) < 2:
            msgs.append("・「2つ目の text encoder を使用」を ON にし、"
                        "Text encoder 1/2 に SDXL 用（clip_l と clip_g）を"
                        "選択してください")
        if self._vae_family(params.vae) != "sdxl":
            msgs.append("・VAE に SDXL 用（sdxl_vae 等）を選択してください")
        if not msgs:
            return None
        return ("選択中のモデルは SDXL です。次を直してください:\n\n"
                + "\n".join(msgs))

    def _config_warning(self, params: GenParams) -> Optional[str]:
        """Family-mismatch pre-flight: returns a message when the current
        model needs a different TE/CLIP/VAE configuration."""
        # フルチェックポイントで内蔵 VAE/CLIP を使う場合は検証不要。
        if params.checkpoint and not params.vae and not params.te:
            return None
        return (self._krea2_config_warning(params)
                or self._sdxl_config_warning(params))

    # ----- model merge (マージモデル) ---------------------------------------
    def _load_merges(self) -> list[dict]:
        """Load merge entries from settings, migrating the old single-recipe
        keys (merge_models/merge_quant/merge_fp8/merge_low_memory)."""
        try:
            merges = json.loads(str(self.settings.get("merges", "[]")))
            out = []
            for e in merges:
                out.append({
                    "id": int(e["id"]),
                    "name": str(e["name"]),
                    "models": [(str(n), float(w)) for n, w in e["models"]],
                    "quant": str(e.get("quant", "")),
                    "low_memory": bool(e.get("low_memory", False)),
                })
            if out:
                return out
        except (ValueError, TypeError, KeyError):
            pass
        # Migration: a pre-multi-merge settings file with a single recipe.
        try:
            old = [(str(n), float(w)) for n, w in
                   json.loads(str(self.settings.get("merge_models", "[]")))]
        except (ValueError, TypeError):
            old = []
        if len(old) < 2:
            return []
        quant = str(self.settings.get("merge_quant", ""))
        if not quant and self.settings.get("merge_fp8"):
            quant = "fp8"
        return [{"id": 1, "name": "マージモデル1", "models": old,
                 "quant": quant,
                 "low_memory": bool(self.settings.get("merge_low_memory", False))}]

    def _merge_family(self, entry: dict) -> str:
        """Family of a merge entry, judged from its first source model
        (same convention as _krea2_config_warning)."""
        if not entry.get("models"):
            return "unknown"
        return self._diffusion_family(entry["models"][0][0])

    def _merge_selected(self) -> bool:
        data = self.cb_diffusion.currentData()
        return isinstance(data, str) and data.startswith(MERGE_TOKEN)

    def _selected_merge_entry(self) -> Optional[dict]:
        data = self.cb_diffusion.currentData()
        if not (isinstance(data, str) and data.startswith(MERGE_TOKEN)):
            return None
        try:
            entry_id = int(data.split(":", 1)[1])
        except (IndexError, ValueError):
            return None
        return self._merge_entry_by_id(entry_id)

    def _merge_entry_by_id(self, entry_id: int) -> Optional[dict]:
        for e in self._merges:
            if int(e["id"]) == entry_id:
                return e
        return None

    def _push_merge_state(self) -> None:
        """Refresh the merge window's entry list (if it is open)."""
        if self._merge_dlg is None:
            return
        try:
            self._merge_dlg.set_entries(self._merges, self._merge_built_ids)
        except RuntimeError:  # underlying Qt object was deleted
            self._merge_dlg = None

    def _open_merge_dialog(self) -> None:
        from .merge_dialog import MergeDialog
        # Non-modal, at most one instance; re-opening replaces it so the model
        # list is always current.
        if self._merge_dlg is not None:
            try:
                self._merge_dlg.close()
                self._merge_dlg.deleteLater()
            except RuntimeError:
                pass
        # Parentless on purpose: a QDialog with a parent always stays on top
        # of it, but this window should be able to go behind the main window.
        dlg = MergeDialog(
            self._all_models.get("diffusion_models", []),
            self._diffusion_family,
            config.models_root() / "diffusion_models", None)
        dlg.merge_requested.connect(self._on_merge_requested)
        dlg.save_requested.connect(self._on_save_requested)
        dlg.delete_requested.connect(self._on_merge_delete)
        dlg.rename_requested.connect(self._on_merge_rename)
        dlg.free_memory_requested.connect(self._on_free_memory)
        self._merge_dlg = dlg
        self._push_merge_state()
        if self._merge_running:
            dlg.set_merge_running(True)
        dlg.show()

    def _on_merge_requested(self, entries, quant: str, low_memory: bool) -> None:
        """マージ button: register a new entry and build it in backend RAM."""
        models = [(str(n), float(w)) for n, w in entries]
        try:
            graph = build_merge_graph(models, quant, low_memory)
        except ValueError as e:
            QMessageBox.warning(self, "入力不足", str(e))
            return
        self._merge_seq += 1
        entry = {"id": self._merge_seq, "name": f"マージモデル{self._merge_seq}",
                 "models": models, "quant": str(quant),
                 "low_memory": bool(low_memory)}
        self._merges.append(entry)
        self._schedule_save()
        self._apply_preset_filter()  # the new entry appears in the dropdown
        self._push_merge_state()
        if self._merge_dlg is not None:
            self._merge_dlg.select_entry(entry["id"])
        self._run_merge(graph, saving=False, entry_id=entry["id"])

    def _on_save_requested(self, entries, quant: str, low_memory: bool,
                           filename: str) -> None:
        models = [(str(n), float(w)) for n, w in entries]
        try:
            graph = build_merge_graph(models, quant, low_memory,
                                      save_to=filename)
        except ValueError as e:
            QMessageBox.warning(self, "入力不足", str(e))
            return
        self._run_merge(graph, saving=True, entry_id=None)

    def _on_merge_delete(self, entry_id: int) -> None:
        entry = self._merge_entry_by_id(entry_id)
        self._merges = [e for e in self._merges
                        if int(e["id"]) != entry_id]
        self._merge_built_ids.discard(entry_id)
        # Free its pinned model from backend RAM (per-entry, via our node's
        # management route).
        if entry is not None and self.backend.is_running():
            try:
                self.backend.release_merge(
                    merge_recipe(entry["models"]), entry["quant"],
                    entry["low_memory"])
            except OSError as e:
                self.append_log(f"メモリ解放に失敗: {e}")
        was_selected = (self.cb_diffusion.currentData()
                        == f"{MERGE_TOKEN}:{entry_id}")
        self._apply_preset_filter()
        if was_selected:
            # Fall back to the first regular model (after the merge entries
            # still shown under the current preset).
            first_regular = 0
            for i in range(self.cb_diffusion.count()):
                data = self.cb_diffusion.itemData(i)
                if not (isinstance(data, str) and data.startswith(MERGE_TOKEN)):
                    first_regular = i
                    break
            self.cb_diffusion.setCurrentIndex(
                min(first_regular, self.cb_diffusion.count() - 1))
        self._schedule_save()
        self._push_merge_state()
        name = entry["name"] if entry else f"id={entry_id}"
        self.append_log(f"{name} を一覧から削除し、メモリを解放しました")

    def _on_merge_rename(self, entry_id: int, name: str) -> None:
        e = self._merge_entry_by_id(entry_id)
        if e is None or not name.strip():
            return
        e["name"] = name.strip()
        self._apply_preset_filter()  # dropdown labels follow the new name
        self._schedule_save()
        self._push_merge_state()

    def _on_free_memory(self) -> None:
        if not self.backend.is_running():
            QMessageBox.warning(self, "未準備", "バックエンドが起動していません。")
            return
        try:
            self.backend.release_all_merges()
            self.backend.free_memory()
        except OSError as e:
            self.append_log(f"メモリ解放に失敗: {e}")
            return
        self._merge_built_ids.clear()
        self._push_merge_state()
        self.append_log("バックエンドのキャッシュとモデルをすべて解放しました"
                        "（次回使用時に自動で再構築されます）")

    def _sync_merge_states(self) -> None:
        """Refresh ●/○ from the backend's pin cache (the source of truth)."""
        if not self.backend.is_running():
            return
        try:
            pinned = set(self.backend.merge_pinned())
        except OSError:
            return
        self._merge_built_ids = {
            int(e["id"]) for e in self._merges
            if merge_pin_key(e["models"], e["quant"], e["low_memory"])
            in pinned}
        self._push_merge_state()

    def _run_merge(self, graph: dict, saving: bool,
                   entry_id: Optional[int]) -> None:
        """Run a merge-only prompt (build in RAM / save to file) off-thread."""
        if not self.backend.is_running():
            QMessageBox.warning(
                self, "未準備",
                "バックエンドがまだ起動していません。起動完了後、"
                "生成時に自動でマージされます。")
            return
        if self._gen_thread is not None or self._xyz_thread is not None:
            QMessageBox.information(
                self, "実行中",
                "生成の完了後に実行してください（生成時に自動でマージされます）。")
            return
        self._last_gen_ok = False  # keep continuous mode from chaining a merge
        self.btn_generate.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        note = ("マージモデルを保存中…" if saving
                else "マージモデルをメインメモリ上に構築中…")
        self.status.showMessage(note)
        self.append_log(note)

        self._gen_thread = QThread(self)
        self._gen_worker = _GenWorker(self.backend, graph)
        self._gen_worker.moveToThread(self._gen_thread)
        self._gen_thread.started.connect(self._gen_worker.run)
        self._gen_worker.progress.connect(self._on_progress)
        # NOTE: must be a bound method, not a lambda. Signals connected to a
        # plain callable run in the EMITTING (worker) thread; only QObject
        # bound methods get queued to the GUI thread. A lambda here executed
        # _on_merge_done -> dialog widget updates off the GUI thread
        # ("Cannot set parent ... different thread" warnings).
        self._merge_run_ctx = (saving, entry_id)
        self._merge_running = True
        self._gen_worker.progress.connect(self._on_merge_progress)
        self._gen_worker.done.connect(self._on_merge_worker_done)
        self._gen_worker.failed.connect(self._on_gen_failed)
        self._gen_worker.done.connect(self._end_merge_progress)
        self._gen_worker.failed.connect(self._end_merge_progress)
        self._gen_worker.done.connect(self._gen_thread.quit)
        self._gen_worker.failed.connect(self._gen_thread.quit)
        self._gen_thread.finished.connect(self._cleanup_gen_thread)
        self._gen_thread.start()
        if self._merge_dlg is not None:
            try:
                self._merge_dlg.set_merge_running(True)
            except RuntimeError:
                self._merge_dlg = None

    def _on_merge_worker_done(self, _images: list) -> None:
        saving, entry_id = getattr(self, "_merge_run_ctx", (False, None))
        self._on_merge_done(saving, entry_id)

    # NOTE: どちらも bound method で接続すること（_run_merge の NOTE 参照）。
    def _on_merge_progress(self, p: Progress) -> None:
        """Mirror merge progress onto the merge window's progress bar."""
        if self._merge_dlg is None:
            return
        try:
            self._merge_dlg.set_merge_progress(p.value, p.maximum, p.note)
        except RuntimeError:
            self._merge_dlg = None

    def _end_merge_progress(self, *_a) -> None:
        self._merge_running = False
        if self._merge_dlg is None:
            return
        try:
            self._merge_dlg.set_merge_running(False)
        except RuntimeError:
            self._merge_dlg = None

    def _on_merge_done(self, saved: bool, entry_id: Optional[int]) -> None:
        if entry_id is not None:
            self._merge_built_ids.add(entry_id)
        self._sync_merge_states()
        if saved:
            self.status.showMessage("マージモデルを保存しました")
            self.append_log("マージモデルを models/diffusion_models に保存しました")
            self.refresh_models()  # the new file appears in the dropdowns
        else:
            e = self._merge_entry_by_id(entry_id) if entry_id else None
            name = e["name"] if e else "マージモデル"
            self.status.showMessage(f"{name} を構築しました")
            self.append_log(
                f"{name} をメインメモリ上に構築しました"
                "（同じ構成での生成はこのモデルを再利用します）")
            # Building it means the user intends to generate with it.
            idx = self.cb_diffusion.findData(f"{MERGE_TOKEN}:{entry_id}")
            if idx >= 0:
                self.cb_diffusion.setCurrentIndex(idx)
        self._push_merge_state()

    # ----- LoRA -------------------------------------------------------------
    def _open_lora_dialog(self) -> None:
        from .lora_dialog import LoraDialog
        # Non-modal, at most one instance; re-opening replaces it so the file
        # list is always current (parentless so it can go behind us).
        if self._lora_dlg is not None:
            try:
                self._lora_dlg.close()
                self._lora_dlg.deleteLater()
            except RuntimeError:
                pass
        dlg = LoraDialog(config.models_root() / "loras",
                         self.paths.user_data / "lora_cache",
                         self._current_preset, None)
        dlg.apply_requested.connect(self._on_lora_apply)
        dlg.remove_requested.connect(self._on_lora_remove)
        dlg.toggle_prompt_requested.connect(self._on_lora_toggle_prompt)
        self._lora_dlg = dlg
        self._push_lora_state()
        self._push_lora_inserted()
        dlg.show()

    def _push_lora_state(self) -> None:
        """Refresh the LoRA window's applied marks (if it is open)."""
        if self._lora_dlg is None:
            return
        try:
            self._lora_dlg.set_applied(
                {e["name"]: float(e["strength"]) for e in self._loras})
        except RuntimeError:
            self._lora_dlg = None

    def _on_lora_apply(self, name: str, strength: float) -> None:
        for e in self._loras:
            if e["name"] == name:
                e["strength"] = float(strength)
                break
        else:
            self._loras.append({"name": name, "strength": float(strength)})
            self.append_log(f"LoRA を適用: {name} ×{strength:g}")
        self._rebuild_lora_rows()
        self._push_lora_state()
        self._schedule_save()

    def _on_lora_remove(self, name: str) -> None:
        before = len(self._loras)
        self._loras = [e for e in self._loras if e["name"] != name]
        if len(self._loras) != before:
            self.append_log(f"LoRA を解除: {name}")
        self._rebuild_lora_rows()
        self._push_lora_state()
        self._schedule_save()

    def _on_lora_strength_changed(self, name: str, value: float) -> None:
        for e in self._loras:
            if e["name"] == name:
                e["strength"] = float(value)
        self._push_lora_state()
        self._schedule_save()

    def _on_lora_toggle_prompt(self, token: str, negative: bool,
                               text: str) -> None:
        """LoRA のトリガーワードをトグルする。未挿入なら挿入（区別マーク付き）、
        挿入済みなら（ユーザー編集後でも）その区間ごと削除する。"""
        field = self.txt_negative if negative else self.txt_prompt
        regions = self._find_token_regions(field, token)
        if regions:
            self._remove_token_regions(field, regions)
        else:
            self._insert_token_words(field, token, text)
        self._push_lora_inserted()

    @staticmethod
    def _plain_char_format() -> QTextCharFormat:
        """token を持たない通常書式（区切りや以降の入力がハイライトされない
        ようにするため）。"""
        fmt = QTextCharFormat()
        fmt.clearBackground()
        return fmt

    def _token_char_format(self, token: str) -> QTextCharFormat:
        fmt = QTextCharFormat()
        fmt.setBackground(_LORA_INSERT_BG)
        fmt.setForeground(_LORA_INSERT_FG)
        fmt.setProperty(_LORA_TOKEN_PROP, token)
        return fmt

    @staticmethod
    def _find_token_regions(field, token: str) -> list[tuple[int, int]]:
        """指定 token の文字書式を持つ連続区間 (start, end) を左から順に返す。
        内部編集でフラグメントが分割されていても隣接分は1区間に統合する。"""
        doc = field.document()
        frags: list[tuple[int, int]] = []
        block = doc.begin()
        while block != doc.end():
            it = block.begin()
            while not it.atEnd():
                frag = it.fragment()
                if frag.isValid() and \
                        frag.charFormat().property(_LORA_TOKEN_PROP) == token:
                    start = frag.position()
                    frags.append((start, start + frag.length()))
                it += 1
            block = block.next()
        frags.sort()
        regions: list[tuple[int, int]] = []
        for s, e in frags:
            if regions and s <= regions[-1][1]:
                regions[-1] = (regions[-1][0], max(regions[-1][1], e))
            else:
                regions.append((s, e))
        return regions

    def _remove_token_regions(self, field,
                              regions: list[tuple[int, int]]) -> None:
        """token 区間を削除する。挿入時に付けた直後の色なしスペースと、
        隣接する区切り ", " も1つ巻き込んで取り除き、", ," や余分な空白が
        残らないようにする。位置ズレを避けるため右端の区間から削除する。"""
        text = field.toPlainText()
        cursor = field.textCursor()
        for start, end in sorted(regions, reverse=True):
            s, e = start, end
            lead = text[s - 2:s] == ", "
            # 挿入時の色なし後続スペースを巻き込む。ただし前側の区切りも
            # 取る場合、スペースの先にユーザーの追記があるなら残す
            # （両方消すと前後のテキストが癒着するため）。
            if text[e:e + 1] == " " and (not lead or not text[e + 1:].strip()):
                e += 1
            if lead:                        # 直前の区切りを巻き込む
                s -= 2
            elif text[e:e + 2] == ", ":     # 先頭要素なら直後の区切りを
                e += 2
            cursor.setPosition(s)
            cursor.setPosition(e, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
        field.setCurrentCharFormat(self._plain_char_format())

    def _insert_token_words(self, field, token: str, text: str) -> None:
        """欄末尾に、区別マーク付きでワードを追記する。

        色付き区間の直後に入力すると Qt は左隣の書式（=色）を引き継ぐため、
        区間の直後に色なしスペースを1つ置く。前側は色なしの ", " 区切りが
        同じ役割を果たす。これで続けて追記しても色は付かない。
        """
        words = [w.strip() for w in text.split(",") if w.strip()]
        if not words:
            return
        joined = ", ".join(words)
        full = field.toPlainText()
        stripped = full.rstrip()
        cursor = field.textCursor()
        # 末尾の余分な空白を除いてから追記する。
        cursor.setPosition(len(stripped))
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        if stripped:
            cursor.insertText(", ", self._plain_char_format())
        cursor.insertText(joined, self._token_char_format(token))
        cursor.insertText(" ", self._plain_char_format())
        field.setTextCursor(cursor)
        # 続けて入力してもハイライトされないよう書式を通常へ戻す。
        field.setCurrentCharFormat(self._plain_char_format())

    def _active_lora_tokens(self) -> set[str]:
        """両プロンプト欄に現在挿入されている LoRA トリガーワードの token 集合。"""
        tokens: set[str] = set()
        for field in (self.txt_prompt, self.txt_negative):
            doc = field.document()
            block = doc.begin()
            while block != doc.end():
                it = block.begin()
                while not it.atEnd():
                    frag = it.fragment()
                    if frag.isValid():
                        tok = frag.charFormat().property(_LORA_TOKEN_PROP)
                        if tok:
                            tokens.add(str(tok))
                    it += 1
                block = block.next()
        return tokens

    def _push_lora_inserted(self) -> None:
        """挿入済み token を LoRA ウィンドウ・カードのポップアップへ通知
        （リンクの挿入済みハイライト更新用）。"""
        tokens = self._active_lora_tokens()
        if self._lora_dlg is not None:
            try:
                self._lora_dlg.set_inserted(tokens)
            except RuntimeError:
                self._lora_dlg = None
        if self._lora_popup is not None and self._lora_popup.isVisible():
            self._lora_popup.set_active(tokens)

    # ----- LoRA card hover popup -------------------------------------------
    def _ensure_lora_popup(self):
        if self._lora_popup is None:
            from .lora_dialog import TriggerWordsPopup
            self._lora_popup = TriggerWordsPopup(self)
            self._lora_popup.toggle_requested.connect(self._on_lora_toggle_prompt)
            self._lora_popup.hover_changed.connect(self._on_lora_popup_hover)
        return self._lora_popup

    def _show_lora_popup(self, relname: str, anchor) -> None:
        pos, neg = lora_meta.effective_trigger_words(
            relname, config.models_root() / "loras",
            self.paths.user_data / "lora_cache")
        popup = self._ensure_lora_popup()
        popup.set_content(relname, pos, neg, self._active_lora_tokens())
        if not popup.has_words():        # トリガーワードが無ければ出さない
            popup.hide()
            return
        self._lora_pop_timer.stop()
        self._lora_pop_anchor = anchor
        # サイズは set_content で確定済み。画面内に収まる位置へクランプする
        # （はみ出すと OS 側で再配置され、setGeometry 警告の原因になる）。
        below = anchor.mapToGlobal(QPoint(0, anchor.height() + 2))
        screen = (anchor.screen() or self.screen()).availableGeometry()
        y = below.y()
        if y + popup.height() - 1 > screen.bottom():
            # 下にはみ出すならアンカーの上側へ回す。
            y = anchor.mapToGlobal(QPoint(0, 0)).y() - popup.height() - 2
        # 最終的に画面内へ確実に収める（上下・左右とも）。
        x = max(screen.left(),
                min(below.x(), screen.right() - popup.width() + 1))
        y = max(screen.top(), min(y, screen.bottom() - popup.height() + 1))
        popup.move(x, y)
        popup.show()
        popup.raise_()

    def _on_lora_popup_hover(self, over: bool) -> None:
        # ポップアップ上にマウスがある間は閉じない。離れたら猶予後に閉じる。
        if over:
            self._lora_pop_timer.stop()
        else:
            self._lora_pop_timer.start()

    def _hide_lora_popup(self, force: bool = False) -> None:
        """カーソルがまだカード or ポップアップ上にあれば閉じない（チップの
        子ウィジェット＝強度スピン等の上でも Leave が飛ぶため、実際の位置で
        判定する）。force=True は無条件で閉じる（チップ再構築時など）。"""
        if self._lora_popup is None or not self._lora_popup.isVisible():
            return
        if not force:
            gp = QCursor.pos()
            for w in (self._lora_popup, getattr(self, "_lora_pop_anchor", None)):
                try:
                    if (w is not None and w.isVisible()
                            and w.rect().contains(w.mapFromGlobal(gp))):
                        self._lora_pop_timer.start()   # まだ上にある → 保持
                        return
                except RuntimeError:
                    pass                               # チップが破棄済み
        self._lora_popup.hide()
        self._lora_pop_anchor = None

    def _rebuild_lora_rows(self) -> None:
        """Rebuild the applied-LoRA chips after the LoRA button (flow layout;
        index 0 is the button itself)."""
        self._hide_lora_popup(force=True)
        while self._lora_flow.count() > 1:
            item = self._lora_flow.takeAt(1)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for e in self._loras:
            name = e["name"]
            chip = QFrame()
            chip.setObjectName("loraChip")
            chip.setStyleSheet(
                "#loraChip { border: 1px solid #666; border-radius: 4px; }")
            # ホバーでトリガーワードのポップアップを出す（Enter/Leave を監視）。
            # _lora_chip は常にポップアップの位置基準（子から入っても同じ場所）。
            chip._lora_name = name
            chip._lora_chip = chip
            chip.installEventFilter(self)
            h = QHBoxLayout(chip)
            h.setContentsMargins(6, 1, 4, 1)
            h.setSpacing(4)
            trash = QPushButton("🗑")
            trash.setFixedWidth(24)
            trash.setFlat(True)
            trash.setToolTip("この LoRA を解除")
            trash.clicked.connect(
                lambda *_a, n=name: self._on_lora_remove(n))
            lbl = QLabel(Path(name).stem)
            lbl.setToolTip(name)
            spin = QDoubleSpinBox()
            spin.setRange(-4.0, 4.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(float(e["strength"]))
            spin.setFixedWidth(64)
            spin.setToolTip("LoRA の適用強度（model / clip 共通）")
            spin.valueChanged.connect(
                lambda v, n=name: self._on_lora_strength_changed(n, v))
            h.addWidget(trash)
            h.addWidget(lbl)
            h.addWidget(spin)
            # 子ウィジェットに直接カーソルが入ってもポップアップが出るように
            # 同じ監視をぶら下げる（Enter はカーソル直下のウィジェットに届く）。
            for child in (trash, lbl, spin):
                child._lora_name = name
                child._lora_chip = chip
                child.installEventFilter(self)
            self._lora_flow.addWidget(chip)

    # ----- XYZ plot ---------------------------------------------------------
    def _open_xyz_dialog(self) -> None:
        from .xyz_dialog import XyzDialog
        # 実行中は既存ウィンドウを前面に出すだけ（作り直すと進捗表示が切れる）。
        if self._xyz_thread is not None and self._xyz_dlg is not None:
            try:
                self._xyz_dlg.show()
                self._xyz_dlg.raise_()
                self._xyz_dlg.activateWindow()
                return
            except RuntimeError:
                self._xyz_dlg = None
        if self._xyz_dlg is not None:
            try:
                self._xyz_dlg.close()
                self._xyz_dlg.deleteLater()
            except RuntimeError:
                pass
        try:
            state = json.loads(str(self.settings.get("xyz", "{}")))
        except (ValueError, TypeError):
            state = {}
        # Parentless on purpose: same as the merge window, so it can go
        # behind the main window. Model-axis choices list merge entries
        # first (as "マージモデル：<name>" tokens), like the main dropdown.
        choices = ([MERGE_PREFIX + e["name"] for e in self._merges]
                   + self._all_models.get("diffusion_models", []))
        dlg = XyzDialog(choices, state, None)
        dlg.run_requested.connect(self._on_xyz_requested)
        dlg.cancel_requested.connect(self._on_xyz_cancel_requested)
        # チェック状態は実行しなくても記憶する（次回開いたとき再現）。
        # コンストラクタでの復元後に接続するので、復元自体では発火しない。
        dlg.chk_save_cells.toggled.connect(
            lambda c: self._persist_xyz_flag("save_cells", c))
        dlg.chk_show_grid.toggled.connect(
            lambda c: self._persist_xyz_flag("show_grid", c))
        dlg.chk_save_grid.toggled.connect(
            lambda c: self._persist_xyz_flag("save_grid", c))
        self._xyz_dlg = dlg
        if self._xyz_thread is not None and self._xyz_ctx is not None:
            dlg.set_running(True, int(self._xyz_ctx.get("total", 0)))
        dlg.show()

    def _persist_xyz_flag(self, key: str, checked: bool) -> None:
        """XYZ ウィンドウのチェック状態だけを即座に永続化する（他の入力欄は
        実行時にまとめて保存されるので、途中入力を上書きしないよう merge）。"""
        try:
            st = json.loads(str(self.settings.get("xyz", "{}")))
            if not isinstance(st, dict):
                st = {}
        except (ValueError, TypeError):
            st = {}
        st[key] = bool(checked)
        self.settings["xyz"] = json.dumps(st, ensure_ascii=False)
        self._schedule_save()

    def _xyz_parent(self):
        """Message-box parent: the XYZ window if alive, else the main window."""
        if self._xyz_dlg is not None:
            try:
                if self._xyz_dlg.isVisible():
                    return self._xyz_dlg
            except RuntimeError:
                self._xyz_dlg = None
        return self

    def _push_xyz_running(self, running: bool, total: int = 0) -> None:
        if self._xyz_dlg is None:
            return
        try:
            self._xyz_dlg.set_running(running, total)
        except RuntimeError:
            self._xyz_dlg = None

    def _on_xyz_requested(self, spec: dict, auto: bool = False) -> None:
        """Start an XYZ run. ``auto`` marks a continuous-mode chained run
        (skips the many-cells confirmation so the loop keeps going)."""
        if not self.backend.is_running():
            QMessageBox.warning(self._xyz_parent(), "未準備",
                                "バックエンドがまだ起動していません。")
            return
        if self._gen_thread is not None or self._xyz_thread is not None:
            QMessageBox.information(self._xyz_parent(), "実行中",
                                    "生成の完了後に実行してください。")
            return
        try:
            base = self._collect_params()
            base.batch_size = 1  # 1セル = 1枚
            if base.hires_enabled:
                self.append_log(
                    "XYZ: Hires fix 有効（各セルが2段生成になります）")
            axes = [xyz.axis_by_id(a["id"]) for a in spec["axes"]]
            values = [a["values"] for a in spec["axes"]]
            has_model_axis = any(a.id == "model" for a in axes)
            if not has_model_axis:
                # モデル軸があるときは全セルでモデルが差し替わるので、ベース
                # 選択に対する系統チェックは意味を持たない（整合は下で
                # セルごとに取る）。
                warn = self._config_warning(base)
                if warn:
                    QMessageBox.warning(self._xyz_parent(),
                                        "モデル設定を確認してください", warn)
                    return
            plan = xyz.plan_cells(base, axes, values)
            if has_model_axis:
                self._xyz_align_logged = set()
                # 系統ごとのプリセット設定（メモリ上）。アクティブな表示モデル
                # は現在の UI 値、他は切り替え時に退避した値を使う（設定
                # ファイルからは読まない）。
                confs = dict(self._preset_conf)
                confs[self._current_preset()] = self._collect_preset_conf()
                # 軸で振っているパラメータはプリセット値で上書きしない。
                locked = set()
                for a in axes:
                    locked |= _XYZ_AXIS_FIELDS.get(a.id, set())
                plan = [(idx, self._resolve_xyz_model_cell(p, confs, locked))
                        for idx, p in plan]
            jobs = [(idx, build_graph(p)) for idx, p in plan]
        except ValueError as e:
            QMessageBox.warning(self._xyz_parent(), "入力エラー", str(e))
            return
        total = len(jobs)
        if total > 64 and not auto:
            res = QMessageBox.question(
                self._xyz_parent(), "確認",
                f"{total} 枚の画像を生成します。実行しますか？")
            if res != QMessageBox.Yes:
                return
        # ウィンドウの入力状態を保存（次回開いたとき復元される）。
        if self._xyz_dlg is not None:
            try:
                self.settings["xyz"] = json.dumps(
                    self._xyz_dlg.state(), ensure_ascii=False)
                self._schedule_save()
            except RuntimeError:
                self._xyz_dlg = None

        from datetime import datetime
        # 個別保存はセル完成のたびに行う（キャンセルしても済んだ分は残る）。
        # フォーマットは開始時点の設定で run 全体を通して固定する。
        cell_fmt = None
        if spec.get("save_cells"):
            fmt = self.cb_img_format.currentText()
            if fmt == NO_SAVE:
                self.append_log("Format = 保存しない のため個別画像は保存しません")
            else:
                cell_fmt = (fmt, *self._encode_params(fmt))
        self._xyz_ctx = {
            "spec": spec,
            "base": base,
            "params": dict(plan),
            "nx": len(values[0]), "ny": len(values[1]), "nz": len(values[2]),
            "total": total,
            "stamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "cell_fmt": cell_fmt,   # (fmt, ext, quality) | None
            "cells_saved": 0,
            "embed": self.chk_embed_meta.isChecked(),  # run 全体で固定
        }
        self._xyz_last_spec = spec   # 連続モードの次回実行用
        self._xyz_last_ok = False
        self._last_seed = base.seed
        self._last_params = base
        self._last_graph = None
        self._last_gen_ok = False  # 連続モードに拾わせない
        self.btn_generate.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.status.showMessage(f"XYZ プロット生成中… (0/{total})")
        self.append_log(
            f"XYZ プロット開始: {total} セル "
            f"({self._xyz_ctx['nx']}x{self._xyz_ctx['ny']}x"
            f"{self._xyz_ctx['nz']}) seed={base.seed}")
        self._push_xyz_running(True, total)

        self._xyz_thread = QThread(self)
        self._xyz_worker = _XyzWorker(self.backend, jobs)
        self._xyz_worker.moveToThread(self._xyz_thread)
        self._xyz_thread.started.connect(self._xyz_worker.run)
        self._xyz_worker.progress.connect(self._on_progress)
        self._xyz_worker.preview.connect(self._on_preview_frame)
        self._xyz_worker.timing.connect(self._on_gen_timing)
        self._xyz_worker.cell_done.connect(self._on_xyz_cell_done)
        self._xyz_worker.cell_image.connect(self._on_xyz_cell_image)
        self._xyz_worker.cell_error.connect(self._on_xyz_cell_error)
        self._xyz_worker.done.connect(self._on_xyz_done)
        self._xyz_worker.failed.connect(self._on_xyz_failed)
        self._xyz_worker.done.connect(self._xyz_thread.quit)
        self._xyz_worker.failed.connect(self._xyz_thread.quit)
        self._xyz_thread.finished.connect(self._cleanup_xyz_thread)
        self._xyz_thread.start()

    def _resolve_xyz_model_cell(self, p: GenParams, confs: dict,
                                locked: set) -> GenParams:
        """モデル軸のセルを解決する: マージ展開 + プリセット適用 + 系統整合。

        値が「マージモデル：<名前>」なら登録済みマージエントリのレシピに
        展開する。次に、そのモデルの系統に対応する表示モデル（プリセット）
        の Models + 設定値（``confs`` = メモリ上の記憶。アクティブな系統は
        現在の UI 値）をセルへ適用する — ただし軸で振っている ``locked``
        フィールドと seed（比較可能性のため全セル共通）は上書きしない。
        最後に TE / CLIP / VAE の系統整合を検証し、食い違いは補正する
        （整合が取れないと conditioning の次元不一致でバックエンドが
        "mat1 and mat2 shapes cannot be multiplied" を出す）。
        """
        from dataclasses import replace
        if p.diffusion.startswith(MERGE_PREFIX):
            name = p.diffusion[len(MERGE_PREFIX):]
            entry = next((e for e in self._merges if e["name"] == name), None)
            if entry is None:
                raise ValueError(
                    f"モデル軸のマージモデル「{name}」が見つかりません"
                    "（削除または名前変更されていませんか）")
            p = replace(p, diffusion="", checkpoint=False,
                        merge_models=[(str(n), float(w))
                                      for n, w in entry["models"]],
                        merge_quant=str(entry["quant"]),
                        merge_low_memory=bool(entry["low_memory"]))
            fam = self._merge_family(entry)
        else:
            fam = self._diffusion_family(p.diffusion)
            # フルチェックポイント（VAE/CLIP 内蔵）はセルでも内蔵を使う。
            # TE/VAE の系統整列は不要（むしろ内蔵を上書きしてしまう）。
            if p.diffusion and modelinfo.is_checkpoint(
                    config.models_root() / "diffusion_models" / p.diffusion):
                return replace(p, checkpoint=True, vae="", te=[])
            # 非チェックポイントのセルは内蔵フラグを必ず落とす（基準 p が
            # チェックポイントでも、このセルは分割ロードで整列する）。
            p = replace(p, checkpoint=False)

        conf = confs.get(fam) if fam in ("anima", "krea2", "sdxl") else None
        if conf:
            p = self._apply_preset_conf_to_cell(p, conf, locked)
            note = ("preset", fam)
            if note not in self._xyz_align_logged:
                self._xyz_align_logged.add(note)
                self.append_log(
                    f"XYZ: {fam} 系セルには表示モデル {fam} の設定を適用します")

        if fam == "sdxl":
            te_ok = (p.clip_type == "sdxl" and len(p.te) == 2
                     and all(self._te_family(t) == "sdxl" for t in p.te))
            vae_ok = self._vae_family(p.vae) == "sdxl"
            if te_ok and vae_ok:
                return p
            clip_l, clip_g = self._sdxl_te_pair()
            if clip_l is None or clip_g is None:
                raise ValueError(
                    "モデル軸に SDXL 系がありますが、SDXL 用 text encoder"
                    "（clip_l / clip_g）が text_encoders に見つかりません。")
            vae = p.vae if vae_ok else self._first_vae(("sdxl",))
            if vae is None:
                raise ValueError(
                    "モデル軸に SDXL 系がありますが、SDXL 用 VAE が vae に"
                    "見つかりません。")
            self._log_xyz_align(fam, f"{clip_l} + {clip_g}", "sdxl", vae)
            return replace(p, te=[clip_l, clip_g], clip_type="sdxl", vae=vae)

        if fam not in ("anima", "krea2"):
            return p
        te_ok = (len(p.te) == 1 and self._te_family(p.te[0]) == fam)
        clip_ok = (p.clip_type != "sdxl"
                   and (p.clip_type == "krea2") == (fam == "krea2"))
        vae_ok = self._vae_family(p.vae) in (fam, "shared")
        if te_ok and clip_ok and vae_ok:
            return p
        te = list(p.te)
        if not te_ok:
            for name in self._all_models.get("text_encoders", []):
                if self._te_family(name) == fam:
                    te = [name]
                    break
            else:
                raise ValueError(
                    f"モデル軸の {p.diffusion} は {fam} 系ですが、対応する "
                    "text encoder が見つかりません。「設定…」から"
                    "ダウンロードしてください。")
        clip = p.clip_type if clip_ok else (
            "krea2" if fam == "krea2" else "stable_diffusion")
        vae = p.vae if vae_ok else self._first_vae(("shared", fam))
        if vae is None:
            raise ValueError(f"{fam} 用の VAE が見つかりません。")
        self._log_xyz_align(fam, te[0], clip, vae)
        return replace(p, te=te, clip_type=clip, vae=vae)

    def _apply_preset_conf_to_cell(self, p: GenParams, conf: dict,
                                   locked: set) -> GenParams:
        """Apply a preset's remembered Models/設定 values to one XYZ cell.

        モデル（diffusion / マージ指定）は軸の値、seed はベース値のまま。
        ``locked`` のフィールド（軸で振っているもの）も触らない。ここで
        入った値は続く系統整合チェックで検証される（古いファイル名等は
        そこで補正される）。
        """
        from dataclasses import replace
        kw = {}
        te = [str(conf.get("te1", ""))]
        if conf.get("dual_te") and str(conf.get("te2", "")).strip():
            te.append(str(conf.get("te2")).strip())
        te = [t for t in te if t]
        if te:
            kw["te"] = te
        if str(conf.get("clip_type", "")):
            kw["clip_type"] = str(conf["clip_type"])
        if str(conf.get("vae", "")):
            kw["vae"] = str(conf["vae"])
        try:
            fields = (("width", "width", int), ("height", "height", int),
                      ("steps", "steps", int), ("cfg", "cfg", float),
                      ("sampler", "sampler", str),
                      ("scheduler", "scheduler", str),
                      ("weight_dtype", "dtype", str),
                      ("hires_enabled", "hires_enabled", bool),
                      ("hires_scale", "hires_scale", float),
                      ("hires_denoise", "hires_denoise", float),
                      ("hires_steps", "hires_steps", int),
                      ("hires_method", "hires_method", str))
            for field, key, cast in fields:
                if field not in locked and key in conf:
                    kw[field] = cast(conf[key])
        except (TypeError, ValueError):
            pass  # 壊れた保存値: 適用できた分だけ使う
        return replace(p, **kw)

    def _log_xyz_align(self, fam: str, te_desc: str, clip: str,
                       vae: str) -> None:
        note = (fam, te_desc, clip, vae)
        if note not in self._xyz_align_logged:
            self._xyz_align_logged.add(note)
            self.append_log(
                f"XYZ: {fam} 系モデルには text encoder {te_desc} / "
                f"CLIP type {clip} / VAE {vae} を使用します")

    def _on_xyz_cell_done(self, done: int, total: int) -> None:
        self.status.showMessage(f"XYZ プロット生成中… ({done}/{total})")
        if self._xyz_dlg is not None:
            try:
                self._xyz_dlg.set_progress(done)
            except RuntimeError:
                self._xyz_dlg = None

    def _on_xyz_cell_image(self, idx: int, data: bytes) -> None:
        """個別保存 ON のとき、セルが出来た直後にその画像を保存する。"""
        ctx = self._xyz_ctx
        if not ctx or not ctx.get("cell_fmt") or not metadata.AVAILABLE:
            return
        fmt, ext, quality = ctx["cell_fmt"]
        path = (self.paths.output_dir
                / f"xyz_{ctx['stamp']}_c{idx:03d}.{ext}")
        cell_text, cell_extra = self._build_metadata(
            ctx.get("params", {}).get(idx))
        try:
            metadata.save_with_metadata(
                data, path, fmt, quality, cell_text, extra=cell_extra,
                embed=bool(ctx.get("embed", True)))
            ctx["cells_saved"] += 1
        except Exception as e:  # noqa: BLE001
            self.append_log(f"保存に失敗: {e}")

    def _on_xyz_cell_error(self, idx: int, msg: str) -> None:
        self.append_log(f"XYZ セル {idx} でエラー（グレーで継続）: {msg}")

    def _on_xyz_done(self, results: dict) -> None:
        ctx = self._xyz_ctx or {}
        spec = ctx.get("spec", {})
        nx, ny, nz = ctx.get("nx", 1), ctx.get("ny", 1), ctx.get("nz", 1)
        n_cells = nx * ny * nz
        ok = sum(1 for v in results.values() if v)
        self.status.showMessage(f"XYZ プロット完了（{ok}/{n_cells} セル）")
        self.append_log(f"XYZ プロット完了: {ok}/{n_cells} セル成功")
        if ctx.get("cell_fmt"):
            self.append_log(f"個別画像を {ctx.get('cells_saved', 0)} 枚"
                            "保存しました（各セル生成直後に保存）")
        self._xyz_last_ok = ok > 0  # 連続モードは成功時のみ続行
        # モデル軸でマージモデルを使ったならピン状態 ●/○ を最新化。
        if any(p.merge_models for p in ctx.get("params", {}).values()):
            self._sync_merge_states()
        # 比較グリッドの合成/保存はワーカースレッドで行う（大きなグリッド
        # では数秒かかり、UI スレッドで行うとフリーズして見えるため）。
        # 表示 OFF なら合成自体をスキップ（保存も表示が前提）。
        if not spec.get("show_grid", True):
            return
        cells: list = [None] * n_cells
        for idx, data in results.items():
            cells[idx] = data or None
        labels = [a["labels"] for a in spec.get("axes", [])] or [[""]] * 3
        save_path = None
        if spec.get("save_grid", True):
            if metadata.AVAILABLE:
                from datetime import datetime
                stamp = (ctx.get("stamp")
                         or datetime.now().strftime("%Y%m%d_%H%M%S"))
                # グリッドは本機能の成果物なので Format 設定に関わらず PNG。
                save_path = (self.paths.output_dir
                             / f"xyz_{stamp}_{self._last_seed}.png")
            else:
                self.append_log("警告: Pillow/piexif が無いため保存できません")
        # メタデータ文字列は UI の値を読むためここ（UIスレッド）で組み立てる。
        params_text, extra = self._build_metadata(ctx.get("base"))
        for name, a in zip(("x", "y", "z"), spec.get("axes", [])):
            if a["id"] != "none":
                extra[f"xyz_{name}_type"] = a["id"]
                extra[f"xyz_{name}_values"] = ", ".join(a["labels"])
        self._start_xyz_compose(
            cells, nx, ny, nz, labels, int(spec.get("margin", 0)),
            save_path, params_text, extra, bool(ctx.get("embed", True)))

    def _start_xyz_compose(self, cells, nx, ny, nz, labels, margin,
                           save_path, params_text, extra, embed) -> None:
        thread = QThread(self)
        worker = _XyzComposeWorker(cells, nx, ny, nz, labels, True, margin,
                                   save_path, params_text, extra, embed)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self._on_xyz_grid_ready)
        worker.done.connect(thread.quit)
        # 参照を保持（GC 防止）。連続モードで次の実行と重なっても互いに独立。
        self._xyz_compose_jobs.append((thread, worker))
        thread.finished.connect(
            lambda t=thread, w=worker: self._xyz_compose_jobs.remove((t, w)))
        thread.start()

    def _on_xyz_grid_ready(self, res: dict) -> None:
        self.append_log(f"比較画像を合成しました（{res['w']}x{res['h']}px）")
        for line in res.get("log", []):
            self.append_log(line)
        self._last_images = [res["png"]]
        self._show_image(res["png"])

    def _on_xyz_failed(self, msg: str) -> None:
        self.status.showMessage("XYZ プロット失敗")
        self.append_log("エラー: " + msg)
        if "キャンセル" not in msg:
            QMessageBox.critical(self._xyz_parent(), "XYZ プロットエラー", msg)

    def _cleanup_xyz_thread(self) -> None:
        self._xyz_thread = None
        self._xyz_worker = None
        self._xyz_ctx = None
        self.btn_generate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._push_xyz_running(False)
        # XYZ 連続モード: 成功（またはスキップによる中断）で終わり、
        # ウィンドウの連続がONなら同じ設定で次の実行を開始する。
        skip = self._xyz_skip
        self._xyz_skip = False
        spec = getattr(self, "_xyz_last_spec", None)
        if not ((getattr(self, "_xyz_last_ok", False) or skip)
                and spec is not None and self._xyz_dlg is not None):
            return
        try:
            chained = self._xyz_dlg.chk_continuous.isChecked()
        except RuntimeError:
            self._xyz_dlg = None
            return
        if chained:
            QTimer.singleShot(
                0, lambda: self._on_xyz_requested(spec, auto=True))

    # ----- preset filtering (family judged from file content) -------------
    def _current_preset(self) -> str:
        return self.cb_preset.currentData() or "anima"

    def _set_preset(self, token: str) -> None:
        i = self.cb_preset.findData(token)
        self.cb_preset.setCurrentIndex(i if i >= 0 else 0)

    def _filter_for_preset(self, kind: str, files: list[str], preset: str) -> list[str]:
        if preset == "all":
            return files
        # "shared" は anima/krea2 の共通ファイル（Qwen-Image VAE）を指す。
        # sdxl プリセットには含めない。
        allowed = {preset, "shared"} if preset in ("anima", "krea2") \
            else {preset}
        root = config.models_root() / kind
        return [f for f in files
                if modelinfo.family(kind, root / f) in allowed]

    def _apply_preset_filter(self) -> None:
        """Repopulate the model combos with only the current preset's files."""
        if not hasattr(self, "_all_models"):
            return
        preset = self._current_preset()
        prev_data = self.cb_diffusion.currentData()
        self._fill_combo(self.cb_diffusion, self._filter_for_preset(
            "diffusion_models", self._all_models["diffusion_models"], preset))
        # Merge entries go at the top of the list, filtered like regular
        # models (family judged from the first source model), with a green
        # tint to stand out.
        self.cb_diffusion.blockSignals(True)
        shown = [e for e in self._merges
                 if preset == "all" or self._merge_family(e) in (preset, "shared")]
        for i, e in enumerate(shown):
            self.cb_diffusion.insertItem(
                i, MERGE_PREFIX + e["name"], f"{MERGE_TOKEN}:{e['id']}")
            self.cb_diffusion.setItemData(
                i, QBrush(QColor(MERGE_COLOR)), Qt.ForegroundRole)
        if isinstance(prev_data, str) and prev_data.startswith(MERGE_TOKEN):
            idx = self.cb_diffusion.findData(prev_data)
            if idx >= 0:
                self.cb_diffusion.setCurrentIndex(idx)
        self.cb_diffusion.blockSignals(False)
        self._fill_combo(self.cb_vae, self._filter_for_preset(
            "vae", self._all_models["vae"], preset))
        te = self._filter_for_preset(
            "text_encoders", self._all_models["text_encoders"], preset)
        self._fill_combo(self.cb_te1, te)
        self._fill_combo(self.cb_te2, te, allow_empty=True)

    def _on_preset_changed(self, *_) -> None:
        preset = self._current_preset()
        if not self._loading:
            # 現在の Models + 設定の内容を旧プリセットの枠に退避する。
            # 絞り込み（_apply_preset_filter）はコンボの選択を書き換えるので、
            # 必ずその前に退避すること。
            old = getattr(self, "_active_preset", None)
            if old and old != preset:
                self._preset_conf[old] = self._collect_preset_conf()
            self._active_preset = preset
        self._apply_preset_filter()
        if self._loading:
            return  # startup restores exact saved picks, no auto-defaulting
        # 新プリセットの保存値（無ければ既定値）を復元する。
        stored = self._preset_conf.get(preset)
        prev_loading = self._loading
        self._loading = True  # 復元中の自動整合・自動保存を抑止
        try:
            if stored:
                self._apply_preset_conf(stored)
            else:
                self._select_preset_defaults(preset)
        finally:
            self._loading = prev_loading
        self._sync_builtin_for_diffusion()  # 新 diffusion に内蔵チェックを追従
        if not self.cb_diffusion.count():
            self.append_log(
                f"{preset} のモデルが見つかりません。"
                "「設定…」からダウンロードしてください。")
        self._schedule_save()

    # ----- per-preset Models/設定 memory -----------------------------------
    def _collect_preset_conf(self) -> dict:
        """Current Models + 設定 values (persisted per preset)."""
        diffusion = (self.cb_diffusion.currentData()
                     if self._merge_selected()
                     else self.cb_diffusion.currentText())
        return {
            "diffusion": str(diffusion),
            "vae": self.cb_vae.currentText(),
            "te1": self.cb_te1.currentText(),
            "te2": self.cb_te2.currentText(),
            "dual_te": self.chk_dual_te.isChecked(),
            "clip_type": self.cb_clip_type.currentText(),
            "width": self.sp_width.value(),
            "height": self.sp_height.value(),
            "steps": self.sp_steps.value(),
            "cfg": float(self.sp_cfg.value()),
            "batch": self.sp_batch.value(),
            "sampler": self.cb_sampler.currentText(),
            "scheduler": self.cb_scheduler.currentText(),
            "seed": self.ed_seed.text().strip() or "-1",
            "dtype": self.cb_dtype.currentText(),
            "hires_enabled": self.grp_hires.isChecked(),
            "hires_scale": float(self.sp_hires_scale.value()),
            "hires_denoise": float(self.sp_hires_denoise.value()),
            "hires_steps": self.sp_hires_steps.value(),
            "hires_method": self.cb_hires_method.currentText(),
        }

    def _apply_preset_conf(self, c: dict) -> None:
        """Restore Models + 設定 values saved for a preset (combos are
        already filtered to that preset; missing files are skipped)."""
        try:
            saved_diffusion = str(c.get("diffusion", ""))
            if saved_diffusion.startswith(MERGE_TOKEN):
                idx = self.cb_diffusion.findData(saved_diffusion)
                if idx >= 0:
                    self.cb_diffusion.setCurrentIndex(idx)
            elif saved_diffusion:
                self.cb_diffusion.setCurrentText(saved_diffusion)
            self.cb_vae.setCurrentText(str(c.get("vae", "")))
            self.cb_te1.setCurrentText(str(c.get("te1", "")))
            self.cb_te2.setCurrentText(str(c.get("te2", "")))
            self.chk_dual_te.setChecked(bool(c.get("dual_te", False)))
            self._on_dual_toggled(self.chk_dual_te.isChecked())
            self.cb_clip_type.setCurrentText(
                str(c.get("clip_type", "stable_diffusion")))
            self.sp_width.setValue(int(c.get("width", self.sp_width.value())))
            self.sp_height.setValue(int(c.get("height", self.sp_height.value())))
            self.sp_steps.setValue(int(c.get("steps", self.sp_steps.value())))
            self.sp_cfg.setValue(float(c.get("cfg", self.sp_cfg.value())))
            self.sp_batch.setValue(int(c.get("batch", self.sp_batch.value())))
            self.cb_sampler.setCurrentText(str(c.get("sampler", "er_sde")))
            self.cb_scheduler.setCurrentText(str(c.get("scheduler", "simple")))
            self.ed_seed.setText(str(c.get("seed", "-1")))
            self.cb_dtype.setCurrentText(str(c.get("dtype", "default")))
            self.grp_hires.setChecked(bool(c.get("hires_enabled", False)))
            self.sp_hires_scale.setValue(float(c.get("hires_scale", 1.5)))
            self.sp_hires_denoise.setValue(
                float(c.get("hires_denoise", 0.55)))
            self.sp_hires_steps.setValue(int(c.get("hires_steps", 0)))
            self.cb_hires_method.setCurrentText(
                str(c.get("hires_method", "bislerp")))
        except (TypeError, ValueError):
            pass  # 壊れた保存値は途中まで適用（以後の保存で正される）

    def _select_preset_defaults(self, preset: str) -> None:
        """Pick sensible model/vae/te/CLIP defaults for the chosen family.

        The combos are already content-filtered to this family, so the text
        encoder and VAE simply default to the first entry; the diffusion model
        prefers a turbo/base variant by name when several are present."""
        if preset == "sdxl":
            # SDXL は dual TE (clip_l + clip_g) + CLIP type 'sdxl'。
            self.chk_dual_te.setChecked(True)
            self._select_preferred(self.cb_diffusion, ())
            if self.cb_vae.count():
                self.cb_vae.setCurrentIndex(0)
            clip_l, clip_g = self._sdxl_te_pair()
            if clip_l:
                self.cb_te1.setCurrentText(clip_l)
            if clip_g:
                self.cb_te2.setCurrentText(clip_g)
            idx = self.cb_clip_type.findText("sdxl")
            if idx >= 0:
                self.cb_clip_type.setCurrentIndex(idx)
            return
        self.chk_dual_te.setChecked(False)  # anima/krea2 use a single CLIPLoader
        prefer = ("base",) if preset == "anima" else ("turbo",)
        self._select_preferred(self.cb_diffusion, prefer)
        if self.cb_vae.count():
            self.cb_vae.setCurrentIndex(0)
        if self.cb_te1.count():
            self.cb_te1.setCurrentIndex(0)
        clip = "krea2" if preset == "krea2" else "stable_diffusion"
        idx = self.cb_clip_type.findText(clip)
        if idx >= 0:
            self.cb_clip_type.setCurrentIndex(idx)

    def _select_preferred(self, combo: QComboBox, prefer: tuple[str, ...]) -> None:
        for i in range(combo.count()):
            if any(s in combo.itemText(i).lower() for s in prefer):
                combo.setCurrentIndex(i)
                return
        if combo.count():
            combo.setCurrentIndex(0)

    # ----- settings persistence -------------------------------------------
    def _apply_settings(self) -> None:
        """Apply loaded settings to the widgets (called once, at startup)."""
        s = self.settings
        # Preset first: it filters the model dropdowns before we restore picks.
        # 旧バージョンの「すべて」("all") は保存されていたモデルの系統に移行。
        preset = str(s.get("preset", ""))
        if preset not in ("anima", "krea2", "sdxl"):
            saved_diffusion = str(s.get("diffusion", ""))
            if saved_diffusion.startswith(MERGE_TOKEN):
                try:
                    entry = self._merge_entry_by_id(
                        int(saved_diffusion.split(":", 1)[1]))
                except (IndexError, ValueError):
                    entry = None
                fam = self._merge_family(entry) if entry else "unknown"
            else:
                fam = self._diffusion_family(saved_diffusion)
            preset = fam if fam in ("anima", "krea2", "sdxl") else "anima"
        self._set_preset(preset)
        # Merge selections are stored as their data token ("__merge__:<id>");
        # they are restorable because unbuilt entries auto-merge at generation.
        saved_diffusion = str(s.get("diffusion", ""))
        if saved_diffusion.startswith(MERGE_TOKEN):
            idx = self.cb_diffusion.findData(saved_diffusion)
            if idx >= 0:
                self.cb_diffusion.setCurrentIndex(idx)
        else:
            self.cb_diffusion.setCurrentText(saved_diffusion)
        self.cb_vae.setCurrentText(str(s.get("vae", "")))
        self.cb_te1.setCurrentText(str(s.get("te1", "")))
        self.cb_te2.setCurrentText(str(s.get("te2", "")))
        self.chk_dual_te.setChecked(bool(s.get("dual_te", False)))
        self._on_dual_toggled(self.chk_dual_te.isChecked())  # sync clip list
        self.cb_clip_type.setCurrentText(str(s.get("clip_type", "stable_diffusion")))
        # prompt/negative: startup values come from the first prompts.csv
        # entry (already loaded); without one the constructor defaults stay.
        presets = getattr(self, "_prompt_presets", [])
        if presets:
            _name, prompt, negative = presets[0]
            self.txt_prompt.setPlainText(prompt)
            self.txt_negative.setPlainText(negative)
        self.sp_width.setValue(int(s.get("width", 1024)))
        self.sp_height.setValue(int(s.get("height", 1024)))
        self.sp_steps.setValue(int(s.get("steps", 30)))
        self.sp_cfg.setValue(float(s.get("cfg", 4.0)))
        self.sp_batch.setValue(int(s.get("batch", 1)))
        self.cb_sampler.setCurrentText(str(s.get("sampler", "er_sde")))
        self.cb_scheduler.setCurrentText(str(s.get("scheduler", "simple")))
        self.ed_seed.setText(str(s.get("seed", "-1")))
        self.cb_dtype.setCurrentText(str(s.get("dtype", "default")))
        # Hires fix (latent)
        self.grp_hires.setChecked(bool(s.get("hires_enabled", False)))
        self.sp_hires_scale.setValue(float(s.get("hires_scale", 1.5)))
        self.sp_hires_denoise.setValue(float(s.get("hires_denoise", 0.55)))
        self.sp_hires_steps.setValue(int(s.get("hires_steps", 0)))
        self.cb_hires_method.setCurrentText(
            str(s.get("hires_method", "bislerp")))
        # image output
        self.cb_img_format.setCurrentText(str(s.get("image_format", "png")))
        self.stack_quality.setCurrentIndex(self.cb_img_format.currentIndex())
        self.sp_png_compress.setValue(int(s.get("png_compress", 6)))
        self.sp_jpg_quality.setValue(int(s.get("jpg_quality", 92)))
        self.sp_webp_quality.setValue(int(s.get("webp_quality", 90)))
        self.chk_embed_meta.setChecked(bool(s.get("embed_metadata", True)))
        # Applied LoRAs は永続化しない（毎回まっさらで起動する仕様）。
        self._loras = []
        self._rebuild_lora_rows()
        # Per-preset Models/設定 memory (settings "preset_conf" JSON).
        try:
            pc = json.loads(str(s.get("preset_conf", "{}")))
            self._preset_conf = pc if isinstance(pc, dict) else {}
        except (ValueError, TypeError):
            self._preset_conf = {}
        self._active_preset = self._current_preset()

    def _connect_autosave(self) -> None:
        """Save whenever any persisted control changes."""
        for combo in (self.cb_diffusion, self.cb_vae, self.cb_te1, self.cb_te2,
                      self.cb_clip_type, self.cb_sampler, self.cb_scheduler,
                      self.cb_dtype, self.cb_img_format, self.cb_hires_method):
            combo.currentTextChanged.connect(self._schedule_save)
        for spin in (self.sp_width, self.sp_height, self.sp_steps, self.sp_batch,
                     self.sp_png_compress, self.sp_jpg_quality, self.sp_webp_quality,
                     self.sp_hires_steps):
            spin.valueChanged.connect(self._schedule_save)
        self.sp_cfg.valueChanged.connect(self._schedule_save)
        self.sp_hires_scale.valueChanged.connect(self._schedule_save)
        self.sp_hires_denoise.valueChanged.connect(self._schedule_save)
        self.grp_hires.toggled.connect(self._schedule_save)
        self.chk_dual_te.toggled.connect(self._schedule_save)
        self.chk_embed_meta.toggled.connect(self._schedule_save)
        self.ed_seed.textChanged.connect(self._schedule_save)

    def _schedule_save(self, *args) -> None:
        if self._loading:
            return
        self._save_timer.start()  # debounce rapid changes (e.g. spinbox drag)

    def _do_save(self) -> None:
        if getattr(self, "_settings_broken", False):
            return  # don't overwrite a hand-broken settings.toml
        # Persist everything except prompt/negative, which keep their file values.
        # Merge selections persist as their data token, not the display text.
        diffusion = (self.cb_diffusion.currentData()
                     if self._merge_selected()
                     else self.cb_diffusion.currentText())
        # Keep the active preset's slot in sync with the widgets.
        self._preset_conf[self._current_preset()] = self._collect_preset_conf()
        data = {
            "preset": self._current_preset(),
            "preset_conf": json.dumps(self._preset_conf, ensure_ascii=False),
            "diffusion": str(diffusion),
            "vae": self.cb_vae.currentText(),
            "te1": self.cb_te1.currentText(),
            "te2": self.cb_te2.currentText(),
            "dual_te": self.chk_dual_te.isChecked(),
            "clip_type": self.cb_clip_type.currentText(),
            "merges": json.dumps(
                [{"id": e["id"], "name": e["name"],
                  "models": [[n, w] for n, w in e["models"]],
                  "quant": e["quant"], "low_memory": e["low_memory"]}
                 for e in self._merges], ensure_ascii=False),
            "merge_seq": int(self._merge_seq),
            "width": self.sp_width.value(),
            "height": self.sp_height.value(),
            "steps": self.sp_steps.value(),
            "cfg": float(self.sp_cfg.value()),
            "batch": self.sp_batch.value(),
            "sampler": self.cb_sampler.currentText(),
            "scheduler": self.cb_scheduler.currentText(),
            "seed": self.ed_seed.text().strip() or "-1",
            "dtype": self.cb_dtype.currentText(),
            "hires_enabled": self.grp_hires.isChecked(),
            "hires_scale": float(self.sp_hires_scale.value()),
            "hires_denoise": float(self.sp_hires_denoise.value()),
            "hires_steps": self.sp_hires_steps.value(),
            "hires_method": self.cb_hires_method.currentText(),
            "image_format": self.cb_img_format.currentText(),
            "embed_metadata": self.chk_embed_meta.isChecked(),
            "png_compress": self.sp_png_compress.value(),
            "jpg_quality": self.sp_jpg_quality.value(),
            "webp_quality": self.sp_webp_quality.value(),
        }
        self.settings.update(data)
        try:
            settings.save(self.paths.settings_path, self.settings)
        except OSError as e:
            self.append_log(f"設定の保存に失敗: {e}")

    # ----- backend start ---------------------------------------------------
    def start_backend(self) -> None:
        # SageAttention の設定を起動フラグへ反映（未導入なら backend 側で無視）。
        self.backend.use_sage_attention = bool(
            self.settings.get("sage_attention", False))
        self._start_thread = QThread(self)
        worker = _StartWorker(self.backend)
        worker.moveToThread(self._start_thread)
        self._start_thread.started.connect(worker.run)
        worker.log.connect(self.append_log)
        worker.done.connect(self._on_backend_ready)
        worker.failed.connect(self._on_backend_failed)
        worker.done.connect(self._start_thread.quit)
        worker.failed.connect(self._start_thread.quit)
        self._start_worker = worker  # keep ref
        self._start_thread.start()

    def _on_backend_ready(self) -> None:
        self.status.showMessage(f"バックエンド準備完了: {self.backend.base_url}")
        self.append_log("バックエンド準備完了")

    def _on_backend_failed(self, msg: str) -> None:
        self.status.showMessage("バックエンドの起動に失敗")
        self.append_log("エラー: " + msg)
        QMessageBox.critical(self, "バックエンドエラー", msg)

    # ----- generation ------------------------------------------------------
    def _seed_value(self) -> int:
        try:
            return int(self.ed_seed.text().strip())
        except ValueError:
            return -1

    def _collect_params(self) -> GenParams:
        te = [self.cb_te1.currentText().strip()]
        if self.chk_dual_te.isChecked() and self.cb_te2.currentText().strip():
            te.append(self.cb_te2.currentText().strip())

        # -1 は生成のたびに内部でランダム値へ解決する（欄は書き換えない。
        # 使った値はログ・画像メタデータに残る）。
        seed = self._seed_value()
        if seed < 0:
            seed = random.randint(0, MAX_SEED)

        entry = self._selected_merge_entry()
        self._last_merge_id = int(entry["id"]) if entry else None
        # フルチェックポイントで内蔵 ON のときは VAE/TE を空にして内蔵を使う。
        checkpoint = entry is None and self._diffusion_is_checkpoint()
        use_builtin = checkpoint and self.chk_builtin.isChecked()
        vae = "" if use_builtin else self.cb_vae.currentText().strip()
        te_list = [] if use_builtin else [t for t in te if t]
        return GenParams(
            diffusion="" if entry else self.cb_diffusion.currentText().strip(),
            checkpoint=checkpoint,
            merge_models=list(entry["models"]) if entry else [],
            merge_quant=entry["quant"] if entry else "",
            merge_low_memory=entry["low_memory"] if entry else False,
            vae=vae,
            te=te_list,
            clip_type=self.cb_clip_type.currentText(),
            loras=[(e["name"], float(e["strength"])) for e in self._loras],
            prompt=self.txt_prompt.toPlainText(),
            negative=self.txt_negative.toPlainText(),
            width=self.sp_width.value(),
            height=self.sp_height.value(),
            steps=self.sp_steps.value(),
            cfg=self.sp_cfg.value(),
            sampler=self.cb_sampler.currentText(),
            scheduler=self.cb_scheduler.currentText(),
            seed=seed,
            batch_size=self.sp_batch.value(),
            weight_dtype=self.cb_dtype.currentText(),
            hires_enabled=self.grp_hires.isChecked(),
            hires_scale=float(self.sp_hires_scale.value()),
            hires_denoise=float(self.sp_hires_denoise.value()),
            hires_steps=self.sp_hires_steps.value(),
            hires_method=self.cb_hires_method.currentText(),
        )

    # ----- prompt presets (prompts.csv) -------------------------------------
    def _reload_prompt_presets(self, *_args, quiet: bool = False) -> None:
        try:
            # Created on first run so the sample preset seeds the prompt
            # fields at startup (the first entry is the startup prompt).
            prompt_presets.ensure_file(self.paths.prompts_path)
            self._prompt_presets = prompt_presets.load(self.paths.prompts_path)
        except (OSError, ValueError) as e:
            self.append_log(f"prompts.csv の読み込みエラー: {e}")
            return
        current = self.cb_prompt_preset.currentText()
        self.cb_prompt_preset.blockSignals(True)
        self.cb_prompt_preset.clear()
        self.cb_prompt_preset.addItems([n for n, _p, _n in self._prompt_presets])
        idx = self.cb_prompt_preset.findText(current)
        if idx >= 0:
            self.cb_prompt_preset.setCurrentIndex(idx)
        self.cb_prompt_preset.blockSignals(False)
        if not quiet:
            self.append_log(
                f"プロンプトプリセットを {len(self._prompt_presets)} 件読み込みました")

    def _apply_prompt_preset(self) -> None:
        i = self.cb_prompt_preset.currentIndex()
        if i < 0 or i >= len(getattr(self, "_prompt_presets", [])):
            return
        _name, prompt, negative = self._prompt_presets[i]
        self._append_prompt_text(self.txt_prompt, prompt)
        self._append_prompt_text(self.txt_negative, negative)

    @staticmethod
    def _append_prompt_text(edit, text: str) -> None:
        """Append preset text, comma-separated after any existing content."""
        if not text:
            return
        current = edit.toPlainText()
        if current.strip():
            sep = "" if current.rstrip().endswith(",") else ","
            edit.setPlainText(current.rstrip() + sep + " " + text)
        else:
            edit.setPlainText(text)

    def _open_prompt_csv(self) -> None:
        path = self.paths.prompts_path
        try:
            prompt_presets.ensure_file(path)
        except OSError as e:
            self.append_log(f"prompts.csv を作成できませんでした: {e}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt signature)
        # Shift+Enter in a prompt field triggers generation.
        if (obj in (self.txt_prompt, self.txt_negative)
                and event.type() == QEvent.KeyPress
                and event.key() in (Qt.Key_Return, Qt.Key_Enter)
                and event.modifiers() & Qt.ShiftModifier):
            self.on_generate()
            return True
        # LoRA チップのホバーでトリガーワードのポップアップを開閉する。
        name = getattr(obj, "_lora_name", None)
        if name is not None:
            if event.type() == QEvent.Enter:
                self._show_lora_popup(name, getattr(obj, "_lora_chip", obj))
            elif event.type() == QEvent.Leave:
                self._lora_pop_timer.start()   # 猶予後に閉じる（保持判定つき）
        return super().eventFilter(obj, event)

    def on_generate(self) -> None:
        """生成ボタン / Shift+Enter。アイドルなら即開始、生成中なら現在の
        UI設定のスナップショットをタスクとして積む（完了後に順次消費）。"""
        if not self.backend.is_running():
            QMessageBox.warning(self, "未準備", "バックエンドがまだ起動していません。")
            return
        if self._xyz_thread is not None:
            return  # XYZ 実行中はタスク積み対象外（ボタンも無効）
        if self._merge_running:
            QMessageBox.information(
                self, "マージ実行中", "マージの完了後に生成してください。")
            return
        if self._merge_selected() and self._selected_merge_entry() is None:
            QMessageBox.warning(self, "マージ未設定",
                                "選択中のマージモデルが見つかりません。")
            self._open_merge_dialog()
            return
        try:
            params = self._collect_params()
            warn = self._config_warning(params)
            if warn:
                QMessageBox.warning(self, "モデル設定を確認してください", warn)
                return
            build_graph(params)   # 積む場合も入力不足はこの場で検出する
        except ValueError as e:
            QMessageBox.warning(self, "入力不足", str(e))
            return
        if self._gen_thread is not None:
            # 生成中: タスクに積む（押した時点の設定・seed を保持）。
            self._gen_queue.append(params)
            self.append_log(
                f"生成をタスクに積みました（待機 {len(self._gen_queue)} 件, "
                f"seed={params.seed}）")
            self._update_generate_button()
            return
        self._start_generation(params)

    def _start_generation(self, params: GenParams) -> None:
        """1件の生成を開始する（params は確定済みのスナップショット）。"""
        try:
            graph = build_graph(params)
        except ValueError as e:
            # 積んだ後に前提が消えた場合（マージ削除等）のみ起こり得る。
            QMessageBox.warning(self, "入力不足", str(e))
            self._update_generate_button()
            return
        self._merge_was_cached = False
        self._last_seed = params.seed
        self._last_params = params
        self._last_graph = graph
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.status.showMessage("生成中…")
        self.append_log(f"生成 seed={params.seed} {params.width}x{params.height}")

        self._gen_thread = QThread(self)
        self._gen_worker = _GenWorker(self.backend, graph)
        self._gen_worker.moveToThread(self._gen_thread)
        self._gen_thread.started.connect(self._gen_worker.run)
        self._gen_worker.progress.connect(self._on_progress)
        self._gen_worker.preview.connect(self._on_preview_frame)
        self._gen_worker.cached.connect(self._on_cached_nodes)
        self._gen_worker.timing.connect(self._on_gen_timing)
        self._gen_worker.done.connect(self._on_gen_done)
        self._gen_worker.failed.connect(self._on_gen_failed)
        self._gen_worker.done.connect(self._gen_thread.quit)
        self._gen_worker.failed.connect(self._gen_thread.quit)
        self._gen_thread.finished.connect(self._cleanup_gen_thread)
        self._gen_thread.start()
        self._update_generate_button()

    def _update_generate_button(self, *_a) -> None:
        """生成中はボタンを「生成をタスクに積む (待機数)」表示にする。"""
        if self._gen_thread is not None:
            self.btn_generate.setText(
                f"生成をタスクに積む ({len(self._gen_queue)})")
            self.btn_generate.setToolTip(
                "現在の設定・プロンプトのスナップショットを待機タスクとして"
                "積みます。現在の生成が終わると順番に実行されます")
        else:
            self.btn_generate.setText("生成")
            self.btn_generate.setToolTip("")

    def on_cancel(self) -> None:
        """メイン画面のキャンセル/スキップボタン。

        連続 ON のときは「スキップ」: 現在の生成だけ中断し、連続はそのまま
        次の生成へ進む。OFF のときは従来どおりのキャンセル。XYZ 実行中に
        押された場合は XYZ を完全に停止する（XYZ のスキップは XYZ ウィンドウ
        のボタンで行う）。"""
        if self._gen_worker:
            if self.btn_continuous.isChecked():
                self._gen_skip = True
                self.append_log("スキップ: 現在の生成を中断して次へ進みます")
            else:
                self.append_log("キャンセルを要求しました")
            self._gen_worker.cancel()
        if self._xyz_worker:
            # メイン側からのキャンセルは XYZ ループ終了の合図。
            if self._xyz_dlg is not None:
                try:
                    self._xyz_dlg.chk_continuous.setChecked(False)
                except RuntimeError:
                    self._xyz_dlg = None
            self._xyz_worker.cancel()
            self.append_log("XYZ プロットのキャンセルを要求しました")

    def _on_xyz_cancel_requested(self) -> None:
        """XYZ ウィンドウのキャンセル/スキップボタン。

        XYZ の連続 ON なら「スキップ」: 現在の実行だけ中断し、連続は維持
        したまま次の実行へ進む。OFF なら従来どおりのキャンセル。"""
        if not self._xyz_worker:
            return
        xyz_cont = False
        if self._xyz_dlg is not None:
            try:
                xyz_cont = self._xyz_dlg.chk_continuous.isChecked()
            except RuntimeError:
                self._xyz_dlg = None
        if xyz_cont:
            self._xyz_skip = True
            self.append_log("スキップ: 現在の XYZ 実行を中断して次の実行へ進みます")
        else:
            self.append_log("XYZ プロットのキャンセルを要求しました")
        self._xyz_worker.cancel()

    def _update_cancel_button(self, *_a) -> None:
        """連続 ON のときはキャンセルボタンを「スキップ」表示にする。"""
        cont = self.btn_continuous.isChecked()
        self.btn_cancel.setText("スキップ" if cont else "キャンセル")
        self.btn_cancel.setToolTip(
            "現在の生成を中断して次の生成に進みます（連続は続行）" if cont
            else "")

    def _on_progress(self, p: Progress) -> None:
        if p.maximum:
            self.progress.setMaximum(p.maximum)
            self.progress.setValue(p.value)
            self.progress.setFormat(f"{p.note} {p.value}/{p.maximum}")

    def _on_gen_timing(self, secs: float) -> None:
        """純粋な推論時間（サンプリングのみ）をログとステータスバーへ。"""
        self.lbl_gen_time.setText(f"推論 {secs:.2f} 秒")
        self.append_log(f"推論時間: {secs:.2f} 秒")

    def _swap_size(self) -> None:
        w, h = self.sp_width.value(), self.sp_height.value()
        self.sp_width.setValue(h)
        self.sp_height.setValue(w)

    def _on_preview_frame(self, data: bytes) -> None:
        # Live latent preview during sampling; does not replace the final image.
        self._show_image(data)

    def _on_gen_done(self, images: list) -> None:
        self.status.showMessage(f"完了（{len(images)} 枚）")
        self.append_log(f"完了: {len(images)} 枚")
        self._last_gen_ok = True
        self._last_images = images
        # A merge generation guarantees the merged model is (now) in backend
        # RAM — auto-built, reused from the pin cache, or a plain cache hit.
        # Sync ●/○ from the backend's pin cache (the source of truth).
        if (self._last_params is not None and self._last_params.merge_models
                and self._last_merge_id is not None):
            self._merge_built_ids.add(self._last_merge_id)
            self._sync_merge_states()
        if images:
            self._show_image(images[0])
            self._save_outputs(images)

    def _on_cached_nodes(self, nodes: list) -> None:
        # Node "4" is the merge node in every merge graph; seeing it in the
        # execution_cached list means the merged model came straight from RAM.
        if "4" in nodes:
            self._merge_was_cached = True

    def _save_outputs(self, images: list) -> list:
        """Save each generated image to output/ with embedded metadata."""
        from datetime import datetime
        fmt = self.cb_img_format.currentText()
        if fmt == NO_SAVE:
            self.append_log("画像は保存しません（Format = 保存しない）")
            return []
        if not metadata.AVAILABLE:
            self.append_log("警告: Pillow/piexif が無いため保存できません")
            return []
        out = self.paths.output_dir
        ext, quality = self._encode_params(fmt)
        params_text, extra = self._build_metadata()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = []
        embed = self.chk_embed_meta.isChecked()
        for i, data in enumerate(images):
            path = out / f"scom_{stamp}_{self._last_seed}_{i}.{ext}"
            try:
                metadata.save_with_metadata(
                    data, path, fmt, quality, params_text,
                    extra=extra, comfy_prompt=self._last_graph, embed=embed,
                )
                saved.append(path)
            except Exception as e:  # noqa: BLE001
                self.append_log(f"保存に失敗: {e}")
        if saved:
            note = "メタデータ付き" if embed else "メタデータなし"
            self.append_log(f"{out} に {len(saved)} 枚保存しました（{note}）")
        else:
            self.append_log("警告: 画像を保存できませんでした")
        return saved

    def _encode_params(self, fmt: str) -> tuple[str, int]:
        """Return (extension, quality). For PNG, quality is the compress level."""
        if fmt == "jpg":
            return "jpg", self.sp_jpg_quality.value()
        if fmt == "webp":
            return "webp", self.sp_webp_quality.value()
        return "png", self.sp_png_compress.value()  # 0..9 compress level

    def _build_metadata(self, params: Optional[GenParams] = None) -> tuple[str, dict]:
        """Build the parameters string + structured dict from the last run
        (or from ``params`` when given, e.g. per-cell XYZ metadata)."""
        p = params if params is not None else self._last_params
        if p is None:
            return "", {}
        if p.merge_models:
            # Record the merge recipe so the image stays reproducible.
            model_name = ("merge(" + ", ".join(
                f"{n}:{w:g}" for n, w in p.merge_models) + ")")
            if p.merge_quant:
                model_name += f" {p.merge_quant}"
        else:
            model_name = p.diffusion
        # webui 互換: メタデータ上のプロンプトには適用中 LoRA を
        # <lora:名前:強度> タグとして末尾に入れ込む（生成に使う実際の
        # プロンプトには含まれない — 適用は LoraLoader ノードで行う）。
        prompt_meta = p.prompt
        if p.loras:
            tags = " ".join(f"<lora:{Path(n).stem}:{w:g}>"
                            for n, w in p.loras)
            body = prompt_meta.rstrip()
            prompt_meta = (body + ", " + tags) if body.strip() else tags
        meta = {
            "prompt": prompt_meta,
            "negative": p.negative,
            "steps": p.steps,
            "sampler": p.sampler,
            "scheduler": p.scheduler,
            "cfg": p.cfg,
            "seed": p.seed,
            "width": p.width,
            "height": p.height,
            "model": model_name,
            "vae": p.vae,
            "text_encoder": ", ".join(p.te),
            "clip_type": p.clip_type,
            "batch": p.batch_size,
            "dtype": p.weight_dtype,
        }
        if p.hires_enabled:
            # webui 互換フィールド名（Hires upscale / Hires steps / Denoising
            # strength）に latent 方式の情報を足す。
            meta["hires_upscale"] = f"{p.hires_scale:g}"
            meta["hires_steps"] = (int(p.hires_steps) if p.hires_steps > 0
                                   else p.steps)
            meta["denoising_strength"] = f"{p.hires_denoise:g}"
            meta["hires_method"] = f"Latent ({p.hires_method})"
        if p.loras:
            meta["loras"] = ", ".join(f"{n}:{w:g}" for n, w in p.loras)
            # webui の "Lora hashes" フィールド（AutoV2 = SHA256 先頭10桁）。
            # ハッシュは LoRA ブラウザが計算・キャッシュ済みのものだけ使う
            # （ここで数GBのハッシュ計算を始めない）。
            cache = lora_meta.LoraCache(self.paths.user_data / "lora_cache")
            hashes = []
            for n, _w in p.loras:
                e = cache.lookup(n, config.models_root() / "loras" / n)
                if e and e.get("sha256"):
                    hashes.append(f"{Path(n).stem}: {e['sha256'][:10]}")
            if hashes:
                meta["lora_hashes"] = ", ".join(hashes)
        params_text = metadata.build_parameters(meta)
        return params_text, {"app": "scom", **meta}

    def _on_gen_failed(self, msg: str) -> None:
        self.status.showMessage("生成失敗")
        self.append_log("エラー: " + msg)
        self._last_gen_ok = False
        if "キャンセル" not in msg:
            QMessageBox.critical(self, "生成エラー", msg)

    def _cleanup_gen_thread(self) -> None:
        self._gen_thread = None
        self._gen_worker = None
        self.btn_generate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        skip = self._gen_skip
        self._gen_skip = False
        proceed = self._last_gen_ok or skip
        if proceed and self._gen_queue:
            # 待機タスクを最優先で消費（連続 ON でもタスクが先）。
            params = self._gen_queue.pop(0)
            self.append_log(
                f"待機タスクを開始します（残り {len(self._gen_queue)} 件）")
            self._update_generate_button()
            QTimer.singleShot(0, lambda p=params: self._start_generation(p))
            return
        if proceed and self.btn_continuous.isChecked():
            # Continuous mode: start the next with the current UI values.
            self._update_generate_button()
            QTimer.singleShot(0, self.on_generate)
            return
        if self._gen_queue:
            # キャンセル/エラーで停止したときは待機タスクも破棄する。
            n = len(self._gen_queue)
            self._gen_queue.clear()
            self.append_log(f"停止したため待機中のタスク {n} 件を破棄しました")
        self._update_generate_button()

    def _show_image(self, data: bytes) -> None:
        img = QImage.fromData(data)
        if img.isNull():
            return
        pix = QPixmap.fromImage(img)
        self.preview.setPixmap(
            pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    # ----- misc ------------------------------------------------------------
    def append_log(self, text: str) -> None:
        ansi_log.append_ansi(self.log_view, text)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        super().resizeEvent(event)
        if self._last_images:
            self._show_image(self._last_images[0])

    def closeEvent(self, event) -> None:  # noqa: N802
        # The merge/XYZ/LoRA windows are parentless (so they can go behind
        # us); close them explicitly or the app would keep running after the
        # main window.
        for dlg in (self._merge_dlg, self._xyz_dlg, self._lora_dlg):
            if dlg is not None:
                try:
                    dlg.close()
                except RuntimeError:
                    pass
        self.append_log("バックエンドを終了中…")
        if self._gen_worker:
            self._gen_worker.cancel()
        if self._xyz_worker:
            self._xyz_worker.cancel()
        self.backend.stop()
        super().closeEvent(event)

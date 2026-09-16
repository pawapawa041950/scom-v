"""PySide6 main window: MiniMax H3 video generation (t2v / i2v / r2v)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QRegularExpression, QTimer, QEvent, QUrl,
    QPoint,
)
from PySide6.QtGui import (
    QColor, QCursor, QDesktopServices, QImage, QPixmap,
    QRegularExpressionValidator, QTextCharFormat, QTextCursor, QTextFormat,
)
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSizePolicy, QSpinBox, QSplitter, QStackedWidget,
    QVBoxLayout, QWidget,
)

from .. import config, settings, prompt_presets, workflow
from .. import lora as lora_meta
from . import ansi_log
from .widgets import (
    FlowLayout, GrowingTextEdit, PlaceholderListWidget, WideComboBox,
)
from ..bootstrap import environment
from ..bootstrap import models as models_mod
from ..bootstrap.setup import (
    SetupError, install_sage_attention, sage_installed, _Manifest,
)
from ..comfy_backend import ComfyBackend, BackendError, Progress
from ..workflow import (
    GenParams, build_graph, frames_for_seconds, size_for_aspect,
    size_for_image, ASPECT_PRESETS, SAMPLERS, SCHEDULERS, FPS,
    SPARSE_METHODS,
)

MAX_SEED = 2**63 - 1

# LoRA トリガーワードをプロンプト欄に挿入したときの区別用マーキング。
# 挿入した文字範囲に専用の文字書式（背景色 + token プロパティ）を付け、
# 見た目と区間追跡の両方で元のプロンプトと区別する。token を持つ区間は
# ユーザーが編集しても書式が残るので、編集後のワードごとまとめて削除できる。
_LORA_TOKEN_PROP = QTextFormat.UserProperty + 17
_LORA_INSERT_BG = QColor("#e7edf5")   # 明るい背景
_LORA_INSERT_FG = QColor("#22456e")   # 濃い文字色

MODES = [("t2v", "テキストから動画 (t2v)"),
         ("i2v", "画像から動画 (i2v)"),
         ("r2v", "参照から動画 (r2v)"),
         ("chain", "長尺チェーン (ContexLoop)")]

_IMAGE_FILTER = "画像 (*.png *.jpg *.jpeg *.webp *.bmp);;すべて (*.*)"
_VIDEO_FILTER = "動画 (*.mp4 *.webm *.mkv *.mov *.avi);;すべて (*.*)"
_AUDIO_FILTER = "音声 (*.wav *.mp3 *.flac *.ogg *.m4a);;すべて (*.*)"
_VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov", ".avi")


# SageAttention インストールジョブ（親なしスレッド）の生存参照。UI 側の
# 状態に関係なくインストールを完走させるために保持する。
_SAGE_JOBS: list = []


class _SageInstallWorker(QObject):
    log = Signal(str)
    done = Signal()
    failed = Signal(str)

    def __init__(self, paths: config.AppPaths):
        super().__init__()
        self.paths = paths

    def run(self) -> None:
        try:
            install_sage_attention(self.paths, self.log.emit)
            self.done.emit()
        except SetupError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _ModelDownloadWorker(QObject):
    """1ファイルをバックグラウンドでダウンロードする（Turbo LoRA 用）。"""
    progress = Signal(float, float)   # done, total bytes
    done = Signal(str)                # filename
    failed = Signal(str, str)         # filename, message

    def __init__(self, paths: config.AppPaths, item: models_mod.ModelFile):
        super().__init__()
        self.paths = paths
        self.item = item
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            models_mod.download_model(
                self.paths, self.item,
                on_progress=lambda d, t: self.progress.emit(float(d), float(t)),
                cancel=lambda: self._cancel)
            self.done.emit(self.item.filename)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(self.item.filename, str(e))


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
    timing = Signal(float)   # 純粋な推論(サンプリング)時間 [秒]
    done = Signal(list)      # list[Path] 保存された出力ファイル
    failed = Signal(str)

    def __init__(self, backend: ComfyBackend, graph: dict,
                 extra_pnginfo: Optional[dict] = None):
        super().__init__()
        self.backend = backend
        self.graph = graph
        self.extra_pnginfo = extra_pnginfo
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            files = self.backend.generate(
                self.graph,
                on_progress=self.progress.emit,
                on_preview=self.preview.emit,
                cancel=lambda: self._cancel,
                on_timing=self.timing.emit,
                extra_pnginfo=self.extra_pnginfo,
            )
            self.done.emit(list(files))
        except BackendError as e:
            self.failed.emit(str(e))
        except Exception as e:  # pragma: no cover - defensive
            self.failed.emit(f"unexpected error: {e}")


class MainWindow(QMainWindow):
    _NOT_READY_TIP = "バックエンド（ComfyUI）の準備が完了するまで生成できません"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("scom-v - 動画生成")

        self.paths = config.AppPaths()
        self.backend = ComfyBackend(self.paths)
        self._start_thread: Optional[QThread] = None
        self._gen_thread: Optional[QThread] = None
        self._gen_worker: Optional[_GenWorker] = None
        self._last_video: Optional[Path] = None
        self._last_seed: int = 0
        self._last_gen_ok: bool = False
        self._gen_skip: bool = False
        # 生成中に積まれた待機タスク（押した時点の GenParams スナップショット）。
        self._gen_queue: list[GenParams] = []
        # ローカルパス -> (mtime, アップロード済み名) のキャッシュ。連続生成で
        # 同じ参照ファイルを毎回アップロードし直さないため。
        self._upload_cache: dict[str, tuple[float, str]] = {}
        self._all_models: dict[str, list[str]] = {}
        # ContexLoop の設定（専用ウィンドウが保持する plan 辞書）。
        self._chain_plan: Optional[dict] = None
        self._chain_dlg = None
        # Turbo LoRA のダウンロードジョブ（同時に1つ）。
        self._turbo_job: Optional[tuple] = None
        self._turbo_last_pct = -1
        # fl2v 用 Turbo LoRA の版（r2v 表示中もこの値を保持し、保存する）。
        self._turbo_fl2v_variant = "8step"
        # 版ごとのステップ数（設定から復元、スピンボックス変更で更新）。
        self._turbo_steps: dict[str, int] = dict(models_mod.TURBO_DEFAULT_STEPS)
        # 適用中 LoRA（チェックポイント別に記憶: fl2va = t2v/i2v, ref2va = r2v。
        # scom と同様、アプリ再起動では保存しない）。
        self._loras_by_family: dict[str, list[dict]] = {
            "fl2va": [], "ref2va": []}
        self._lora_dlg = None            # 非モーダル LoraDialog（最大1個）
        self._lora_popup = None          # チップホバーのポップアップ（遅延生成）
        self._lora_pop_anchor = None
        self._lora_pop_timer = QTimer(self)
        self._lora_pop_timer.setSingleShot(True)
        self._lora_pop_timer.setInterval(220)   # 離脱後この時間で閉じる
        self._lora_pop_timer.timeout.connect(self._hide_lora_popup)

        self.settings, settings_error = settings.load(self.paths.settings_path)
        self._loading = True
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(250)
        self._save_timer.timeout.connect(self._do_save)

        self._build_ui()
        # 既定サイズ（settings.py の window_size 既定と同値）。保存済みの
        # window_size / pane_sizes があれば _apply_settings で上書き復元される。
        self.resize(1209, 675)
        self.refresh_models()
        self._reload_prompt_presets(quiet=True)
        self._apply_settings()
        self._loading = False
        self._connect_autosave()

        if settings_error:
            self.append_log(f"settings.toml の読み込みエラー: {settings_error}")
            QMessageBox.warning(
                self, "settings.toml の読み込みに失敗",
                "settings.toml に文法エラーがあるため、既定値で起動します。\n"
                "ファイルを修正するまで自動保存は行いません。\n\n"
                f"エラー: {settings_error}",
            )
        else:
            self._do_save()
        self._settings_broken = bool(settings_error)

        self.start_backend()

    # ----- UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # 3ペイン構成: 左（モード/Models/高速化/参照入力)・中央（設定/Prompt)・
        # 右（ログ/プレビュー）。それぞれスプリッターで幅を調整できる。
        pane_left = QWidget()
        lv = QVBoxLayout(pane_left)
        pane_center = QWidget()
        cv = QVBoxLayout(pane_center)

        # モード（左ペイン上部に配置。addLayout は後段の cell_lt で行う）
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("モード:"))
        self.cb_mode = WideComboBox()
        for token, label in MODES:
            self.cb_mode.addItem(label, token)
        self.cb_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.cb_mode, stretch=1)

        # Models（左ペイン）
        box_models = QGroupBox("Models")
        form = QFormLayout(box_models)
        self.cb_diffusion = WideComboBox()
        # FastH3 選択時は高速化設定を固定表示にする（_sync_fasth3_controls）。
        self.cb_diffusion.currentTextChanged.connect(
            lambda *_a: self._sync_fasth3_controls())
        self.cb_te = WideComboBox()
        self.cb_vae_video = WideComboBox()
        self.cb_vae_audio = WideComboBox()
        self.lbl_diffusion = QLabel("Diffusion (fl2va):")
        form.addRow(self.lbl_diffusion, self.cb_diffusion)
        form.addRow("Text encoder:", self.cb_te)
        form.addRow("動画 VAE:", self.cb_vae_video)
        form.addRow("音声 VAE:", self.cb_vae_audio)
        # 再スキャン / 設定… は Models の末尾に置く
        btn_rescan = QPushButton("再スキャン")
        btn_rescan.clicked.connect(self.refresh_models)
        btn_manage = QPushButton("設定…")
        btn_manage.clicked.connect(self.open_models_dialog)
        model_btns = QHBoxLayout()
        model_btns.addStretch(1)
        model_btns.addWidget(btn_rescan)
        model_btns.addWidget(btn_manage)
        form.addRow(model_btns)

        # 左ペイン: モード行 → Models → 高速化設定
        lv.addLayout(mode_row)
        lv.addWidget(box_models)
        lv.addWidget(self._build_speed_box())

        # Prompt（中央ペイン下段）
        box_prompt = self.box_prompt = QGroupBox("Prompt")
        pv = QVBoxLayout(box_prompt)
        # 公式推奨のプロンプトは350〜500語と長くなるため、一定行数を超えたら
        # スクロールバー表示に切り替えてウィンドウの肥大化を防ぐ。
        self.txt_prompt = GrowingTextEdit(min_lines=5, max_lines=14)
        self.txt_prompt.setPlaceholderText(
            "動画の内容を文章で記述… (r2v では <Picture 1> <Video 1> <Audio 1> "
            "のタグで参照を指せます)")
        self.txt_prompt.installEventFilter(self)  # Shift+Enter で生成
        pv.addWidget(self.txt_prompt)
        # ユーザーが挿入済みハイライトを手で消したときも LoRA 窓の表示を追従。
        self.txt_prompt.textChanged.connect(self._push_lora_inserted)
        # LoRA（選択ボタン + 適用中チップ）。
        pv.addWidget(QLabel("LoRA"))
        lora_row = QWidget()
        self._lora_flow = FlowLayout(lora_row, hspacing=6, vspacing=4)
        self._lora_flow.setContentsMargins(0, 0, 0, 0)
        self.btn_lora = QPushButton("LoRA選択…")
        self.btn_lora.setToolTip(
            "LoRA の一覧（サムネイル・トリガーワード付き）を開いて"
            "適用する LoRA を選びます")
        self.btn_lora.clicked.connect(self._open_lora_dialog)
        self._lora_flow.addWidget(self.btn_lora)
        pv.addWidget(lora_row)
        preset_row = QHBoxLayout()
        self.cb_prompt_preset = WideComboBox()
        self.cb_prompt_preset.setToolTip(
            "prompts.csv のプリセット（1列目: 設定名、2列目: プロンプト）")
        btn_apply = QPushButton("書込み")
        btn_apply.setToolTip("選択中のプリセットをプロンプト欄に追記")
        btn_apply.clicked.connect(self._apply_prompt_preset)
        btn_edit = QPushButton("編集")
        btn_edit.clicked.connect(self._open_prompt_csv)
        btn_reload = QPushButton("再読込み")
        btn_reload.clicked.connect(self._reload_prompt_presets)
        preset_row.addWidget(self.cb_prompt_preset, stretch=1)
        preset_row.addWidget(btn_apply)
        preset_row.addWidget(btn_edit)
        preset_row.addWidget(btn_reload)
        pv.addLayout(preset_row)
        cv.addWidget(box_prompt)
        cv.addStretch(1)

        # モード別入力（左ペイン下段: 参照/入力設定）
        self.stack_mode = QStackedWidget()
        self.stack_mode.addWidget(self._build_t2v_page())
        self.stack_mode.addWidget(self._build_i2v_page())
        self.stack_mode.addWidget(self._build_r2v_page())
        self.stack_mode.addWidget(self._build_chain_page())
        lv.addWidget(self.stack_mode)
        lv.addStretch(1)

        # 中央ペイン: 設定を最上部に置く（その下にガイド、Prompt）
        cv.insertWidget(0, self._build_settings_box())
        cv.insertWidget(1, self._build_guides_box())

        # 右カラム: ログ / プレビュー / アクション
        right = QWidget()
        rv = QVBoxLayout(right)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setPlaceholderText("バックエンドログ…")
        ansi_log.style_log(self.log_view)
        rv.addWidget(self.log_view, stretch=1)

        # プレビュー: 生成中のフレーム静止画のみ表示する。動画の再生は
        # 外部プレーヤーに任せる（アプリ内再生はファイルをロックするため廃止）。
        self.preview = QLabel(
            "プレビュー\n（生成中のフレームがここに表示されます。"
            "ダブルクリック=外部プレーヤーで再生）")
        # 【一時措置】ペイン調整のため最小サイズを緩和（元: 480x360）。
        self.preview.setMinimumSize(80, 60)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(
            "QLabel { background:#1e1e1e; color:#888; border:1px solid #333; }")
        self.preview.installEventFilter(self)
        rv.addWidget(self.preview, stretch=3)

        act = QHBoxLayout()
        self.btn_continuous = QCheckBox("連続")
        self.btn_continuous.setToolTip(
            "ONの間、生成が終わるたびに自動で次を生成します"
            "（ON中のキャンセルボタンは「スキップ」= 現在の生成だけ中断）")
        self.btn_continuous.setMinimumHeight(40)
        self.btn_continuous.toggled.connect(self._update_cancel_button)
        self.btn_generate = QPushButton("生成")
        self.btn_generate.clicked.connect(self.on_generate)
        # バックエンド（ComfyUI）の準備が終わるまで押せない。
        self.btn_generate.setEnabled(False)
        self.btn_generate.setToolTip(self._NOT_READY_TIP)
        self.btn_cancel = QPushButton("キャンセル")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.on_cancel)
        for b in (self.btn_generate, self.btn_cancel):
            b.setMinimumHeight(40)
            sp = b.sizePolicy()
            sp.setHorizontalPolicy(QSizePolicy.Ignored)
            b.setSizePolicy(sp)
        act.addWidget(self.btn_continuous)
        act.addWidget(self.btn_generate, stretch=2)
        act.addWidget(self.btn_cancel, stretch=1)
        rv.addLayout(act)

        splitter.addWidget(pane_left)
        splitter.addWidget(pane_center)
        splitter.addWidget(right)
        splitter.setStretchFactor(2, 1)
        # ペインは内容の自然な最小幅を超えて自由に縮められるようにする
        # （ドラッグ中の実サイズはステータスバーに表示）。
        for p in (pane_left, pane_center, right):
            p.setMinimumWidth(80)
        splitter.splitterMoved.connect(
            lambda *_a: self.status.showMessage(
                f"ペイン幅 [左, 中央, 右] = {splitter.sizes()}"))
        self.splitter = splitter
        self.setCentralWidget(splitter)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setMaximumWidth(220)
        self.lbl_gen_time = QLabel("")
        self.lbl_gen_time.setToolTip(
            "直近の生成の推論時間（サンプリングのみ。モデル読み込み・"
            "テキストエンコード・VAEデコード等は含みません）")
        self.status = self.statusBar()
        self.status.addPermanentWidget(self.lbl_gen_time)
        self.status.addPermanentWidget(self.progress)
        self.status.showMessage("バックエンドを起動中…")

    def _build_chain_page(self) -> QWidget:
        """ContexLoop モード: 専用ウィンドウを開くボタンと概要だけを置く。"""
        page = QGroupBox("長尺チェーン (ContexLoop)")
        v = QVBoxLayout(page)
        note = QLabel(
            "15秒を超える動画を、シーンをつないで生成します。"
            "シーン・参照・音声の設定は専用ウィンドウで行います。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888;")
        v.addWidget(note)
        self.btn_chain = QPushButton("チェーン設定を開く…")
        self.btn_chain.setMinimumHeight(34)
        self.btn_chain.clicked.connect(self.open_chain_dialog)
        v.addWidget(self.btn_chain)
        self.lbl_chain = QLabel("")
        self.lbl_chain.setWordWrap(True)
        v.addWidget(self.lbl_chain)
        v.addStretch(1)
        return page

    def _build_t2v_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        note = QLabel("プロンプトのみから動画+音声を生成します。")
        note.setStyleSheet("color:#888;")
        v.addWidget(note)
        return page

    def _build_i2v_page(self) -> QWidget:
        # 縦に余裕があるので「ラベル / パス欄（全幅）/ ボタン行」の3行構成に
        # して、長いファイルパスが読めるようにする。
        page = QGroupBox("i2v 入力画像")
        v = QVBoxLayout(page)
        self.ed_first_frame = QLineEdit()
        self.ed_first_frame.setReadOnly(True)
        self.ed_last_frame = QLineEdit()
        self.ed_last_frame.setReadOnly(True)
        self._last_frame_widgets: list = []
        for i, (label, ed, tip) in enumerate((
            ("開始フレーム:", self.ed_first_frame,
             "動画の最初のフレームになる画像"),
            ("終端フレーム(任意):", self.ed_last_frame,
             "指定すると動画の最後がこの画像へ収束します（両方指定で補間的な生成）"),
        )):
            if i:
                v.addSpacing(8)
            ed.setToolTip(tip)
            ed.setPlaceholderText("未選択")
            lbl = QLabel(label)
            lbl.setToolTip(tip)
            v.addWidget(lbl)
            v.addWidget(ed)
            btn_sel = QPushButton("参照…")
            btn_clr = QPushButton("クリア")
            btn_sel.clicked.connect(
                lambda *_a, e=ed: self._pick_file(e, _IMAGE_FILTER))
            btn_clr.clicked.connect(lambda *_a, e=ed: self._clear_frame(e))
            row = QHBoxLayout()
            row.addWidget(btn_sel)
            row.addWidget(btn_clr)
            row.addStretch(1)
            v.addLayout(row)
            if i:
                # 終端フレーム側は「開始と同じ」ON でまとめてグレーアウトする。
                self._last_frame_widgets += [lbl, ed, btn_sel, btn_clr]
        # 開始と終端に同じ画像を使うオプション（ループ動画向け）。
        self.chk_same_frame = QCheckBox("終端フレームに開始フレームと同じ画像を使う")
        self.chk_same_frame.setToolTip(
            "ON にすると終端フレームは開始フレームと同じ画像になります"
            "（先頭と末尾がつながるループ的な動画向け）")
        self.chk_same_frame.toggled.connect(self._on_same_frame_toggled)
        v.addWidget(self.chk_same_frame)
        v.addStretch(1)
        self.ed_first_frame.textChanged.connect(self._update_size_label)
        self.ed_first_frame.textChanged.connect(self._sync_same_frame)
        return page

    def _on_same_frame_toggled(self, checked: bool) -> None:
        for w in getattr(self, "_last_frame_widgets", []):
            w.setEnabled(not checked)
        if checked:
            self._saved_last_frame = self.ed_last_frame.text()
        self._sync_same_frame()
        if not checked:
            # 手動指定に戻すときは、ON にする前の値を復元する。
            self.ed_last_frame.setText(getattr(self, "_saved_last_frame", ""))
        self._schedule_save()

    def _sync_same_frame(self, *_a) -> None:
        if getattr(self, "chk_same_frame", None) is None:
            return
        if self.chk_same_frame.isChecked():
            self.ed_last_frame.setText(self.ed_first_frame.text())

    def _build_r2v_page(self) -> QWidget:
        page = QGroupBox("r2v 参照")
        page.setToolTip(
            "プロンプト内で <Picture i> <Video k> <Audio j> のタグで参照します")
        grid = QGridLayout(page)

        def make_list(title: str, lst_tip: str, max_n: int, flt: str,
                      checkable: bool = False, placeholder: str = ""):
            lst = PlaceholderListWidget(placeholder)
            lst.setMaximumHeight(72)
            lst.setToolTip(lst_tip)
            lbl = QLabel()
            lbl.setToolTip(lst_tip)

            def update_lbl(*_a):
                lbl.setText(f"{title} {lst.count()}/{max_n}:")

            update_lbl()
            btn_add = QPushButton("+")
            btn_del = QPushButton("-")
            btn_add.setFixedWidth(28)
            btn_del.setFixedWidth(28)
            btn_add.setToolTip(f"{title}を追加（複数選択可・最大 {max_n} 件）")
            btn_del.setToolTip("選択した項目を削除")

            def add(*_a):
                remain = max_n - lst.count()
                if remain <= 0:
                    QMessageBox.information(self, "上限",
                                            f"{title}は最大 {max_n} 件です。")
                    return
                paths, _ = QFileDialog.getOpenFileNames(
                    self, f"{title}を追加", "", flt)
                for path in paths[:remain]:
                    item = QListWidgetItem(Path(path).name)
                    item.setData(Qt.UserRole, path)
                    item.setToolTip(path)
                    if checkable:
                        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                        item.setCheckState(Qt.Checked)
                    lst.addItem(item)
                if len(paths) > remain:
                    QMessageBox.information(
                        self, "上限",
                        f"{title}は最大 {max_n} 件です。"
                        f"超過した {len(paths) - remain} 件は追加されませんでした。")
                update_lbl()

            def remove(*_a):
                for it in lst.selectedItems():
                    lst.takeItem(lst.row(it))
                update_lbl()

            btn_add.clicked.connect(add)
            btn_del.clicked.connect(remove)
            # ラベルの下に +/- を横並びで置く（左列に集約して横幅を節約）。
            col = QVBoxLayout()
            col.addWidget(lbl)
            btns = QHBoxLayout()
            btns.addWidget(btn_add)
            btns.addWidget(btn_del)
            btns.addStretch(1)
            col.addLayout(btns)
            col.addStretch(1)
            return lst, col

        self.lst_ref_images, col1 = make_list(
            "画像", "キャラクター・画風などの参照画像（最大9枚）", 9,
            _IMAGE_FILTER)
        self.lst_ref_videos, col2 = make_list(
            "動画", "参照動画 2〜15秒（最大3本）。チェックONでその動画の"
            "音声も参照に含めます", 3, _VIDEO_FILTER, checkable=True,
            placeholder="チェックONでその動画の音声も参照に含める")
        self.lst_ref_audios, col3 = make_list(
            "音声", "単体の参照音声（最大3本）", 3, _AUDIO_FILTER)

        for r, (lst, col) in enumerate((
                (self.lst_ref_images, col1),
                (self.lst_ref_videos, col2),
                (self.lst_ref_audios, col3))):
            grid.addLayout(col, r, 0)
            grid.addWidget(lst, r, 1)
        grid.setColumnStretch(1, 1)

        # 参照画像の取り込み解像度（ノードの ref_image_size）。
        size_row = QHBoxLayout()
        lbl_rs = QLabel("画像参照解像度")
        self.cb_ref_size = WideComboBox()
        self.cb_ref_size.addItem("match（生成解像度に合わせる・速い）", "match")
        self.cb_ref_size.addItem("max（短辺2048px・忠実度優先）", "max")
        tip = ("参照画像をどの解像度でモデルに渡すか。\n"
               "match: 生成解像度の画素数へ縮小（速い）\n"
               "max: 短辺2048pxまで保持。人物などの同一性再現は最良だが、\n"
               "参照トークンが全ステップに乗るため生成が数倍遅くなることがあります")
        lbl_rs.setToolTip(tip)
        self.cb_ref_size.setToolTip(tip)
        size_row.addWidget(lbl_rs)
        size_row.addWidget(self.cb_ref_size, stretch=1)
        grid.addLayout(size_row, 3, 0, 1, 2)
        # TE のみ参照（v0.35〜: vae/audio_vae を繋がない）
        self.chk_ref_te_only = QCheckBox(
            "TE のみ参照（VAE を通さない・同一性より内容重視）")
        self.chk_ref_te_only.setToolTip(
            "参照素材をテキストエンコーダ（Qwen3-VL）にだけ渡し、VAE の\n"
            "参照トークンを乗せません。人物などの見た目の一致は弱くなり、\n"
            "内容や雰囲気の理解だけを参照に使う形になります。\n"
            "速度差は小さく、参照解像度 max や参照動画が多いときだけ効きます")
        grid.addWidget(self.chk_ref_te_only, 4, 0, 1, 2)
        return page

    def _build_settings_box(self) -> QGroupBox:
        box = QGroupBox("設定")
        grid = QGridLayout(box)

        self.cb_aspect = WideComboBox()
        for name, aw, ah in ASPECT_PRESETS:
            self.cb_aspect.addItem(name, (aw, ah))
        self.cb_aspect.setCurrentIndex(1)  # 16:9
        self.cb_aspect.setToolTip(
            "出力動画のアスペクト比。\n"
            "i2v では入力画像の縦横比が使われるため無効になります")
        self.cb_aspect.currentIndexChanged.connect(self._update_size_label)
        # 解像度 = 目標メガピクセル。0.1刻みのスピンボックス。
        self.sp_quality = QDoubleSpinBox()
        self.sp_quality.setRange(0.1, 3.0)
        self.sp_quality.setSingleStep(0.1)
        self.sp_quality.setDecimals(1)
        self.sp_quality.setValue(1.0)
        self.sp_quality.setSuffix(" MP")
        self.sp_quality.setToolTip(
            "目標画素数（メガピクセル）。1.0 ≒ 768p級（公式標準）、"
            "0.4 = 軽量・高速。\n"
            "1.1以上は生成時間がかかるうえ品質が壊れる可能性があります")
        self._last_quality_val = 1.0
        self.sp_quality.valueChanged.connect(self._on_quality_spin_changed)
        self.chk_size_manual = QCheckBox("手動")
        self.chk_size_manual.setToolTip(
            "出力解像度を手動で指定します（32の倍数へ丸められます）。\n"
            "ONの間は解像度・アスペクト比は使われません")
        self.chk_size_manual.toggled.connect(self._on_size_manual_toggled)

        self.lbl_size = QLabel("")
        self.sp_length = QDoubleSpinBox()
        self.sp_length.setRange(0.3, 15.0)
        self.sp_length.setSingleStep(0.5)
        self.sp_length.setDecimals(1)
        self.sp_length.setValue(5.0)
        self.sp_length.setSuffix(" 秒")
        self.sp_length.setToolTip(
            "動画の長さ。24fps の 17n+5 フレームグリッドへスナップされます"
            "（学習の中心は約5秒）")
        self.sp_length.valueChanged.connect(self._update_size_label)

        self.sp_steps = QSpinBox()
        self.sp_steps.setRange(1, 100)
        self.sp_steps.setValue(20)
        self.cb_sampler = WideComboBox(); self.cb_sampler.addItems(SAMPLERS)
        self.cb_scheduler = WideComboBox(); self.cb_scheduler.addItems(SCHEDULERS)

        self.ed_seed = QLineEdit("-1")
        self.ed_seed.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"-1|\d{1,19}")))
        self.ed_seed.setToolTip(
            "-1 = 生成ごとにランダム（欄は書き換わりません。使われた値は"
            "ログで確認できます）")
        self.cb_dtype = WideComboBox()
        self.cb_dtype.addItems(["default", "fp8_e4m3fn", "fp8_e5m2"])

        r = 0
        self.lbl_aspect = QLabel("アスペクト比")
        grid.addWidget(self.lbl_aspect, r, 0)
        grid.addWidget(self.cb_aspect, r, 1)
        grid.addWidget(QLabel("解像度"), r, 2)
        grid.addWidget(self.sp_quality, r, 3)
        r += 1
        grid.addWidget(QLabel("出力"), r, 0)
        size_row = QWidget()
        size_row.setToolTip(
            "出力解像度。「手動」をONにすると編集できます"
            "（32の倍数へ丸められます）。\n"
            "OFFではアスペクト比と解像度から自動計算した値が入ります")
        sr = QHBoxLayout(size_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.addWidget(self.chk_size_manual)
        self.sp_out_w = QSpinBox()
        self.sp_out_h = QSpinBox()
        for sp in (self.sp_out_w, self.sp_out_h):
            sp.setRange(32, 4096)
            sp.setSingleStep(32)
            sp.setEnabled(False)   # 手動OFF中はグレーアウト（自動値を表示）
        self.sp_out_w.setValue(1344)
        self.sp_out_h.setValue(768)
        sr.addWidget(self.sp_out_w)
        sr.addWidget(QLabel("×"))
        sr.addWidget(self.sp_out_h)
        sr.addStretch(1)
        grid.addWidget(size_row, r, 1, 1, 3)
        r += 1
        grid.addWidget(QLabel("長さ"), r, 0)
        grid.addWidget(self.sp_length, r, 1)
        grid.addWidget(QLabel("フレーム"), r, 2)
        grid.addWidget(self.lbl_size, r, 3)
        r += 1
        grid.addWidget(QLabel("Steps"), r, 0)
        grid.addWidget(self.sp_steps, r, 1)
        grid.addWidget(QLabel("Sampler"), r, 2)
        grid.addWidget(self.cb_sampler, r, 3)
        r += 1
        grid.addWidget(QLabel("Scheduler"), r, 0)
        grid.addWidget(self.cb_scheduler, r, 1)
        grid.addWidget(QLabel("Seed"), r, 2)
        grid.addWidget(self.ed_seed, r, 3)
        r += 1
        grid.addWidget(QLabel("UNet dtype"), r, 0)
        grid.addWidget(self.cb_dtype, r, 1)

        # Sigma shift（OFF ならモデル既定値のまま）
        self.grp_shift = QGroupBox("Sigma Shift 調整")
        self.grp_shift.setCheckable(True)
        self.grp_shift.setChecked(False)
        self.grp_shift.setToolTip(
            "動画/音声の flow shift を上書きします（OFF = モデル既定）")
        sh = QHBoxLayout(self.grp_shift)
        self.sp_shift_video = QDoubleSpinBox()
        self.sp_shift_video.setRange(0.01, 100.0)
        self.sp_shift_video.setValue(12.0)
        self.sp_shift_audio = QDoubleSpinBox()
        self.sp_shift_audio.setRange(0.01, 100.0)
        self.sp_shift_audio.setValue(3.0)
        sh.addWidget(QLabel("video"))
        sh.addWidget(self.sp_shift_video)
        sh.addWidget(QLabel("audio"))
        sh.addWidget(self.sp_shift_audio)
        sh.addStretch(1)
        # 全4列にまたがる独立行にして、設定ボックスの横幅を抑える。
        r += 1
        grid.addWidget(self.grp_shift, r, 0, 1, 4)

        self._update_size_label()
        return box

    def _build_speed_box(self) -> QGroupBox:
        """高速化設定: SageAttention と EasyCache をまとめたカテゴリ。"""
        box = QGroupBox("高速化設定")
        v = QVBoxLayout(box)

        # SageAttention（バックエンド起動フラグ。切替は再起動後に反映）
        self.chk_sage = QCheckBox("SageAttention（出力が僅かに変化）")
        self.chk_sage.setToolTip(
            "量子化 attention による推論高速化。\n"
            "未導入の場合は ON にしたときにダウンロードの確認を出します。\n"
            "切替の反映にはアプリの再起動が必要です。")
        v.addWidget(self.chk_sage)
        self._init_sage_checkbox()

        # EasyCache（ステップスキップによる高速化）— 1行構成
        ec = QHBoxLayout()
        self.chk_easycache = QCheckBox("EasyCache")
        self.chk_easycache.setToolTip(
            "変化の小さいサンプリングステップをスキップして高速化します。\n"
            "閾値を上げるほど速くなりますが品質が低下します（既定 0.2）。\n"
            "スキップ数はログの \"skipped N/M steps\" で確認できます")
        ec.addWidget(self.chk_easycache)
        ec.addWidget(QLabel("閾値"))
        self.sp_easycache = QDoubleSpinBox()
        self.sp_easycache.setRange(0.0, 3.0)
        self.sp_easycache.setSingleStep(0.05)
        self.sp_easycache.setDecimals(2)
        self.sp_easycache.setValue(0.2)
        self.sp_easycache.setEnabled(False)
        self.chk_easycache.toggled.connect(self.sp_easycache.setEnabled)
        ec.addWidget(self.sp_easycache)
        ec.addStretch(1)
        v.addLayout(ec)

        # Turbo (PDD) LoRA — 少ステップ蒸留。ON の間は Steps 欄の代わりに
        # ここのステップ数を使う。1行構成。
        tr = QHBoxLayout()
        self.chk_turbo = QCheckBox("Turbo LoRA")
        self.chk_turbo.setToolTip(
            "公式の蒸留 LoRA（各 1.96GB）で 4〜8 ステップ生成にします。\n"
            "ON の間は設定の Steps ではなくこの行のステップ数を使います。\n"
            "t2v/i2v は fl2v 用（8step: 544p 学習・推奨 8 ステップ / "
            "4step 768p: 768p 学習・shift 6/3 を自動適用）、\n"
            "r2v は ref2v 用 4step 版が自動で選ばれます。\n"
            "EasyCache との併用は品質が崩れやすいので避けてください。\n"
            "未ダウンロードなら ON にしたときに確認を出します。")
        tr.addWidget(self.chk_turbo)
        # 版の選択肢はモード（fl2v / ref2v）に応じて _refill_turbo_variants
        # が入れ替える。ref2v は 1 種類だが、何が使われるか分かるよう表示する。
        self.cb_turbo_variant = WideComboBox()
        self.cb_turbo_variant.setToolTip(
            "使う Turbo LoRA の版。\n"
            "t2v/i2v: fl2v 8step（汎用・4〜8 ステップ）/ fl2v 4step 768p"
            "（768p 向け 4 ステップ特化）\n"
            "r2v: ref2v 4step（現在この 1 種類のみ）")
        self.cb_turbo_variant.setEnabled(False)
        tr.addWidget(self.cb_turbo_variant)
        tr.addWidget(QLabel("Steps"))
        self.sp_turbo_steps = QSpinBox()
        self.sp_turbo_steps.setRange(1, 12)
        self.sp_turbo_steps.setValue(4)
        self.sp_turbo_steps.setEnabled(False)
        self.sp_turbo_steps.setToolTip(
            "Turbo LoRA 使用時のステップ数（版ごとに記憶）。\n"
            "既定: fl2v 8step = 8（公式テンプレは 6）、fl2v 4step 768p = 4、"
            "ref2v 4step = 4")
        self.sp_turbo_steps.valueChanged.connect(self._on_turbo_steps_changed)
        tr.addWidget(self.sp_turbo_steps)
        tr.addStretch(1)
        v.addLayout(tr)
        # 版の選択肢とステップ数はコンボ/スピン両方が出来てから入れる。
        self._refill_turbo_variants()
        self.chk_turbo.toggled.connect(self._on_turbo_toggled)
        self.cb_turbo_variant.currentIndexChanged.connect(
            self._on_turbo_variant_changed)

        # Block-sparse attention（v0.35〜）— 1行構成
        sa = QHBoxLayout()
        self.chk_sparse = QCheckBox("Sparse Attention")
        self.chk_sparse.setToolTip(
            "attention をブロック単位で間引いて高速化します（実験的機能）。\n"
            "長尺・高解像度ほど効果が大きく、短い動画では速くなりません。\n"
            "sol-attn: 学習不要（既定）。sla / vsa: 専用に学習された重み向け")
        sa.addWidget(self.chk_sparse)
        sa.addWidget(QLabel("方式"))
        self.cb_sparse_method = WideComboBox()
        for m in SPARSE_METHODS:
            self.cb_sparse_method.addItem(m, m)
        self.cb_sparse_method.setEnabled(False)
        self.chk_sparse.toggled.connect(self.cb_sparse_method.setEnabled)
        sa.addWidget(self.cb_sparse_method)
        sa.addStretch(1)
        v.addLayout(sa)

        # FastH3（蒸留+VSA 学習済み checkpoint）選択時の案内。学習条件に
        # 合わせた設定を生成時に自動適用するため、上の行は操作不可にする。
        self.lbl_fasth3 = QLabel(
            "FastH3 選択中: 8 ステップ / Sigma Shift 10・3 / "
            "Sparse Attention vsa 保持 20% を自動適用します"
            "（Turbo LoRA・EasyCache は使われません）")
        self.lbl_fasth3.setWordWrap(True)
        self.lbl_fasth3.setStyleSheet("color:#39c;")
        self.lbl_fasth3.setVisible(False)
        v.addWidget(self.lbl_fasth3)
        return box

    # ----- FastH3 ------------------------------------------------------------
    def _fasth3_selected(self) -> bool:
        return models_mod.is_fasth3(self.cb_diffusion.currentText())

    def _sync_fasth3_controls(self) -> None:
        """FastH3 選択中は Turbo / EasyCache / Sparse / Steps を固定表示にする。"""
        if not hasattr(self, "lbl_fasth3"):
            return
        on = self._fasth3_selected() and self._mode() == "t2v"
        self.lbl_fasth3.setVisible(on)
        for w in (self.chk_turbo, self.chk_easycache, self.chk_sparse):
            w.setEnabled(not on)
        self.cb_turbo_variant.setEnabled(
            not on and self.chk_turbo.isChecked())
        self.sp_turbo_steps.setEnabled(not on and self.chk_turbo.isChecked())
        self.sp_easycache.setEnabled(not on and self.chk_easycache.isChecked())
        self.cb_sparse_method.setEnabled(
            not on and self.chk_sparse.isChecked())
        self.sp_steps.setEnabled(not on and not self.chk_turbo.isChecked())

    # ----- Turbo LoRA ------------------------------------------------------
    def _turbo_ckpt_kind(self) -> str:
        """現在のモードで使うチェックポイント系統（fl2va / ref2va）。"""
        mode = self._mode()
        if mode == "r2v":
            return "ref2va"
        if mode == "chain":
            plan = self._current_chain_plan() or self._chain_plan or {}
            if plan.get("chain_type") == "r2v":
                return "ref2va"
        return "fl2va"

    def _turbo_lora_name(self) -> str:
        """現在のモード/版に対応する Turbo LoRA のファイル名。"""
        kind = self._turbo_ckpt_kind()
        variant = (self._turbo_fl2v_variant if kind == "fl2va" else "4step")
        return models_mod.turbo_lora_for(kind, variant)

    _TURBO_VARIANTS = {
        "fl2va": [("fl2v 8step", "8step"), ("fl2v 4step 768p", "4step_768p")],
        "ref2va": [("ref2v 4step", "4step")],
    }

    def _refill_turbo_variants(self) -> None:
        """モードに応じた版の選択肢に入れ替える（fl2v の選択は記憶）。"""
        kind = self._turbo_ckpt_kind()
        items = self._TURBO_VARIANTS[kind]
        cb = self.cb_turbo_variant
        cb.blockSignals(True)
        cb.clear()
        for label, data in items:
            cb.addItem(label, data)
        if kind == "fl2va":
            i = cb.findData(self._turbo_fl2v_variant)
            cb.setCurrentIndex(i if i >= 0 else 0)
        cb.blockSignals(False)
        self._load_turbo_steps()

    def _current_turbo_variant(self) -> str:
        """現在表示中の版キー（8step / 4step_768p / 4step）。"""
        return str(self.cb_turbo_variant.currentData() or "8step")

    def _load_turbo_steps(self) -> None:
        """表示中の版に記憶したステップ数をスピンボックスへ出す。"""
        v = self._current_turbo_variant()
        steps = int(self._turbo_steps.get(
            v, models_mod.TURBO_DEFAULT_STEPS.get(v, 4)))
        self.sp_turbo_steps.blockSignals(True)
        self.sp_turbo_steps.setValue(steps)
        self.sp_turbo_steps.blockSignals(False)

    def _on_turbo_steps_changed(self, val: int) -> None:
        self._turbo_steps[self._current_turbo_variant()] = int(val)
        self._schedule_save()

    def _turbo_spec(self) -> dict:
        """現在の Turbo LoRA の学習条件（steps / shift）。"""
        return models_mod.TURBO_LORA_SPECS.get(self._turbo_lora_name(), {})

    def _turbo_lora_present(self, name: str) -> bool:
        p = self.paths.models / "loras" / name
        if not p.is_file():
            return False
        m = next((m for m in models_mod.load_manifest(self.paths)
                  if m.filename == name), None)
        return not (m and m.size) or p.stat().st_size == m.size

    def _sync_turbo_controls(self) -> None:
        on = self.chk_turbo.isChecked()
        self.sp_steps.setEnabled(not on)
        self.sp_steps.setToolTip(
            "Turbo LoRA が ON のため、高速化設定のステップ数が使われます"
            if on else "")
        self.sp_turbo_steps.setEnabled(on)
        self._refill_turbo_variants()
        self.cb_turbo_variant.setEnabled(on)
        # FastH3 選択中は上書きで固定表示に戻す。
        self._sync_fasth3_controls()

    def _on_turbo_toggled(self, checked: bool) -> None:
        self._sync_turbo_controls()
        if self._loading:
            return
        self._schedule_save()
        if checked:
            self._ensure_turbo_lora()

    def _on_turbo_variant_changed(self, *_a) -> None:
        if self._turbo_ckpt_kind() == "fl2va":
            self._turbo_fl2v_variant = (
                self.cb_turbo_variant.currentData() or "8step")
        self._load_turbo_steps()
        if self._loading:
            return
        self._schedule_save()
        if self.chk_turbo.isChecked():
            self._ensure_turbo_lora()

    def _ensure_turbo_lora(self) -> bool:
        """必要な Turbo LoRA が無ければダウンロードを確認して開始する。
        既にある（または開始済み）なら True。"""
        name = self._turbo_lora_name()
        if self._turbo_lora_present(name):
            return True
        if self._turbo_job is not None:
            return False
        m = next((m for m in models_mod.load_manifest(self.paths)
                  if m.filename == name), None)
        if m is None:
            QMessageBox.warning(
                self, "Turbo LoRA",
                f"models.json に {name} の定義がありません。")
            return False
        from .model_selector import fmt_size
        ret = QMessageBox.question(
            self, "Turbo LoRA",
            f"Turbo LoRA が未ダウンロードです。\n{name}"
            f"（{fmt_size(m.size)}）\nダウンロードしますか？",
            QMessageBox.Yes | QMessageBox.Cancel)
        if ret != QMessageBox.Yes:
            return False
        self._start_turbo_download(m)
        return False

    def _start_turbo_download(self, m: models_mod.ModelFile) -> None:
        self.append_log(f"Turbo LoRA をダウンロードしています: {m.filename}")
        thread = QThread()
        worker = _ModelDownloadWorker(self.paths, m)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_turbo_progress)
        worker.done.connect(self._on_turbo_downloaded)
        worker.failed.connect(self._on_turbo_download_failed)
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
        self._turbo_job = (thread, worker)
        self._turbo_last_pct = -1
        thread.finished.connect(self._on_turbo_thread_finished)
        thread.start()

    def _on_turbo_thread_finished(self) -> None:
        self._turbo_job = None

    def _on_turbo_progress(self, done: float, total: float) -> None:
        if total <= 0:
            return
        pct = int(done * 100 / total)
        if pct // 10 != self._turbo_last_pct // 10:
            self._turbo_last_pct = pct
            self.append_log(f"Turbo LoRA ダウンロード {pct}%")

    def _on_turbo_downloaded(self, name: str) -> None:
        self.append_log(f"Turbo LoRA のダウンロードが完了しました: {name}")

    def _on_turbo_download_failed(self, name: str, msg: str) -> None:
        self.append_log(f"Turbo LoRA のダウンロードに失敗: {name}: {msg}")
        QMessageBox.warning(
            self, "Turbo LoRA", f"ダウンロードに失敗しました:\n{msg}")

    # ----- ガイド（AddGuide）------------------------------------------------
    _MAX_GUIDES = 8

    def _build_guides_box(self) -> QGroupBox:
        """画像/音声を任意フレームに固定するガイド行の一覧。"""
        box = self.box_guides = QGroupBox("ガイド（フレーム固定）")
        box.setToolTip(
            "画像や音声を動画の指定フレームに固定します（MiniMaxH3AddGuide）。\n"
            "秒: 0 = 先頭、負の値は末尾から数えます（-0.1 ≒ 最後の数フレーム）。\n"
            "画像は 32 の倍数の出力解像度へ中央クロップで合わせられます。\n"
            "長尺チェーンでは使われません")
        v = QVBoxLayout(box)
        head = QHBoxLayout()
        self.lbl_guides = QLabel("")
        self.lbl_guides.setStyleSheet("color:#888;")
        head.addWidget(self.lbl_guides, stretch=1)
        btn_img = QPushButton("+ 画像")
        btn_img.setToolTip("指定フレームに固定する画像を追加")
        btn_img.clicked.connect(lambda: self._add_guide("image"))
        btn_aud = QPushButton("+ 音声")
        btn_aud.setToolTip("指定フレームから流す音声を追加")
        btn_aud.clicked.connect(lambda: self._add_guide("audio"))
        head.addWidget(btn_img)
        head.addWidget(btn_aud)
        v.addLayout(head)
        self._guides_layout = QVBoxLayout()
        self._guides_layout.setContentsMargins(0, 0, 0, 0)
        v.addLayout(self._guides_layout)
        self._guides: list[dict] = []
        self._update_guides_label()
        return box

    def _update_guides_label(self) -> None:
        n = len(self._guides)
        self.lbl_guides.setText(
            f"{n}/{self._MAX_GUIDES} 件" if n else
            "画像/音声を指定フレームに固定できます（任意）")

    def _add_guide(self, kind: str) -> None:
        if len(self._guides) >= self._MAX_GUIDES:
            QMessageBox.information(
                self, "上限", f"ガイドは最大 {self._MAX_GUIDES} 件です。")
            return
        flt = _IMAGE_FILTER if kind == "image" else _AUDIO_FILTER
        path, _ = QFileDialog.getOpenFileName(
            self, "ガイドに使う" + ("画像" if kind == "image" else "音声"),
            "", flt)
        if not path:
            return
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        lbl_kind = QLabel("画像" if kind == "image" else "音声")
        lbl_kind.setStyleSheet("color:#888;")
        lbl_name = QLabel(Path(path).name)
        lbl_name.setToolTip(path)
        sp_sec = QDoubleSpinBox()
        sp_sec.setRange(-60.0, 60.0)
        sp_sec.setSingleStep(0.5)
        sp_sec.setDecimals(2)
        sp_sec.setValue(0.0)
        sp_sec.setSuffix(" 秒")
        sp_sec.setToolTip(
            "固定するフレームの時刻。24fps でフレーム番号へ丸めます。\n"
            "負の値は末尾から数えます")
        btn_del = QPushButton("🗑")
        btn_del.setFixedWidth(28)
        btn_del.setToolTip("このガイドを削除")
        h.addWidget(lbl_kind)
        h.addWidget(lbl_name, stretch=1)
        h.addWidget(sp_sec)
        h.addWidget(btn_del)
        entry = {"kind": kind, "path": path, "row": row, "spin": sp_sec}
        btn_del.clicked.connect(lambda: self._remove_guide(entry))
        self._guides.append(entry)
        self._guides_layout.addWidget(row)
        self._update_guides_label()

    def _remove_guide(self, entry: dict) -> None:
        if entry not in self._guides:
            return
        self._guides.remove(entry)
        row = entry["row"]
        self._guides_layout.removeWidget(row)
        row.deleteLater()
        self._update_guides_label()

    def _guide_params(self) -> list[dict]:
        """ガイド行を GenParams.guides 形式へ（ファイルはアップロード）。"""
        out = []
        for e in self._guides:
            sec = float(e["spin"].value())
            frame_idx = int(round(sec * FPS))
            if sec < 0 and frame_idx == 0:
                frame_idx = -1
            out.append({"kind": e["kind"],
                        "name": self._upload(e["path"]),
                        "frame_idx": frame_idx})
        return out

    def _init_sage_checkbox(self) -> None:
        """環境の対応可否を判定して初期状態を決める（対応外なら無効化）。"""
        gpu = environment.detect_gpu()
        torch_tag = str(_Manifest(self.paths.manifest_path)
                        .get("torch_tag") or "")
        ok, reason = environment.sage_supported(gpu, torch_tag)
        if not ok:
            self.chk_sage.setEnabled(False)
            self.chk_sage.setToolTip(reason)
            return
        # 接続はここで行い、設定復元中の発火はハンドラ側で _loading を見る。
        self.chk_sage.toggled.connect(self._on_sage_toggled)

    def _on_sage_toggled(self, checked: bool) -> None:
        if self._loading:
            return
        if checked and not sage_installed(self.paths):
            ret = QMessageBox.question(
                self, "SageAttention",
                "SageAttention の必要コンポーネントが未インストールです。\n"
                "ダウンロードしてインストールしますか？",
                QMessageBox.Yes | QMessageBox.Cancel)
            if ret != QMessageBox.Yes:
                self.chk_sage.blockSignals(True)
                self.chk_sage.setChecked(False)
                self.chk_sage.blockSignals(False)
                return
            self.settings["sage_attention"] = True
            self._schedule_save()
            self._start_sage_install()
            return
        self.settings["sage_attention"] = bool(checked)
        self._schedule_save()
        QMessageBox.information(
            self, "SageAttention",
            "SageAttention を{}にしました。\n"
            "反映にはアプリの再起動が必要です。".format(
                "有効" if checked else "無効"))

    def _start_sage_install(self) -> None:
        self.chk_sage.setEnabled(False)
        self.append_log("SageAttention をインストールしています…")
        thread = QThread()
        worker = _SageInstallWorker(self.paths)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self.append_log)
        worker.done.connect(self._on_sage_installed)
        worker.failed.connect(self._on_sage_install_failed)
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
        _SAGE_JOBS.append((thread, worker))
        thread.finished.connect(
            lambda t=thread, w=worker: _SAGE_JOBS.remove((t, w)))
        thread.start()

    def _on_sage_installed(self) -> None:
        self.chk_sage.setEnabled(True)
        self.append_log("SageAttention のインストールが完了しました")
        QMessageBox.information(
            self, "SageAttention",
            "インストールが完了しました。\n"
            "反映にはアプリの再起動が必要です。")

    def _on_sage_install_failed(self, msg: str) -> None:
        self.chk_sage.setEnabled(True)
        self.chk_sage.blockSignals(True)
        self.chk_sage.setChecked(False)
        self.chk_sage.blockSignals(False)
        self.settings["sage_attention"] = False
        self._schedule_save()
        self.append_log("SageAttention のインストールに失敗: " + msg)
        QMessageBox.warning(
            self, "SageAttention", f"インストールに失敗しました:\n{msg}")

    # ----- helpers ---------------------------------------------------------
    def _mode(self) -> str:
        return self.cb_mode.currentData() or "t2v"

    def _pick_file(self, ed: QLineEdit, flt: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "ファイルを選択", "", flt)
        if path:
            ed.setText(path)

    def _clear_frame(self, ed: QLineEdit) -> None:
        ed.clear()

    def _on_mode_changed(self, *_a) -> None:
        mode = self._mode()
        self.stack_mode.setCurrentIndex(
            {"t2v": 0, "i2v": 1, "r2v": 2, "chain": 3}[mode])
        # チェーンではシーンごとのプロンプトを専用ウィンドウで編集するので、
        # メイン側のプロンプト欄は隠す。
        self.box_prompt.setVisible(mode != "chain")
        # ガイドはチェーン（シーン単位の条件付け）では使わない。
        if hasattr(self, "box_guides"):
            self.box_guides.setVisible(mode != "chain")
        self._update_chain_summary()
        # Turbo LoRA はチェックポイント系統（fl2v/ref2v）で別ファイル。
        if hasattr(self, "chk_turbo"):
            self._sync_turbo_controls()
            if not self._loading and self.chk_turbo.isChecked():
                self._ensure_turbo_lora()
        self.lbl_diffusion.setText(
            "Diffusion (ref2va):" if mode == "r2v" else "Diffusion (fl2va):")
        # アスペクト比の有効/無効（i2v・手動で無効）とサイズ再計算。
        self._update_size_controls()
        self._refill_diffusion()
        self._update_size_label()
        # 適用 LoRA はチェックポイント別（fl2va / ref2va）に記憶している。
        if hasattr(self, "_lora_flow"):
            self._rebuild_lora_rows()
            self._push_lora_state()
        self._schedule_save()

    def _auto_size(self) -> tuple[int, int]:
        """アスペクト比/解像度（i2v は入力画像）から出力サイズを計算する。"""
        mp = float(self.sp_quality.value())
        if self._mode() == "i2v" and self.ed_first_frame.text():
            img = QImage(self.ed_first_frame.text())
            if not img.isNull():
                return size_for_image(img.width(), img.height(), mp)
        aw, ah = self.cb_aspect.currentData() or (16, 9)
        return size_for_aspect(aw, ah, mp)

    def _size_is_manual(self) -> bool:
        return self.chk_size_manual.isChecked()

    def _update_size_label(self, *_a) -> None:
        frames = frames_for_seconds(self.sp_length.value())
        secs = frames / workflow.FPS
        self.lbl_size.setText(f"{frames}f ≈ {secs:.1f}s")
        # 手動指定でないときは算出値をスピンボックスへ反映する
        # （保存は発火させない）。
        if not self._size_is_manual():
            w, h = self._auto_size()
            for sp, v in ((self.sp_out_w, w), (self.sp_out_h, h)):
                sp.blockSignals(True)
                sp.setValue(int(v))
                sp.blockSignals(False)

    def _update_size_controls(self) -> None:
        """手動チェック・モードに応じて解像度まわりの有効/無効を切り替える。"""
        manual = self._size_is_manual()
        self.sp_quality.setEnabled(not manual)
        self.sp_out_w.setEnabled(manual)
        self.sp_out_h.setEnabled(manual)
        # アスペクト比は「i2v（画像基準）」か「手動」では使われない。
        en = (self._mode() != "i2v") and not manual
        self.cb_aspect.setEnabled(en)
        self.lbl_aspect.setEnabled(en)
        self._update_size_label()

    def _on_size_manual_toggled(self, *_a) -> None:
        self._update_size_controls()
        self._schedule_save()

    def _on_quality_spin_changed(self, val: float) -> None:
        # 1.0（公式標準）を超えた瞬間に一度だけ警告する。
        if (not self._loading and val > 1.0
                and self._last_quality_val <= 1.0):
            self.append_log(
                "\x1b[93m警告: 解像度 1.1 以上は時間がかかる上に品質が"
                "壊れる可能性があるためお勧めしません\x1b[0m")
        self._last_quality_val = val
        self._update_size_label()
        self._schedule_save()

    # ----- model scan ------------------------------------------------------
    def refresh_models(self) -> None:
        self._all_models = {
            "diffusion_models": config.scan_models("diffusion_models"),
            "vae": config.scan_models("vae"),
            "text_encoders": config.scan_models("text_encoders"),
        }
        self.append_log(
            "モデルスキャン: "
            f"diffusion={len(self._all_models['diffusion_models'])} "
            f"vae={len(self._all_models['vae'])} "
            f"te={len(self._all_models['text_encoders'])}")
        self._refill_diffusion()
        self._fill_combo(self.cb_te, self._all_models["text_encoders"])
        vaes = self._all_models["vae"]
        self._fill_combo(self.cb_vae_video, vaes)
        self._fill_combo(self.cb_vae_audio, vaes)
        self._auto_pick(self.cb_vae_video, "video")
        self._auto_pick(self.cb_vae_audio, "audio")

    def _refill_diffusion(self) -> None:
        files = self._all_models.get("diffusion_models", [])
        mode = self._mode()
        if mode == "r2v":
            wanted = [f for f in files if "ref2va" in f.lower()]
        else:
            wanted = [f for f in files if "ref2va" not in f.lower()]
        # FastH3 は t2av 専用の蒸留なので t2v 以外の一覧からは外す。
        if mode != "t2v":
            wanted = [f for f in wanted if not models_mod.is_fasth3(f)]
        self._fill_combo(self.cb_diffusion, wanted or files)
        self._sync_fasth3_controls()

    @staticmethod
    def _fill_combo(combo: QComboBox, items: list[str]) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    @staticmethod
    def _auto_pick(combo: QComboBox, substr: str) -> None:
        """現在の選択が substr を含まないとき、含む項目へ自動で合わせる。"""
        if substr in combo.currentText().lower():
            return
        for i in range(combo.count()):
            if substr in combo.itemText(i).lower():
                combo.setCurrentIndex(i)
                return

    # ----- 設定ウィンドウ --------------------------------------------------
    def open_models_dialog(self) -> None:
        from .models_dialog import ModelsDialog
        dlg = ModelsDialog(self.paths, parent=self)
        dlg.exec()
        self.refresh_models()

    # ----- ContexLoop（長尺チェーン）---------------------------------------
    def _ensure_contex_loop(self) -> bool:
        """チェーン用のカスタムノードパックを確認し、無ければ確認して導入。"""
        from ..comfy_custom_nodes import (
            CONTEX_LOOP_REPO, CONTEX_LOOP_VERSION, contex_loop_installed,
            install_contex_loop,
        )
        if contex_loop_installed(self.paths):
            return True
        ret = QMessageBox.question(
            self, "ContexLoop",
            "長尺チェーンには第三者製の追加コンポーネントが必要です。\n"
            f"{CONTEX_LOOP_REPO} ({CONTEX_LOOP_VERSION}) を"
            "ダウンロードしますか？\n"
            "（導入後、反映にはアプリの再起動が必要です）",
            QMessageBox.Yes | QMessageBox.Cancel)
        if ret != QMessageBox.Yes:
            return False
        try:
            install_contex_loop(self.paths, self.append_log)
        except Exception as e:  # noqa: BLE001
            self.append_log(f"ContexLoop の導入に失敗: {e}")
            QMessageBox.warning(self, "ContexLoop",
                                f"導入に失敗しました:\n{e}")
            return False
        QMessageBox.information(
            self, "ContexLoop",
            "追加コンポーネントを導入しました。\n"
            "反映にはアプリの再起動が必要です。")
        return True

    def open_chain_dialog(self) -> None:
        """チェーン設定ウィンドウを開く（非モーダル・1個だけ）。"""
        if not self._ensure_contex_loop():
            return
        if self._chain_dlg is None:
            from .chain_dialog import ChainDialog
            dlg = ChainDialog(parent=self)
            if self._chain_plan:
                dlg.load_plan(self._chain_plan)
            dlg.changed.connect(self._update_chain_summary)
            # 閉じても設定は保持する（次に開くと続きから編集できる）。
            dlg.finished.connect(self._on_chain_dialog_closed)
            self._chain_dlg = dlg
        self._chain_dlg.show()
        self._chain_dlg.raise_()
        self._chain_dlg.activateWindow()
        self._update_chain_summary()

    def _on_chain_dialog_closed(self, *_a) -> None:
        if self._chain_dlg is not None:
            self._chain_plan = self._chain_dlg.plan()
        self._update_chain_summary()

    def _current_chain_plan(self) -> Optional[dict]:
        """生成時に使う設定。ウィンドウが開いていればその場の内容を使う。"""
        if self._chain_dlg is not None:
            return self._chain_dlg.plan()
        return self._chain_plan

    def _update_chain_summary(self) -> None:
        if self._mode() != "chain" or not hasattr(self, "lbl_chain"):
            return
        plan = self._current_chain_plan()
        if not plan:
            self.lbl_chain.setText("未設定（ボタンから設定してください）")
            self.lbl_chain.setStyleSheet("color:#c33;")
            return
        _raw, delivered = workflow.chain_frames(plan)
        kind = {"i2v": "画像から開始", "r2v": "参照から生成",
                "t2v": "プロンプトのみ"}.get(plan.get("chain_type"), "")
        try:
            workflow.validate_chain(plan)
            warn = ""
            self.lbl_chain.setStyleSheet("color:#888;")
        except ValueError as e:
            warn = f"\n⚠ {e}"
            self.lbl_chain.setStyleSheet("color:#c33;")
        self.lbl_chain.setText(
            f"{plan.get('run_name')} / {kind} / "
            f"{len(plan.get('shots', []))} シーン / "
            f"実尺 {delivered / workflow.FPS:.1f} 秒" + warn)

    def _upload_chain_files(self, plan: dict) -> dict:
        """チェーン設定内のローカルパスをアップロード済み名へ置き換える。"""
        out = dict(plan)
        out["shots"] = [dict(s) for s in plan.get("shots", [])]
        out["references"] = [dict(r) for r in plan.get("references", [])]
        if out["shots"] and out["shots"][0].get("first_frame"):
            out["shots"][0]["first_frame"] = self._upload(
                out["shots"][0]["first_frame"])
        for ref in out["references"]:
            if ref.get("path"):
                ref["name"] = self._upload(ref["path"])
        if out.get("audio_file"):
            out["audio_file"] = self._upload(out["audio_file"])
        return out

    # ----- LoRA ------------------------------------------------------------
    def _lora_family(self) -> str:
        """適用リストのキー。t2v/i2v は fl2va を共有、r2v は ref2va。"""
        return "ref2va" if self._mode() == "r2v" else "fl2va"

    def _current_loras(self) -> list[dict]:
        return self._loras_by_family[self._lora_family()]

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
                         self.paths.user_data / "lora_cache", None)
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
                {e["name"]: float(e["strength"])
                 for e in self._current_loras()})
        except RuntimeError:
            self._lora_dlg = None

    def _on_lora_apply(self, name: str, strength: float) -> None:
        loras = self._current_loras()
        for e in loras:
            if e["name"] == name:
                e["strength"] = float(strength)
                break
        else:
            loras.append({"name": name, "strength": float(strength)})
            self.append_log(
                f"LoRA を適用 [{self._lora_family()}]: {name} ×{strength:g}")
        self._rebuild_lora_rows()
        self._push_lora_state()

    def _on_lora_remove(self, name: str) -> None:
        loras = self._current_loras()
        before = len(loras)
        loras[:] = [e for e in loras if e["name"] != name]
        if len(loras) != before:
            self.append_log(f"LoRA を解除: {name}")
        self._rebuild_lora_rows()
        self._push_lora_state()

    def _on_lora_strength_changed(self, name: str, value: float) -> None:
        for e in self._current_loras():
            if e["name"] == name:
                e["strength"] = float(value)
        self._push_lora_state()

    def _rebuild_lora_rows(self) -> None:
        """Rebuild the applied-LoRA chips after the LoRA button (flow layout;
        index 0 is the button itself)."""
        self._hide_lora_popup(force=True)
        while self._lora_flow.count() > 1:
            item = self._lora_flow.takeAt(1)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for e in self._current_loras():
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
            trash = QPushButton("\U0001f5d1")
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
            spin.setToolTip("LoRA の適用強度（model / TE 共通）")
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

    # ----- LoRA trigger-word insertion (colored tokens) --------------------
    def _on_lora_toggle_prompt(self, token: str, text: str) -> None:
        """LoRA のトリガーワードをトグルする。未挿入なら挿入（区別マーク付き）、
        挿入済みなら（ユーザー編集後でも）その区間ごと削除する。"""
        field = self.txt_prompt
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
        cursor.setPosition(len(stripped))
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        if stripped:
            cursor.insertText(", ", self._plain_char_format())
        cursor.insertText(joined, self._token_char_format(token))
        cursor.insertText(" ", self._plain_char_format())
        field.setTextCursor(cursor)
        field.setCurrentCharFormat(self._plain_char_format())

    def _active_lora_tokens(self) -> set[str]:
        """プロンプト欄に現在挿入されている LoRA トリガーワードの token 集合。"""
        tokens: set[str] = set()
        doc = self.txt_prompt.document()
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
        """挿入済み token を LoRA ウィンドウ・チップのポップアップへ通知
        （リンクの挿入済みハイライト更新用）。"""
        tokens = self._active_lora_tokens()
        if self._lora_dlg is not None:
            try:
                self._lora_dlg.set_inserted(tokens)
            except RuntimeError:
                self._lora_dlg = None
        if self._lora_popup is not None and self._lora_popup.isVisible():
            self._lora_popup.set_active(tokens)

    # ----- LoRA chip hover popup -------------------------------------------
    def _ensure_lora_popup(self):
        if self._lora_popup is None:
            from .lora_dialog import TriggerWordsPopup
            self._lora_popup = TriggerWordsPopup(self)
            self._lora_popup.toggle_requested.connect(
                self._on_lora_toggle_prompt)
            self._lora_popup.hover_changed.connect(self._on_lora_popup_hover)
        return self._lora_popup

    def _show_lora_popup(self, relname: str, anchor) -> None:
        pos, _neg = lora_meta.effective_trigger_words(
            relname, config.models_root() / "loras",
            self.paths.user_data / "lora_cache")
        popup = self._ensure_lora_popup()
        popup.set_content(relname, pos, self._active_lora_tokens())
        if not popup.has_words():        # トリガーワードが無ければ出さない
            popup.hide()
            return
        self._lora_pop_timer.stop()
        self._lora_pop_anchor = anchor
        # サイズは set_content で確定済み。画面内に収まる位置へクランプする。
        below = anchor.mapToGlobal(QPoint(0, anchor.height() + 2))
        screen = (anchor.screen() or self.screen()).availableGeometry()
        y = below.y()
        if y + popup.height() - 1 > screen.bottom():
            y = anchor.mapToGlobal(QPoint(0, 0)).y() - popup.height() - 2
        x = max(screen.left(),
                min(below.x(), screen.right() - popup.width() + 1))
        y = max(screen.top(), min(y, screen.bottom() - popup.height() + 1))
        popup.move(x, y)
        popup.show()
        popup.raise_()

    def _on_lora_popup_hover(self, over: bool) -> None:
        if over:
            self._lora_pop_timer.stop()
        else:
            self._lora_pop_timer.start()

    def _hide_lora_popup(self, force: bool = False) -> None:
        """カーソルがまだチップ or ポップアップ上にあれば閉じない（チップの
        子ウィジェット上でも Leave が飛ぶため、実際の位置で判定する）。
        force=True は無条件で閉じる（チップ再構築時など）。"""
        if self._lora_popup is None or not self._lora_popup.isVisible():
            return
        if not force:
            gp = QCursor.pos()
            for w in (self._lora_popup, self._lora_pop_anchor):
                try:
                    if (w is not None and w.isVisible()
                            and w.rect().contains(w.mapFromGlobal(gp))):
                        self._lora_pop_timer.start()   # まだ上にある → 保持
                        return
                except RuntimeError:
                    pass                               # チップが破棄済み
        self._lora_popup.hide()
        self._lora_pop_anchor = None

    # ----- prompt presets --------------------------------------------------
    def _reload_prompt_presets(self, *_args, quiet: bool = False) -> None:
        prompt_presets.ensure_file(self.paths.prompts_path)
        self._presets = prompt_presets.load(self.paths.prompts_path)
        self.cb_prompt_preset.clear()
        self.cb_prompt_preset.addItems([name for name, _p, _n in self._presets])
        if not quiet:
            self.append_log(
                f"プロンプトプリセットを再読込み（{len(self._presets)} 件）")
        # 起動時はプリセット1個目のプロンプトを初期値にする。
        if quiet and self._presets and not self.txt_prompt.toPlainText():
            self.txt_prompt.setPlainText(self._presets[0][1])

    def _apply_prompt_preset(self) -> None:
        i = self.cb_prompt_preset.currentIndex()
        if not (0 <= i < len(self._presets)):
            return
        _name, prompt, _neg = self._presets[i]
        if not prompt:
            return
        cur = self.txt_prompt.toPlainText().rstrip()
        self.txt_prompt.setPlainText(
            (cur + ", " + prompt) if cur else prompt)

    def _open_prompt_csv(self) -> None:
        prompt_presets.ensure_file(self.paths.prompts_path)
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.paths.prompts_path)))

    # ----- settings persistence -------------------------------------------
    def _apply_settings(self) -> None:
        s = self.settings
        mode = str(s.get("mode", "t2v"))
        idx = self.cb_mode.findData(mode)
        if idx >= 0:
            self.cb_mode.setCurrentIndex(idx)
        self._on_mode_changed()
        # モデル選択（モード別の diffusion は fl2va/ref2va 両方復元）
        fl = str(s.get("diffusion_fl2va", ""))
        rf = str(s.get("diffusion_ref2va", ""))
        want = rf if mode == "r2v" else fl
        if want:
            self.cb_diffusion.setCurrentText(want)
        self._saved_fl2va = fl
        self._saved_ref2va = rf
        if str(s.get("te", "")):
            self.cb_te.setCurrentText(str(s.get("te")))
        if str(s.get("vae_video", "")):
            self.cb_vae_video.setCurrentText(str(s.get("vae_video")))
        if str(s.get("vae_audio", "")):
            self.cb_vae_audio.setCurrentText(str(s.get("vae_audio")))
        # 保存値が食い違っていたら（音声 VAE に動画 VAE が入っている等）
        # 名前で選び直す。同じファイルを両方に使うと音声デコードで落ちる。
        if (self.cb_vae_audio.currentText() == self.cb_vae_video.currentText()
                or "audio" not in self.cb_vae_audio.currentText().lower()):
            self._auto_pick(self.cb_vae_audio, "audio")
        if "audio" in self.cb_vae_video.currentText().lower():
            self._auto_pick(self.cb_vae_video, "video")
        ai = self.cb_aspect.findText(str(s.get("aspect", "16:9")))
        if ai >= 0:
            self.cb_aspect.setCurrentIndex(ai)
        self.sp_quality.setValue(float(s.get("quality_mp", 1.0)))
        self._last_quality_val = float(self.sp_quality.value())
        self.sp_out_w.setValue(int(s.get("size_w", 1344)))
        self.sp_out_h.setValue(int(s.get("size_h", 768)))
        self.chk_size_manual.setChecked(bool(s.get("size_manual", False)))
        self._update_size_controls()
        self.sp_length.setValue(float(s.get("length_sec", 5.0)))
        self.sp_steps.setValue(int(s.get("steps", 20)))
        self.cb_sampler.setCurrentText(str(s.get("sampler", "res_multistep")))
        self.cb_scheduler.setCurrentText(str(s.get("scheduler", "simple")))
        self.ed_seed.setText(str(s.get("seed", "-1")))
        self.cb_dtype.setCurrentText(str(s.get("dtype", "default")))
        self.grp_shift.setChecked(bool(s.get("shift_enabled", False)))
        self.sp_shift_video.setValue(float(s.get("shift_video", 12.0)))
        self.sp_shift_audio.setValue(float(s.get("shift_audio", 3.0)))
        self.chk_easycache.setChecked(bool(s.get("easycache_enabled", False)))
        self.sp_easycache.setValue(float(s.get("easycache_threshold", 0.2)))
        if self.chk_sage.isEnabled():
            self.chk_sage.setChecked(bool(s.get("sage_attention", False)))
        tv = str(s.get("turbo_variant", "8step"))
        if tv in dict(self._TURBO_VARIANTS["fl2va"]).values():
            self._turbo_fl2v_variant = tv
        # 版ごとのステップ数（"版:数,…"）。壊れていれば既定のまま。
        for part in str(s.get("turbo_steps_map", "")).split(","):
            if ":" in part:
                k, _, n = part.partition(":")
                if k.strip() in self._turbo_steps and n.strip().isdigit():
                    self._turbo_steps[k.strip()] = max(1, min(12, int(n)))
        self.chk_turbo.setChecked(bool(s.get("turbo_enabled", False)))
        self._sync_turbo_controls()
        si = self.cb_sparse_method.findData(str(s.get("sparse_method", "sol-attn")))
        if si >= 0:
            self.cb_sparse_method.setCurrentIndex(si)
        self.chk_sparse.setChecked(bool(s.get("sparse_enabled", False)))
        self.chk_ref_te_only.setChecked(bool(s.get("ref_te_only", False)))
        self.chk_same_frame.setChecked(
            bool(s.get("same_first_last_frame", False)))
        ri = self.cb_ref_size.findData(str(s.get("ref_image_size", "match")))
        if ri >= 0:
            self.cb_ref_size.setCurrentIndex(ri)
        # ウィンドウ/ペインサイズの復元（保存が無ければ既定のまま）。
        ws = str(s.get("window_size", ""))
        if "x" in ws:
            try:
                ww, hh = (int(v) for v in ws.split("x", 1))
                self.resize(max(400, ww), max(300, hh))
            except ValueError:
                pass
        ps = str(s.get("pane_sizes", ""))
        if ps:
            try:
                sizes = [int(v) for v in ps.split(",")]
                if len(sizes) == 3 and all(v > 0 for v in sizes):
                    self.splitter.setSizes(sizes)
            except ValueError:
                pass
        self._update_size_label()
        self._sync_fasth3_controls()

    def _connect_autosave(self) -> None:
        for combo in (self.cb_mode, self.cb_diffusion, self.cb_te,
                      self.cb_vae_video, self.cb_vae_audio, self.cb_aspect,
                      self.cb_sampler, self.cb_scheduler,
                      self.cb_dtype):
            combo.currentTextChanged.connect(self._schedule_save)
        self.sp_length.valueChanged.connect(self._schedule_save)
        self.sp_out_w.valueChanged.connect(self._schedule_save)
        self.sp_out_h.valueChanged.connect(self._schedule_save)
        self.sp_steps.valueChanged.connect(self._schedule_save)
        self.sp_shift_video.valueChanged.connect(self._schedule_save)
        self.sp_shift_audio.valueChanged.connect(self._schedule_save)
        self.grp_shift.toggled.connect(self._schedule_save)
        self.chk_easycache.toggled.connect(self._schedule_save)
        self.sp_easycache.valueChanged.connect(self._schedule_save)
        self.ed_seed.textChanged.connect(self._schedule_save)
        self.cb_ref_size.currentIndexChanged.connect(self._schedule_save)
        self.chk_ref_te_only.toggled.connect(self._schedule_save)
        self.chk_sparse.toggled.connect(self._schedule_save)
        self.cb_sparse_method.currentIndexChanged.connect(self._schedule_save)
        self.splitter.splitterMoved.connect(self._schedule_save)

    def _schedule_save(self, *args) -> None:
        if self._loading:
            return
        self._save_timer.start()

    def _do_save(self) -> None:
        if getattr(self, "_settings_broken", False):
            return
        mode = self._mode()
        # モード別の diffusion 選択を保持する。
        fl = getattr(self, "_saved_fl2va", "")
        rf = getattr(self, "_saved_ref2va", "")
        if mode == "r2v":
            rf = self.cb_diffusion.currentText()
        else:
            fl = self.cb_diffusion.currentText()
        self._saved_fl2va, self._saved_ref2va = fl, rf
        data = {
            "mode": mode,
            "diffusion_fl2va": fl,
            "diffusion_ref2va": rf,
            "te": self.cb_te.currentText(),
            "vae_video": self.cb_vae_video.currentText(),
            "vae_audio": self.cb_vae_audio.currentText(),
            "aspect": self.cb_aspect.currentText(),
            "quality_mp": float(self.sp_quality.value()),
            "size_manual": self._size_is_manual(),
            "size_w": int(self.sp_out_w.value()),
            "size_h": int(self.sp_out_h.value()),
            "length_sec": float(self.sp_length.value()),
            "steps": self.sp_steps.value(),
            "sampler": self.cb_sampler.currentText(),
            "scheduler": self.cb_scheduler.currentText(),
            "seed": self.ed_seed.text().strip() or "-1",
            "dtype": self.cb_dtype.currentText(),
            "shift_enabled": self.grp_shift.isChecked(),
            "shift_video": float(self.sp_shift_video.value()),
            "shift_audio": float(self.sp_shift_audio.value()),
            "easycache_enabled": self.chk_easycache.isChecked(),
            "easycache_threshold": float(self.sp_easycache.value()),
            "same_first_last_frame": self.chk_same_frame.isChecked(),
            "ref_image_size": self.cb_ref_size.currentData() or "match",
            "ref_te_only": self.chk_ref_te_only.isChecked(),
            "turbo_enabled": self.chk_turbo.isChecked(),
            "turbo_steps_map": ",".join(
                f"{k}:{int(v)}" for k, v in self._turbo_steps.items()),
            "turbo_variant": self._turbo_fl2v_variant,
            "sparse_enabled": self.chk_sparse.isChecked(),
            "sparse_method": self.cb_sparse_method.currentData() or "sol-attn",
        }
        # ジオメトリはウィンドウ表示後のみ保存する。未表示（起動処理中）の
        # splitter.sizes() はレイアウト未確定の仮値で、保存すると復元済みの
        # 正しい値を上書きで壊してしまう。
        if self.isVisible():
            data["window_size"] = f"{self.width()}x{self.height()}"
            data["pane_sizes"] = ",".join(
                str(v) for v in self.splitter.sizes())
        self.settings.update(data)
        try:
            settings.save(self.paths.settings_path, self.settings)
        except OSError as e:
            self.append_log(f"設定の保存に失敗: {e}")

    # ----- backend ---------------------------------------------------------
    def start_backend(self) -> None:
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
        # 準備完了で初めて生成ボタンを有効にする。
        self.btn_generate.setEnabled(True)
        self._update_generate_button()

    def _on_backend_failed(self, msg: str) -> None:
        self.status.showMessage("バックエンドの起動に失敗")
        self.append_log("エラー: " + msg)
        self.btn_generate.setEnabled(False)
        self.btn_generate.setToolTip("バックエンドの起動に失敗したため生成できません")
        QMessageBox.critical(self, "バックエンドエラー", msg)

    # ----- generation ------------------------------------------------------
    def _seed_value(self) -> int:
        try:
            return int(self.ed_seed.text().strip())
        except ValueError:
            return -1

    def _upload(self, local_path: str) -> str:
        """ローカルファイルをバックエンド input へ（キャッシュ付きで）上げる。"""
        p = Path(local_path)
        if not p.is_file():
            raise ValueError(f"ファイルが見つかりません: {local_path}")
        mtime = p.stat().st_mtime
        cached = self._upload_cache.get(str(p))
        if cached and cached[0] == mtime:
            return cached[1]
        name = self.backend.upload_input_file(p)
        self._upload_cache[str(p)] = (mtime, name)
        return name

    def _collect_params(self) -> GenParams:
        mode = self._mode()
        seed = self._seed_value()
        if seed < 0:
            seed = random.randint(0, MAX_SEED)

        mp = float(self.sp_quality.value())
        first = last = ""
        if mode == "i2v":
            if self.ed_first_frame.text():
                first = self._upload(self.ed_first_frame.text())
            if self.ed_last_frame.text():
                last = self._upload(self.ed_last_frame.text())
        # 解像度: 手動指定が最優先。自動は i2v なら開始（無ければ終端）
        # フレーム画像基準、他はプリセットから計算。
        base_img = (self.ed_first_frame.text() or self.ed_last_frame.text()) \
            if mode == "i2v" else ""
        if self._size_is_manual():
            # H3 のノード入力は32刻みのため丸める（UI のステップも32）。
            width = max(32, round(self.sp_out_w.value() / 32) * 32)
            height = max(32, round(self.sp_out_h.value() / 32) * 32)
        elif base_img:
            img = QImage(base_img)
            if img.isNull():
                raise ValueError(f"画像を読み込めません: {base_img}")
            width, height = size_for_image(img.width(), img.height(), mp)
        else:
            aw, ah = self.cb_aspect.currentData() or (16, 9)
            width, height = size_for_aspect(aw, ah, mp)

        chain = None
        if mode == "chain":
            # 「生成」を押した時点のチェーン設定をそのまま使う。
            plan = self._current_chain_plan()
            if not plan:
                raise ValueError(
                    "チェーンが未設定です。「チェーン設定を開く…」から"
                    "シーンを作成してください")
            workflow.validate_chain(plan)
            self._chain_plan = plan
            chain = self._upload_chain_files(plan)

        ref_images: list[str] = []
        ref_videos: list[dict] = []
        ref_audios: list[str] = []
        if mode == "r2v":
            for i in range(self.lst_ref_images.count()):
                it = self.lst_ref_images.item(i)
                ref_images.append(self._upload(it.data(Qt.UserRole)))
            for i in range(self.lst_ref_videos.count()):
                it = self.lst_ref_videos.item(i)
                ref_videos.append({
                    "name": self._upload(it.data(Qt.UserRole)),
                    "use_audio": it.checkState() == Qt.Checked,
                })
            for i in range(self.lst_ref_audios.count()):
                it = self.lst_ref_audios.item(i)
                ref_audios.append(self._upload(it.data(Qt.UserRole)))

        # Turbo LoRA: モード（チェーンは plan の種類）に応じたファイル。
        # 未ダウンロードならここで止める（ダウンロード中も含む）。
        turbo_lora = ""
        steps = self.sp_steps.value()
        shift_enabled = self.grp_shift.isChecked()
        shift_video = float(self.sp_shift_video.value())
        shift_audio = float(self.sp_shift_audio.value())
        if self.chk_turbo.isChecked():
            turbo_lora = self._turbo_lora_name()
            if not self._turbo_lora_present(turbo_lora):
                raise ValueError(
                    f"Turbo LoRA がまだダウンロードされていません:\n{turbo_lora}\n"
                    "高速化設定の Turbo LoRA を入れ直してダウンロードするか、"
                    "Models の「設定…」から Turbo LoRA セットを取得してください")
            steps = int(self.sp_turbo_steps.value())
            # 学習時の shift が既定（12/3）と違う版は、Sigma Shift 未指定なら
            # その値を自動適用する（手動 ON ならユーザー値を優先）。
            spec = self._turbo_spec()
            if not shift_enabled and spec.get("shift_video"):
                shift_enabled = True
                shift_video = float(spec["shift_video"])
                shift_audio = float(spec.get("shift_audio") or shift_audio)
                self.append_log(
                    f"Turbo LoRA: 学習条件に合わせ Sigma Shift を "
                    f"video {shift_video:g} / audio {shift_audio:g} に自動設定")
            if self.chk_easycache.isChecked():
                self.append_log(
                    "\x1b[93m警告: Turbo LoRA（少ステップ）と EasyCache の併用は"
                    "ステップスキップの影響が大きく品質が崩れやすいです\x1b[0m")

        guides = self._guide_params() if mode != "chain" else []

        # Sparse Attention（FastH3 は学習条件で上書き）
        sparse_enabled = self.chk_sparse.isChecked()
        sparse_method = self.cb_sparse_method.currentData() or "sol-attn"
        sparse_keep = 10.0
        sparse_start = 0.2
        easycache_enabled = self.chk_easycache.isChecked()
        diffusion = self.cb_diffusion.currentText().strip()
        if models_mod.is_fasth3(diffusion):
            if mode != "t2v":
                raise ValueError(
                    "FastH3 は t2v 専用です（i2v / r2v / チェーンの蒸留は"
                    "含まれていません）。Diffusion モデルを切り替えてください")
            spec = models_mod.FASTH3_SPEC
            turbo_lora = ""
            easycache_enabled = False
            steps = int(spec["steps"])
            if not self.grp_shift.isChecked():
                shift_enabled = True
                shift_video = float(spec["shift_video"])
                shift_audio = float(spec["shift_audio"])
            sparse_enabled = True
            sparse_method = str(spec["sparse_method"])
            sparse_keep = float(spec["sparse_keep_percent"])
            sparse_start = float(spec["sparse_start"])
            self.append_log(
                f"FastH3: {steps} ステップ / Sigma Shift video {shift_video:g} "
                f"audio {shift_audio:g} / Sparse Attention {sparse_method} "
                f"保持 {sparse_keep:g}% を自動適用")

        return GenParams(
            mode=mode,
            diffusion=diffusion,
            te=self.cb_te.currentText().strip(),
            vae_video=self.cb_vae_video.currentText().strip(),
            vae_audio=self.cb_vae_audio.currentText().strip(),
            prompt=self.txt_prompt.toPlainText(),
            width=width,
            height=height,
            frames=frames_for_seconds(self.sp_length.value()),
            steps=steps,
            sampler=self.cb_sampler.currentText(),
            scheduler=self.cb_scheduler.currentText(),
            seed=seed,
            weight_dtype=self.cb_dtype.currentText(),
            shift_enabled=shift_enabled,
            shift_video=shift_video,
            shift_audio=shift_audio,
            loras=[(e["name"], float(e["strength"]))
                   for e in self._current_loras()],
            chain=chain,
            easycache_enabled=easycache_enabled,
            easycache_threshold=float(self.sp_easycache.value()),
            first_frame=first,
            last_frame=last,
            ref_image_size=self.cb_ref_size.currentData() or "match",
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_audios=ref_audios,
            ref_te_only=self.chk_ref_te_only.isChecked(),
            turbo_lora=turbo_lora,
            sparse_enabled=sparse_enabled,
            sparse_method=sparse_method,
            sparse_keep_percent=sparse_keep,
            sparse_start=sparse_start,
            guides=guides,
        )

    def on_generate(self) -> None:
        """生成ボタン / Shift+Enter。アイドルなら即開始、生成中なら現在の
        UI設定のスナップショットをタスクとして積む（完了後に順次消費）。"""
        if not self.backend.is_running():
            QMessageBox.warning(self, "未準備", "バックエンドがまだ起動していません。")
            return
        try:
            params = self._collect_params()
            build_graph(params)   # 積む場合も入力不足はこの場で検出する
        except ValueError as e:
            QMessageBox.warning(self, "入力不足", str(e))
            return
        except BackendError as e:
            QMessageBox.warning(self, "アップロード失敗", str(e))
            return
        if self._gen_thread is not None:
            self._gen_queue.append(params)
            self.append_log(
                f"生成をタスクに積みました（待機 {len(self._gen_queue)} 件, "
                f"seed={params.seed}）")
            self._update_generate_button()
            return
        self._start_generation(params)

    def _start_generation(self, params: GenParams) -> None:
        try:
            graph = build_graph(params)
        except ValueError as e:
            QMessageBox.warning(self, "入力不足", str(e))
            self._update_generate_button()
            return
        self._last_seed = params.seed
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.status.showMessage("生成中…")
        if params.mode == "chain" and params.chain:
            raw, delivered = workflow.chain_frames(params.chain)
            self.append_log(
                f"生成 [chain] {params.chain.get('run_name')} "
                f"{len(params.chain.get('shots', []))} シーン "
                f"{params.width}x{params.height} "
                f"生成 {raw}f / 実尺 {delivered}f "
                f"({delivered / workflow.FPS:.1f}s)")
        else:
            secs = params.frames / workflow.FPS
            self.append_log(
                f"生成 [{params.mode}] seed={params.seed} "
                f"{params.width}x{params.height} {params.frames}f "
                f"({secs:.1f}s)")

        self._gen_thread = QThread(self)
        # mp4 のメタデータは ComfyUI 標準（workflow JSON）に任せ、生成アプリ
        # の識別用に software タグだけを追加する。
        self._gen_worker = _GenWorker(
            self.backend, graph,
            # チェーンの成果物は SaveVideo を通らないが、注入した
            # カスタムノードが Contex Loop 側の書き出しに同じタグを混ぜる。
            extra_pnginfo={"software": config.APP_SIGNATURE})
        self._gen_worker.moveToThread(self._gen_thread)
        self._gen_thread.started.connect(self._gen_worker.run)
        self._gen_worker.progress.connect(self._on_progress)
        self._gen_worker.preview.connect(self._on_preview_frame)
        self._gen_worker.timing.connect(self._on_gen_timing)
        self._gen_worker.done.connect(self._on_gen_done)
        self._gen_worker.failed.connect(self._on_gen_failed)
        self._gen_worker.done.connect(self._gen_thread.quit)
        self._gen_worker.failed.connect(self._gen_thread.quit)
        self._gen_thread.finished.connect(self._cleanup_gen_thread)
        self._gen_thread.start()
        self._update_generate_button()

    def _update_generate_button(self, *_a) -> None:
        # 待機タスクの有無でキャンセル/スキップの表示も変わるため同時に更新。
        self._update_cancel_button()
        if self._gen_thread is not None:
            self.btn_generate.setText(
                f"生成をタスクに積む ({len(self._gen_queue)})")
            self.btn_generate.setToolTip(
                "現在の設定・プロンプトのスナップショットを待機タスクとして"
                "積みます。現在の生成が終わると順番に実行されます")
        else:
            self.btn_generate.setText("生成")
            self.btn_generate.setToolTip(
                "" if self.backend.is_running() else self._NOT_READY_TIP)

    def _skip_mode(self) -> bool:
        """「スキップ」として振る舞うか（連続 ON、または待機タスクあり）。"""
        return bool(self.btn_continuous.isChecked() or self._gen_queue)

    def _update_cancel_button(self, *_a) -> None:
        skip = self._skip_mode()
        self.btn_cancel.setText("スキップ" if skip else "キャンセル")
        self.btn_cancel.setToolTip(
            "現在の生成を中断して次へ進みます"
            "（待機タスク・連続生成はそのまま続行）" if skip else "")

    def on_cancel(self) -> None:
        """連続 ON か待機タスクがあるときは「スキップ」: 現在の生成だけ中断し、
        残りのタスク・連続生成はそのまま続ける。どちらも無ければ通常の
        キャンセル。"""
        if not self._gen_worker:
            return
        if self._skip_mode():
            self._gen_skip = True
            n = len(self._gen_queue)
            self.append_log(
                "スキップ: 現在の生成を中断して次へ進みます"
                + (f"（待機タスク {n} 件）" if n else ""))
        else:
            self.append_log("キャンセルを要求しました")
        self._gen_worker.cancel()

    def _on_progress(self, p: Progress) -> None:
        if p.maximum:
            self.progress.setMaximum(p.maximum)
            self.progress.setValue(p.value)
            self.progress.setFormat(f"{p.note} {p.value}/{p.maximum}")

    def _on_gen_timing(self, secs: float) -> None:
        self.lbl_gen_time.setText(f"推論 {secs:.2f} 秒")
        self.append_log(f"推論時間: {secs:.2f} 秒")

    def _on_preview_frame(self, data: bytes) -> None:
        img = QImage.fromData(data)
        if img.isNull():
            return
        pix = QPixmap.fromImage(img)
        self.preview.setPixmap(pix.scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_gen_done(self, files: list) -> None:
        videos = [Path(f) for f in files
                  if str(f).lower().endswith(_VIDEO_EXTS)]
        self._last_gen_ok = True
        if videos:
            self._last_video = videos[0]
            self.status.showMessage(f"完了: {videos[0].name}")
            for v in videos:
                self.append_log(f"{v} に保存しました")
            self.preview.setToolTip(
                f"{videos[0]}\nダブルクリック=外部プレーヤーで再生")
        else:
            self.status.showMessage("完了（出力ファイルが見つかりません）")
            self.append_log("警告: 出力動画が見つかりませんでした")

    def _on_gen_failed(self, msg: str) -> None:
        self.status.showMessage("生成失敗")
        self.append_log("エラー: " + msg)
        self._last_gen_ok = False
        if "キャンセル" not in msg:
            QMessageBox.critical(self, "生成エラー", msg)

    def _cleanup_gen_thread(self) -> None:
        self._gen_thread = None
        self._gen_worker = None
        self.btn_generate.setEnabled(self.backend.is_running())
        self.btn_cancel.setEnabled(False)
        skip = self._gen_skip
        self._gen_skip = False
        proceed = self._last_gen_ok or skip
        if proceed and self._gen_queue:
            params = self._gen_queue.pop(0)
            self.append_log(
                f"待機タスクを開始します（残り {len(self._gen_queue)} 件）")
            self._update_generate_button()
            QTimer.singleShot(0, lambda p=params: self._start_generation(p))
            return
        if proceed and self.btn_continuous.isChecked():
            self._update_generate_button()
            QTimer.singleShot(0, self.on_generate)
            return
        if self._gen_queue:
            n = len(self._gen_queue)
            self._gen_queue.clear()
            self.append_log(f"停止したため待機中のタスク {n} 件を破棄しました")
        self._update_generate_button()

    # ----- misc ------------------------------------------------------------
    def append_log(self, text: str) -> None:
        ansi_log.append_ansi(self.log_view, text)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt signature)
        # 注意: UI 構築中にも呼ばれるため、属性は getattr で安全に参照する。
        # Shift+Enter in the prompt field triggers generation.
        if (obj is getattr(self, "txt_prompt", None)
                and event.type() == QEvent.KeyPress
                and event.key() in (Qt.Key_Return, Qt.Key_Enter)
                and event.modifiers() & Qt.ShiftModifier):
            self.on_generate()
            return True
        # プレビュー欄: ダブルクリックで直近の動画を外部プレーヤーで再生。
        if (obj is getattr(self, "preview", None)
                and event.type() == QEvent.MouseButtonDblClick):
            if self._last_video and self._last_video.exists():
                QDesktopServices.openUrl(
                    QUrl.fromLocalFile(str(self._last_video)))
            return True
        # LoRA チップのホバーでトリガーワードのポップアップを開閉する。
        name = getattr(obj, "_lora_name", None)
        if name is not None:
            if event.type() == QEvent.Enter:
                self._show_lora_popup(name, getattr(obj, "_lora_chip", obj))
            elif event.type() == QEvent.Leave:
                self._lora_pop_timer.start()   # 猶予後に閉じる（保持判定つき）
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        super().resizeEvent(event)
        # ウィンドウサイズも自動保存（構築中・設定復元中は _loading が守る）。
        if not getattr(self, "_loading", True):
            self._schedule_save()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        if self._chain_dlg is not None:
            self._chain_dlg.close()
        if self._gen_worker:
            self._gen_worker.cancel()
        try:
            self.backend.stop()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)

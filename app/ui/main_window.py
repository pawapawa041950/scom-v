"""PySide6 main window: MiniMax H3 video generation (t2v / i2v / r2v)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    Qt, QThread, Signal, QObject, QRegularExpression, QTimer, QEvent, QUrl,
)
from PySide6.QtGui import (
    QDesktopServices, QImage, QPixmap, QRegularExpressionValidator,
)
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSizePolicy, QSpinBox, QSplitter, QStackedWidget,
    QVBoxLayout, QWidget,
)

from .. import config, settings, prompt_presets, workflow
from . import ansi_log
from .widgets import GrowingTextEdit, WideComboBox
from ..comfy_backend import ComfyBackend, BackendError, Progress
from ..workflow import (
    GenParams, build_graph, frames_for_seconds, size_for_aspect,
    size_for_image, ASPECT_PRESETS, QUALITY_PRESETS, SAMPLERS, SCHEDULERS,
)

MAX_SEED = 2**63 - 1

MODES = [("t2v", "テキストから動画 (t2v)"),
         ("i2v", "画像から動画 (i2v)"),
         ("r2v", "参照から動画 (r2v)")]

_IMAGE_FILTER = "画像 (*.png *.jpg *.jpeg *.webp *.bmp);;すべて (*.*)"
_VIDEO_FILTER = "動画 (*.mp4 *.webm *.mkv *.mov *.avi);;すべて (*.*)"
_AUDIO_FILTER = "音声 (*.wav *.mp3 *.flac *.ogg *.m4a);;すべて (*.*)"
_VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov", ".avi")


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

    def __init__(self, backend: ComfyBackend, graph: dict):
        super().__init__()
        self.backend = backend
        self.graph = graph
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
            )
            self.done.emit(list(files))
        except BackendError as e:
            self.failed.emit(str(e))
        except Exception as e:  # pragma: no cover - defensive
            self.failed.emit(f"unexpected error: {e}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("scom-v - 動画生成")
        self.resize(1180, 760)

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

        self.settings, settings_error = settings.load(self.paths.settings_path)
        self._loading = True
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(250)
        self._save_timer.timeout.connect(self._do_save)

        self._build_ui()
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

        left = QWidget()
        lv = QVBoxLayout(left)

        # モード + 管理ボタン
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("モード:"))
        self.cb_mode = WideComboBox()
        for token, label in MODES:
            self.cb_mode.addItem(label, token)
        self.cb_mode.currentIndexChanged.connect(self._on_mode_changed)
        top_row.addWidget(self.cb_mode, stretch=1)
        btn_rescan = QPushButton("再スキャン")
        btn_rescan.clicked.connect(self.refresh_models)
        btn_manage = QPushButton("設定…")
        btn_manage.clicked.connect(self.open_models_dialog)
        top_row.addWidget(btn_rescan)
        top_row.addWidget(btn_manage)
        lv.addLayout(top_row)

        # Models
        box_models = QGroupBox("Models")
        form = QFormLayout(box_models)
        self.cb_diffusion = WideComboBox()
        self.cb_te = WideComboBox()
        self.cb_vae_video = WideComboBox()
        self.cb_vae_audio = WideComboBox()
        self.lbl_diffusion = QLabel("Diffusion (fl2va):")
        form.addRow(self.lbl_diffusion, self.cb_diffusion)
        form.addRow("Text encoder:", self.cb_te)
        form.addRow("動画 VAE:", self.cb_vae_video)
        form.addRow("音声 VAE:", self.cb_vae_audio)
        lv.addWidget(box_models)

        # Prompt
        box_prompt = QGroupBox("Prompt")
        pv = QVBoxLayout(box_prompt)
        self.txt_prompt = GrowingTextEdit(min_lines=5)
        self.txt_prompt.setPlaceholderText(
            "動画の内容を文章で記述… (r2v では <Picture 1> <Video 1> <Audio 1> "
            "のタグで参照を指せます)")
        self.txt_prompt.installEventFilter(self)  # Shift+Enter で生成
        pv.addWidget(self.txt_prompt)
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
        lv.addWidget(box_prompt)

        # モード別入力
        self.stack_mode = QStackedWidget()
        self.stack_mode.addWidget(self._build_t2v_page())
        self.stack_mode.addWidget(self._build_i2v_page())
        self.stack_mode.addWidget(self._build_r2v_page())
        lv.addWidget(self.stack_mode)

        # 設定
        lv.addWidget(self._build_settings_box())
        lv.addStretch(1)
        left.setMinimumWidth(560)

        # 右カラム: ログ / プレビュー / アクション
        right = QWidget()
        rv = QVBoxLayout(right)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setPlaceholderText("バックエンドログ…")
        ansi_log.style_log(self.log_view)
        rv.addWidget(self.log_view, stretch=1)

        self.preview = QLabel("プレビュー\n（完成後はダブルクリックで動画を再生）")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(480, 360)
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

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
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

    def _build_t2v_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        note = QLabel("プロンプトのみから動画+音声を生成します。")
        note.setStyleSheet("color:#888;")
        v.addWidget(note)
        return page

    def _build_i2v_page(self) -> QWidget:
        page = QGroupBox("i2v 入力画像")
        grid = QGridLayout(page)
        self.ed_first_frame = QLineEdit()
        self.ed_first_frame.setReadOnly(True)
        self.ed_last_frame = QLineEdit()
        self.ed_last_frame.setReadOnly(True)
        for r, (label, ed, tip) in enumerate((
            ("開始フレーム:", self.ed_first_frame,
             "動画の最初のフレームになる画像"),
            ("終端フレーム(任意):", self.ed_last_frame,
             "指定すると動画の最後がこの画像へ収束します（両方指定で補間的な生成）"),
        )):
            ed.setToolTip(tip)
            btn_sel = QPushButton("参照…")
            btn_clr = QPushButton("クリア")
            btn_sel.clicked.connect(
                lambda *_a, e=ed: self._pick_file(e, _IMAGE_FILTER))
            btn_clr.clicked.connect(lambda *_a, e=ed: self._clear_frame(e))
            grid.addWidget(QLabel(label), r, 0)
            grid.addWidget(ed, r, 1)
            grid.addWidget(btn_sel, r, 2)
            grid.addWidget(btn_clr, r, 3)
        self.ed_first_frame.textChanged.connect(self._update_size_label)
        return page

    def _build_r2v_page(self) -> QWidget:
        page = QGroupBox("r2v 参照 (プロンプト内で <Picture i> <Video k> <Audio j> で参照)")
        grid = QGridLayout(page)

        def make_list(label: str, lst_tip: str, max_n: int, flt: str,
                      checkable: bool = False):
            lst = QListWidget()
            lst.setMaximumHeight(72)
            lst.setToolTip(lst_tip)
            btn_add = QPushButton("追加…")
            btn_del = QPushButton("削除")

            def add(*_a):
                if lst.count() >= max_n:
                    QMessageBox.information(self, "上限",
                                            f"{label}は最大 {max_n} 件です。")
                    return
                path, _ = QFileDialog.getOpenFileName(self, label, "", flt)
                if not path:
                    return
                item = QListWidgetItem(Path(path).name)
                item.setData(Qt.UserRole, path)
                item.setToolTip(path)
                if checkable:
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked)
                lst.addItem(item)

            def remove(*_a):
                for it in lst.selectedItems():
                    lst.takeItem(lst.row(it))

            btn_add.clicked.connect(add)
            btn_del.clicked.connect(remove)
            col = QVBoxLayout()
            col.addWidget(btn_add)
            col.addWidget(btn_del)
            col.addStretch(1)
            return lst, col

        self.lst_ref_images, col1 = make_list(
            "参照画像", "キャラクター・画風などの参照画像（最大9枚）", 9,
            _IMAGE_FILTER)
        self.lst_ref_videos, col2 = make_list(
            "参照動画", "参照動画 2〜15秒（最大3本）。チェックONでその動画の"
            "音声も参照に含めます", 3, _VIDEO_FILTER, checkable=True)
        self.lst_ref_audios, col3 = make_list(
            "参照音声", "単体の参照音声（最大3本）", 3, _AUDIO_FILTER)

        grid.addWidget(QLabel("参照画像 (≤9):"), 0, 0)
        grid.addWidget(self.lst_ref_images, 0, 1)
        grid.addLayout(col1, 0, 2)
        grid.addWidget(QLabel("参照動画 (≤3)\nチェック=音声も含める:"), 1, 0)
        grid.addWidget(self.lst_ref_videos, 1, 1)
        grid.addLayout(col2, 1, 2)
        grid.addWidget(QLabel("参照音声 (≤3):"), 2, 0)
        grid.addWidget(self.lst_ref_audios, 2, 1)
        grid.addLayout(col3, 2, 2)
        return page

    def _build_settings_box(self) -> QGroupBox:
        box = QGroupBox("設定")
        grid = QGridLayout(box)

        self.cb_aspect = WideComboBox()
        for name, aw, ah in ASPECT_PRESETS:
            self.cb_aspect.addItem(name, (aw, ah))
        self.cb_aspect.setCurrentIndex(1)  # 16:9
        self.cb_aspect.currentIndexChanged.connect(self._update_size_label)
        self.cb_quality = WideComboBox()
        for name, mp in QUALITY_PRESETS:
            self.cb_quality.addItem(name, mp)
        self.cb_quality.currentIndexChanged.connect(self._update_size_label)

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
        grid.addWidget(QLabel("アスペクト比"), r, 0)
        grid.addWidget(self.cb_aspect, r, 1)
        grid.addWidget(QLabel("解像度"), r, 2)
        grid.addWidget(self.cb_quality, r, 3)
        r += 1
        grid.addWidget(QLabel("長さ"), r, 0)
        grid.addWidget(self.sp_length, r, 1)
        grid.addWidget(QLabel("出力"), r, 2)
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
        grid.addWidget(self.grp_shift, r, 2, 1, 2)

        self._update_size_label()
        return box

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
            {"t2v": 0, "i2v": 1, "r2v": 2}[mode])
        self.lbl_diffusion.setText(
            "Diffusion (ref2va):" if mode == "r2v" else "Diffusion (fl2va):")
        self._refill_diffusion()
        self._update_size_label()
        self._schedule_save()

    def _update_size_label(self, *_a) -> None:
        mode = self._mode()
        mp = float(self.cb_quality.currentData() or 1.0)
        frames = frames_for_seconds(self.sp_length.value())
        secs = frames / workflow.FPS
        if mode == "i2v" and self.ed_first_frame.text():
            img = QImage(self.ed_first_frame.text())
            if not img.isNull():
                w, h = size_for_image(img.width(), img.height(), mp)
                self.lbl_size.setText(
                    f"{w}x{h}（画像基準）  {frames}f ≈ {secs:.1f}s")
                return
        aw, ah = self.cb_aspect.currentData() or (16, 9)
        w, h = size_for_aspect(aw, ah, mp)
        note = "（i2v は画像基準）" if mode == "i2v" else ""
        self.lbl_size.setText(f"{w}x{h}{note}  {frames}f ≈ {secs:.1f}s")

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
        if self._mode() == "r2v":
            wanted = [f for f in files if "ref2va" in f.lower()]
        else:
            wanted = [f for f in files if "ref2va" not in f.lower()]
        self._fill_combo(self.cb_diffusion, wanted or files)

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

    # ----- 設定ウィンドウ / SageAttention ----------------------------------
    def open_models_dialog(self) -> None:
        from .models_dialog import ModelsDialog
        dlg = ModelsDialog(
            self.paths,
            sage_enabled=bool(self.settings.get("sage_attention", False)),
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
        ai = self.cb_aspect.findText(str(s.get("aspect", "16:9")))
        if ai >= 0:
            self.cb_aspect.setCurrentIndex(ai)
        qmp = float(s.get("quality_mp", 1.0))
        for i in range(self.cb_quality.count()):
            if abs(float(self.cb_quality.itemData(i)) - qmp) < 1e-6:
                self.cb_quality.setCurrentIndex(i)
                break
        self.sp_length.setValue(float(s.get("length_sec", 5.0)))
        self.sp_steps.setValue(int(s.get("steps", 20)))
        self.cb_sampler.setCurrentText(str(s.get("sampler", "res_multistep")))
        self.cb_scheduler.setCurrentText(str(s.get("scheduler", "simple")))
        self.ed_seed.setText(str(s.get("seed", "-1")))
        self.cb_dtype.setCurrentText(str(s.get("dtype", "default")))
        self.grp_shift.setChecked(bool(s.get("shift_enabled", False)))
        self.sp_shift_video.setValue(float(s.get("shift_video", 12.0)))
        self.sp_shift_audio.setValue(float(s.get("shift_audio", 3.0)))
        self._ref_image_size = str(s.get("ref_image_size", "match"))
        self._update_size_label()

    def _connect_autosave(self) -> None:
        for combo in (self.cb_mode, self.cb_diffusion, self.cb_te,
                      self.cb_vae_video, self.cb_vae_audio, self.cb_aspect,
                      self.cb_quality, self.cb_sampler, self.cb_scheduler,
                      self.cb_dtype):
            combo.currentTextChanged.connect(self._schedule_save)
        self.sp_length.valueChanged.connect(self._schedule_save)
        self.sp_steps.valueChanged.connect(self._schedule_save)
        self.sp_shift_video.valueChanged.connect(self._schedule_save)
        self.sp_shift_audio.valueChanged.connect(self._schedule_save)
        self.grp_shift.toggled.connect(self._schedule_save)
        self.ed_seed.textChanged.connect(self._schedule_save)

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
            "quality_mp": float(self.cb_quality.currentData() or 1.0),
            "length_sec": float(self.sp_length.value()),
            "steps": self.sp_steps.value(),
            "sampler": self.cb_sampler.currentText(),
            "scheduler": self.cb_scheduler.currentText(),
            "seed": self.ed_seed.text().strip() or "-1",
            "dtype": self.cb_dtype.currentText(),
            "shift_enabled": self.grp_shift.isChecked(),
            "shift_video": float(self.sp_shift_video.value()),
            "shift_audio": float(self.sp_shift_audio.value()),
            "ref_image_size": getattr(self, "_ref_image_size", "match"),
        }
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

        mp = float(self.cb_quality.currentData() or 1.0)
        first = last = ""
        if mode == "i2v":
            if self.ed_first_frame.text():
                first = self._upload(self.ed_first_frame.text())
            if self.ed_last_frame.text():
                last = self._upload(self.ed_last_frame.text())
        # 解像度: i2v は開始（無ければ終端）フレーム画像基準、他はプリセット。
        base_img = (self.ed_first_frame.text() or self.ed_last_frame.text()) \
            if mode == "i2v" else ""
        if base_img:
            img = QImage(base_img)
            if img.isNull():
                raise ValueError(f"画像を読み込めません: {base_img}")
            width, height = size_for_image(img.width(), img.height(), mp)
        else:
            aw, ah = self.cb_aspect.currentData() or (16, 9)
            width, height = size_for_aspect(aw, ah, mp)

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

        return GenParams(
            mode=mode,
            diffusion=self.cb_diffusion.currentText().strip(),
            te=self.cb_te.currentText().strip(),
            vae_video=self.cb_vae_video.currentText().strip(),
            vae_audio=self.cb_vae_audio.currentText().strip(),
            prompt=self.txt_prompt.toPlainText(),
            width=width,
            height=height,
            frames=frames_for_seconds(self.sp_length.value()),
            steps=self.sp_steps.value(),
            sampler=self.cb_sampler.currentText(),
            scheduler=self.cb_scheduler.currentText(),
            seed=seed,
            weight_dtype=self.cb_dtype.currentText(),
            shift_enabled=self.grp_shift.isChecked(),
            shift_video=float(self.sp_shift_video.value()),
            shift_audio=float(self.sp_shift_audio.value()),
            first_frame=first,
            last_frame=last,
            ref_image_size=getattr(self, "_ref_image_size", "match"),
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_audios=ref_audios,
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
        secs = params.frames / workflow.FPS
        self.append_log(
            f"生成 [{params.mode}] seed={params.seed} "
            f"{params.width}x{params.height} {params.frames}f ({secs:.1f}s)")

        self._gen_thread = QThread(self)
        self._gen_worker = _GenWorker(self.backend, graph)
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
        if self._gen_thread is not None:
            self.btn_generate.setText(
                f"生成をタスクに積む ({len(self._gen_queue)})")
            self.btn_generate.setToolTip(
                "現在の設定・プロンプトのスナップショットを待機タスクとして"
                "積みます。現在の生成が終わると順番に実行されます")
        else:
            self.btn_generate.setText("生成")
            self.btn_generate.setToolTip("")

    def _update_cancel_button(self, *_a) -> None:
        cont = self.btn_continuous.isChecked()
        self.btn_cancel.setText("スキップ" if cont else "キャンセル")
        self.btn_cancel.setToolTip(
            "現在の生成を中断して次の生成に進みます（連続は続行）" if cont
            else "")

    def on_cancel(self) -> None:
        """連続 ON のときは「スキップ」: 現在の生成だけ中断し、連続はそのまま
        次の生成へ進む。OFF のときは通常のキャンセル。"""
        if not self._gen_worker:
            return
        if self.btn_continuous.isChecked():
            self._gen_skip = True
            self.append_log("スキップ: 現在の生成を中断して次へ進みます")
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
                f"{videos[0]}\nダブルクリックで再生")
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
        self.btn_generate.setEnabled(True)
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
        # プレビューのダブルクリックで最後の動画を再生。
        if (obj is getattr(self, "preview", None)
                and event.type() == QEvent.MouseButtonDblClick):
            if self._last_video and self._last_video.exists():
                QDesktopServices.openUrl(
                    QUrl.fromLocalFile(str(self._last_video)))
            return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        if self._gen_worker:
            self._gen_worker.cancel()
        try:
            self.backend.stop()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)

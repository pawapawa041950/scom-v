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
from ..bootstrap.setup import (
    SetupError, install_sage_attention, sage_installed, _Manifest,
)
from ..comfy_backend import ComfyBackend, BackendError, Progress
from ..workflow import (
    GenParams, build_graph, frames_for_seconds, size_for_aspect,
    size_for_image, ASPECT_PRESETS, SAMPLERS, SCHEDULERS,
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
         ("r2v", "参照から動画 (r2v)")]

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
        box_prompt = QGroupBox("Prompt")
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
        lv.addWidget(self.stack_mode)
        lv.addStretch(1)

        # 中央ペイン: 設定を最上部に置く（Prompt はその下）
        cv.insertWidget(0, self._build_settings_box())

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
        v.addStretch(1)
        self.ed_first_frame.textChanged.connect(self._update_size_label)
        return page

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
        return box

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
            {"t2v": 0, "i2v": 1, "r2v": 2}[mode])
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

    # ----- 設定ウィンドウ --------------------------------------------------
    def open_models_dialog(self) -> None:
        from .models_dialog import ModelsDialog
        dlg = ModelsDialog(self.paths, parent=self)
        dlg.exec()
        self.refresh_models()

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
            "ref_image_size": self.cb_ref_size.currentData() or "match",
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
            loras=[(e["name"], float(e["strength"]))
                   for e in self._current_loras()],
            easycache_enabled=self.chk_easycache.isChecked(),
            easycache_threshold=float(self.sp_easycache.value()),
            first_frame=first,
            last_frame=last,
            ref_image_size=self.cb_ref_size.currentData() or "match",
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
        # mp4 のメタデータは ComfyUI 標準（workflow JSON）に任せ、生成アプリ
        # の識別用に software タグだけを追加する。
        self._gen_worker = _GenWorker(
            self.backend, graph,
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
        if self._gen_worker:
            self._gen_worker.cancel()
        try:
            self.backend.stop()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)

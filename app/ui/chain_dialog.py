"""Contex Loop（長尺チェーン）の設定ウィンドウ — たたき台。

まだバックエンドとは繋がっていない UI のみの実装。設定は plan 辞書
（Contex Loop の Plan JSON に近い形）として保持する。

構成:
  ヘッダ    run_name / チェーン種別 / 実尺サマリ
  シーンタブ 上=共通プロンプト、下=3ペイン（一覧 / 編集 / 参照）
  他タブ    つなぎ・音声 / 出力・再開
  フッタ    実尺の内訳と警告

参照は「素材プール + シーンごとの使用チェック」で持ち、Contex Loop の
`scenes` セレクタ（例 "1,3,5:8"）へは plan() で変換する。
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from PySide6.QtCore import (
    QEvent, QPoint, QSize, Qt, QTimer, QUrl, Signal,
)
from PySide6.QtGui import QCursor, QDesktopServices, QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSpinBox,
    QSplitter, QStyle, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from .. import config
from ..workflow import FPS, frames_for_seconds
from .widgets import FlowLayout, WideComboBox
from .window_state import bind_geometry

CONTEXT_LENGTHS = [1, 5, 22, 39]
AUDIO_MODES = [
    ("generated_audio", "H3 が音声も生成（既定）"),
    ("source_track", "外部音源に合わせる（ミュージックビデオ向け）"),
    ("source_plus_timeline", "外部音源 + 生成音声（実験的）"),
]
_IMAGE_FILTER = "画像 (*.png *.jpg *.jpeg *.webp *.bmp);;すべて (*.*)"
_VIDEO_FILTER = "動画 (*.mp4 *.webm *.mkv *.mov *.avi);;すべて (*.*)"
_AUDIO_FILTER = "音声 (*.wav *.mp3 *.flac *.ogg *.m4a);;すべて (*.*)"

# 参照素材の種別: (キー, 表示名, フィルタ, 上限, タグ接頭辞)
REF_KINDS = [
    ("image", "画像", _IMAGE_FILTER, 9, "picture"),
    ("video", "動画", _VIDEO_FILTER, 3, "video"),
    ("audio", "音声", _AUDIO_FILTER, 3, "audio"),
]


class _PreviewPopup(QLabel):
    """参照素材のサムネイルを出す軽量ポップアップ（枠付きのツールチップ窓）。"""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setStyleSheet(
            "QLabel { background:#2b2b2b; border:1px solid #555; padding:4px; }")
        self.setAlignment(Qt.AlignCenter)

    def show_pixmap(self, pix: QPixmap) -> None:
        self.setPixmap(pix)
        self.resize(pix.width() + 10, pix.height() + 10)
        self._move_near_cursor()
        self.show()

    def show_text(self, text: str) -> None:
        self.setPixmap(QPixmap())
        self.setText(text)
        self.setStyleSheet(
            "QLabel { background:#2b2b2b; color:#ddd; border:1px solid #555;"
            " padding:6px; }")
        self.adjustSize()
        self._move_near_cursor()
        self.show()

    def _move_near_cursor(self) -> None:
        pos = QCursor.pos() + QPoint(16, 16)
        screen = QGuiApplication.screenAt(QCursor.pos())
        if screen is not None:
            g = screen.availableGeometry()
            x = min(pos.x(), g.right() - self.width() - 4)
            y = min(pos.y(), g.bottom() - self.height() - 4)
            pos = QPoint(max(g.left() + 4, x), max(g.top() + 4, y))
        self.move(pos)


def _new_scene(index: int) -> dict:
    return {"id": f"clip_{index:04d}", "prompt": "", "duration_seconds": 15.0,
            "steps": 0, "seed": "", "refs": [], "first_frame": ""}


class ChainDialog(QDialog):
    # シーン構成・つなぎ設定が変わったときに発火（メイン側の概要表示用）。
    changed = Signal()

    def __init__(self, parent=None):
        # 親（オーナー）を持つウィンドウは OS 側で常に親より前面に固定される
        # ため、親を渡さず独立したトップレベルウィンドウにする。メイン
        # ウィンドウの裏へ回せるようになり、タスクバーにも並ぶ。
        # 生存管理は呼び出し側（MainWindow が参照を保持し、終了時に閉じる）。
        super().__init__(None)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("Contex Loop - 長尺チェーン設定")
        self.resize(1180, 760)
        bind_geometry(self, "chain")

        self.scenes: list[dict] = [_new_scene(1), _new_scene(2)]
        # 参照素材プール: {kind: [{"id": int, "path": str, "tag": str}, ...]}
        self.refs: dict[str, list[dict]] = {k: [] for k, *_ in REF_KINDS}
        self._next_ref_id = 1
        self._loading = False
        self._popup: _PreviewPopup | None = None
        self._thumbs: dict[str, QPixmap] = {}

        root = QVBoxLayout(self)
        root.addWidget(self._build_header())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_scenes_tab(), "シーン")
        self.tabs.addTab(self._build_chain_tab(), "つなぎ・音声")
        self.tabs.addTab(self._build_output_tab(), "出力・再開")
        root.addWidget(self.tabs, stretch=1)

        self.lbl_footer = QLabel("")
        self.lbl_footer.setWordWrap(True)
        root.addWidget(self.lbl_footer)

        # 非モーダル。設定は「生成」を押した時点の内容がそのまま使われるため、
        # OK/キャンセルは持たない。設定の保存/読込は明示的に行う。
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_import = QPushButton("インポート…")
        btn_import.setToolTip("保存したチェーン設定（JSON）を読み込みます")
        btn_import.clicked.connect(self.import_plan)
        btn_export = QPushButton("エクスポート…")
        btn_export.setToolTip("現在のチェーン設定を JSON ファイルに保存します")
        btn_export.clicked.connect(self.export_plan)
        btn_close = QPushButton("閉じる")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(btn_import)
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

        self._reload_scene_list()
        self._select_scene(0)
        self._on_chain_type_changed()

    # ----- header ----------------------------------------------------------
    def _build_header(self) -> QWidget:
        box = QGroupBox("チェーン")
        h = QHBoxLayout(box)
        h.addWidget(QLabel("名前"))
        self.ed_run_name = QLineEdit("h3_chain")
        self.ed_run_name.setToolTip(
            "出力フォルダ名になります（output/h3_chains/<名前>/）")
        h.addWidget(self.ed_run_name, stretch=1)
        h.addSpacing(12)
        h.addWidget(QLabel("種別"))
        self.cb_chain_type = WideComboBox()
        self.cb_chain_type.addItem("画像から開始 (i2v)", "i2v")
        self.cb_chain_type.addItem("参照から生成 (r2v)", "r2v")
        self.cb_chain_type.addItem("プロンプトのみ (t2v)", "t2v")
        self.cb_chain_type.setToolTip(
            "i2v: 1枚目の画像から始める（2シーン目以降は動きの文脈で継続）\n"
            "r2v: 参照画像/動画/音声をシーン単位で出し入れする\n"
            "t2v: 参照なし。プロンプトのみ")
        self.cb_chain_type.currentIndexChanged.connect(
            self._on_chain_type_changed)
        h.addWidget(self.cb_chain_type, stretch=1)
        h.addSpacing(12)
        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet("font-weight:bold;")
        h.addWidget(self.lbl_summary)
        return box

    # ----- tab: scenes -----------------------------------------------------
    def _build_scenes_tab(self) -> QWidget:
        """左（共通プロンプト + シーン一覧/編集）と、タブの上端から下端まで
        通した右の参照ペインに分ける。"""
        page = QWidget()
        v = QVBoxLayout(page)
        main = QSplitter(Qt.Horizontal)

        left = QSplitter(Qt.Vertical)
        # 左上: 共通プロンプト（どのシーンを選んでいても常に見える・狭め）
        box_prefix = QGroupBox("共通プロンプト（全シーンの先頭に自動で付きます）")
        pv = QVBoxLayout(box_prefix)
        self.txt_prefix = QTextEdit()
        self.txt_prefix.setAcceptRichText(False)
        self.txt_prefix.setPlaceholderText(
            "人物・服装・画風など、シーンをまたいで変わってほしくない情報。\n"
            "例) subject_definitions:\n    <Subject 1> is …")
        pv.addWidget(self.txt_prefix)
        left.addWidget(box_prefix)

        # 左下: シーン一覧 / シーン編集
        inner = QSplitter(Qt.Horizontal)
        inner.addWidget(self._build_scene_list_pane())
        inner.addWidget(self._build_scene_edit_pane())
        inner.setStretchFactor(1, 1)
        inner.setSizes([230, 560])
        left.addWidget(inner)
        left.setStretchFactor(1, 1)
        left.setSizes([110, 560])

        main.addWidget(left)
        main.addWidget(self._build_refs_pane())
        main.setStretchFactor(0, 1)
        main.setSizes([830, 330])
        v.addWidget(main)
        return page

    def _build_scene_list_pane(self) -> QWidget:
        w = QWidget()
        lv = QVBoxLayout(w)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("シーン"))
        self.lst_scenes = QListWidget()
        self.lst_scenes.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lst_scenes.currentRowChanged.connect(self._on_scene_selected)
        lv.addWidget(self.lst_scenes, stretch=1)
        row = QHBoxLayout()
        for text, tip, slot in (
                ("+", "シーンを追加", self._add_scene),
                ("-", "選択シーンを削除", self._del_scene),
                ("複製", "選択シーンを複製", self._dup_scene),
                ("↑", "上へ移動", lambda: self._move_scene(-1)),
                ("↓", "下へ移動", lambda: self._move_scene(1))):
            b = QPushButton(text)
            b.setToolTip(tip)
            if len(text) == 1:
                b.setFixedWidth(30)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        lv.addLayout(row)
        return w

    def _build_scene_edit_pane(self) -> QWidget:
        w = QWidget()
        rv = QVBoxLayout(w)
        rv.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.ed_scene_id = QLineEdit()
        self.ed_scene_id.setToolTip("チェックポイント名（省略時は自動）")
        self.ed_scene_id.textChanged.connect(self._scene_field_changed)
        form.addRow("ID", self.ed_scene_id)

        len_row = QHBoxLayout()
        self.sp_scene_len = QDoubleSpinBox()
        self.sp_scene_len.setRange(0.3, 149.6)
        self.sp_scene_len.setDecimals(1)
        self.sp_scene_len.setSingleStep(0.5)
        self.sp_scene_len.setSuffix(" 秒")
        self.sp_scene_len.setToolTip(
            "このシーンの生成長。24fps の 17n+5 フレームへ切り上げられます")
        self.sp_scene_len.valueChanged.connect(self._scene_field_changed)
        self.lbl_scene_frames = QLabel("")
        self.lbl_scene_frames.setStyleSheet("color:#888;")
        len_row.addWidget(self.sp_scene_len)
        len_row.addWidget(self.lbl_scene_frames)
        len_row.addStretch(1)
        form.addRow("長さ", len_row)

        steps_row = QHBoxLayout()
        self.sp_scene_steps = QSpinBox()
        self.sp_scene_steps.setRange(0, 100)
        self.sp_scene_steps.setSpecialValueText("共通設定に従う")
        self.sp_scene_steps.setToolTip(
            "0 = 「つなぎ・音声」タブの既定 steps を使う")
        self.sp_scene_steps.valueChanged.connect(self._scene_field_changed)
        steps_row.addWidget(self.sp_scene_steps)
        steps_row.addSpacing(12)
        steps_row.addWidget(QLabel("Seed"))
        self.ed_scene_seed = QLineEdit()
        self.ed_scene_seed.setPlaceholderText("空 = 自動")
        self.ed_scene_seed.textChanged.connect(self._scene_field_changed)
        steps_row.addWidget(self.ed_scene_seed, stretch=1)
        form.addRow("Steps", steps_row)
        rv.addLayout(form)

        # このシーンで使う参照素材のラベル（クリックでプロンプトへ挿入）。
        self.box_scene_refs = QGroupBox("このシーンの参照（クリックで挿入）")
        srv = QVBoxLayout(self.box_scene_refs)
        srv.setContentsMargins(6, 2, 6, 4)
        self._scene_ref_host = QWidget()
        self._scene_ref_flow = FlowLayout(self._scene_ref_host,
                                          hspacing=8, vspacing=4)
        srv.addWidget(self._scene_ref_host)
        rv.addWidget(self.box_scene_refs)

        rv.addWidget(QLabel("このシーンのプロンプト"))
        self.txt_scene_prompt = QTextEdit()
        self.txt_scene_prompt.setAcceptRichText(False)
        self.txt_scene_prompt.setPlaceholderText(
            "前シーンの動作・カメラ・照明を引き継ぐ形で書き始め、"
            "動作の途中で終えると繋がりが良くなります")
        self.txt_scene_prompt.textChanged.connect(self._scene_field_changed)
        rv.addWidget(self.txt_scene_prompt, stretch=1)
        return w

    def _build_refs_pane(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)

        # i2v: シーン1の開始フレーム画像
        self.box_first = QGroupBox("開始フレーム（シーン1）")
        fv = QVBoxLayout(self.box_first)
        self.ed_first_frame = QLineEdit()
        self.ed_first_frame.setReadOnly(True)
        self.ed_first_frame.setPlaceholderText("未選択")
        fv.addWidget(self.ed_first_frame)
        frow = QHBoxLayout()
        b_pick = QPushButton("参照…")
        b_pick.clicked.connect(self._pick_first_frame)
        b_clr = QPushButton("クリア")
        b_clr.clicked.connect(lambda: self._set_first_frame(""))
        frow.addWidget(b_pick)
        frow.addWidget(b_clr)
        frow.addStretch(1)
        fv.addLayout(frow)
        self.lbl_first_note = QLabel(
            "2シーン目以降は前シーンの動きを引き継ぐため画像は使いません")
        self.lbl_first_note.setWordWrap(True)
        self.lbl_first_note.setStyleSheet("color:#888;")
        fv.addWidget(self.lbl_first_note)
        v.addWidget(self.box_first)

        # r2v: 素材プール（チェック = 選択中シーンで使用）
        self.box_pool = QGroupBox("参照素材")
        pv = QVBoxLayout(self.box_pool)
        self.lbl_pool_note = QLabel(
            "チェック = 選択中のシーンで使う素材。"
            "各行の「使用」は、その素材を使うシーン番号です。")
        self.lbl_pool_note.setWordWrap(True)
        self.lbl_pool_note.setStyleSheet("color:#888;")
        pv.addWidget(self.lbl_pool_note)
        self.lst_refs: dict[str, QListWidget] = {}
        for kind, title, flt, maxn, _pfx in REF_KINDS:
            head = QHBoxLayout()
            head.addWidget(QLabel(f"{title}（最大{maxn}）"))
            head.addStretch(1)
            b_all = QPushButton("全シーンで使う")
            b_all.setToolTip(f"プール内の{title}素材を全シーンで使う")
            b_all.clicked.connect(lambda *_a, k=kind: self._use_in_all(k))
            head.addWidget(b_all)
            b_add = QPushButton("+")
            b_add.setFixedWidth(30)
            b_add.setToolTip(f"{title}素材を追加")
            b_add.clicked.connect(
                lambda *_a, k=kind, f=flt, n=maxn: self._add_ref(k, f, n))
            head.addWidget(b_add)
            pv.addLayout(head)
            lst = QListWidget()
            # 行ウィジェットをリスト幅いっぱいに広げるためリサイズを監視。
            lst.viewport().installEventFilter(self)
            pv.addWidget(lst, stretch=1)
            self.lst_refs[kind] = lst
        v.addWidget(self.box_pool, stretch=1)

        self.lbl_refs_off = QLabel(
            "この種別では参照素材を使いません。")
        self.lbl_refs_off.setWordWrap(True)
        self.lbl_refs_off.setStyleSheet("color:#888;")
        v.addWidget(self.lbl_refs_off)
        return w

    # ----- tab: chain / audio ---------------------------------------------
    def _build_chain_tab(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)

        box_join = QGroupBox("シーンのつなぎ")
        f1 = QFormLayout(box_join)
        self.cb_context = WideComboBox()
        for n in CONTEXT_LENGTHS:
            self.cb_context.addItem(f"{n} フレーム（{n / FPS:.2f} 秒）", n)
        self.cb_context.setCurrentIndex(CONTEXT_LENGTHS.index(22))
        self.cb_context.setToolTip(
            "前シーンから引き継ぐ動きの文脈。長いほど繋ぎが自然になりますが、"
            "その分だけ2シーン目以降の実尺が短くなります")
        self.cb_context.currentIndexChanged.connect(self._update_summary)
        f1.addRow("引き継ぎフレーム数", self.cb_context)
        v.addWidget(box_join)

        box_audio = QGroupBox("音声")
        f2 = QFormLayout(box_audio)
        self.cb_audio_mode = WideComboBox()
        for token, label in AUDIO_MODES:
            self.cb_audio_mode.addItem(label, token)
        self.cb_audio_mode.currentIndexChanged.connect(
            self._on_audio_mode_changed)
        f2.addRow("音声モード", self.cb_audio_mode)
        arow = QHBoxLayout()
        self.ed_audio_file = QLineEdit()
        self.ed_audio_file.setReadOnly(True)
        self.ed_audio_file.setPlaceholderText("未選択")
        self.btn_audio_file = QPushButton("参照…")
        self.btn_audio_file.clicked.connect(self._pick_audio)
        arow.addWidget(self.ed_audio_file, stretch=1)
        arow.addWidget(self.btn_audio_file)
        self.lbl_audio_file = QLabel("外部音源")
        f2.addRow(self.lbl_audio_file, arow)
        self.sp_audio_context = QSpinBox()
        self.sp_audio_context.setRange(0, 128)
        self.sp_audio_context.setValue(22)
        self.sp_audio_context.setFixedWidth(120)
        self.sp_audio_context.setToolTip("音声の引き継ぎ長（既定22）")
        f2.addRow("音声の引き継ぎ", self.sp_audio_context)
        v.addWidget(box_audio)

        box_def = QGroupBox("シーンの既定値（シーン側で未指定のときに使われます）")
        f3 = QFormLayout(box_def)
        self.sp_def_steps = QSpinBox()
        self.sp_def_steps.setRange(1, 100)
        self.sp_def_steps.setValue(20)
        self.sp_def_steps.setFixedWidth(120)
        f3.addRow("既定 Steps", self.sp_def_steps)
        self.ed_base_seed = QLineEdit("0")
        self.ed_base_seed.setFixedWidth(220)
        self.ed_base_seed.setToolTip(
            "seed 未指定のシーンは、この値とシーン番号から自動で決まります")
        f3.addRow("ベース seed", self.ed_base_seed)
        v.addWidget(box_def)

        note = QLabel(
            "解像度・モデル・LoRA・sampler・Sigma Shift・EasyCache は"
            "メインウィンドウの設定がチェーン全体に適用されます"
            "（シーンごとには変えられません）。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888;")
        v.addWidget(note)
        v.addStretch(1)
        return page

    # ----- tab: output -----------------------------------------------------
    def _build_output_tab(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)

        box_out = QGroupBox("出力")
        f = QFormLayout(box_out)
        self.ed_final_name = QLineEdit("final")
        self.ed_final_name.setFixedWidth(260)
        f.addRow("最終ファイル名", self.ed_final_name)
        self.sp_crf = QSpinBox()
        self.sp_crf.setRange(0, 51)
        self.sp_crf.setValue(18)
        self.sp_crf.setFixedWidth(120)
        self.sp_crf.setToolTip("各シーンの保存画質（小さいほど高画質・大容量）")
        f.addRow("シーン保存画質 (CRF)", self.sp_crf)
        self.sp_bitrate = QSpinBox()
        self.sp_bitrate.setRange(64, 512)
        self.sp_bitrate.setValue(256)
        self.sp_bitrate.setSuffix(" kbps")
        self.sp_bitrate.setFixedWidth(120)
        f.addRow("最終音声ビットレート", self.sp_bitrate)
        v.addWidget(box_out)

        box_resume = QGroupBox("再開・部分生成")
        f2 = QFormLayout(box_resume)
        self.ed_scene_range = QLineEdit()
        self.ed_scene_range.setFixedWidth(260)
        self.ed_scene_range.setPlaceholderText("空 = 全シーン（例: 3 や 3:8）")
        self.ed_scene_range.setToolTip(
            "途中のシーンだけ作り直すときに使います。開始が2以上の場合、"
            "直前シーンのチェックポイントが必要です")
        f2.addRow("シーン範囲", self.ed_scene_range)
        self.chk_keep_ckpt = QCheckBox(
            "各シーンのチェックポイントを残す（再開・部分やり直しに必要）")
        self.chk_keep_ckpt.setChecked(True)
        f2.addRow(self.chk_keep_ckpt)
        v.addWidget(box_resume)

        note = QLabel(
            "チェーンは1回の生成で全シーンを順に処理します。"
            "ディスクには シーンごとの latent・音声・mp4 が保存されるため、"
            "長いチェーンでは数十GBになることがあります。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888;")
        v.addWidget(note)
        v.addStretch(1)
        return page

    # ----- scene list ------------------------------------------------------
    def _current_scene(self) -> dict | None:
        row = self.lst_scenes.currentRow()
        return self.scenes[row] if 0 <= row < len(self.scenes) else None

    def _reload_scene_list(self) -> None:
        cur = self.lst_scenes.currentRow()
        self.lst_scenes.blockSignals(True)
        self.lst_scenes.clear()
        for i, s in enumerate(self.scenes, 1):
            frames = frames_for_seconds(float(s["duration_seconds"]))
            marks = []
            if s.get("first_frame"):
                marks.append("画")
            if s.get("refs"):
                marks.append(f"参{len(s['refs'])}")
            tail = ("  [" + " ".join(marks) + "]") if marks else ""
            self.lst_scenes.addItem(
                QListWidgetItem(f"{i}. {s['id']}   {frames / FPS:.1f}s{tail}"))
        # 選択の復元まで signals を止める。ここで currentRowChanged が飛ぶと
        # 編集中でも _on_scene_selected がプロンプト欄を再設定してしまい、
        # カーソル位置が先頭へ戻る（入力・タグ挿入の位置がずれる）。
        if 0 <= cur < self.lst_scenes.count():
            self.lst_scenes.setCurrentRow(cur)
        self.lst_scenes.blockSignals(False)
        self._update_summary()

    def _select_scene(self, row: int) -> None:
        """シーンを選び、編集ペインを必ず作り直す。

        行番号が変わらないと currentRowChanged が飛ばないため（例: 先頭を
        削除して再び先頭を選ぶ）、明示的に反映する。
        """
        if not (0 <= row < len(self.scenes)):
            return
        self.lst_scenes.setCurrentRow(row)
        self._on_scene_selected(row)

    def _on_scene_selected(self, row: int) -> None:
        if not (0 <= row < len(self.scenes)):
            return
        s = self.scenes[row]
        self._loading = True
        self.ed_scene_id.setText(s["id"])
        self.sp_scene_len.setValue(float(s["duration_seconds"]))
        self.sp_scene_steps.setValue(int(s["steps"]))
        self.ed_scene_seed.setText(str(s["seed"]))
        self.txt_scene_prompt.setPlainText(s["prompt"])
        self.ed_first_frame.setText(str(s.get("first_frame", "")))
        self._loading = False
        self._update_frame_label()
        self._reload_ref_lists()
        self._reload_scene_ref_labels()
        self._update_first_frame_box()

    def _scene_field_changed(self, *_a) -> None:
        if self._loading:
            return
        row = self.lst_scenes.currentRow()
        if not (0 <= row < len(self.scenes)):
            return
        self.scenes[row].update(
            id=self.ed_scene_id.text().strip() or f"clip_{row + 1:04d}",
            duration_seconds=float(self.sp_scene_len.value()),
            steps=int(self.sp_scene_steps.value()),
            seed=self.ed_scene_seed.text().strip(),
            prompt=self.txt_scene_prompt.toPlainText())
        self._update_frame_label()
        self._reload_scene_list()

    def _update_frame_label(self) -> None:
        frames = frames_for_seconds(float(self.sp_scene_len.value()))
        self.lbl_scene_frames.setText(
            f"→ {frames} フレーム（{frames / FPS:.2f} 秒）")

    def _add_scene(self) -> None:
        self.scenes.append(_new_scene(len(self.scenes) + 1))
        self._reload_scene_list()
        self._select_scene(len(self.scenes) - 1)

    def _del_scene(self) -> None:
        row = self.lst_scenes.currentRow()
        if len(self.scenes) <= 1 or not (0 <= row < len(self.scenes)):
            return
        self.scenes.pop(row)
        self._reload_scene_list()
        self._select_scene(min(row, len(self.scenes) - 1))

    def _dup_scene(self) -> None:
        row = self.lst_scenes.currentRow()
        if not (0 <= row < len(self.scenes)):
            return
        s = dict(self.scenes[row])
        s["refs"] = list(s.get("refs", []))
        s["id"] = f"clip_{len(self.scenes) + 1:04d}"
        self.scenes.insert(row + 1, s)
        self._reload_scene_list()
        self._select_scene(row + 1)

    def _move_scene(self, delta: int) -> None:
        row = self.lst_scenes.currentRow()
        new = row + delta
        if not (0 <= row < len(self.scenes) and 0 <= new < len(self.scenes)):
            return
        self.scenes[row], self.scenes[new] = self.scenes[new], self.scenes[row]
        self._reload_scene_list()
        self._select_scene(new)

    # ----- first frame (i2v) -----------------------------------------------
    def _update_first_frame_box(self) -> None:
        is_i2v = self.cb_chain_type.currentData() == "i2v"
        first_selected = self.lst_scenes.currentRow() == 0
        self.box_first.setVisible(is_i2v)
        self.box_first.setEnabled(first_selected)
        if is_i2v and not first_selected:
            self.lbl_first_note.setText(
                "開始フレームはシーン1でだけ指定できます"
                "（2シーン目以降は前シーンの動きを引き継ぎます）")
        else:
            self.lbl_first_note.setText(
                "2シーン目以降は前シーンの動きを引き継ぐため画像は使いません")

    def _pick_first_frame(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "開始フレーム画像を選択", "", _IMAGE_FILTER)
        if path:
            self._set_first_frame(path)

    def _set_first_frame(self, path: str) -> None:
        s = self._current_scene()
        if s is None:
            return
        s["first_frame"] = path
        self.ed_first_frame.setText(path)
        self._reload_scene_list()

    # ----- scene reference labels ------------------------------------------
    # プロンプト内のタグ名。有効な素材だけが種別ごとに 1 から採番される
    # （Contex Loop 側と同じ規則）。
    _TAG_NAMES = {"image": "Picture", "video": "Video", "audio": "Audio"}
    _THUMB_H = 30          # 小サムネイルの高さ（テキスト2行ぶん）

    def _scene_ref_tags(self, scene: dict) -> list[tuple[str, dict, str]]:
        """(タグ文字列, ref, kind) のリスト。並びは 画像→動画→音声。"""
        used = set(scene.get("refs", []))
        out = []
        for kind, *_ in REF_KINDS:
            n = 0
            for ref in self.refs[kind]:
                if ref["id"] not in used:
                    continue
                n += 1
                out.append((f"<{self._TAG_NAMES[kind]} {n}>", ref, kind))
        return out

    def _clear_scene_ref_labels(self) -> None:
        while self._scene_ref_flow.count():
            item = self._scene_ref_flow.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _reload_scene_ref_labels(self) -> None:
        if not hasattr(self, "_scene_ref_flow"):
            return
        self._clear_scene_ref_labels()
        scene = self._current_scene()
        if scene is None or self.cb_chain_type.currentData() != "r2v":
            self._scene_ref_host.setVisible(False)
            return
        entries = self._scene_ref_tags(scene)
        self._scene_ref_host.setVisible(True)
        if not entries:
            hint = QLabel("（右の一覧でチェックすると、ここに参照が並びます）")
            hint.setStyleSheet("color:#888;")
            self._scene_ref_flow.addWidget(hint)
            return
        for tag, ref, kind in entries:
            self._scene_ref_flow.addWidget(self._make_ref_chip(tag, ref, kind))

    def _make_ref_chip(self, tag: str, ref: dict, kind: str) -> QWidget:
        chip = QWidget()
        chip.setToolTip(f"{Path(ref['path']).name}\nクリックで {tag} を挿入")
        chip.setCursor(Qt.PointingHandCursor)
        chip.setStyleSheet(
            "QWidget { background:#eef2f7; border:1px solid #c6d0dc;"
            " border-radius:3px; }"
            "QLabel { border:0; background:transparent; }")
        h = QHBoxLayout(chip)
        h.setContentsMargins(4, 2, 6, 2)
        h.setSpacing(5)
        if kind in ("image", "video"):
            thumb = QLabel()
            pix = self._thumbnail(kind, ref["path"])
            if pix is not None and not pix.isNull():
                thumb.setPixmap(pix.scaledToHeight(
                    self._THUMB_H, Qt.SmoothTransformation))
            else:
                thumb.setFixedSize(self._THUMB_H, self._THUMB_H)
                thumb.setText("?")
                thumb.setAlignment(Qt.AlignCenter)
            # サムネイルの上だけホバーで拡大プレビューを出す。
            thumb._thumb_path = ref["path"]
            thumb._thumb_kind = kind
            thumb.installEventFilter(self)
            h.addWidget(thumb)
        label = QLabel(tag)
        label.setStyleSheet("color:#22456e;")
        h.addWidget(label)
        # クリックはチップ本体だけで拾う。子のラベルはマウスイベントを
        # 受け取らないので親へ伝播し、1回だけ挿入される（両方に付けると
        # ラベル→チップの順に2回発火してしまう）。
        chip._insert_tag = tag
        chip.installEventFilter(self)
        return chip

    # ----- reference pool ---------------------------------------------------
    def _scenes_using(self, ref_id: int) -> list[int]:
        return [i for i, s in enumerate(self.scenes, 1)
                if ref_id in s.get("refs", [])]

    def _reload_ref_lists(self) -> None:
        """素材プールを描き直す。各行は「使用チェック + ゴミ箱」の
        ウィジェットで、チェックは選択中シーンでの使用有無を表す。"""
        cur = self._current_scene()
        used = set(cur.get("refs", [])) if cur else set()
        trash = self.style().standardIcon(QStyle.SP_TrashIcon)
        for kind, *_ in REF_KINDS:
            lst = self.lst_refs[kind]
            lst.clear()
            for ref in self.refs[kind]:
                scenes = self._scenes_using(ref["id"])
                label = f"{Path(ref['path']).name}  @{ref['tag']}"
                if scenes:
                    label += "   使用: " + ",".join(map(str, scenes))
                item = QListWidgetItem()
                row = QWidget()
                h = QHBoxLayout(row)
                h.setContentsMargins(4, 1, 2, 1)
                h.setSpacing(4)
                # チェックはインジケータのみ。テキストは別ラベルにして、
                # そこだけホバーでプレビューを出す。
                cb = QCheckBox()
                cb.setChecked(ref["id"] in used)
                cb.setToolTip("このシーンで使う")
                cb.toggled.connect(
                    lambda on, rid=ref["id"]: self._set_ref_used(rid, on))
                h.addWidget(cb)
                text = QLabel(label)
                text.setToolTip(ref["path"])
                text.setCursor(Qt.PointingHandCursor)
                # クリックで使用トグル（プレビューは編集ペイン側で出す）。
                text._ref_path = ref["path"]
                text._ref_kind = kind
                text._ref_checkbox = cb
                text.installEventFilter(self)
                h.addWidget(text, stretch=1)
                if kind == "audio":
                    play = QPushButton()
                    play.setIcon(self.style().standardIcon(
                        QStyle.SP_MediaPlay))
                    play.setFlat(True)
                    play.setFixedWidth(26)
                    play.setToolTip("既定のプレーヤーで再生")
                    play.clicked.connect(
                        lambda *_a, path=ref["path"]: QDesktopServices.openUrl(
                            QUrl.fromLocalFile(path)))
                    h.addWidget(play)
                btn = QPushButton()
                btn.setIcon(trash)
                btn.setFlat(True)
                btn.setFixedWidth(26)
                btn.setToolTip("この素材をプールから削除")
                btn.clicked.connect(
                    lambda *_a, k=kind, rid=ref["id"]:
                        self._del_ref_by_id(k, rid))
                h.addWidget(btn)
                item.setSizeHint(QSize(0, row.sizeHint().height()))
                lst.addItem(item)
                lst.setItemWidget(item, row)
            self._fit_ref_rows(lst)
            # 初回はレイアウト確定前なので、次のイベントループでもう一度。
            QTimer.singleShot(0, lambda w=lst: self._fit_ref_rows(w))

    @staticmethod
    def _fit_ref_rows(lst: QListWidget) -> None:
        """行をリストの表示幅いっぱいに広げる（ゴミ箱を右端に揃えるため）。"""
        width = lst.viewport().width()
        for i in range(lst.count()):
            item = lst.item(i)
            item.setSizeHint(QSize(width, item.sizeHint().height()))

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt signature)
        if event.type() == QEvent.Resize:
            for lst in self.lst_refs.values():
                if obj is lst.viewport():
                    self._fit_ref_rows(lst)
                    break
        # 参照素材のテキスト部分: ホバーでプレビュー、クリックで使用切替。
        path = getattr(obj, "_ref_path", None)
        if path is not None and event.type() == QEvent.MouseButtonRelease:
            cb = getattr(obj, "_ref_checkbox", None)
            if cb is not None:
                cb.setChecked(not cb.isChecked())
        # シーン編集ペインの参照ラベル: サムネのホバーで拡大、クリックで挿入。
        thumb = getattr(obj, "_thumb_path", None)
        if thumb is not None:
            if event.type() == QEvent.Enter:
                self._show_preview(getattr(obj, "_thumb_kind", ""), thumb)
            elif event.type() == QEvent.Leave:
                self._hide_preview()
        tag = getattr(obj, "_insert_tag", None)
        if tag is not None and event.type() == QEvent.MouseButtonRelease:
            self.txt_scene_prompt.insertPlainText(tag)
            self.txt_scene_prompt.setFocus()
        return super().eventFilter(obj, event)

    # ----- preview ---------------------------------------------------------
    def _show_preview(self, kind: str, path: str) -> None:
        if kind not in ("image", "video"):
            return
        if self._popup is None:
            self._popup = _PreviewPopup(self)
        pix = self._thumbnail(kind, path)
        if pix is None or pix.isNull():
            self._popup.show_text("プレビューを作れませんでした\n" + path)
        else:
            self._popup.show_pixmap(pix)

    def _hide_preview(self) -> None:
        if self._popup is not None:
            self._popup.hide()

    def _thumbnail(self, kind: str, path: str) -> QPixmap | None:
        """画像はそのまま、動画は先頭フレームを取り出してキャッシュする。"""
        cached = self._thumbs.get(path)
        if cached is not None:
            return cached
        pix = QPixmap()
        if kind == "image":
            pix.load(path)
        else:
            frame = self._video_frame(path)
            if frame:
                pix.load(frame)
        if pix.isNull():
            return None
        pix = pix.scaled(360, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._thumbs[path] = pix
        return pix

    @staticmethod
    def _video_frame(path: str) -> str:
        """動画の先頭フレームを PNG に書き出してパスを返す（失敗時は空）。

        GUI 側の venv には PyAV が無いため、バックエンド venv の python を
        1回だけ呼んで取り出す。
        """
        py = config.AppPaths().backend_python
        if not Path(py).exists():
            return ""
        out = Path(tempfile.gettempdir()) / (
            "scomv_thumb_%08x.png" % (abs(hash(path)) & 0xFFFFFFFF))
        if out.exists():
            return str(out)
        code = (
            "import av, sys\n"
            "c = av.open(sys.argv[1])\n"
            "f = next(c.decode(video=0))\n"
            "f.to_image().save(sys.argv[2])\n"
        )
        try:
            subprocess.run([str(py), "-c", code, path, str(out)],
                           timeout=20, capture_output=True,
                           creationflags=getattr(subprocess,
                                                 "CREATE_NO_WINDOW", 0))
        except Exception:  # noqa: BLE001
            return ""
        return str(out) if out.exists() else ""

    def _set_ref_used(self, ref_id: int, used: bool) -> None:
        s = self._current_scene()
        if s is None:
            return
        refs = [r for r in s.get("refs", []) if r != ref_id]
        if used:
            refs.append(ref_id)
        s["refs"] = refs
        # 行ウィジェットのシグナル内なので、描き直しは次のイベントループで。
        QTimer.singleShot(0, self._refresh_refs_and_list)

    def _refresh_refs_and_list(self) -> None:
        self._reload_ref_lists()
        self._reload_scene_ref_labels()
        self._reload_scene_list()

    def _add_ref(self, kind: str, flt: str, max_n: int) -> None:
        remain = max_n - len(self.refs[kind])
        if remain <= 0:
            return
        prefix = next(p for k, _t, _f, _n, p in REF_KINDS if k == kind)
        paths, _ = QFileDialog.getOpenFileNames(self, "参照素材を追加", "", flt)
        for p in paths[:remain]:
            self.refs[kind].append({
                "id": self._next_ref_id,
                "path": p,
                "tag": f"{prefix}_{len(self.refs[kind]) + 1}",
            })
            self._next_ref_id += 1
        self._reload_ref_lists()

    def _del_ref_by_id(self, kind: str, ref_id: int) -> None:
        self.refs[kind] = [r for r in self.refs[kind] if r["id"] != ref_id]
        for s in self.scenes:
            s["refs"] = [r for r in s.get("refs", []) if r != ref_id]
        QTimer.singleShot(0, self._refresh_refs_and_list)

    def _use_in_all(self, kind: str) -> None:
        """この種別の素材を全シーンで使う（一括チェック）。"""
        ids = [r["id"] for r in self.refs[kind]]
        for s in self.scenes:
            refs = list(s.get("refs", []))
            for i in ids:
                if i not in refs:
                    refs.append(i)
            s["refs"] = refs
        self._reload_ref_lists()
        self._reload_scene_list()

    # ----- reactions -------------------------------------------------------
    def _on_chain_type_changed(self, *_a) -> None:
        kind = self.cb_chain_type.currentData()
        self.box_pool.setVisible(kind == "r2v")
        self.box_scene_refs.setVisible(kind == "r2v")
        self.lbl_refs_off.setVisible(kind == "t2v")
        self._reload_scene_ref_labels()
        self._update_first_frame_box()
        self._update_summary()

    def _on_audio_mode_changed(self, *_a) -> None:
        needs_file = self.cb_audio_mode.currentData() in (
            "source_track", "source_plus_timeline")
        for w in (self.ed_audio_file, self.btn_audio_file, self.lbl_audio_file):
            w.setEnabled(needs_file)

    def _pick_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "外部音源を選択", "", _AUDIO_FILTER)
        if path:
            self.ed_audio_file.setText(path)

    def _update_summary(self, *_a) -> None:
        ctx = int(self.cb_context.currentData() or 22)
        raw = [frames_for_seconds(float(s["duration_seconds"]))
               for s in self.scenes]
        total_raw = sum(raw)
        delivered = total_raw - ctx * (len(raw) - 1)
        self.lbl_summary.setText(
            f"{len(raw)} シーン / 実尺 {delivered / FPS:.1f} 秒")
        short = [i for i, f in enumerate(raw[:-1], 1) if f <= ctx]
        warn = ""
        if short:
            warn = ("　⚠ シーン " + ", ".join(map(str, short))
                    + " は引き継ぎフレーム数以下のため生成できません")
        self.lbl_footer.setText(
            f"生成 {total_raw} フレーム − 引き継ぎ {ctx}×{len(raw) - 1} = "
            f"実尺 {delivered} フレーム（{delivered / FPS:.2f} 秒 / 24fps）"
            + warn)
        self.lbl_footer.setStyleSheet(
            "color:#c33;" if short else "color:#888;")
        if not self._loading:
            self.changed.emit()

    # ----- import / export -------------------------------------------------
    PLAN_FORMAT = "scomv_chain_plan_v1"
    _PLAN_FILTER = "チェーン設定 (*.json);;すべて (*.*)"

    def export_plan(self) -> None:
        """現在の設定を JSON ファイルへ保存する。"""
        name = (self.ed_run_name.text().strip() or "h3_chain") + ".json"
        path, _ = QFileDialog.getSaveFileName(
            self, "チェーン設定をエクスポート", name, self._PLAN_FILTER)
        if not path:
            return
        doc = {"format": self.PLAN_FORMAT, "plan": self.plan()}
        try:
            Path(path).write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self, "エクスポート",
                                f"保存に失敗しました:\n{e}")
            return
        QMessageBox.information(self, "エクスポート",
                                f"保存しました:\n{path}")

    def import_plan(self) -> None:
        """JSON ファイルから設定を読み込む（現在の内容は置き換わる）。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "チェーン設定をインポート", "", self._PLAN_FILTER)
        if not path:
            return
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "インポート",
                                f"読み込めませんでした:\n{e}")
            return
        plan = doc.get("plan") if isinstance(doc, dict) else None
        if not isinstance(plan, dict) or not plan.get("shots"):
            QMessageBox.warning(
                self, "インポート",
                "チェーン設定のファイルではないようです。")
            return
        if QMessageBox.question(
                self, "インポート",
                "現在のチェーン設定を、読み込んだ内容で置き換えます。\n"
                "よろしいですか？",
                QMessageBox.Yes | QMessageBox.Cancel) != QMessageBox.Yes:
            return
        self.load_plan(plan)
        # 参照素材や開始フレームは絶対パスで保存されるため、別マシンや
        # 移動後は見つからないことがある。欠けているものを知らせる。
        missing = self._missing_files()
        if missing:
            QMessageBox.warning(
                self, "インポート",
                "読み込みました。ただし次のファイルが見つかりません:\n"
                + "\n".join(missing[:10])
                + ("\n…" if len(missing) > 10 else "")
                + "\n\n該当の素材を選び直してください。")
        self.changed.emit()

    def _missing_files(self) -> list[str]:
        out = []
        for s in self.scenes:
            f = str(s.get("first_frame") or "")
            if f and not Path(f).exists():
                out.append(f)
        for kind, *_ in REF_KINDS:
            for ref in self.refs[kind]:
                if ref["path"] and not Path(ref["path"]).exists():
                    out.append(ref["path"])
        audio = self.ed_audio_file.text().strip()
        if audio and not Path(audio).exists():
            out.append(audio)
        return out

    # ----- plan ------------------------------------------------------------
    @staticmethod
    def _scenes_selector(indices: list[int], total: int) -> str:
        """[1,2,3,5] -> "1:3,5"（全シーンなら空文字 = 常時有効）。"""
        if not indices or len(indices) == total:
            return ""
        parts, start, prev = [], indices[0], indices[0]
        for n in indices[1:] + [None]:
            if n == prev + 1:
                prev = n
                continue
            parts.append(str(start) if start == prev else f"{start}:{prev}")
            if n is not None:
                start = prev = n
        return ",".join(parts)

    def load_plan(self, plan: dict) -> None:
        """plan() が返した辞書から設定を復元する。"""
        self._loading = True
        self.ed_run_name.setText(str(plan.get("run_name") or "h3_chain"))
        i = self.cb_chain_type.findData(plan.get("chain_type") or "t2v")
        if i >= 0:
            self.cb_chain_type.setCurrentIndex(i)
        self.txt_prefix.setPlainText(str(plan.get("prompt_prefix") or ""))
        ci = self.cb_context.findData(int(plan.get("context_length") or 22))
        if ci >= 0:
            self.cb_context.setCurrentIndex(ci)
        ai = self.cb_audio_mode.findData(plan.get("audio_mode"))
        if ai >= 0:
            self.cb_audio_mode.setCurrentIndex(ai)
        self.ed_audio_file.setText(str(plan.get("audio_file") or ""))
        self.sp_audio_context.setValue(
            int(plan.get("audio_context_length") or 22))
        self.sp_def_steps.setValue(int(plan.get("default_steps") or 20))
        self.ed_base_seed.setText(str(plan.get("base_seed") or "0"))
        self.sp_crf.setValue(int(plan.get("segment_crf") or 18))
        self.ed_final_name.setText(str(plan.get("final_name") or "final"))
        self.sp_bitrate.setValue(int(plan.get("audio_bitrate") or 256))
        self.ed_scene_range.setText(str(plan.get("scene_range") or ""))

        self.scenes = [dict(s) for s in plan.get("shots") or []] or [
            _new_scene(1)]
        for s in self.scenes:
            s.setdefault("refs", [])
            s.setdefault("first_frame", "")
        self.refs = {k: [] for k, *_ in REF_KINDS}
        max_id = 0
        for ref in plan.get("references") or []:
            kind = ref.get("kind")
            if kind not in self.refs:
                continue
            rid = int(ref.get("id") or 0) or (max_id + 1)
            max_id = max(max_id, rid)
            self.refs[kind].append({"id": rid, "path": ref.get("path", ""),
                                    "tag": ref.get("tag", "")})
        self._next_ref_id = max_id + 1
        self._loading = False
        self._reload_scene_list()
        self._select_scene(0)
        self._on_chain_type_changed()
        self._on_audio_mode_changed()

    def plan(self) -> dict:
        """現在の設定を Contex Loop の Plan 相当の辞書で返す（未接続）。"""
        total = len(self.scenes)
        references = []
        for kind, *_ in REF_KINDS:
            for ref in self.refs[kind]:
                references.append({
                    "kind": kind,
                    "id": ref["id"],
                    "path": ref["path"],
                    "name": "",          # アップロード後の名前が入る
                    "tag": ref["tag"],
                    "scenes": self._scenes_selector(
                        self._scenes_using(ref["id"]), total),
                })
        return {
            "run_name": self.ed_run_name.text().strip() or "h3_chain",
            "chain_type": self.cb_chain_type.currentData(),
            "prompt_prefix": self.txt_prefix.toPlainText(),
            "context_length": int(self.cb_context.currentData() or 22),
            "audio_mode": self.cb_audio_mode.currentData(),
            "audio_file": self.ed_audio_file.text(),
            "audio_context_length": int(self.sp_audio_context.value()),
            "default_steps": int(self.sp_def_steps.value()),
            "base_seed": self.ed_base_seed.text().strip() or "0",
            "segment_crf": int(self.sp_crf.value()),
            "final_name": self.ed_final_name.text().strip() or "final",
            "audio_bitrate": int(self.sp_bitrate.value()),
            "scene_range": self.ed_scene_range.text().strip(),
            "references": references,
            "shots": [dict(s) for s in self.scenes],
        }

"""モデルマージ管理ウィンドウ（非モーダル・2ペイン）。

左ペイン: 「＋ 新規作成」と作成済みマージモデルの一覧（各行: ゴミ箱 +
RAM状態マーク ●/○ + 名前。右クリックで名前の変更）。下部に「メモリ全解放」。
右ペイン: 新規作成時は編集可能なレシピエディタ（1行 = モデル + 相対比率、
量子化・省メモリオプション）。既存エントリ選択時は同じUIを読み取り専用で表示
し、「複製して新規作成」で内容を引き継いだ新規作成に移れる。

このウィンドウはレシピの管理だけを行い、実行はシグナル経由でメインウィンドウ
に依頼する。エントリ一覧とRAM状態は set_entries() で常にメイン側から供給される。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout, QInputDialog,
    QLabel, QListWidget, QListWidgetItem, QMenu, QMessageBox, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)

from .widgets import WideComboBox
from .window_state import bind_geometry

# Returns "anima" | "krea2" | "shared" | "unknown" for a diffusion file name.
FamilyFn = Callable[[str], str]

NOTE_TEXT = (
    "マージモデルは画像生成時にキャッシュに残っていなければ再マージが走ります。"
    "キャッシュ上のマージモデルはRAMが逼迫してきたときに使われていないモデル"
    "から自動で削除されます。"
)


class MergeDialog(QDialog):
    # Recipe execution / management requests, handled by the main window.
    merge_requested = Signal(object, str, bool)        # entries, quant, low_mem
    save_requested = Signal(object, str, bool, str)    # + filename
    delete_requested = Signal(int)                     # entry id
    rename_requested = Signal(int, str)                # entry id, new name
    free_memory_requested = Signal()

    def __init__(self, models: list[str], family_fn: FamilyFn,
                 model_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("モデルマージ")
        self.setMinimumWidth(780)
        bind_geometry(self, "merge")
        self._models = list(models)
        self._family = family_fn
        self._model_dir = model_dir
        self._rows: list[dict] = []
        self._entries: list[dict] = []
        self._built_ids: set[int] = set()
        self._editable = True

        root = QVBoxLayout(self)
        note = QLabel(NOTE_TEXT)
        note.setWordWrap(True)
        root.addWidget(note)
        panes = QHBoxLayout()
        root.addLayout(panes, stretch=1)

        # マージ実行中の進捗（メインウィンドウから set_merge_* で供給）。
        # 非表示でも高さを確保する — 表示/非表示で最小サイズが変わると
        # QWindowsWindow::setGeometry の警告が出るため（XYZ ウィンドウと同様）。
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        sp = self.progress.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.progress.setSizePolicy(sp)
        root.addWidget(self.progress)

        # ----- left pane: entry list --------------------------------------
        left = QVBoxLayout()
        self.lst = QListWidget()
        self.lst.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lst.customContextMenuRequested.connect(self._on_context_menu)
        self.lst.currentRowChanged.connect(self._on_row_changed)
        left.addWidget(self.lst, stretch=1)
        btn_free = QPushButton("メモリ全解放")
        btn_free.setToolTip(
            "バックエンドのキャッシュとロード済みモデルをすべて解放します。"
            "マージモデルの一覧は残り、次回使用時に自動で再構築されます")
        btn_free.clicked.connect(lambda: self.free_memory_requested.emit())
        left.addWidget(btn_free)
        panes.addLayout(left, 1)

        # ----- right pane: recipe editor / read-only view ------------------
        right = QVBoxLayout()
        right.addWidget(QLabel(
            "マージするモデルと比率（相対値）を指定してください。"
            "比率は合計に対する割合として扱われます。"))

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(4)
        right.addLayout(self._rows_layout)

        self.btn_add = QPushButton("＋ モデルを追加")
        self.btn_add.clicked.connect(lambda: (self._add_row(), self._refresh()))
        add_row = QHBoxLayout()
        add_row.addWidget(self.btn_add)
        add_row.addStretch(1)
        right.addLayout(add_row)

        self.lbl_warn = QLabel("アーキテクチャの違うモデルが選択されています")
        self.lbl_warn.setStyleSheet("color: red;")
        self.lbl_warn.hide()
        right.addWidget(self.lbl_warn)

        self.lbl_info = QLabel("")
        right.addWidget(self.lbl_info)

        self.chk_quant = QCheckBox("量子化する")
        self.chk_quant.setToolTip(
            "マージ結果の重みを低精度化してメモリ/VRAM を節約します")
        self.cb_quant = WideComboBox()
        self.cb_quant.addItem("fp8", "fp8")
        self.cb_quant.addItem("int8 ConvRot", "int8_convrot")
        self.cb_quant.addItem("int4 ConvRot（ハイブリッド）", "int4_convrot")
        self.cb_quant.addItem("int4 ConvRot（完全4bit）", "int4_convrot_full")
        self.cb_quant.setItemData(
            0, "Linear 層を float8_e4m3fn（per-tensor scale）に量子化。"
               "サイズ約1/2、わずかに精度低下", Qt.ToolTipRole)
        self.cb_quant.setItemData(
            1, "回転（Hadamard）ベースの INT8。ほぼ無損失で fp8 より高品質、"
               "RTX 30 系以上の INT8 Tensor Core で高速。"
               "DiT ブロックのみ量子化（sensitive 層は高精度維持）",
            Qt.ToolTipRole)
        self.cb_quant.setItemData(
            2, "回転（Hadamard）ベースの INT4/INT8 ハイブリッド（convrot_w4a4）。"
               "Q/K/V/up 系は int4、誤差の大きい出力側 projection は int8、"
               "行列積は int8 で実行（速度は int8 と同等・サイズはより小さい）。"
               "int8 よりわずかに精度低下（adaLN 等の sensitive 層は高精度維持）",
            Qt.ToolTipRole)
        self.cb_quant.setItemData(
            3, "対象の Linear 層をすべて int4 にする最小サイズ版。"
               "krea2 級の大モデル向き。anima 等の小モデルでは劣化が"
               "目立つためハイブリッド推奨（adaLN 等の sensitive 層は"
               "高精度維持）", Qt.ToolTipRole)
        self.cb_quant.setEnabled(False)
        self.chk_quant.toggled.connect(self._sync_quant_enabled)
        quant_row = QHBoxLayout()
        quant_row.addWidget(self.chk_quant)
        quant_row.addWidget(self.cb_quant)
        quant_row.addStretch(1)
        right.addLayout(quant_row)

        self.chk_lowmem = QCheckBox("省メモリモード（1モデルずつ逐次マージ）")
        self.chk_lowmem.setToolTip(
            "モデルを1個ずつ読み込んで順に合成します。ピークメモリが"
            "「結果1個分＋ソース1個分」程度まで下がる代わり、途中経過を"
            "bf16 で保持するためごくわずかに精度が落ちます。"
            "OFF は全モデルを同時に開いて fp32 で一括合成します（最高精度）")
        right.addWidget(self.chk_lowmem)

        right.addStretch(1)

        btns = QHBoxLayout()
        self.btn_merge = QPushButton("マージ")
        self.btn_merge.setToolTip(
            "この構成を一覧に登録し、マージモデルをメインメモリ上に構築します")
        self.btn_merge.clicked.connect(self._on_merge)
        self.btn_save = QPushButton("ファイルとして保存")
        self.btn_save.setToolTip(
            "表示中の構成でマージを実行し、models/diffusion_models に "
            "safetensors として保存します")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_dup = QPushButton("複製して新規作成")
        self.btn_dup.setToolTip("この構成をコピーした新規作成に切り替えます")
        self.btn_dup.clicked.connect(self._on_duplicate)
        btn_close = QPushButton("閉じる")
        btn_close.clicked.connect(self.close)
        btns.addStretch(1)
        btns.addWidget(self.btn_merge)
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_dup)
        btns.addWidget(btn_close)
        right.addLayout(btns)
        panes.addLayout(right, 2)

        while len(self._rows) < 2:
            self._add_row()
        self.set_entries([], set())

    # ----- left pane -------------------------------------------------------
    def set_entries(self, entries: list[dict], built_ids: set[int]) -> None:
        """Rebuild the entry list (called by the main window on any change)."""
        self._entries = [dict(e) for e in entries]
        self._built_ids = set(built_ids)
        selected = self.selected_entry_id()
        self.lst.blockSignals(True)
        self.lst.clear()
        self.lst.addItem(QListWidgetItem("＋ 新規作成"))
        for e in self._entries:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, int(e["id"]))
            self.lst.addItem(item)
            self.lst.setItemWidget(item, self._make_row_widget(e))
        row = 0
        if selected is not None:
            for i in range(1, self.lst.count()):
                if self.lst.item(i).data(Qt.UserRole) == selected:
                    row = i
        self.lst.setCurrentRow(row)
        self.lst.blockSignals(False)
        self._on_row_changed(self.lst.currentRow())

    def _make_row_widget(self, entry: dict) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 2, 4, 2)
        trash = QPushButton("🗑")
        trash.setFixedWidth(28)
        trash.setToolTip("このマージモデルを一覧から削除し、メモリからも解放します")
        trash.clicked.connect(
            lambda *_a, eid=int(entry["id"]): self._on_trash(eid))
        mark = "●" if int(entry["id"]) in self._built_ids else "○"
        lbl = QLabel(f"{mark} {entry['name']}")
        lbl.setToolTip("● = メモリ上に構築済み / ○ = 未構築（生成時に自動マージ）")
        h.addWidget(trash)
        h.addWidget(lbl, stretch=1)
        return w

    def selected_entry_id(self) -> Optional[int]:
        item = self.lst.currentItem()
        if item is None:
            return None
        data = item.data(Qt.UserRole)
        return int(data) if data is not None else None

    def select_entry(self, entry_id: int) -> None:
        for i in range(1, self.lst.count()):
            if self.lst.item(i).data(Qt.UserRole) == entry_id:
                self.lst.setCurrentRow(i)
                return

    def _entry_by_id(self, entry_id: int) -> Optional[dict]:
        for e in self._entries:
            if int(e["id"]) == entry_id:
                return e
        return None

    def _on_trash(self, entry_id: int) -> None:
        e = self._entry_by_id(entry_id)
        name = e["name"] if e else str(entry_id)
        res = QMessageBox.question(
            self, "削除の確認", f"「{name}」を一覧から削除しますか？")
        if res == QMessageBox.Yes:
            self.delete_requested.emit(entry_id)

    def _on_context_menu(self, pos) -> None:
        item = self.lst.itemAt(pos)
        if item is None or item.data(Qt.UserRole) is None:
            return  # no menu on 新規作成
        entry_id = int(item.data(Qt.UserRole))
        e = self._entry_by_id(entry_id)
        if e is None:
            return
        menu = QMenu(self)
        act = QAction("名前の変更", menu)
        menu.addAction(act)
        if menu.exec(self.lst.mapToGlobal(pos)) is act:
            name, ok = QInputDialog.getText(
                self, "名前の変更", "新しい名前:", text=e["name"])
            name = name.strip()
            if ok and name:
                self.rename_requested.emit(entry_id, name)

    def _on_row_changed(self, row: int) -> None:
        entry_id = self.selected_entry_id()
        if entry_id is None:
            self._set_editable(True)
            self._refresh()
            return
        e = self._entry_by_id(entry_id)
        if e is None:
            return
        self._load_recipe([(n, float(w)) for n, w in e["models"]],
                          str(e.get("quant", "")),
                          bool(e.get("low_memory", False)))
        self._set_editable(False)

    # ----- right pane: rows --------------------------------------------------
    def _add_row(self, name: str = "", weight: float = 1.0) -> None:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)

        combo = WideComboBox()
        combo.addItem("")
        combo.addItems(self._models)
        combo.setCurrentText(name)
        combo.currentTextChanged.connect(self._refresh)

        spin = QDoubleSpinBox()
        spin.setRange(0.01, 1000.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setValue(weight)
        spin.valueChanged.connect(self._refresh)

        pct = QLabel("")
        pct.setFixedWidth(44)
        pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        trash = QPushButton("🗑")
        trash.setFixedWidth(32)
        trash.setToolTip("この行を削除")
        trash.clicked.connect(lambda *_a, widget=w: self._remove_row(widget))
        if len(self._rows) < 2:
            # The first two rows are permanent: no trash icon. Keep its space
            # reserved so the combos/spinboxes line up across all rows.
            sp = trash.sizePolicy()
            sp.setRetainSizeWhenHidden(True)
            trash.setSizePolicy(sp)
            trash.hide()

        h.addWidget(combo, stretch=1)
        h.addWidget(spin)
        h.addWidget(pct)
        h.addWidget(trash)

        self._rows_layout.addWidget(w)
        self._rows.append({"widget": w, "combo": combo, "spin": spin,
                           "pct": pct, "trash": trash})

    def _remove_row(self, widget: QWidget) -> None:
        if len(self._rows) <= 2:
            return
        for row in self._rows:
            if row["widget"] is widget:
                self._rows.remove(row)
                self._rows_layout.removeWidget(widget)
                widget.deleteLater()
                break
        self._refresh()

    def _load_recipe(self, models: list[tuple[str, float]], quant: str,
                     low_memory: bool) -> None:
        """Fill the right pane with a recipe (used for view and duplicate)."""
        while len(self._rows) > max(2, len(models)):
            self._remove_row(self._rows[-1]["widget"])
        while len(self._rows) < max(2, len(models)):
            self._add_row()
        for row, (name, w) in zip(self._rows, models + [("", 1.0)] * 2):
            row["combo"].setCurrentText(name)
            row["spin"].setValue(float(w))
        self.chk_quant.setChecked(bool(quant))
        if quant:
            idx = self.cb_quant.findData(quant)
            if idx >= 0:
                self.cb_quant.setCurrentIndex(idx)
        self.chk_lowmem.setChecked(bool(low_memory))
        self._refresh()

    def _set_editable(self, editable: bool) -> None:
        self._editable = editable
        for row in self._rows:
            row["combo"].setEnabled(editable)
            row["spin"].setEnabled(editable)
            row["trash"].setEnabled(editable)
        self.btn_add.setEnabled(editable)
        self.chk_quant.setEnabled(editable)
        self.chk_lowmem.setEnabled(editable)
        self._sync_quant_enabled()
        self.btn_merge.setEnabled(editable)
        self.btn_dup.setVisible(not editable)

    def _sync_quant_enabled(self, *_a) -> None:
        self.cb_quant.setEnabled(self._editable and self.chk_quant.isChecked())

    # ----- state --------------------------------------------------------------
    def entries(self) -> list[tuple[str, float]]:
        """Selected (model, weight) pairs; rows with no model are skipped."""
        out = []
        for row in self._rows:
            name = row["combo"].currentText().strip()
            if name:
                out.append((name, float(row["spin"].value())))
        return out

    def quant_value(self) -> str:
        """Selected quantization mode: "" | "fp8" | "int8_convrot"
        | "int4_convrot" | "int4_convrot_full"."""
        if not self.chk_quant.isChecked():
            return ""
        return self.cb_quant.currentData() or ""

    def _refresh(self, *_args) -> None:
        entries = self.entries()
        total = sum(w for _n, w in entries)
        for row in self._rows:
            name = row["combo"].currentText().strip()
            w = float(row["spin"].value())
            row["pct"].setText(f"{w / total * 100:.0f}%"
                               if name and total > 0 else "")
            if row["trash"].isVisible():
                row["trash"].setEnabled(self._editable and len(self._rows) > 2)

        # Architecture check: warn when known families disagree (rows are
        # deliberately unfiltered, so mixing anima/krea2 files is possible).
        fams = {self._family(n) for n, _w in entries}
        self.lbl_warn.setVisible(len(fams & {"anima", "krea2"}) > 1)

        size = 0
        for n, _w in entries:
            try:
                size += (self._model_dir / n).stat().st_size
            except OSError:
                pass
        self.lbl_info.setText(
            f"選択モデル合計: {size / 1e9:.1f} GB" if size else "")

    # ----- actions --------------------------------------------------------------
    def _displayed_recipe(self) -> tuple[list[tuple[str, float]], str, bool]:
        """The recipe currently shown (editor draft or selected entry)."""
        entry_id = self.selected_entry_id()
        if entry_id is not None:
            e = self._entry_by_id(entry_id)
            if e is not None:
                return ([(n, float(w)) for n, w in e["models"]],
                        str(e.get("quant", "")),
                        bool(e.get("low_memory", False)))
        return self.entries(), self.quant_value(), self.chk_lowmem.isChecked()

    def _validate(self, entries) -> bool:
        # 1 model is fine: that is a quantize-only (or copy-only) build.
        if len(entries) < 1:
            QMessageBox.warning(self, "入力不足",
                                "モデルを1個以上選択してください。")
            return False
        return True

    def _on_merge(self) -> None:
        entries = self.entries()
        if not self._validate(entries):
            return
        self.merge_requested.emit(entries, self.quant_value(),
                                  self.chk_lowmem.isChecked())

    def _on_save(self) -> None:
        entries, quant, low_memory = self._displayed_recipe()
        if not self._validate(entries):
            return
        name, ok = QInputDialog.getText(
            self, "ファイルとして保存",
            "保存ファイル名（models/diffusion_models 内）:",
            text="merged.safetensors")
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if not name.endswith(".safetensors"):
            name += ".safetensors"
        if (self._model_dir / name).exists():
            res = QMessageBox.question(
                self, "上書き確認", f"{name} は既に存在します。上書きしますか？")
            if res != QMessageBox.Yes:
                return
        self.save_requested.emit(entries, quant, low_memory, name)

    def _on_duplicate(self) -> None:
        entries, quant, low_memory = self._displayed_recipe()
        self.lst.setCurrentRow(0)  # switch to 新規作成 (makes pane editable)
        self._load_recipe(entries, quant, low_memory)

    # ----- progress supplied by the main window ------------------------------
    def set_merge_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        if running:
            # 最初の進捗イベントが届くまではビジー（不定）表示。
            self.progress.setMaximum(0)
            self.progress.setValue(0)

    def set_merge_progress(self, value: int, maximum: int,
                           note: str = "") -> None:
        if not maximum:
            return
        self.progress.setVisible(True)
        self.progress.setMaximum(maximum)
        self.progress.setValue(value)
        self.progress.setFormat(f"{note} {value}/{maximum}".strip())

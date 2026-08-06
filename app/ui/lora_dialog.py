"""LoRA ブラウザウィンドウ（非モーダル・親なし）。

models/loras のファイルをサムネイル付きグリッドで一覧表示する。サムネと
メタ情報（トリガーワード・ベースモデル等）は SHA256 を元に civitai から
取得し userdata/lora_cache/ にキャッシュされる（app/lora.py）。ハッシュ
計算とネットワークアクセスはワーカースレッドで行い、グリッドはまず
ファイル名だけで即表示 → 取得できた順にサムネが埋まる。

適用そのものはメインウィンドウが行う（apply/remove シグナル）。適用中の
状態は set_applied() で常にメイン側から供給される。
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, QSize, QThread, QUrl, Signal, QObject
from PySide6.QtGui import (
    QBrush, QColor, QDesktopServices, QIcon, QPalette, QPixmap, QTextDocument,
)
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QScrollArea,
    QSplitter, QVBoxLayout, QWidget,
)

from .window_state import bind_geometry
from .. import config, lora, modelinfo

# ヘッダ判定の系統 → 一覧/詳細ペインでの表示名
_FAMILY_LABELS = {
    modelinfo.ANIMA: "anima",
    modelinfo.KREA2: "krea2",
    modelinfo.SDXL: "sdxl",
    modelinfo.OTHER: "対応外",
    modelinfo.UNKNOWN: "不明",
}

_ICON_SIZE = 144
_GRID_SIZE = QSize(168, 196)
_APPLIED_BG = QColor("#1a4d2e")  # ✓ 適用中バッジの背景（緑系）

# トリガーワードの衣装グループ区切り: カンマが2個以上連続（間にスペース可）。
# civitai は複数衣装の LoRA を "服A, tie, ,服B, pants,, 服C" のように登録する
# ことが多く、この区切りごとに別リンクへ分ける。
_GROUP_SEP = re.compile(r",\s*(?:,\s*)+")
_LINK_COLOR = "#22456e"        # 通常のリンク文字色（明るい背景に合わせた濃紺）
_LINK_HOVER = "#8a4b00"        # ホバー中のリンク文字色（どのワードが入るか強調）
# 各リンクは同一色のボックス（1セルのテーブル）で囲む。cellpadding で内側の
# 余白、cellspacing でボックス間の間隔を取り、ホバー時のみ背景色を変える。
_GROUP_BG = "#e7edf5"          # 全リンク共通の背景色（明るい）
_GROUP_BG_HOVER = "#ffe6a6"    # ホバー中の背景（明るい琥珀系）
# メインのプロンプトに現在挿入済みのグループ（もう一度押すと削除）を示す色。
_GROUP_BG_ACTIVE = "#cfe9d4"   # 挿入済みの背景（明るい緑）
_LINK_ACTIVE = "#1c5b32"       # 挿入済みの文字色


def split_groups(text: str) -> list[str]:
    """カンマ2個以上（間にスペース可）を衣装グループの区切りとして分割。
    各グループは前後の空白・余分なカンマを除いて返す。詳細ペインとカード
    ホバーのポップアップで共有する。"""
    groups = []
    for part in _GROUP_SEP.split(text):
        g = part.strip().strip(",").strip()
        if g:
            groups.append(g)
    return groups


def groups_html(scheme: str, groups: list[str], hovered: str,
                is_active) -> str:
    """各グループを同一色のボックス（1セルのテーブル行）で囲む HTML を返す。
    cellpadding で内側の余白、cellspacing でボックス間の間隔を取る。ホバー中
    のボックス（hovered == "pos:1" 等）だけ、挿入済み（is_active(i)==True）の
    ボックスだけ、それぞれ色を変える。リンク色は QLabel だと palette の Link
    色が優先されがちなので span に明示色を付けて確実に効かせる。"""
    rows = []
    for i, g in enumerate(groups):
        href = f"{scheme}:{i}"
        if href == hovered:                 # ホバー最優先
            fg, bg = _LINK_HOVER, _GROUP_BG_HOVER
        elif is_active(i):                  # 挿入済み
            fg, bg = _LINK_ACTIVE, _GROUP_BG_ACTIVE
        else:
            fg, bg = _LINK_COLOR, _GROUP_BG
        rows.append(
            f'<tr><td bgcolor="{bg}">'
            f'<a href="{href}" style="text-decoration:none">'
            f'<span style="color:{fg}">{html.escape(g)}</span></a>'
            f'</td></tr>')
    return ('<table width="100%" cellspacing="4" cellpadding="5">'
            + "".join(rows) + '</table>')


def _to_edit_text(text: str) -> str:
    """編集ダイアログ表示用。衣装グループの区切り（2連以上のカンマ）を改行に
    置き換え、どこで分割されるかを見やすくする。"""
    return _GROUP_SEP.sub("\n", text).strip()


def _from_edit_text(text: str) -> str:
    """編集ダイアログ保存用。改行区切りを保存形式（2連カンマ）に戻す。各行は
    前後の空白を除き、空行は捨てる。"""
    lines = [ln.strip() for ln in text.split("\n")]
    return ",,".join(ln for ln in lines if ln)


class _MetaWorker(QObject):
    """LoRA ごとのハッシュ計算 + civitai 取得（1本のスレッドで順次実行）。"""
    info = Signal(str, dict)     # relname, {"sha", "thumb", "meta"}
    status = Signal(str)
    done = Signal()

    def __init__(self, names: list[str], lora_dir: Path, cache_dir: Path):
        super().__init__()
        self._names = list(names)
        self._dir = Path(lora_dir)
        self._cache_dir = Path(cache_dir)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        cache = lora.LoraCache(self._cache_dir)
        offline = 0  # 連続ネットワーク失敗数; 2回で以後の問い合わせを諦める
        for n, relname in enumerate(self._names):
            if self._cancel:
                break
            path = self._dir / relname
            entry = cache.lookup(relname, path)
            if entry is None:
                self.status.emit(
                    f"ハッシュ計算中 ({n + 1}/{len(self._names)}): {relname}")
                try:
                    sha = lora.sha256_file(path)
                except OSError:
                    continue
                meta: Optional[dict] = None
                if offline < 2:
                    self.status.emit(
                        f"civitai 照会中 ({n + 1}/{len(self._names)}): "
                        f"{relname}")
                    try:
                        meta = lora.fetch_civitai_meta(sha)
                        offline = 0
                    except (OSError, ValueError):
                        offline += 1
                        meta = "error"  # ネットワーク失敗: キャッシュしない
                if meta == "error" or offline >= 2:
                    # ハッシュだけ即席エントリで通知（次回開いたとき再照会）
                    self.info.emit(relname, {"sha": sha, "thumb": "",
                                             "meta": {}})
                    continue
                cache.store(relname, path, sha,
                            meta if meta else {"found": False})
                entry = cache.lookup(relname, path)
                if entry is None:
                    continue
            sha = str(entry.get("sha256", ""))
            meta = entry.get("meta") or {}
            thumb = ""
            if meta.get("found"):
                try:
                    thumb = cache.ensure_thumb(
                        sha, str(meta.get("preview_url", "")))
                except (OSError, ValueError):
                    thumb = ""
            self.info.emit(relname, {"sha": sha, "thumb": thumb,
                                     "meta": dict(meta)})
        self.status.emit("")
        self.done.emit()


class LoraDialog(QDialog):
    apply_requested = Signal(str, float)     # relname, strength
    remove_requested = Signal(str)           # relname
    # トリガーワードのトグル: (token, negative, text)。token はどのリンク由来か
    # を一意に表す識別子。メイン側は token が未挿入なら挿入、挿入済みなら削除
    # する。negative=True でネガティブ欄が対象。
    toggle_prompt_requested = Signal(str, bool, str)

    def __init__(self, lora_dir: Path, cache_dir: Path,
                 preset_fn: Callable[[], str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("LoRA")
        self.setMinimumSize(900, 560)
        bind_geometry(self, "lora")
        self._lora_dir = Path(lora_dir)
        self._cache_dir = Path(cache_dir)
        self._preset_fn = preset_fn
        self._items: dict[str, QListWidgetItem] = {}
        self._meta: dict[str, dict] = {}   # relname -> {"sha","thumb","meta"}
        # 系統はヘッダ（キー名+形状）から自前判定する。civitai の baseModel
        # は作者の自己申告なので参考表示のみ（フィルタには使わない）。
        self._fams: dict[str, str] = {}    # relname -> modelinfo family
        self._applied: dict[str, float] = {}
        # 詳細ペインのトリガーワード表示状態（衣装グループ分割 + ホバー）。
        self._pos_groups: list[str] = []
        self._neg_groups: list[str] = []
        self._pos_empty = ""          # ポジティブが空のときの表示文言
        self._hover = ""              # 現在ホバー中のリンク href（"pos:1" 等）
        self._current_rel = ""        # 詳細表示中の relname（token 生成に使う）
        self._inserted: set[str] = set()  # メインに挿入済みのグループ token
        # ユーザー編集のトリガーワード（civitai キャッシュとは別保存）。
        self._prompts = lora.LoraPrompts(self._cache_dir / "user_prompts.json")
        self._thread: Optional[QThread] = None
        self._worker: Optional[_MetaWorker] = None
        self._placeholder = self._make_placeholder()

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("検索…")
        self.ed_search.textChanged.connect(self._apply_filter)
        self.cb_filter = QComboBox()
        self.cb_filter.addItem("表示モデルに合わせる", "preset")
        self.cb_filter.addItem("すべて", "all")
        self.cb_filter.setToolTip(
            "LoRA ファイルのヘッダ（学習対象モジュールの構造）から系統を"
            "判定し、現在の表示モデルで使えるものだけを表示します"
            "（判定できないものは常に表示）")
        self.cb_filter.currentIndexChanged.connect(self._apply_filter)
        btn_rescan = QPushButton("再スキャン")
        btn_rescan.clicked.connect(self.rescan)
        top.addWidget(self.ed_search, stretch=1)
        top.addWidget(QLabel("ベースモデル:"))
        top.addWidget(self.cb_filter)
        top.addWidget(btn_rescan)
        root.addLayout(top)

        # 左右ペインの境界はドラッグで調整できる（QSplitter）。
        panes = QSplitter(Qt.Horizontal)
        root.addWidget(panes, stretch=1)

        # ----- left: thumbnail grid ---------------------------------------
        self.lst = QListWidget()
        self.lst.setViewMode(QListWidget.IconMode)
        self.lst.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self.lst.setGridSize(_GRID_SIZE)
        self.lst.setResizeMode(QListWidget.Adjust)
        self.lst.setMovement(QListWidget.Static)
        self.lst.setWordWrap(True)
        self.lst.setUniformItemSizes(True)
        self.lst.setMinimumWidth(220)
        self.lst.currentItemChanged.connect(self._on_selection)
        self.lst.itemDoubleClicked.connect(self._on_double_click)
        panes.addWidget(self.lst)

        # ----- right: detail pane ------------------------------------------
        detail = QVBoxLayout()
        self.lbl_name = QLabel("")
        self.lbl_name.setWordWrap(True)
        self.lbl_name.setStyleSheet("font-weight: bold;")
        self.lbl_base = QLabel("")
        self.lbl_base.setWordWrap(True)
        # トリガーワード（ポジティブ / ネガティブ）の表示。ポジティブは
        # civitai の初期値、ネガティブはユーザー編集時のみ。
        self.lbl_edited = QLabel("✎ 編集済み")
        self.lbl_edited.setStyleSheet("color:#5a9;")
        self.lbl_edited.setVisible(False)
        # 衣装グループごとにリンク。クリックでそのグループを対応欄へ挿入する。
        self.lbl_pos = QLabel("")
        self.lbl_pos.setWordWrap(True)
        self.lbl_pos.setTextFormat(Qt.RichText)
        self.lbl_pos.setToolTip(
            "クリックでポジティブ欄へ挿入／もう一度で削除（挿入中は緑表示）")
        self.lbl_pos.linkActivated.connect(self._insert)
        self.lbl_pos.linkHovered.connect(self._on_link_hovered)
        self.lbl_neg = QLabel("")
        self.lbl_neg.setWordWrap(True)
        self.lbl_neg.setTextFormat(Qt.RichText)
        self.lbl_neg.setToolTip(
            "クリックでネガティブ欄へ挿入／もう一度で削除（挿入中は緑表示）")
        self.lbl_neg.linkActivated.connect(self._insert)
        self.lbl_neg.linkHovered.connect(self._on_link_hovered)
        self.btn_edit = QPushButton("編集")
        self.btn_edit.setEnabled(False)
        self.btn_edit.setToolTip("このLoRAのトリガーワードを編集します")
        self.btn_edit.clicked.connect(self._edit_words)
        self.btn_civitai = QPushButton("civitai を開く")
        self.btn_civitai.setEnabled(False)
        self.btn_civitai.clicked.connect(self._open_civitai)

        self.sp_strength = QDoubleSpinBox()
        self.sp_strength.setRange(-4.0, 4.0)
        self.sp_strength.setDecimals(2)
        self.sp_strength.setSingleStep(0.05)
        self.sp_strength.setValue(1.0)
        self.btn_apply = QPushButton("適用")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._on_apply_clicked)

        words_head = QHBoxLayout()
        words_head.addWidget(self.lbl_edited)
        words_head.addStretch(1)

        # LoRA名〜ネガティブの情報はスクロール領域に入れ、長くても切れない
        # ようにする。ボタン類（編集/civitai/強度/適用）は常時見える位置に置く。
        info_col = QVBoxLayout()
        info_col.setContentsMargins(0, 0, 0, 0)
        info_col.addWidget(self.lbl_name)
        info_col.addWidget(self.lbl_base)
        info_col.addLayout(words_head)
        info_col.addWidget(self.lbl_pos)
        info_col.addWidget(self.lbl_neg)
        info_col.addStretch(1)
        info_widget = QWidget()
        info_widget.setLayout(info_col)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(info_widget)

        # 編集ボタンと civitai ボタンを1行に（縦を節約）。
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_edit)
        btn_row.addWidget(self.btn_civitai)
        btn_row.addStretch(1)
        # 強度設定と適用ボタンを1行に（縦を節約）。
        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("強度"))
        strength_row.addWidget(self.sp_strength)
        strength_row.addStretch(1)
        strength_row.addWidget(self.btn_apply)

        detail.addWidget(scroll, stretch=1)
        detail.addLayout(btn_row)
        detail.addLayout(strength_row)
        side = QWidget()
        side.setLayout(detail)
        side.setMinimumWidth(240)
        panes.addWidget(side)
        # 初期比率（グリッド優先で広く）。境界はドラッグで変更可能。
        panes.setStretchFactor(0, 1)
        panes.setStretchFactor(1, 0)
        panes.setSizes([620, 280])

        bottom = QHBoxLayout()
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #888;")
        self.lbl_applied = QLabel("適用中: なし")
        btn_close = QPushButton("閉じる")
        btn_close.clicked.connect(self.close)
        bottom.addWidget(self.lbl_status, stretch=1)
        bottom.addWidget(self.lbl_applied)
        bottom.addWidget(btn_close)
        root.addLayout(bottom)

        self.rescan()

    # ----- grid population ---------------------------------------------------
    @staticmethod
    def _make_placeholder() -> QIcon:
        pix = QPixmap(_ICON_SIZE, _ICON_SIZE)
        pix.fill(QColor("#2a2a2a"))
        return QIcon(pix)

    def rescan(self) -> None:
        """(Re)list models/loras and restart the metadata worker."""
        self._stop_worker()
        names = config.scan_models("loras")
        self.lst.clear()
        self._items.clear()
        # 既知メタは残す（ワーカーがキャッシュから同じ内容を再供給する）
        for relname in names:
            # 系統判定はヘッダ読みだけ（数KB・ハッシュ不要）なのでこの場で
            # 行う。結果は modelinfo 側で size+mtime キャッシュされる。
            self._fams[relname] = modelinfo.family(
                "loras", self._lora_dir / relname)
            item = QListWidgetItem(self._placeholder, relname)
            item.setData(Qt.UserRole, relname)
            item.setToolTip(
                f"{relname}\n系統: "
                f"{_FAMILY_LABELS.get(self._fams[relname], '不明')}")
            self.lst.addItem(item)
            self._items[relname] = item
        self._refresh_applied_marks()
        self._apply_filter()
        if not names:
            self.lbl_status.setText(
                f"LoRA がありません: {self._lora_dir} に配置してください")
            return
        self._worker = _MetaWorker(names, self._lora_dir, self._cache_dir)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.info.connect(self._on_info)
        self._worker.status.connect(self.lbl_status.setText)
        self._worker.done.connect(self._thread.quit)
        self._thread.finished.connect(self._on_worker_finished)
        self._thread.start()

    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(10_000)
        self._thread = None
        self._worker = None

    def _on_worker_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _on_info(self, relname: str, info: dict) -> None:
        self._meta[relname] = info
        item = self._items.get(relname)
        if item is None:
            return
        thumb = str(info.get("thumb", ""))
        if thumb:
            pix = QPixmap(thumb)
            if not pix.isNull():
                item.setIcon(QIcon(pix))
        self._apply_filter()
        if item is self.lst.currentItem():
            self._show_detail(relname)

    # ----- filtering -----------------------------------------------------------
    def _visible(self, relname: str) -> bool:
        text = self.ed_search.text().strip().lower()
        if text and text not in relname.lower():
            return False
        if self.cb_filter.currentData() == "preset":
            fam = self._fams.get(relname, modelinfo.UNKNOWN)
            # 判定不能だけは表示（誤って隠すより安全側）。「対応外」
            # (SD1.x/Flux/Qwen-Image 20B など) は確定情報なので隠す。
            if fam != modelinfo.UNKNOWN and fam != self._preset_fn():
                return False
        return True

    def _apply_filter(self, *_a) -> None:
        for relname, item in self._items.items():
            item.setHidden(not self._visible(relname))

    # ----- applied state (supplied by the main window) --------------------------
    def set_applied(self, applied: dict[str, float]) -> None:
        self._applied = dict(applied)
        self._refresh_applied_marks()
        if applied:
            self.lbl_applied.setText("適用中: " + ", ".join(
                f"{Path(n).stem}×{w:g}" for n, w in applied.items()))
        else:
            self.lbl_applied.setText("適用中: なし")
        current = self.lst.currentItem()
        if current is not None:
            self._show_detail(str(current.data(Qt.UserRole)))

    def _refresh_applied_marks(self) -> None:
        for relname, item in self._items.items():
            if relname in self._applied:
                item.setText(f"✓ {relname}")
                item.setBackground(_APPLIED_BG)
            else:
                item.setText(relname)
                item.setBackground(QBrush())

    # ----- detail pane -----------------------------------------------------------
    def _on_selection(self, current, _previous) -> None:
        if current is None:
            self.btn_apply.setEnabled(False)
            self.btn_edit.setEnabled(False)
            return
        self._show_detail(str(current.data(Qt.UserRole)))

    def _show_detail(self, relname: str) -> None:
        info = self._meta.get(relname) or {}
        meta = info.get("meta") or {}
        fam = self._fams.get(relname, modelinfo.UNKNOWN)
        fam_text = f"系統: {_FAMILY_LABELS.get(fam, '不明')}"
        if meta.get("found"):
            title = meta.get("name") or relname
            if meta.get("version"):
                title += f"（{meta['version']}）"
            self.lbl_name.setText(title)
            base = str(meta.get("base_model", ""))
            if base:
                fam_text += f"（civitai 登録: {base}）"
            self.lbl_base.setText(fam_text)
            self._civitai_url = str(meta.get("url", ""))
        else:
            self.lbl_name.setText(relname)
            self.lbl_base.setText(fam_text)
            self._civitai_url = ""
        self._refresh_words(relname)
        self.btn_civitai.setEnabled(bool(self._civitai_url))
        applied = relname in self._applied
        if applied:
            self.sp_strength.setValue(float(self._applied[relname]))
        self.btn_apply.setText("解除" if applied else "適用")
        self.btn_apply.setEnabled(True)

    def _civitai_words(self, relname: str) -> str:
        """civitai のトリガーワード（ポジティブ）をカンマ区切り文字列で。"""
        meta = (self._meta.get(relname) or {}).get("meta") or {}
        return ", ".join(str(w) for w in (meta.get("trained_words") or []))

    def _effective_words(self, relname: str) -> tuple[str, str]:
        """(positive, negative)。ユーザー編集があればそれを、無ければ civitai
        のトリガーワードをポジティブとして返す（ネガティブは編集時のみ）。"""
        override = self._prompts.get(relname)
        if override is not None:
            return override["positive"], override["negative"]
        return self._civitai_words(relname), ""

    def _group_token(self, scheme: str, i: int) -> str:
        """リンク（グループ）を一意に表す識別子。メイン側の挿入区間の追跡に使う。
        ファイル名に '|' は使えないので区切りに使える。"""
        return f"{self._current_rel}|{scheme}|{i}"

    def _refresh_words(self, relname: str) -> None:
        """トリガーワード表示（衣装グループごとのリンク）を更新する。"""
        self._current_rel = relname
        pos, neg = self._effective_words(relname)
        self.lbl_edited.setVisible(self._prompts.get(relname) is not None)
        self._pos_groups = split_groups(pos)
        self._neg_groups = split_groups(neg)
        meta = (self._meta.get(relname) or {}).get("meta")
        self._pos_empty = ("ポジティブ: （情報取得中…）" if not meta
                           else "ポジティブ: （なし）")
        self._hover = ""              # 選択が変わったらホバー強調はリセット
        self._render_words()
        self.btn_edit.setEnabled(True)

    def _groups_html(self, scheme: str, groups: list[str]) -> str:
        return groups_html(
            scheme, groups, self._hover,
            lambda i: self._group_token(scheme, i) in self._inserted)

    def _render_words(self) -> None:
        if self._pos_groups:
            self.lbl_pos.setText(
                "ポジティブ:" + self._groups_html("pos", self._pos_groups))
        else:
            self.lbl_pos.setText(self._pos_empty)
        if self._neg_groups:
            self.lbl_neg.setText(
                "ネガティブ:" + self._groups_html("neg", self._neg_groups))
            self.lbl_neg.setVisible(True)
        else:
            self.lbl_neg.setText("")
            self.lbl_neg.setVisible(False)

    def _on_link_hovered(self, href: str) -> None:
        if href == self._hover:
            return
        self._hover = href            # 離脱時は "" になり強調が消える
        self._render_words()

    def _insert(self, href: str) -> None:
        """クリックされた衣装グループを、メイン側でトグル（挿入/削除）させる。"""
        scheme, _, idx = href.partition(":")
        groups = self._pos_groups if scheme == "pos" else self._neg_groups
        try:
            i = int(idx)
            g = groups[i]
        except (ValueError, IndexError):
            return
        self.toggle_prompt_requested.emit(
            self._group_token(scheme, i), scheme == "neg", g)

    def set_inserted(self, tokens: set[str]) -> None:
        """メイン側で現在プロンプトに挿入済みのグループ token 一覧を受け取り、
        リンクの表示（挿入済みハイライト）を更新する。"""
        tokens = set(tokens)
        if tokens == self._inserted:
            return
        self._inserted = tokens
        if self._current_rel:
            self._render_words()

    def _edit_words(self) -> None:
        """トリガーワード（ポジティブ/ネガティブ）を編集して保存する。"""
        relname = self._current_relname()
        if relname is None:
            return
        pos, neg = self._effective_words(relname)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"トリガーワード編集 — {relname}")
        dlg.setMinimumWidth(440)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            "ポジティブ（カンマ区切り／改行で衣装グループを分割）:"))
        ed_pos = QPlainTextEdit(_to_edit_text(pos))
        ed_pos.setMinimumHeight(70)
        lay.addWidget(ed_pos)
        lay.addWidget(QLabel(
            "ネガティブ（カンマ区切り／改行で衣装グループを分割）:"))
        ed_neg = QPlainTextEdit(_to_edit_text(neg))
        ed_neg.setMinimumHeight(70)
        lay.addWidget(ed_neg)
        btns = QHBoxLayout()
        b_reset = QPushButton("civitai の初期値に戻す")
        b_reset.setToolTip("編集内容を破棄し、civitai のトリガーワードに戻します")
        b_cancel = QPushButton("キャンセル")
        b_ok = QPushButton("保存")
        b_ok.setDefault(True)
        btns.addWidget(b_reset)
        btns.addStretch(1)
        btns.addWidget(b_cancel)
        btns.addWidget(b_ok)
        lay.addLayout(btns)
        b_reset.clicked.connect(
            lambda: (ed_pos.setPlainText(
                         _to_edit_text(self._civitai_words(relname))),
                     ed_neg.setPlainText("")))
        b_cancel.clicked.connect(dlg.reject)
        b_ok.clicked.connect(dlg.accept)
        if dlg.exec() != QDialog.Accepted:
            return
        self._prompts.set(relname, _from_edit_text(ed_pos.toPlainText()),
                          _from_edit_text(ed_neg.toPlainText()))
        self._refresh_words(relname)

    def _open_civitai(self) -> None:
        if getattr(self, "_civitai_url", ""):
            QDesktopServices.openUrl(QUrl(self._civitai_url))

    # ----- apply / remove -----------------------------------------------------
    def _current_relname(self) -> Optional[str]:
        item = self.lst.currentItem()
        return None if item is None else str(item.data(Qt.UserRole))

    def _on_apply_clicked(self) -> None:
        relname = self._current_relname()
        if relname is None:
            return
        if relname in self._applied:
            self.remove_requested.emit(relname)
        else:
            self.apply_requested.emit(relname, self.sp_strength.value())

    def _on_double_click(self, item) -> None:
        relname = str(item.data(Qt.UserRole))
        if relname in self._applied:
            self.remove_requested.emit(relname)
        else:
            self.apply_requested.emit(relname, self.sp_strength.value())

    # ----- lifecycle -------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        self._stop_worker()
        super().closeEvent(event)


class TriggerWordsPopup(QFrame):
    """LoRA カードにマウスホバーしたとき出す、トリガーワードのポップアップ。
    LoRA ウィンドウの詳細ペインと同じ描画（groups_html）・同じトグル操作を
    行う。クリックでメインのプロンプト欄への挿入/削除をトグルする。"""

    # LoRA ウィンドウと同じ (token, negative, text)。
    toggle_requested = Signal(str, bool, str)
    hover_changed = Signal(bool)   # マウスがポップアップ上にあるか（保持判定用）

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint
                         | Qt.WindowStaysOnTopHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setObjectName("twPopup")
        # 枠の背景はメインウィンドウ（親）と同じ色に合わせる。
        pal = parent.palette() if parent is not None else self.palette()
        bg = pal.color(QPalette.Window).name()
        self.setStyleSheet(
            f"#twPopup {{ background:{bg}; border:1px solid #666;"
            " border-radius:6px; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        self.lbl = QLabel("")
        self.lbl.setTextFormat(Qt.RichText)
        self.lbl.setWordWrap(True)
        self.lbl.linkActivated.connect(self._on_link)
        self.lbl.linkHovered.connect(self._on_hover)
        lay.addWidget(self.lbl)
        self._relname = ""
        self._pos: list[str] = []
        self._neg: list[str] = []
        self._active: set[str] = set()
        self._hover = ""

    def set_content(self, relname: str, positive: str, negative: str,
                    active_tokens: set) -> None:
        self._relname = relname
        self._pos = split_groups(positive)
        self._neg = split_groups(negative)
        self._active = set(active_tokens)
        self._hover = ""
        self._render()

    def has_words(self) -> bool:
        return bool(self._pos or self._neg)

    def set_active(self, active_tokens: set) -> None:
        self._active = set(active_tokens)
        self._render()

    def _token(self, scheme: str, i: int) -> str:
        return f"{self._relname}|{scheme}|{i}"

    def _render(self) -> None:
        parts = []
        if self._pos:
            parts.append("ポジティブ:" + groups_html(
                "pos", self._pos, self._hover,
                lambda i: self._token("pos", i) in self._active))
        if self._neg:
            parts.append("ネガティブ:" + groups_html(
                "neg", self._neg, self._hover,
                lambda i: self._token("neg", i) in self._active))
        html_text = "<br>".join(parts)
        self.lbl.setText(html_text)
        # ラベル(=ポップアップ)のサイズを一意に確定させる。wordWrap ラベルの
        # sizeHint は高さを過小評価するため、表示後にレイアウトが広げ直して
        # QWindowsWindow::setGeometry 警告が出る。QTextDocument で実際の描画
        # サイズを測り、幅・高さを固定して食い違いを無くす。
        doc = QTextDocument()
        doc.setHtml(html_text)
        w = int(min(max(doc.idealWidth(), 160), 340))
        doc.setTextWidth(w)
        self.lbl.setFixedWidth(w)
        self.lbl.setFixedHeight(int(doc.size().height()) + 4)
        self.adjustSize()

    def _on_hover(self, href: str) -> None:
        if href == self._hover:
            return
        self._hover = href
        self._render()

    def _on_link(self, href: str) -> None:
        scheme, _, idx = href.partition(":")
        groups = self._pos if scheme == "pos" else self._neg
        try:
            g = groups[int(idx)]
        except (ValueError, IndexError):
            return
        self.toggle_requested.emit(self._token(scheme, int(idx)),
                                   scheme == "neg", g)

    def enterEvent(self, e) -> None:  # noqa: N802
        self.hover_changed.emit(True)
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:  # noqa: N802
        self.hover_changed.emit(False)
        super().leaveEvent(e)

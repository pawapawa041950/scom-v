"""Reusable model-selection widget shared by the first-run setup and the
settings dialog, so the two screens don't drift apart.

「必須モデルダウンロード」ボックスにプリセット（チェックボックス）を並べ、
各行の右に導入状況/ダウンロード進捗を数値で表示する。``with_progress=True``
（設定ウィンドウ）ではボックス内に「選択をダウンロード」ボタンも持つ。
個別ファイルの選択リストは無い — プリセットのチェックが対応ファイル群
（``self.checks`` の隠しチェック状態）を駆動する。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
    QWidget,
)

from .. import config
from ..bootstrap import models as models_mod

# Quick-select presets: which files a button ticks for a given setup.
PRESET_ANIMA = [
    "anima-base-v1.0.safetensors",
    "qwen_image_vae.safetensors",
    "qwen_3_06b_base.safetensors",
]
PRESET_KREA2 = [
    "krea2_turbo_int8_convrot.safetensors",
    "qwen_image_vae.safetensors",
    "qwen3vl_4b_fp8_scaled.safetensors",
]
# WAI はフル SDXL チェックポイント（VAE/CLIP 内蔵）なので本体のみ。VAE/CLIP は
# モデル内蔵を使うため別途ダウンロードしない。
PRESET_SDXL = [
    "waiIllustriousSDXL_v170.safetensors",
]
QUICK_PRESETS = [
    ("Anima 必須モデル", PRESET_ANIMA),
    ("Krea2 必須モデル int8convrot", PRESET_KREA2),
    ("SDXL WAI-illustrious-SDXL 必須モデル", PRESET_SDXL),
]


def fmt_size(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    return f"{n} B"


class ModelSelector(QWidget):
    """Preset checkboxes with per-preset numeric download status.

    ``checks``/``manifest`` are exposed so an embedding dialog can drive
    downloads; the dialog reports progress back via ``update_progress`` /
    ``mark_file_done`` / ``mark_file_failed``.
    """

    def __init__(self, paths: config.AppPaths, with_progress: bool = False,
                 parent=None):
        super().__init__(parent)
        self._paths = paths
        self._with_progress = with_progress
        self.manifest = models_mod.load_manifest(paths)
        self._by_name = {m.filename: m for m in self.manifest}
        self.checks: dict[str, QCheckBox] = {}
        # ダウンロード中のファイルの (done, total) バイト数と失敗集合。
        self._live: dict[str, tuple[float, float]] = {}
        self._failed: set[str] = set()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        quick = QGroupBox("必須モデルダウンロード")
        quick_layout = QVBoxLayout(quick)
        quick_layout.addWidget(QLabel(
            "チェックした一式をまとめてダウンロード対象にします"))
        # (チェックボックス, ファイル群, 状況ラベル)
        self._quick: list[tuple[QCheckBox, list, QLabel]] = []
        for label, fileset in QUICK_PRESETS:
            qcb = QCheckBox(label)
            qcb.toggled.connect(
                lambda checked, s=fileset: self._on_quick_toggled(s, checked))
            status = QLabel("")
            status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row = QHBoxLayout()
            row.addWidget(qcb)
            row.addStretch(1)
            row.addWidget(status)
            quick_layout.addLayout(row)
            self._quick.append((qcb, fileset, status))
        if with_progress:
            btn_row = QHBoxLayout()
            btn_row.addStretch(1)
            self.btn_download = QPushButton("選択をダウンロード")
            btn_row.addWidget(self.btn_download)
            quick_layout.addLayout(btn_row)
        layout.addWidget(quick)
        layout.addStretch(1)

        self._populate()
        self.refresh_status()

    def _is_present(self, m) -> bool:
        p = models_mod.target_path(self._paths, m)
        return p.exists() and (not m.size or p.stat().st_size == m.size)

    def _populate(self) -> None:
        """Build the (hidden) per-file selection state. Individual files are
        not shown — the preset checkboxes drive them."""
        for m in self.manifest:
            cb = QCheckBox(m.filename)
            if self._is_present(m):
                cb.setEnabled(False)
            self.checks[m.filename] = cb

    # ----- per-preset status ------------------------------------------------
    def refresh_status(self) -> None:
        """各プリセット行の右側に導入状況/進捗を数値で表示する。"""
        for _qcb, fileset, lbl in self._quick:
            done_b = 0.0
            total_b = 0.0
            unknown_total = False
            files_done = 0
            failed = any(fn in self._failed for fn in fileset)
            for fn in fileset:
                m = self._by_name.get(fn)
                if m is None:
                    continue
                if self._is_present(m):
                    files_done += 1
                    sz = m.size or models_mod.target_path(
                        self._paths, m).stat().st_size
                    done_b += sz
                    total_b += sz
                    continue
                d, t = self._live.get(fn, (0.0, float(m.size)))
                done_b += d
                if t:
                    total_b += t
                else:
                    unknown_total = True
            n = len(fileset)
            if failed:
                lbl.setText("失敗（再試行できます）")
                lbl.setStyleSheet("color:#c33;")
                continue
            lbl.setStyleSheet("color:#888;")
            if files_done == n:
                lbl.setText(f"導入済み（{fmt_size(int(total_b))}）")
            else:
                total_txt = ("?" if unknown_total
                             else fmt_size(int(total_b)))
                lbl.setText(f"{files_done}/{n} ファイル  "
                            f"{fmt_size(int(done_b))} / {total_txt}")

    def update_progress(self, filename: str, done: float, total: float) -> None:
        """ダウンロード中のバイト数を反映（埋め込み側から呼ばれる）。"""
        self._live[filename] = (done, total)
        self.refresh_status()

    def mark_file_done(self, filename: str) -> None:
        self._live.pop(filename, None)
        self._failed.discard(filename)
        cb = self.checks.get(filename)
        if cb is not None:
            cb.setChecked(False)
            cb.setEnabled(False)
        self.refresh_status()

    def mark_file_failed(self, filename: str) -> None:
        self._live.pop(filename, None)
        self._failed.add(filename)
        self.refresh_status()

    # ----- selection ----------------------------------------------------------
    def _on_quick_toggled(self, files: list, checked: bool) -> None:
        """Union (OR) selection: checking a preset ticks its files; unchecking
        unticks only the files no other still-checked preset needs — so the
        shared VAE survives until both presets are off.
        """
        if checked:
            for fn in files:
                cb = self.checks.get(fn)
                if cb is not None and cb.isEnabled():
                    cb.setChecked(True)
            return
        keep = set()
        for qcb, f, _lbl in self._quick:
            if qcb.isChecked():
                keep |= set(f)
        for fn in files:
            if fn in keep:
                continue
            cb = self.checks.get(fn)
            if cb is not None and cb.isEnabled():
                cb.setChecked(False)

    def selected_filenames(self) -> list:
        """Filenames the user ticked (excludes already-downloaded ones)."""
        return [fn for fn, cb in self.checks.items()
                if cb.isChecked() and cb.isEnabled()]

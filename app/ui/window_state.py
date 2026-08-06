"""サブウィンドウのジオメトリ（サイズ・位置）を記憶する小さなヘルパー。

各ダイアログは __init__ の初期サイズ設定の直後に ``bind_geometry(self, key)``
を1回呼ぶだけでよい。保存先はアプリのポータブル方針に合わせ、レジストリでは
なく ``userdata/windows.ini``（QSettings の INI 形式）にする。

- 復元: 保存済みジオメトリがあれば適用（無ければダイアログ既定のサイズのまま）。
- 保存: QDialog の ``finished``（OK/キャンセル/×どの閉じ方でも発火）で自動保存。
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSettings
from PySide6.QtWidgets import QWidget

from .. import config

_INI_NAME = "windows.ini"


def _settings() -> QSettings:
    path: Path = config.AppPaths().user_data / _INI_NAME
    return QSettings(str(path), QSettings.IniFormat)


def restore_geometry(widget: QWidget, key: str) -> None:
    geo = _settings().value(f"{key}/geometry")
    if geo:
        widget.restoreGeometry(geo)


def save_geometry(widget: QWidget, key: str) -> None:
    s = _settings()
    s.setValue(f"{key}/geometry", widget.saveGeometry())
    s.sync()


class _GeometryBinder(QObject):
    """Restore geometry on the widget's first show (after its layout is built —
    restoring too early in __init__ doesn't stick), and save on close/finish."""

    def __init__(self, widget: QWidget, key: str):
        super().__init__(widget)          # parented so it lives with the widget
        self._widget = widget
        self._key = key
        self._restored = False
        widget.installEventFilter(self)
        finished = getattr(widget, "finished", None)
        if finished is not None:          # QDialog: どの閉じ方でも発火する
            finished.connect(lambda *_: save_geometry(widget, key))

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if not self._restored and event.type() == QEvent.Show:
            self._restored = True
            restore_geometry(self._widget, self._key)
        return False


def bind_geometry(widget: QWidget, key: str) -> None:
    """Remember this window's size/position across runs (stored per ``key``)."""
    _GeometryBinder(widget, key)

"""Render ANSI-colored log lines into a QPlainTextEdit."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit

from ..textutil import ansi_runs

# Dark, terminal-like style so the ANSI palette renders as intended, with a
# light scrollbar that stays visible against the dark background.
LOG_STYLE = """
QPlainTextEdit { background:#1e1e1e; color:#d4d4d4; }
QScrollBar:vertical, QScrollBar:horizontal { background:#3a3a3a; margin:0; }
QScrollBar:vertical { width:12px; }
QScrollBar:horizontal { height:12px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background:#c8c8c8; border-radius:5px; margin:2px;
}
QScrollBar::handle:vertical { min-height:24px; }
QScrollBar::handle:horizontal { min-width:24px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background:#eaeaea;
}
QScrollBar::add-line, QScrollBar::sub-line { width:0; height:0; }
QScrollBar::add-page, QScrollBar::sub-page { background:none; }
"""


def style_log(widget: QPlainTextEdit) -> None:
    widget.setStyleSheet(LOG_STYLE)
    f = QFont("Consolas")
    f.setStyleHint(QFont.Monospace)
    f.setPointSize(9)
    widget.setFont(f)
    widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)


def append_ansi(widget: QPlainTextEdit, line: str) -> None:
    """Append ``line`` (which may contain ANSI color codes) as colored text.

    Only follows the tail (scrolls to bottom) when the view was already at the
    bottom; if the user has scrolled up, their position is left untouched.
    """
    sb = widget.verticalScrollBar()
    at_bottom = sb.value() >= sb.maximum() - 2

    # Insert through a document cursor so the viewport/visible cursor is not moved.
    cursor = QTextCursor(widget.document())
    cursor.movePosition(QTextCursor.End)
    if not widget.document().isEmpty():
        cursor.insertText("\n")
    for text, color, bold in ansi_runs(line):
        if not text:
            continue
        fmt = QTextCharFormat()
        if color:
            fmt.setForeground(QColor(color))
        if bold:
            fmt.setFontWeight(QFont.Bold)
        cursor.insertText(text, fmt)

    if at_bottom:
        sb.setValue(sb.maximum())

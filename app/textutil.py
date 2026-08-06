"""Small text helpers for log display."""
from __future__ import annotations

import re

# Matches ANSI escape sequences: CSI (colors, cursor moves) and OSC.
_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"      # CSI: e.g. \x1b[32m, \x1b[0m, \x1b[2K
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC: \x1b]...BEL or ...ST
    r"|\x1b[@-Z\\-_]"                 # two-char escapes
)

# OSC and other (non-CSI) escapes — stripped before coloring.
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
# Any CSI sequence, capturing its parameters and final byte.
_CSI_RE = re.compile(r"\x1b\[([0-9;?]*)[ -/]*([@-~])")

# VS Code terminal palette (reads well on a dark log background).
_BASE = {30: "#666666", 31: "#cd3131", 32: "#0dbc79", 33: "#e5e510",
         34: "#3b8eea", 35: "#bc3fbc", 36: "#11a8cd", 37: "#e5e5e5"}
_BRIGHT = {90: "#888888", 91: "#f14c4c", 92: "#23d18b", 93: "#f5f543",
           94: "#3b8eea", 95: "#d670d6", 96: "#29b8db", 97: "#ffffff"}


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes (and stray carriage returns) from a log line."""
    return _ANSI_RE.sub("", text).replace("\r", "")


def ansi_runs(line: str) -> list[tuple[str, str | None, bool]]:
    """Split a line into (text, color_hex_or_None, bold) runs by ANSI SGR codes.

    OSC/cursor escapes and carriage returns are dropped; only SGR (color/bold)
    sequences affect formatting.
    """
    line = _OSC_RE.sub("", line).replace("\r", "")
    runs: list[tuple[str, str | None, bool]] = []
    color: str | None = None
    bold = False
    pos = 0
    for m in _CSI_RE.finditer(line):
        if m.start() > pos:
            runs.append((line[pos:m.start()], color, bold))
        pos = m.end()
        if m.group(2) != "m":
            continue  # non-SGR CSI (cursor move, erase, …): drop, no text
        codes = [int(c) for c in m.group(1).split(";") if c.isdigit()] or [0]
        i = 0
        while i < len(codes):
            n = codes[i]
            if n == 0:
                color, bold = None, False
            elif n == 1:
                bold = True
            elif n == 22:
                bold = False
            elif n == 39:
                color = None
            elif n in _BASE:
                color = _BASE[n]
            elif n in _BRIGHT:
                color = _BRIGHT[n]
            elif n in (38, 48):  # extended color: skip its parameters
                mode = codes[i + 1] if i + 1 < len(codes) else None
                i += 2 if mode == 5 else 4 if mode == 2 else 1
            i += 1
    if pos < len(line):
        runs.append((line[pos:], color, bold))
    return runs

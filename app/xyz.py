"""XYZ plot: axis definitions, value parsing, and grid image composition.

Stable Diffusion WebUI の X/Y/Z plot 相当。3軸それぞれにパラメータの種類と
値リストを指定し、全組み合わせを生成して1枚のラベル付きグリッド画像に
まとめる。このモジュールは UI 非依存のコアロジック:

  * AXES           — 選べる軸の宣言的な定義テーブル
  * parse_values() — 値文字列のパース（CSV + 範囲記法 1-5 / 1-9 (+2) / 1-8 [4]）
  * apply_value()  — GenParams のコピーに軸の値を適用
  * plan_cells()   — 全組み合わせの (グリッド位置, GenParams) 実行計画
  * compose_grid() — セル画像群を凡例付きの1枚の QImage に合成
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from io import StringIO
from itertools import chain
from typing import Optional

from PySide6.QtCore import QBuffer, QIODevice, QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter

from .workflow import GenParams, SAMPLERS, SCHEDULERS

# 範囲記法 (Forge 互換): "1-5" / "1-9 (+2)" = ステップ指定 / "1-8 [4]" = 分割数
_RE_INT_STEP = re.compile(r"([+-]?\d+)\s*-\s*([+-]?\d+)(?:\s*\(\s*([+-]?\d+)\s*\))?")
_RE_INT_COUNT = re.compile(r"([+-]?\d+)\s*-\s*([+-]?\d+)\s*\[\s*(\d+)\s*\]")
_FLOAT = r"[+-]?\d+(?:\.\d*)?"
_RE_FLOAT_STEP = re.compile(rf"({_FLOAT})\s*-\s*({_FLOAT})(?:\s*\(\s*([+-]?{_FLOAT})\s*\))?")
_RE_FLOAT_COUNT = re.compile(rf"({_FLOAT})\s*-\s*({_FLOAT})\s*\[\s*(\d+)\s*\]")

VALUE_SYNTAX_HELP = (
    "カンマ区切りで複数指定。数値軸は範囲記法も使えます:\n"
    "  1-5 … 1,2,3,4,5   /   1-9 (+2) … 1,3,5,7,9   /   0-1 [5] … 0,0.25,0.5,0.75,1\n"
    "カンマを含む値は \"...\" で囲んでください")


@dataclass(frozen=True)
class AxisDef:
    """One selectable axis type.

    kind: "none" | "int" | "float" | "choice" | "size" | "text"
    cost: 値の切替が重い軸ほど大きく（実行順で外側ループに回される）
    """
    id: str
    label: str
    kind: str
    cost: float = 0.0
    choices: tuple[str, ...] = ()
    tooltip: str = ""


AXES: tuple[AxisDef, ...] = (
    AxisDef("none", "なし", "none"),
    AxisDef("seed", "Seed", "int"),
    AxisDef("steps", "Steps", "int"),
    AxisDef("cfg", "CFG", "float"),
    AxisDef("sampler", "Sampler", "choice", choices=tuple(SAMPLERS)),
    AxisDef("scheduler", "Scheduler", "choice", choices=tuple(SCHEDULERS)),
    AxisDef("model", "モデル", "choice", cost=1.0,
            tooltip="値ごとにモデルを読み込み直すため時間がかかります。"
                    "マージモデルは「マージモデル：<名前>」で指定できます"
                    "（候補▾から選択）"),
    AxisDef("size", "サイズ", "size",
            tooltip="幅x高さ で指定（例: 1024x1024, 832x1216）"),
    AxisDef("prompt_sr", "プロンプト S/R", "text",
            tooltip="検索/置換。1個目の値がプロンプト内の検索語、"
                    "2個目以降がその置換値になります（ネガティブにも適用）。\n"
                    "例: apple, banana, cherry → プロンプト中の \"apple\" を"
                    "banana / cherry に置き換えて比較。\n"
                    "末尾の空白は値の一部として扱われます。空白のみの値や"
                    "先頭に空白を含む値は \" \" のように引用符で囲んでください"),
    AxisDef("dtype", "UNet dtype", "choice", cost=0.5,
            choices=("default", "fp8_e4m3fn", "fp8_e5m2"),
            tooltip="UNETLoader の読み込み精度（マージモデル選択時は無効）"),
)


def axis_by_id(axis_id: str) -> AxisDef:
    for a in AXES:
        if a.id == axis_id:
            return a
    raise ValueError(f"unknown axis: {axis_id}")


# ----- value parsing --------------------------------------------------------
def _split_csv(text: str, strip: bool = True) -> list[str]:
    """Split on commas, honoring quotes ("a, b" stays one value).

    strip=False はプロンプト S/R 用: 末尾の空白を値の一部として残す
    （区切りコンマ直後の空白は skipinitialspace が落とす。空白のみ・
    先頭空白付きの値は引用符で囲めば保持される）。
    """
    reader = csv.reader(StringIO(text), skipinitialspace=True)
    values = chain.from_iterable(reader)
    if strip:
        return [s.strip() for s in values if s.strip()]
    return [s for s in values if s != ""]


def split_values(text: str) -> list[str]:
    """値欄の文字列を値リストに分解する（選択式軸のチェック状態の判定用）。"""
    return _split_csv(text)


def join_values(values: list[str]) -> str:
    """値リストを値欄の文字列に戻す。カンマ・引用符・前後の空白を含む値は
    CSV 流儀（"..." で囲み、内部の " は "" に）で引用する。"""
    out = []
    for v in values:
        if "," in v or '"' in v or v != v.strip():
            v = '"' + v.replace('"', '""') + '"'
        out.append(v)
    return ", ".join(out)


def _parse_numeric(vals: list[str], is_float: bool) -> list:
    re_count = _RE_FLOAT_COUNT if is_float else _RE_INT_COUNT
    re_step = _RE_FLOAT_STEP if is_float else _RE_INT_STEP
    num = float if is_float else int
    out: list = []
    for v in vals:
        mc = re_count.fullmatch(v)
        ms = re_step.fullmatch(v)
        if mc is not None:
            start, end, n = num(mc.group(1)), num(mc.group(2)), int(mc.group(3))
            if n < 2:
                out.append(start)
            else:
                step = (end - start) / (n - 1)
                seq = [start + step * i for i in range(n)]
                out += [round(x, 8) for x in seq] if is_float else \
                       [int(round(x)) for x in seq]
        elif ms is not None:
            start, end = num(ms.group(1)), num(ms.group(2))
            step = num(ms.group(3)) if ms.group(3) is not None else num(1)
            if step == 0 or (end - start) * step < 0:
                raise ValueError(f"範囲のステップが不正です: {v}")
            x = start
            # 浮動小数の蓄積誤差で終端を取りこぼさないよう半ステップ許容
            while (x - end) * (1 if step > 0 else -1) <= abs(step) * 0.5:
                out.append(round(x, 8) if is_float else int(x))
                x += step
        else:
            try:
                out.append(num(v))
            except ValueError:
                raise ValueError(f"数値として解釈できません: {v}") from None
    return out


def parse_values(axis: AxisDef, text: str,
                 model_choices: Optional[list[str]] = None) -> list:
    """Parse the value string for ``axis``. Raises ValueError (Japanese)."""
    if axis.kind == "none":
        return [None]
    vals = _split_csv(text, strip=(axis.kind != "text"))
    if not vals:
        raise ValueError(f"{axis.label} 軸の値が空です")
    if axis.kind == "int":
        return _parse_numeric(vals, is_float=False)
    if axis.kind == "float":
        return _parse_numeric(vals, is_float=True)
    if axis.kind == "size":
        out = []
        for v in vals:
            m = re.fullmatch(r"(\d+)\s*[xX×]\s*(\d+)", v)
            if m is None:
                raise ValueError(f"サイズは 幅x高さ で指定してください: {v}")
            out.append((int(m.group(1)), int(m.group(2))))
        return out
    if axis.kind == "choice":
        known = list(model_choices or []) if axis.id == "model" else \
            list(axis.choices)
        for v in vals:
            if v not in known:
                raise ValueError(f"{axis.label} に不明な値があります: {v}")
        return vals
    if axis.kind == "text":
        if axis.id == "prompt_sr" and len(vals) < 2:
            raise ValueError(
                "プロンプト S/R には検索語と置換値の2個以上が必要です")
        return vals
    raise ValueError(f"unknown axis kind: {axis.kind}")


def value_label(axis: AxisDef, value) -> str:
    """Legend text for one value."""
    if axis.kind == "none":
        return ""
    if axis.kind == "size":
        return f"{value[0]}x{value[1]}"
    if axis.kind in ("int", "float"):
        return f"{axis.label}: {value:g}" if isinstance(value, float) \
            else f"{axis.label}: {value}"
    return str(value)


# ----- applying values to GenParams -----------------------------------------
def apply_value(axis: AxisDef, p: GenParams, value, values: list) -> GenParams:
    """Return a copy of ``p`` with one axis value applied."""
    if axis.kind == "none":
        return p
    if axis.id == "seed":
        return replace(p, seed=int(value))
    if axis.id == "steps":
        return replace(p, steps=int(value))
    if axis.id == "cfg":
        return replace(p, cfg=float(value))
    if axis.id == "sampler":
        return replace(p, sampler=str(value))
    if axis.id == "scheduler":
        return replace(p, scheduler=str(value))
    if axis.id == "model":
        # モデル軸は通常のモデルファイル前提: マージ設定は外して差し替える。
        return replace(p, diffusion=str(value), merge_models=[],
                       merge_quant="", merge_low_memory=False)
    if axis.id == "size":
        return replace(p, width=int(value[0]), height=int(value[1]))
    if axis.id == "dtype":
        return replace(p, weight_dtype=str(value))
    if axis.id == "prompt_sr":
        search = str(values[0])
        if search not in p.prompt and search not in p.negative:
            raise ValueError(
                f"プロンプト S/R: 検索語「{search}」がプロンプトにも"
                "ネガティブにも見つかりません")
        return replace(p, prompt=p.prompt.replace(search, str(value)),
                       negative=p.negative.replace(search, str(value)))
    raise ValueError(f"unknown axis: {axis.id}")


def plan_cells(base: GenParams, axes: list[AxisDef],
               values: list[list]) -> list[tuple[int, GenParams]]:
    """Build the execution plan: [(grid_index, params), ...].

    grid_index = ix + iy*nx + iz*nx*ny （グリッド上の位置）。実行順は切替コスト
    が最大の軸が最外ループになるよう並べ替える（モデル軸のリロード回数最小化）。
    既定の入れ子は Z→Y→X。値の適用も X→Y→Z の固定順（順序依存を避ける）。
    """
    nx, ny = len(values[0]), len(values[1])
    # axis index (0=x,1=y,2=z) から最外→最内のループ順へ。等コストなら z,y,x。
    nest = sorted((2, 1, 0), key=lambda i: axes[i].cost, reverse=True)
    plan: list[tuple[int, GenParams]] = []

    def rec(depth: int, picked: dict[int, int]) -> None:
        if depth == 3:
            ix, iy, iz = picked[0], picked[1], picked[2]
            p = base
            for a in (0, 1, 2):
                p = apply_value(axes[a], p, values[a][picked[a]], values[a])
            plan.append((ix + iy * nx + iz * nx * ny, p))
            return
        a = nest[depth]
        for i in range(len(values[a])):
            picked[a] = i
            rec(depth + 1, picked)

    rec(0, {})
    return plan


# ----- grid composition ------------------------------------------------------
_BG = QColor("white")
_FG = QColor("black")
_ERR_BG = QColor("#d0d0d0")


def _fit_font(painter: QPainter, rect: QRect, text: str, base_px: int) -> None:
    """Set painter font to the largest size (<= base_px) whose wrapped text
    fits ``rect``."""
    px = base_px
    while px > 8:
        f = painter.font()
        f.setPixelSize(px)
        painter.setFont(f)
        needed = painter.fontMetrics().boundingRect(
            rect, Qt.TextWordWrap | Qt.AlignCenter, text)
        if needed.width() <= rect.width() and needed.height() <= rect.height():
            return
        px -= 2


def _draw_label(painter: QPainter, rect: QRect, text: str, base_px: int) -> None:
    if not text:
        return
    painter.save()
    _fit_font(painter, rect, text, base_px)
    painter.setPen(_FG)
    painter.drawText(rect, Qt.TextWordWrap | Qt.AlignCenter, text)
    painter.restore()


def compose_grid(cells: list[Optional[QImage]], nx: int, ny: int, nz: int,
                 x_labels: list[str], y_labels: list[str], z_labels: list[str],
                 draw_legend: bool = True, margin: int = 0) -> QImage:
    """Assemble cell images (index = ix + iy*nx + iz*nx*ny) into one image.

    複数 Z はサブグリッドを縦に積み、各サブグリッドの上に Z ラベルの帯を
    描く。None のセルはグレー地に「エラー」。
    """
    valid = [c for c in cells if c is not None and not c.isNull()]
    cw = max((c.width() for c in valid), default=512)
    ch = max((c.height() for c in valid), default=512)

    base_px = max(16, (cw + ch) // 30)
    show_x = draw_legend and any(x_labels)
    show_y = draw_legend and any(y_labels)
    show_z = draw_legend and any(z_labels)
    pad_left = cw * 3 // 8 if show_y else 0
    pad_top = base_px * 3 if show_x else 0
    z_head = base_px * 2 if show_z else 0
    z_gap = max(margin, base_px // 2) if nz > 1 else 0

    grid_w = pad_left + nx * cw + (nx - 1) * margin
    sub_h = z_head + pad_top + ny * ch + (ny - 1) * margin
    grid_h = nz * sub_h + (nz - 1) * z_gap

    out = QImage(grid_w, grid_h, QImage.Format_RGB32)
    out.fill(_BG)
    painter = QPainter(out)
    try:
        for iz in range(nz):
            top = iz * (sub_h + z_gap)
            if show_z:
                _draw_label(painter,
                            QRect(0, top, grid_w, z_head),
                            z_labels[iz], base_px)
            top += z_head
            if show_x:
                for ix in range(nx):
                    _draw_label(
                        painter,
                        QRect(pad_left + ix * (cw + margin), top + 2,
                              cw, pad_top - 4),
                        x_labels[ix], base_px)
            top += pad_top
            for iy in range(ny):
                cy = top + iy * (ch + margin)
                if show_y:
                    _draw_label(painter,
                                QRect(2, cy, pad_left - 4, ch),
                                y_labels[iy], base_px)
                for ix in range(nx):
                    cx = pad_left + ix * (cw + margin)
                    img = cells[ix + iy * nx + iz * nx * ny]
                    if img is None or img.isNull():
                        painter.fillRect(QRect(cx, cy, cw, ch), _ERR_BG)
                        _draw_label(painter, QRect(cx, cy, cw, ch),
                                    "エラー", base_px)
                    else:
                        # 小さいセルは中央寄せ（サイズ軸で解像度が混在し得る）
                        painter.drawImage(cx + (cw - img.width()) // 2,
                                          cy + (ch - img.height()) // 2, img)
    finally:
        painter.end()
    return out


def qimage_png_bytes(img: QImage) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())

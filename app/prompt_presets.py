"""Prompt presets loaded from a user-editable CSV file (prompts.csv).

Columns: 1 = 設定名, 2 = プロンプト（MiniMax H3 にネガティブは無い）.
Rows whose first column starts with ``#`` are comments. The first preset row
doubles as the startup content of the prompt field. Prompts contain commas and
newlines, so values must be quoted — any spreadsheet app does this
automatically. The file is written with a UTF-8 BOM so Excel on Japanese
Windows opens it correctly.
"""
from __future__ import annotations

import csv
from pathlib import Path

# Seed content for a freshly created file: MiniMax H3 公式プロンプトガイド
# （3フィールド形式）に沿ったテンプレート集。（日本語の説明）部分を書き換えて使う。
TEMPLATE = '''\
#1個目の設定が起動時に読み込まれます。
#プロンプトは改行を含むため必ずダブルクォーテーション( " )で囲ってください。
#書式: 設定名, プロンプト
#
#MiniMax H3 のプロンプトは英語の3フィールド形式:
#  integrated_multimodal_description = 映像・動作・カメラ・話者・台詞・同期音を時系列で
#  overall_soundscape = 動画全体の環境音・動作音
#  non_diegetic_music = BGM（登場人物には聞こえない音楽。無ければ N/A）
#台詞は <d>[Language] ...</d>、話者には (S1) (S2) と番号を振ります。
#ショットは [Shot 1]、2個目以降は [Shot 2] At 00:05.000, ... と切替時刻を書きます。
#i2v では冒頭に整列指示文を1行 + 空行が必要です（テンプレート参照）。
#FLF の S.SS 秒（例 5.00）は「長さ」設定と一致させてください。
"T2V テンプレート","integrated_multimodal_description: [Shot 1] (ここに英語で、映像スタイル・構図・被写体と場面・動作・カメラの動きを時系列で記述。台詞がある場合は話者に (S1) を付けて <d>[English] 台詞</d> と書く) [Shot 2] At 00:05.000, (ここに英語で2個目のショットを記述。ショットが1個なら [Shot 2] ごと削除)

overall_soundscape: (ここに英語で、動画全体の環境音・動作音を記述)

non_diegetic_music: (ここに英語でBGMを記述。BGM無しなら N/A)"
"I2V テンプレート（先頭フレーム）","For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] (ここに英語で、<Picture 1> の被写体・構図・場面を起点として、そこから始まる動作と展開を時系列で記述。外見・服装・配置は画像と一致させる)

overall_soundscape: (ここに英語で、動画全体の環境音・動作音を記述)

non_diegetic_music: (ここに英語でBGMを記述。BGM無しなら N/A)"
"I2V テンプレート（先頭+終端 FLF）","How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 5.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] (ここに英語で、Picture 1 の状態から始まり、目に見える中間変化を経て、最後に Picture 2 の構図・ポーズ・照明に着地するまでの流れを記述)

overall_soundscape: (ここに英語で、動画全体の環境音・動作音を記述)

non_diegetic_music: (ここに英語でBGMを記述。BGM無しなら N/A)"
"R2V テンプレート（参照から動画）","subject_definitions:
<Subject 1> is (ここに英語で、<Picture 1> 等の参照から使う人物・物と、その外見の特徴を記述。参照ごとに <Subject 2> ... と行を追加)

summary:
[reference generation] (ここに英語で、動画の概要と各参照の役割を1段落で記述)

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - (ここに英語で、保持される特徴を記述)

detailed_description:
(ここに英語で、映像スタイル・照明・色調を1〜2文で記述)
[Shot 1] (ここに英語で、映像・動作・カメラ・台詞を時系列で記述。<Subject 1> の初登場時に特徴と画面内の位置を書く)

overall_soundscape: (ここに英語で、動画全体の環境音・動作音を記述)

non_diegetic_music: (ここに英語でBGMを記述。BGM無しなら N/A)"
'''


def ensure_file(path: Path) -> None:
    """Create the CSV with an example row if it does not exist yet."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TEMPLATE, encoding="utf-8-sig")


def load(path: Path) -> list[tuple[str, str, str]]:
    """Return [(name, prompt, negative), ...]; missing file -> empty list.

    Rows with an empty first column and comment rows (first column starting
    with ``#``) are skipped, extra columns are ignored, and short rows are
    padded with empty strings.
    """
    if not path.exists():
        return []
    out: list[tuple[str, str, str]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            if row[0].lstrip().startswith("#"):
                continue
            name = row[0].strip()
            prompt = row[1].strip() if len(row) > 1 else ""
            negative = row[2].strip() if len(row) > 2 else ""
            out.append((name, prompt, negative))
    return out

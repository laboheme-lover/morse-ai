"""電報形式(電気通信術の試験の方法)の音源を採点するための正解テキストの準備と、部分ごとの採点。

電気通信術受信練習用原信(audio2)の正解テキストは、次の3つの部分からなる。
    1行目        前文(字数・発信局・発信番号・受付時刻)  例「一四二　四七　七八　セ五、三九」
    続く1〜2行   名あて                                      例「オオイタムセン」マツドサンマル」…」
    残りの行     本文
本文の始まる行は、前文の先頭にある「字数」(= 本文の文字数)を手がかりに決める
(末尾から数えた文字数が字数に最も近い行の始まりを本文の始まりとする)。

従来の採点(evaluate_audio_variants.prepare_ground_truth)との違い:
- 「」」を「。」として扱う。「」」は区切りの信号で、モールス符号は句点「。」と同じ(.-.-..)。
  音だけでは区別できないため、モデルの「。」は正しい出力である。従来は「」」を符号で表せない
  文字として正解から除いていたため、正しい出力が「挿入」の誤りに数えられていた。
- 括弧「（」「）」を正解に残す。括弧は実際に送信されている(本文中の位置に音がある)。
  従来は正解から除いていたため、括弧の音を余分な文字(挿入)として数えていた。
"""

from __future__ import annotations

# ============================================================================
# このモジュールの役割
#   evaluate_telegram.py が使う採点の部品。正解テキストの準備(split_telegram_ground_truth)と、
#   前文・名あて・本文ごとの採点(compare_by_section)の2つからなる。
#
# 採点の考え方
#   認識結果は「どこまでが前文で、どこからが本文か」が分からない(後処理なしの出力は信号を含む)。
#   そこで正解の側を3つに分けておき、認識結果は全体のまま正解全体と1回だけ対応付ける。
#   正解の各文字がどの部分に属するかは分かっているので、一致・置換・脱落を部分ごとに数えられる。
#   後処理あり・なしの両方を、同じ正解・同じ方法で採点できる。
# ============================================================================
import difflib
import re
import unicodedata
from dataclasses import dataclass

from evaluate_real_samples import SMALL_KANA_TO_NORMAL, hiragana_to_katakana
from generate_real_text_synthetic_dataset import KANJI_DIGIT
from morse_table import is_supported

TITLE_LINE_RE = re.compile(r"電気通信術受信練習用原信")
# 区切り「」」は句点と同じ符号なので、採点では句点「。」にそろえる
SEPARATOR_AS_KUTEN = {"」": "。"}
# 括弧は全角にそろえる(NFKCで半角になったものを戻す)
BRACKETS = {"(": "（", ")": "）"}
SECTION_NAMES = ("前文", "名あて", "本文")


def normalize_telegram_text(raw_text: str) -> tuple[str, list[str]]:
    """1つの部分のテキストを、認識結果と比べられる形にする(「」」→「。」、括弧は残す)。"""
    # (1) 漢数字を数字に(例「一四二」→「142」)
    text = "".join(KANJI_DIGIT.get(ch, ch) for ch in raw_text)
    # (2) 全角・半角をそろえ(NFKC。全角括弧「（」は半角「(」になる)、ひらがなをカタカナに
    text = hiragana_to_katakana(unicodedata.normalize("NFKC", text))
    # (3) 小書き仮名を通常の仮名に(和文モールスに小書きの符号は無い)
    text = text.translate(SMALL_KANA_TO_NORMAL)
    # (4) 空白を除き、「」」→「。」、半角括弧→全角括弧(後処理の出力「（」「）」と比べられるように)
    text = "".join(SEPARATOR_AS_KUTEN.get(ch, BRACKETS.get(ch, ch)) for ch in text if not ch.isspace())
    # (5) 符号で表せる文字と括弧だけを残す。除いた文字は記録用に返す
    kept = "".join(ch for ch in text if is_supported(ch) or ch in BRACKETS.values())
    excluded = sorted({ch for ch in text if not (is_supported(ch) or ch in BRACKETS.values())})
    return kept, excluded


@dataclass
class TelegramGroundTruth:
    """部分ごとに分けた正解テキスト。"""

    header: str
    address: str
    body: str
    declared_count: int | None  # 前文の字数(読み取れなければ None)
    excluded: list[str]

    @property
    def text(self) -> str:
        return self.header + self.address + self.body

    @property
    def sections(self) -> list[tuple[str, str]]:
        return list(zip(SECTION_NAMES, (self.header, self.address, self.body)))


def split_telegram_ground_truth(raw_text: str) -> TelegramGroundTruth:
    """正解テキストを前文・名あて・本文に分ける。電報形式でなければ全体を本文とする。"""
    # 送信されない見出し行(「…電気通信術受信練習用原信(和文) 電文1」)と空行を除く
    lines = [line.strip() for line in raw_text.splitlines() if line.strip() and not TITLE_LINE_RE.search(line)]
    excluded: set[str] = set()

    def norm(part: str) -> str:
        text, ex = normalize_telegram_text(part)
        excluded.update(ex)
        return text

    # 1行目の先頭が「数字+空白」なら電報形式の前文とみなし、その数字を字数として読む
    # (本文だけのテキストでは一致しないので、全体を本文として返す)
    count_match = re.match(r"^([〇一二三四五六七八九0-9]+)[\s　]", lines[0]) if lines else None
    if not count_match or len(lines) < 3:
        return TelegramGroundTruth("", "", norm("".join(lines)), None, sorted(excluded))
    declared = int("".join(KANJI_DIGIT.get(ch, ch) for ch in count_match.group(1)))
    # 2行目以降(名あて+本文)。行の中の空白は表示上のものなので除く
    rest = ["".join(line.split()) for line in lines[1:]]
    # 末尾から数えた長さが字数に最も近い行の始まりを本文の始まりとする(名あての行は1行以上残す)
    tail_lengths = [len("".join(rest[i:])) for i in range(1, len(rest))]
    # 例: 字数142、2行目以降の各行の始まりから末尾までの長さが [170, 142, 110, …] なら、
    # 長さ142の2つ目の行(rest[1])から本文。audio2の20件では17件が完全に一致、3件は1〜2文字の差だった
    body_start = 1 + min(range(len(tail_lengths)), key=lambda i: abs(tail_lengths[i] - declared))
    return TelegramGroundTruth(
        header=norm(lines[0]),
        address=norm("".join(rest[:body_start])),
        body=norm("".join(rest[body_start:])),
        declared_count=declared,
        excluded=sorted(excluded),
    )


def compare_by_section(predicted: str, sections: list[tuple[str, str]]) -> dict[str, dict]:
    """全体を1回だけ対応付け、正解の各文字がどの部分に属するかで一致・置換・脱落・挿入を部分ごとに数える。

    部分ごとに別々に対応付けると、認識結果のどこまでがどの部分かを先に決める必要があり、
    その決め方で結果が変わってしまう。全体で対応付ければ、その必要がない。
    挿入(正解に無い余分な文字)は、挿入された位置の直前の正解の文字が属する部分に数える
    (部分の境目に入った区切りの信号は、前の部分の終わりに付いたものとみなす)。
    戻り値は {部分の名前: {"matched", "substituted", "deleted", "inserted", "length", "char_match_rate"}}。
    全体の値は "全体" に入れる。
    """
    # 正解全体と、その各文字がどの部分(前文/名あて/本文)に属するかの対応表(owner)を作る
    gt = "".join(text for _, text in sections)
    owner: list[str] = []
    for name, text in sections:
        owner.extend([name] * len(text))
    counts = {name: {"matched": 0, "substituted": 0, "deleted": 0, "inserted": 0} for name, _ in sections}

    # 挿入を数える部分: 挿入位置の直前の正解の文字が属する部分(先頭への挿入は最初の部分)
    def section_before(i: int) -> str:
        if not owner:
            return sections[-1][0]
        return owner[max(0, min(i - 1, len(owner) - 1))] if i > 0 else owner[0]

    # evaluate_real_samples.compare_texts と同じ対応付け(autojunk=False)。対応の種類ごとに部分へ振り分ける
    matcher = difflib.SequenceMatcher(None, gt, predicted, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                counts[owner[i]]["matched"] += 1
        elif tag == "replace":
            paired = min(i2 - i1, j2 - j1)
            for k, i in enumerate(range(i1, i2)):
                counts[owner[i]]["substituted" if k < paired else "deleted"] += 1
            if j2 - j1 > paired:
                counts[section_before(i2)]["inserted"] += (j2 - j1) - paired
        elif tag == "delete":
            for i in range(i1, i2):
                counts[owner[i]]["deleted"] += 1
        elif tag == "insert":
            counts[section_before(i1)]["inserted"] += j2 - j1

    # 部分ごとの文字一致率 = 一致した文字数 / その部分の正解の文字数(挿入率も同じ分母で別に数える)
    result: dict[str, dict] = {}
    for name, text in sections:
        c = counts[name]
        result[name] = {**c, "length": len(text), "char_match_rate": c["matched"] / max(len(text), 1)}
    total = {k: sum(counts[name][k] for name, _ in sections) for k in ("matched", "substituted", "deleted", "inserted")}
    result["全体"] = {**total, "length": len(gt), "char_match_rate": total["matched"] / max(len(gt), 1)}
    return result

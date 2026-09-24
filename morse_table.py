"""和文モールス符号(Wabun code)の符号表定義モジュール。

短点(トン)を "." 、長点(ツー)を "-" として符号を文字列で表現する。
符号の出典は日本アマチュア無線連盟(JARL)配布の早見表、および複数の
公開資料(benricho.org, ja.wikipedia.org, morsecodes.net)を相互照合して
採用した。
"""

from __future__ import annotations

# 和文カタカナ 48字の基本符号(濁音・半濁音は含まない)
WABUN_KANA_CODE: dict[str, str] = {
    "イ": ".-",
    "ロ": ".-.-",
    "ハ": "-...",
    "ニ": "-.-.",
    "ホ": "-..",
    "ヘ": ".",
    "ト": "..-..",
    "チ": "..-.",
    "リ": "--.",
    "ヌ": "....",
    "ル": "-.--.",
    "ヲ": ".---",
    "ワ": "-.-",
    "カ": ".-..",
    "ヨ": "--",
    "タ": "-.",
    "レ": "---",
    "ソ": "---.",
    "ツ": ".--.",
    "ネ": "--.-",
    "ナ": ".-.",
    "ラ": "...",
    "ム": "-",
    "ウ": "..-",
    "ヰ": ".-..-",
    "ノ": "..--",
    "オ": ".-...",
    "ク": "...-",
    "ヤ": ".--",
    "マ": "-..-",
    "ケ": "-.--",
    "フ": "--..",
    "コ": "----",
    "エ": "-.---",
    "テ": ".-.--",
    "ア": "--.--",
    "サ": "-.-.-",
    "キ": "-.-..",
    "ユ": "-..--",
    "メ": "-...-",
    "ミ": "..-.-",
    "シ": "--.-.",
    "ヱ": ".--..",
    "ヒ": "--..-",
    "モ": "-..-.",
    "セ": ".---.",
    "ス": "---.-",
    "ン": ".-.-.",
}

# 濁点・半濁点・長音符・読点・句点(主要記号)
DAKUTEN_MARK = "゛"
HANDAKUTEN_MARK = "゜"
CHOONPU = "ー"
TOUTEN = "、"
# 句点の符号 .-.-.. は、実録音(Taman_86/57)で符号表に無い記号として現れた
# パターンの出現回数が、正解テキスト中の「。」の個数と一致したことから確認した
KUTEN = "。"

MARK_CODE: dict[str, str] = {
    DAKUTEN_MARK: "..",
    HANDAKUTEN_MARK: "..--.",
    CHOONPU: ".--.-",
    TOUTEN: ".-.-.-",
    KUTEN: ".-.-..",
}

# 数字0-9 (国際モールス符号と共通)
DIGIT_CODE: dict[str, str] = {
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    "0": "-----",
}

# 濁音仮名 -> 基本仮名 (ガ = カ + 濁点 のように展開する)
DAKUTEN_KANA: dict[str, str] = {
    "ガ": "カ", "ギ": "キ", "グ": "ク", "ゲ": "ケ", "ゴ": "コ",
    "ザ": "サ", "ジ": "シ", "ズ": "ス", "ゼ": "セ", "ゾ": "ソ",
    "ダ": "タ", "ヂ": "チ", "ヅ": "ツ", "デ": "テ", "ド": "ト",
    "バ": "ハ", "ビ": "ヒ", "ブ": "フ", "ベ": "ヘ", "ボ": "ホ",
    "ヴ": "ウ",
}

# 半濁音仮名 -> 基本仮名 (パ = ハ + 半濁点 のように展開する)
HANDAKUTEN_KANA: dict[str, str] = {
    "パ": "ハ", "ピ": "ヒ", "プ": "フ", "ペ": "ヘ", "ポ": "ホ",
}

# 全ての「単一符号で送出できる文字」をまとめた符号表
MORSE_CODE_TABLE: dict[str, str] = {
    **WABUN_KANA_CODE,
    **MARK_CODE,
    **DIGIT_CODE,
}

# 符号 -> 文字 の逆引き表
CODE_TO_CHAR: dict[str, str] = {code: ch for ch, code in MORSE_CODE_TABLE.items()}


def is_supported(char: str) -> bool:
    """指定した1文字が符号表(濁音・半濁音の合成仮名を含む)で表現できるか判定する。"""
    return char in MORSE_CODE_TABLE or char in DAKUTEN_KANA or char in HANDAKUTEN_KANA


def char_to_units(char: str) -> list[str]:
    """1文字を送出する際のモールス符号列を返す。

    通常の文字は要素1つのリストを返す。濁音・半濁音の合成仮名(ガ, パ 等)は
    基本仮名の符号と付加記号(濁点/半濁点)の符号の2要素に展開する。
    """
    if char in MORSE_CODE_TABLE:
        return [MORSE_CODE_TABLE[char]]
    if char in DAKUTEN_KANA:
        base = DAKUTEN_KANA[char]
        return [WABUN_KANA_CODE[base], MARK_CODE[DAKUTEN_MARK]]
    if char in HANDAKUTEN_KANA:
        base = HANDAKUTEN_KANA[char]
        return [WABUN_KANA_CODE[base], MARK_CODE[HANDAKUTEN_MARK]]
    raise KeyError(f"未対応の文字です: {char!r}")


def code_to_char(code: str) -> str:
    """単一のモールス符号(例 '.-')から対応する文字を返す。"""
    if code not in CODE_TO_CHAR:
        raise KeyError(f"未対応の符号です: {code!r}")
    return CODE_TO_CHAR[code]


# 逆引き: 基本仮名 -> 濁音/半濁音の合成仮名
_DAKUTEN_BASE_TO_KANA: dict[str, str] = {base: composite for composite, base in DAKUTEN_KANA.items()}
_HANDAKUTEN_BASE_TO_KANA: dict[str, str] = {base: composite for composite, base in HANDAKUTEN_KANA.items()}


def compose_pair(base: str, mark: str) -> str | None:
    """基本仮名と濁点/半濁点の1組を合成仮名(ガ・プ等)へ結合する。結合できなければNone。"""
    if mark == DAKUTEN_MARK:
        return _DAKUTEN_BASE_TO_KANA.get(base)
    if mark == HANDAKUTEN_MARK:
        return _HANDAKUTEN_BASE_TO_KANA.get(base)
    return None


def compose_dakuten(text: str) -> str:
    """「基本仮名」+「濁点/半濁点」の並びを、ガ・プ等の合成仮名へ結合する。

    char_to_units()が行う分解の逆変換にあたる。CNNは基本仮名と濁点/半濁点を
    別々の文字として認識するため(Milestone 4のクラス設計)、連続音声認識
    (predict_sequence.py)の出力をそのまま表示すると「カ゛」のようになって
    しまう。この関数で通常の日本語表記(ガ)へ変換する。
    """
    result: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if i + 1 < len(text):
            composed = compose_pair(ch, text[i + 1])
            if composed is not None:
                result.append(composed)
                i += 2
                continue
        result.append(ch)
        i += 1
    return "".join(result)

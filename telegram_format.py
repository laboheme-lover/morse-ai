"""電報形式(電気通信術の試験の方法)の音声を、認識結果の後処理で前文・名あて・本文に分けて読む。

無線の電報は、本文の前後に決まった信号と前文が付いて送られる(和文電報形式)。
    開始の信号 → 前文(字数・発信局・発信番号・受付時刻)→ 区切り → 名あて → 本文の始まりの信号
    → 本文(60字ごとに信号を送り、約5秒あける)→ 終了の信号
モデルは1文字ずつ・符号表の63文字のどれかとして読むので、符号表に無い信号は最も近い仮名
(本文の始まり -..--- → エ、60字ごと ..--.. → ト、終了 ...-. → チ、括弧 -.--.- → サ など)として出てくる。
また受付時刻の数字は略体(1 = .- など)で送られるため、仮名(イ など)として出てくる。

この後処理は、
- 音から読み取った短点・長点の並び(read_element_patterns)で、符号表に無い信号と括弧を見分け、
- 前文を欄ごとに分けて、受付時刻の欄の中だけ略体の数字を数字に戻し、
- 信号を本文から取り除き、括弧を戻す。
電報形式かどうかは、前文の形(数字 → 局 → 数字 → コ/セ+時刻)と、ほかの信号が1つ以上あることで判定する。
無線以外の送信(本文だけ)では前文が無いので、判定が外れて認識結果はそのまま返る。

信号の符号は、audio2(電気通信術受信練習用原信)の原音20件で、該当する区間の短点・長点の並びを
音から読み取って確かめた(開始 .-.-.- 20件、本文の始まり -..--- 20件、60字ごと ..--.. 34件、
終了 ...-. 20件、括弧 -.--.- / .-..-. 各15件)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from morse_segmentation import compute_frame_rms, detect_active_frames_hysteresis, frames_to_runs
from morse_table import MORSE_CODE_TABLE, compose_dakuten

# ---- 符号表に無い信号(音から読み取った短点・長点の並び → 意味)
BODY_START_CODE = "-..---"  # 名あてと本文の区切り(本文の始まり)
EVERY_60_CODE = "..--.."  # 本文の60字ごとの信号(このあと約5秒あける)
END_CODES = ("...-.", "...-.-")  # 終了の信号(...-.-が続くことがある)
OPEN_BRACKET_CODE = "-.--.-"
CLOSE_BRACKET_CODE = ".-..-."
BRACKET_BY_CODE = {OPEN_BRACKET_CODE: "（", CLOSE_BRACKET_CODE: "）"}

# ---- 符号表の文字として出てくる信号
START_SIGNAL = "、"  # 開始の信号(.-.-.-、読点と同じ符号)
HR = "ヌナ"  # HR(.... .-.)。開始の信号の前に送られることがある
STATION_NUMBER_PREFIX = "ハツ"  # 発信局を番号で表すときに前に付ける
SERIAL_NUMBER_PREFIX = "タナ"  # 発信番号の前に付ける

# 略体の数字(受付時刻で使う)。1 .- / 2 ..- / 3 ...- / 7 -... / 8 -.. / 9 -. / 0 - は仮名と同じ符号になる。
# 4・5・6 は通常の数字と同じ符号(audio2の原音でも4・5・6は通常の符号で送られていた)。
RYAKUTAI_DIGITS = {"イ": "1", "ウ": "2", "ク": "3", "ハ": "7", "ホ": "8", "タ": "9", "ム": "0"}
TIME_CHAR_CLASS = "[" + "".join(RYAKUTAI_DIGITS) + "0-9]"

# 前文: 字数 → (ハツ+局番号[+タナ] | 局名) → 発信番号 → コ/セ → 時、分 → 区切りの「、」
HEADER_RE = re.compile(
    rf"^(?P<count>\d{{1,4}})"
    rf"(?:{STATION_NUMBER_PREFIX}(?P<station_number>\d{{1,4}})(?:{SERIAL_NUMBER_PREFIX})?|(?P<station_name>\D+?))"
    rf"(?P<serial>\d{{1,4}})"
    rf"(?P<ampm>[コセ])(?P<hour>{TIME_CHAR_CLASS}{{1,2}})、(?P<minute>{TIME_CHAR_CLASS}{{1,2}})、"
)

# 60字ごとの信号のあとの「約5秒」の無音を見分ける長さ。最も遅い送信(短点0.19秒)でも
# 文字の間の無音(7単位)は1.3秒程度なので、2秒を超える無音は通常の文字間ではない。
LONG_PAUSE_SECONDS = 2.0
# 音から信号を読み取れなかったとき、字数から予想した本文の始まりの位置の前後この文字数以内にある
# 「エ」を本文の始まりの信号とみなす
BODY_START_SEARCH_WINDOW = 5
# 括弧は開きと閉じが対になる。音から読んだ符号が括弧でも、この区間数以内に相手が無ければ
# (雑音で「ン」.-.-. が .-..-. と読めた場合など)括弧に戻さない
BRACKET_PAIR_MAX_TOKENS = 30

# 雑音の強い音声では、雑音の短いかたまりが短点1つ(ヘ)と読まれて前文の途中に入り、前文の形が崩れる。
# 前文がそのままでは読めないときは、ヘを飛ばして読み直す(前文の欄にヘは現れない。局名の中のヘは失われる)。
# 最初の信号の前・最後の信号の後の無音に乗った雑音も ヘ と読まれるので、開始・終了の信号を探す前に除く
# (電報形式と判定できなかったときは、除かずに認識結果をそのまま返す)。
NOISE_BLIP = "ヘ"

MODES = ("auto", "off", "on")


def read_element_patterns(
    y: np.ndarray, sample_rate: int, ranges: list[tuple[int, int]], dot_seconds: float
) -> list[str]:
    """各区間 (開始, 終了)[サンプル] の中の短点・長点の並びを、音の長さから読み取る("." と "-" の文字列)。

    区間分割(morse_segmentation)と同じ有音/無音の判定を音声全体にかけ、区間の中に収まる有音の
    長さを、短点の2倍以上なら長点、未満なら短点とする(ヘ/ムの補正と同じ基準)。
    雑音が強いと有音の判定が崩れるため、読み取った並びは「符号表に無い信号の符号と完全に一致したとき」
    だけ使う(interpret_telegram)。
    """
    rms, frame_length = compute_frame_rms(y, sample_rate)
    runs = [(s, e) for s, e, on in frames_to_runs(detect_active_frames_hysteresis(rms), frame_length, len(y)) if on]
    starts = np.array([s for s, _ in runs])
    margin = frame_length  # 区間の端と有音の端は1フレーム程度ずれることがある
    patterns: list[str] = []
    for start, end in ranges:
        first = int(np.searchsorted(starts, start - margin))
        elements = []
        for s, e in runs[first:]:
            if s > end + margin:
                break
            if e <= end + margin:
                elements.append("-" if (e - s) / sample_rate >= 2 * dot_seconds else ".")
        patterns.append("".join(elements))
    return patterns


@dataclass
class Token:
    """認識した1区間(モデルの出力文字・音から読んだ符号・直後の無音の長さ)。"""

    text: str
    pattern: str | None = None
    pause_after: float | None = None  # 次の区間までの無音[秒]。最後の区間は None


@dataclass
class TelegramReading:
    """電報形式として読んだ結果。detected が False なら text は認識結果そのまま。"""

    detected: bool
    text: str  # 信号を除き、括弧と時刻の数字を戻した全文(前文の欄 + 名あて + 本文)
    raw_text: str  # 後処理前の認識結果
    header_fields: dict[str, str] = field(default_factory=dict)  # 字数・発信局・発信番号・受付時刻
    header: str = ""  # 前文(採点用の連結: 字数+局+発信番号+コ/セ+時、分)
    address: str = ""
    body: str = ""
    signals: list[str] = field(default_factory=list)  # 見つけた信号(画面表示用)
    warnings: list[str] = field(default_factory=list)


def tokens_from_sequence(sequence, sample_rate: int, patterns: list[str] | None = None) -> list[Token]:
    """SequencePrediction(区間ごとの結果)を Token の列にする。"""
    chars = sequence.characters
    tokens = []
    for i, c in enumerate(chars):
        pause = (chars[i + 1].start_sample - c.end_sample) / sample_rate if i + 1 < len(chars) else None
        tokens.append(Token(c.result.prediction, patterns[i] if patterns else None, pause))
    return tokens


def _is(token: Token, *codes: str) -> bool:
    """音から読んだ符号が codes のどれかと一致し、かつモデルの文字の符号とは違う(=符号表に無い信号)。"""
    return token.pattern in codes and MORSE_CODE_TABLE.get(token.text) != token.pattern


def interpret_telegram(tokens: list[Token], mode: str = "auto") -> TelegramReading:
    """認識した区間の列を電報形式として読む。mode: auto(判定して適用)/ off(しない)/ on(判定せず適用)。"""
    raw_text = compose_dakuten("".join(t.text for t in tokens))
    if mode == "off" or not tokens:
        return TelegramReading(False, raw_text, raw_text)

    signals: list[str] = []
    warnings: list[str] = []
    toks = list(tokens)

    # (0) 最初の信号の前・最後の信号の後の雑音(ヘ)を除く
    first = 0
    while first < len(toks) and toks[first].text == NOISE_BLIP:
        first += 1
    last = len(toks)
    while last > first and toks[last - 1].text == NOISE_BLIP:
        last -= 1
    toks = toks[first:last]

    # (1) 開始の信号: 先頭の HR(ヌナ)の繰り返しと「、」
    i = 0
    labels = "".join(t.text for t in toks)
    while labels.startswith(HR, i):
        i += len(HR)
    has_start = labels.startswith(START_SIGNAL, i)
    if has_start:
        signals.append("開始" + ("(HR付き)" if i > 0 else ""))
        i += len(START_SIGNAL)
    else:
        i = 0
    toks = toks[i:]

    # (2) 終了の信号: 末尾の ...-. / ...-.-(音で読めないときは末尾の「チ」)
    end = len(toks)
    while end > 0 and _is(toks[end - 1], *END_CODES):
        end -= 1
    if end == len(toks) and end > 0 and toks[end - 1].text == "チ" and toks[end - 1].pattern in (None, *END_CODES):
        end -= 1
    has_end = end < len(toks)
    if has_end:
        signals.append("終了")
    toks = toks[:end]

    # (3) 前文
    m, rest_from = _match_header(toks)
    header_fields: dict[str, str] = {}
    header = ""
    if m:
        hour = "".join(RYAKUTAI_DIGITS.get(c, c) for c in m["hour"])
        minute = "".join(RYAKUTAI_DIGITS.get(c, c) for c in m["minute"])
        station = m["station_number"] or compose_dakuten(m["station_name"])
        header_fields = {
            "字数": m["count"],
            "発信局": (f"{station}(番号)" if m["station_number"] else station),
            "発信番号": m["serial"],
            "受付時刻": f"{m['ampm']}{hour}時{minute}分",
        }
        header = f"{m['count']}{station}{m['serial']}{m['ampm']}{hour}、{minute}"
    rest = toks[rest_from:]

    # (4) 本文の始まりの信号(-..---)。音で読めなければ、字数から予想した位置に近い「エ」
    body_start = next((k for k, t in enumerate(rest) if _is(t, BODY_START_CODE)), None)
    if body_start is None and m:
        declared = int(m["count"])
        n_every60 = max(0, (declared - 1) // 60)
        expected = len(rest) - declared - n_every60 - 1
        candidates = [k for k, t in enumerate(rest) if t.text == "エ" and abs(k - expected) <= BODY_START_SEARCH_WINDOW]
        if candidates:
            body_start = min(candidates, key=lambda k: abs(k - expected))
    has_body_start = body_start is not None

    detected = bool(m) and (has_start or has_end or has_body_start)
    if mode == "auto" and not detected:
        return TelegramReading(False, raw_text, raw_text)

    if has_body_start:
        signals.append("本文の始まり")
        address_toks, body_toks = rest[:body_start], rest[body_start + 1 :]
    else:
        address_toks, body_toks = [], rest
        warnings.append("名あてと本文の区切りが見つからないため、前文より後をすべて本文としました")

    # (5) 本文: 60字ごとの信号を除き、括弧を戻す
    #     60字ごとの信号は、必ず約5秒の無音が続く。音の符号だけで決めると、雑音で ヱ(.--..)が ..--.. と
    #     読めたときに本文の文字を消してしまうため、無音が続くことも条件にする。無音の中に雑音(ヘ)が
    #     入ると無音が途切れて見えるので、ヘを飛ばして無音の長さを足し合わせ、そのヘも本文から除く。
    kept_body: list[Token] = []
    n60 = 0
    k = 0
    while k < len(body_toks):
        t = body_toks[k]
        looks_like = _is(t, EVERY_60_CODE) or (t.text == "ト" and t.pattern in (None, EVERY_60_CODE))
        if looks_like:
            pause, after = _pause_skipping_blips(body_toks, k)
            if pause >= LONG_PAUSE_SECONDS:
                n60 += 1
                k = after
                continue
        kept_body.append(t)
        k += 1
    if n60:
        signals.append(f"60字ごと×{n60}")
    address = compose_dakuten("".join(_restore_brackets(address_toks)))
    body = compose_dakuten("".join(_restore_brackets(kept_body)))

    if m and len(body) != int(m["count"]):
        warnings.append(f"本文の文字数({len(body)}字)が前文の字数({m['count']}字)と合いません。要確認")

    return TelegramReading(
        detected=True,
        text=header + address + body,
        raw_text=raw_text,
        header_fields=header_fields,
        header=header,
        address=address,
        body=body,
        signals=signals,
        warnings=warnings,
    )


def _pause_skipping_blips(tokens: list[Token], k: int) -> tuple[float, int]:
    """tokens[k] のあとの無音の長さ(間に挟まった雑音 ヘ を飛ばして足し合わせる)と、無音の次の区間の位置。"""
    pause = 0.0
    j = k
    while j < len(tokens):
        pause += tokens[j].pause_after or 0.0
        j += 1
        if j >= len(tokens) or tokens[j].text != NOISE_BLIP:
            break
    return pause, j


def _match_header(toks: list[Token]) -> tuple[re.Match | None, int]:
    """前文を読む。戻り値は (一致, 前文の次の区間の位置)。読めなければ ヘ を飛ばして読み直す。"""
    m = HEADER_RE.match("".join(t.text for t in toks))
    if m:
        return m, m.end()
    kept = [k for k, t in enumerate(toks) if t.text != NOISE_BLIP]
    m = HEADER_RE.match("".join(toks[k].text for k in kept))
    if not m:
        return None, 0
    return m, (kept[m.end()] if m.end() < len(kept) else len(toks))


def _restore_brackets(tokens: list[Token]) -> list[str]:
    """音から読んだ符号が括弧で、開きと閉じが対になっているものだけを括弧に戻す。"""
    texts = [t.text for t in tokens]
    open_at = None
    for k, t in enumerate(tokens):
        if _is(t, OPEN_BRACKET_CODE):
            open_at = k  # 開きが続いたら、閉じに近いほうを使う
        elif _is(t, CLOSE_BRACKET_CODE) and open_at is not None and k - open_at <= BRACKET_PAIR_MAX_TOKENS:
            texts[open_at], texts[k] = BRACKET_BY_CODE[OPEN_BRACKET_CODE], BRACKET_BY_CODE[CLOSE_BRACKET_CODE]
            open_at = None
    return texts


def read_telegram(sequence, y: np.ndarray, sample_rate: int, dot_seconds: float, mode: str = "auto") -> TelegramReading:
    """認識結果(SequencePrediction)と認識に使った波形から、電報形式として読む(アプリ・評価用の入口)。"""
    if mode == "off":
        tokens = tokens_from_sequence(sequence, sample_rate)
        return interpret_telegram(tokens, "off")
    ranges = [(c.start_sample, c.end_sample) for c in sequence.characters]
    patterns = read_element_patterns(y, sample_rate, ranges, dot_seconds) if ranges else []
    return interpret_telegram(tokens_from_sequence(sequence, sample_rate, patterns), mode)

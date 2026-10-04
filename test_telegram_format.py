import numpy as np

from generate_morse_audio import code_to_waveform
from morse_table import MORSE_CODE_TABLE
from telegram_format import (
    BODY_START_CODE,
    CLOSE_BRACKET_CODE,
    END_CODES,
    EVERY_60_CODE,
    OPEN_BRACKET_CODE,
    Token,
    interpret_telegram,
    read_element_patterns,
)


def _tokens(spec: list) -> list[Token]:
    """文字列はモデルの出力(符号は符号表どおり)、(文字, 符号) は符号表に無い信号を表す。"""
    tokens = []
    for item in spec:
        for ch in item if isinstance(item, str) else [item]:
            if isinstance(ch, tuple):
                tokens.append(Token(ch[0], ch[1], 0.5))
            else:
                tokens.append(Token(ch, MORSE_CODE_TABLE.get(ch), 0.5))
    return tokens


BODY = "アイウ"


def _telegram(body=BODY, address="オタルムセン。サキシママル"):
    return _tokens(
        ["、", f"{len(body)}ハツ74タナ26コホ、5、", address, [("エ", BODY_START_CODE)][0], body, ("チ", END_CODES[0])]
    )


def test_numbered_station_header_is_split_into_fields():
    reading = interpret_telegram(_telegram())
    assert reading.detected
    assert reading.header_fields == {"字数": "3", "発信局": "74(番号)", "発信番号": "26", "受付時刻": "コ8時5分"}
    assert reading.header == "37426コ8、5"
    assert reading.address == "オタルムセン。サキシママル"
    assert reading.body == "アイウ"
    assert reading.text == "37426コ8、5オタルムセン。サキシママルアイウ"
    assert reading.warnings == []


def test_named_station_and_ryakutai_minutes():
    tokens = _tokens(["、", "3アタミマルチヨウシムセン67セイム、クタ、", "センダイ", ("エ", BODY_START_CODE), BODY, ("チ", END_CODES[0])])
    reading = interpret_telegram(tokens)
    assert reading.header_fields["発信局"] == "アタミマルチヨウシムセン"
    assert reading.header_fields["受付時刻"] == "セ10時39分"
    assert reading.header == "3アタミマルチヨウシムセン67セ10、39"


def test_ryakutai_digits_are_converted_only_in_time_field():
    # 本文の「タ」「ハ」は数字に変えない
    reading = interpret_telegram(_telegram(body="タハイ"))
    assert reading.body == "タハイ"


def test_every_60_signal_and_brackets_are_handled_by_sound_pattern():
    body = [("サ", OPEN_BRACKET_CODE), "ア", ("ン", CLOSE_BRACKET_CODE), ("ト", EVERY_60_CODE), "イト"]
    tokens = _tokens(["、", "5ハツ1タナ2コイ、イ、", "ナ", ("エ", BODY_START_CODE), *body, ("チ", END_CODES[0])])
    next(t for t in tokens if t.pattern == EVERY_60_CODE).pause_after = 5.0  # 60字ごとの信号のあとの約5秒
    reading = interpret_telegram(tokens)
    assert reading.body == "（ア）イト"  # 本物の「ト」(符号 ..-..)は残す
    assert "60字ごと×1" in reading.signals


def test_every_60_signal_by_long_pause_when_sound_pattern_is_unknown():
    tokens = _telegram(body="アトイ")
    for t in tokens:
        t.pattern = None  # 音から読めなかった場合
    tokens[-3].pause_after = 5.0  # 「ト」のあとに約5秒の無音
    reading = interpret_telegram(tokens)
    assert reading.body == "アイ"


def test_body_start_found_by_declared_count_when_sound_pattern_is_unknown():
    # 名あてにも「エ」がある(アタゴマエ)。字数から予想した位置に近いほうを本文の始まりとする
    tokens = _telegram(address="アタゴマエ150")
    for t in tokens:
        t.pattern = None
    reading = interpret_telegram(tokens)
    assert reading.address == "アタゴマエ150"
    assert reading.body == BODY


def test_hr_before_start_signal_and_trailing_end_codes():
    tokens = _tokens(["ヌナヌナ"]) + _telegram()[:-1] + _tokens([("チ", END_CODES[0]), ("ミ", END_CODES[1]), ("ミ", END_CODES[1])])
    reading = interpret_telegram(tokens)
    assert reading.signals[0] == "開始(HR付き)"
    assert reading.body == BODY


def test_body_only_text_is_returned_unchanged():
    tokens = _tokens(["コンニチハ、キヨウハハレ。ト"])
    reading = interpret_telegram(tokens)
    assert not reading.detected
    assert reading.text == "コンニチハ、キヨウハハレ。ト"


def test_header_alone_is_not_enough_in_auto_mode():
    tokens = _tokens(["3ハツ74タナ26コホ、5、アイウ"])  # 開始・本文の始まり・終了の信号が無い
    assert not interpret_telegram(tokens).detected
    assert interpret_telegram(tokens, mode="on").detected


def test_off_mode_returns_raw_text():
    reading = interpret_telegram(_telegram(), mode="off")
    assert not reading.detected
    assert reading.text == reading.raw_text


def test_char_count_mismatch_is_warned():
    tokens = _tokens(["、", "5ハツ74タナ26コホ、5、", "オタル", ("エ", BODY_START_CODE), BODY, ("チ", END_CODES[0])])
    reading = interpret_telegram(tokens)
    assert any("字数" in w for w in reading.warnings)


def test_read_element_patterns_from_audio():
    sr, dot = 16000, 0.06
    codes = [BODY_START_CODE, "..-..", OPEN_BRACKET_CODE]
    gap = np.zeros(int(sr * dot * 7), dtype=np.float32)
    pieces, ranges, pos = [], [], 0
    for code in codes:
        w = code_to_waveform(code, dot_duration=dot, sample_rate=sr)
        ranges.append((pos, pos + len(w)))
        pieces += [w, gap]
        pos += len(w) + len(gap)
    y = np.concatenate(pieces)
    assert read_element_patterns(y, sr, ranges, dot) == codes


def test_noise_blips_in_header_are_skipped():
    tokens = _tokens(["、", "3ハツ7ヘ4タヘナ26コホ、5、", "オタル", ("エ", BODY_START_CODE), "ヘイワ", ("チ", END_CODES[0])])
    reading = interpret_telegram(tokens)
    assert reading.header == "37426コ8、5"
    assert reading.address == "オタル"
    assert reading.body == "ヘイワ"  # 本文のヘは残す


def test_unpaired_bracket_pattern_keeps_model_text():
    body = ["アイ", ("ン", CLOSE_BRACKET_CODE), "ウ"]  # 閉じだけ(雑音で ン が括弧の符号に読めた)
    tokens = _tokens(["、", "4ハツ1タナ2コイ、イ、", "ナ", ("エ", BODY_START_CODE), *body, ("チ", END_CODES[0])])
    assert interpret_telegram(tokens).body == "アインウ"


def test_noise_blips_before_start_and_after_end_are_removed():
    tokens = _tokens(["ヘヘ"]) + _telegram() + _tokens(["ヘ"])
    reading = interpret_telegram(tokens)
    assert reading.detected
    assert reading.signals[0] == "開始"
    assert "終了" in reading.signals
    assert reading.text == "37426コ8、5オタルムセン。サキシママルアイウ"


def test_leading_blips_are_kept_when_not_a_telegram():
    tokens = _tokens(["ヘヘイロハ"])
    assert interpret_telegram(tokens).text == "ヘヘイロハ"


def test_every_60_pattern_without_pause_is_kept_as_text():
    # 雑音で ヱ が ..--.. と読めても、あとに約5秒の無音が無ければ本文の文字として残す
    tokens = _telegram(body="ア" + "ヱ" + "イ")
    hit = next(t for t in tokens if t.text == "ヱ")
    hit.pattern = EVERY_60_CODE
    assert interpret_telegram(tokens).body == "アヱイ"


def test_noise_blips_inside_every_60_pause_are_removed():
    tokens = _telegram(body="アトヘヘイ")
    marker = next(t for t in tokens if t.text == "ト")
    marker.pattern, marker.pause_after = EVERY_60_CODE, 1.5
    for t in tokens:
        if t.text == "ヘ":
            t.pause_after = 1.5  # 無音が雑音で3つに途切れて見える(合計4.5秒)
    assert interpret_telegram(tokens).body == "アイ"

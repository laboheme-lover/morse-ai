from telegram_scoring import compare_by_section, normalize_telegram_text, split_telegram_ground_truth

RAW = """一・二総通：電気通信術受信練習用原信（和文）　電文3

一四二　四七　七八　セ五、三九
オオイタムセン」マツドサンマル
アイウ（エオ）カ」キ
"""


def test_separator_becomes_kuten_and_brackets_are_kept():
    text, excluded = normalize_telegram_text("ア」イ（ウ）")
    assert text == "ア。イ（ウ）"
    assert excluded == []


def test_split_uses_declared_count_to_find_body():
    # 字数をわざと本文(10文字)に合わせる
    raw = RAW.replace("一四二", "一〇")
    gt = split_telegram_ground_truth(raw)
    assert gt.declared_count == 10
    assert gt.header == "104778セ5、39"
    assert gt.address == "オオイタムセン。マツドサンマル"
    assert gt.body == "アイウ（エオ）カ。キ"


def test_split_chooses_nearest_line_start_when_count_is_off_by_one():
    raw = RAW.replace("一四二", "一一")  # 実際の本文は10文字
    assert split_telegram_ground_truth(raw).body == "アイウ（エオ）カ。キ"


def test_non_telegram_text_is_all_body():
    gt = split_telegram_ground_truth("アイウエオ\nカキクケコ")
    assert (gt.header, gt.address, gt.body) == ("", "", "アイウエオカキクケコ")
    assert gt.declared_count is None


def test_compare_by_section_counts_per_section():
    sections = [("前文", "12"), ("名あて", "アイ"), ("本文", "ウエオ")]
    # 名あてと本文の境目に余分な「ス」、本文の「エ」を「ケ」に誤り
    result = compare_by_section("12アイスウケオ", sections)
    assert result["前文"]["char_match_rate"] == 1.0
    assert result["名あて"]["inserted"] == 1  # 境目の挿入は前の部分に数える
    assert result["本文"]["substituted"] == 1
    assert result["本文"]["matched"] == 2
    assert result["全体"]["matched"] == 6
    assert result["全体"]["length"] == 7

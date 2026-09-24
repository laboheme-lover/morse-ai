"""morse_table.py のユニットテスト。"""

import pytest

from morse_table import (
    DAKUTEN_KANA,
    HANDAKUTEN_KANA,
    MORSE_CODE_TABLE,
    WABUN_KANA_CODE,
    char_to_units,
    code_to_char,
    compose_dakuten,
    is_supported,
)


def test_kana_count():
    assert len(WABUN_KANA_CODE) == 48


def test_basic_kana_lookup():
    assert MORSE_CODE_TABLE["イ"] == ".-"
    assert MORSE_CODE_TABLE["ロ"] == ".-.-"
    assert MORSE_CODE_TABLE["ヲ"] == ".---"
    assert MORSE_CODE_TABLE["ン"] == ".-.-."


def test_digit_lookup():
    assert MORSE_CODE_TABLE["0"] == "-----"
    assert MORSE_CODE_TABLE["1"] == ".----"
    assert MORSE_CODE_TABLE["9"] == "----."


def test_mark_lookup():
    assert MORSE_CODE_TABLE["゛"] == ".."
    assert MORSE_CODE_TABLE["゜"] == "..--."
    assert MORSE_CODE_TABLE["ー"] == ".--.-"
    assert MORSE_CODE_TABLE["、"] == ".-.-.-"


def test_kuten_code_and_roundtrip():
    """句点(。)は .-.-.. で、ン(.-.-.)・読点(.-.-.-)とは別の符号として扱われる。"""
    assert MORSE_CODE_TABLE["。"] == ".-.-.."
    assert code_to_char(".-.-..") == "。"
    assert char_to_units("。") == [".-.-.."]
    assert is_supported("。")
    assert code_to_char(".-.-.") == "ン"
    assert code_to_char(".-.-.-") == "、"


def test_all_codes_are_unique():
    codes = list(MORSE_CODE_TABLE.values())
    assert len(codes) == len(set(codes))


def test_codes_only_contain_dot_and_dash():
    for code in MORSE_CODE_TABLE.values():
        assert set(code) <= {".", "-"}
        assert len(code) >= 1


def test_code_to_char_roundtrip():
    for char, code in MORSE_CODE_TABLE.items():
        assert code_to_char(code) == char


def test_char_to_units_simple_char():
    assert char_to_units("イ") == [".-"]
    assert char_to_units("1") == [".----"]


def test_char_to_units_dakuten_kana():
    assert char_to_units("ガ") == [".-..", ".."]
    assert char_to_units("ド") == ["..-..", ".."]


def test_char_to_units_handakuten_kana():
    assert char_to_units("パ") == ["-...", "..--."]
    assert char_to_units("ポ") == ["-..", "..--."]


def test_dakuten_and_handakuten_tables_cover_expected_kana():
    assert DAKUTEN_KANA["ガ"] == "カ"
    assert DAKUTEN_KANA["ヴ"] == "ウ"
    assert HANDAKUTEN_KANA["パ"] == "ハ"
    assert len(DAKUTEN_KANA) == 21
    assert len(HANDAKUTEN_KANA) == 5


def test_is_supported():
    assert is_supported("イ")
    assert is_supported("ガ")
    assert is_supported("パ")
    assert is_supported("5")
    assert is_supported("、")
    assert not is_supported("A")
    assert not is_supported("あ")


def test_char_to_units_unknown_char_raises():
    with pytest.raises(KeyError):
        char_to_units("A")


def test_code_to_char_unknown_code_raises():
    with pytest.raises(KeyError):
        code_to_char("....----")


def test_compose_dakuten_recombines_dakuten_and_handakuten():
    assert compose_dakuten("カ゛") == "ガ"
    assert compose_dakuten("ハ゜") == "パ"
    assert compose_dakuten("ウ゛") == "ヴ"


def test_compose_dakuten_leaves_plain_text_unchanged():
    assert compose_dakuten("イロハ") == "イロハ"
    assert compose_dakuten("") == ""


def test_compose_dakuten_handles_mixed_sequence():
    assert compose_dakuten("イカ゛クハ゜") == "イガクパ"


def test_compose_dakuten_is_inverse_of_char_to_units_decomposition():
    for composite in list(DAKUTEN_KANA) + list(HANDAKUTEN_KANA):
        units = char_to_units(composite)
        decomposed_text = "".join(code_to_char(code) for code in units)
        assert compose_dakuten(decomposed_text) == composite


def test_compose_dakuten_mark_without_preceding_base_stays_separate():
    # 対応する基本仮名が直前に無い濁点は結合されない
    assert compose_dakuten("ン゛") == "ン゛"

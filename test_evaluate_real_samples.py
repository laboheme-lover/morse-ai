"""evaluate_real_samples.py のユニットテスト。"""

import numpy as np
import pytest

from evaluate_real_samples import (
    estimate_dot_seconds,
    find_matched_pairs,
    normalize_ground_truth,
)
from generate_morse_audio import SAMPLE_RATE, _silence, sequence_to_waveform


def test_find_matched_pairs_only_returns_numbers_present_in_both_dirs(tmp_path):
    sound_dir = tmp_path / "Sound"
    text_dir = tmp_path / "TEXT"
    sound_dir.mkdir()
    text_dir.mkdir()

    (sound_dir / "Taman_1 [abc-NAp].mp3").write_bytes(b"dummy")
    (sound_dir / "Taman_2 [def-NAp].mp3").write_bytes(b"dummy")
    (sound_dir / "Taman_3 [ghi-NAp].mp3").write_bytes(b"dummy")
    (text_dir / "1.txt").write_text("テスト1", encoding="utf-8")
    (text_dir / "3.txt").write_text("テスト3", encoding="utf-8")
    # 2.txt は存在しない(テキストが無い音声は除外されるべき)

    pairs = find_matched_pairs(sound_dir, text_dir)
    numbers = sorted(n for n, _, _ in pairs)
    assert numbers == [1, 3]


def test_find_matched_pairs_empty_when_no_overlap(tmp_path):
    sound_dir = tmp_path / "Sound"
    text_dir = tmp_path / "TEXT"
    sound_dir.mkdir()
    text_dir.mkdir()
    (sound_dir / "Taman_5 [x-NAp].mp3").write_bytes(b"dummy")
    (text_dir / "9.txt").write_text("テスト", encoding="utf-8")

    assert find_matched_pairs(sound_dir, text_dir) == []


def test_normalize_ground_truth_keeps_supported_characters():
    filtered, excluded = normalize_ground_truth("イロハニホヘト")
    assert filtered == "イロハニホヘト"
    assert excluded == []


def test_normalize_ground_truth_keeps_composite_dakuten_characters():
    """ガ・プ等の合成仮名は基本仮名+濁点/半濁点の2符号で送出できるため、
    未対応文字として除外されてはならない(過去に発見したバグの回帰テスト)。"""
    filtered, excluded = normalize_ground_truth("プラスチック")
    assert "プ" in filtered
    assert "プ" not in excluded


def test_normalize_ground_truth_keeps_kuten():
    filtered, excluded = normalize_ground_truth("コンニチハ。")
    assert filtered == "コンニチハ。"
    assert excluded == []


def test_normalize_ground_truth_excludes_unsupported_punctuation():
    filtered, excluded = normalize_ground_truth("コンニチハ!「A」")
    assert filtered == "コンニチハ"
    assert set(excluded) == {"!", "「", "A", "」"}


def test_normalize_ground_truth_normalizes_small_kana_to_normal_size():
    filtered, _ = normalize_ground_truth("ツカッテ")
    assert filtered == "ツカッテ".replace("ッ", "ツ")


def test_normalize_ground_truth_normalizes_fullwidth_digits():
    filtered, excluded = normalize_ground_truth("５トン")
    assert filtered.startswith("5")
    assert excluded == []


def test_normalize_ground_truth_removes_whitespace():
    filtered, _ = normalize_ground_truth("イロ ハ\nニ　ホ")
    assert filtered == "イロハニホ"


def test_estimate_dot_seconds_matches_known_dot_duration():
    rng = np.random.default_rng(0)
    dot_duration = 0.08
    y = sequence_to_waveform("イロハニホ", dot_duration=dot_duration, rng=rng)
    estimate = estimate_dot_seconds(y, SAMPLE_RATE)
    # 推定値は実際の短点長に近いはず(許容誤差20ms)
    assert abs(estimate - dot_duration) < 0.02


def test_estimate_dot_seconds_falls_back_on_silence():
    y = _silence(1.0, SAMPLE_RATE)
    estimate = estimate_dot_seconds(y, SAMPLE_RATE, fallback=0.08)
    assert estimate == 0.08


def test_compare_texts_is_not_underestimated_for_long_texts():
    """difflibの標準設定(autojunk=True)では、200文字以上の文章で頻出文字が照合から
    外れ、8〜9割一致していても類似度が数%に過小評価される(過去に発生した評価バグ)。"""
    import random

    from evaluate_real_samples import compare_texts

    rng = random.Random(1)
    alphabet = "アイウエオカキクケコサシスセソタチツテトナニヌネノ"
    ground_truth = "".join(rng.choice(alphabet) for _ in range(400))
    predicted = "".join(c if rng.random() > 0.15 else rng.choice(alphabet) for c in ground_truth)
    true_match = sum(a == b for a, b in zip(ground_truth, predicted)) / len(ground_truth)

    result = compare_texts(predicted, ground_truth)
    assert result["similarity"] > 0.8
    assert abs(result["char_match_rate"] - true_match) < 0.03


def test_compare_texts_breakdown_counts_substitution_insertion_deletion():
    from evaluate_real_samples import compare_texts

    # 正解ABCDEFGHIJ に対し、C→X(置換)、E脱落、Kを末尾に挿入
    result = compare_texts("ABXDFGHIJK", "ABCDEFGHIJ")
    assert result["char_match_rate"] == pytest.approx(0.8)
    assert result["char_substitution_rate"] == pytest.approx(0.1)
    assert result["char_deletion_rate"] == pytest.approx(0.1)
    assert result["char_insertion_rate"] == pytest.approx(0.1)


def test_compare_texts_perfect_and_empty():
    from evaluate_real_samples import compare_texts

    assert compare_texts("イロハ", "イロハ")["char_match_rate"] == 1.0
    empty = compare_texts("", "イロハ")
    assert empty["char_match_rate"] == 0.0
    assert empty["char_deletion_rate"] == 1.0


def test_hiragana_to_katakana_converts_only_hiragana():
    from evaluate_real_samples import hiragana_to_katakana

    assert hiragana_to_katakana("さいきん、しょうがっこう") == "サイキン、ショウガッコウ"
    assert hiragana_to_katakana("ランドセル14グラム") == "ランドセル14グラム"  # カタカナ・数字はそのまま
    assert hiragana_to_katakana("ぱぴぷぺぽ・がぎぐげご") == "パピプペポ・ガギグゲゴ"
    assert hiragana_to_katakana("ー、。") == "ー、。"


def test_normalize_ground_truth_keeps_hiragana_transcript_as_katakana():
    """ひらがなで書かれた正解(49.txt)が語彙外として捨てられず、カタカナとして残ること
    (過去に295字が除外され、84字しか残らなかった評価バグの回帰テスト)。"""
    filtered, excluded = normalize_ground_truth("さいきん、しょうがっこうににゅうがくする")
    assert filtered == "サイキン、シヨウガツコウニニユウガクスル"
    assert "さ" not in excluded and "い" not in excluded


def test_normalize_ground_truth_hiragana_small_kana_becomes_normal_size():
    filtered, _ = normalize_ground_truth("がっこう")
    assert filtered == "ガツコウ"

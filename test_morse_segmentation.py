"""morse_segmentation.py のユニットテスト(Milestone 9)。"""

import numpy as np
import pytest

from generate_morse_audio import SAMPLE_RATE, _silence, sequence_to_waveform
from morse_segmentation import (
    compute_frame_rms,
    detect_active_frames,
    detect_active_frames_hysteresis,
    frames_to_runs,
    segment_audio,
    split_into_characters,
)


def _tone(duration_seconds: float, amplitude: float = 0.6, frequency_hz: float = 700.0) -> np.ndarray:
    n = int(round(duration_seconds * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)


def test_compute_frame_rms_shape():
    y = _tone(0.5)
    rms, frame_length = compute_frame_rms(y, SAMPLE_RATE, frame_ms=5.0)
    expected_frames = int(np.ceil(len(y) / frame_length))
    assert len(rms) == expected_frames
    assert rms.max() > 0


def test_detect_active_frames_distinguishes_tone_from_silence():
    tone = _tone(0.2)
    silence = _silence(0.2, SAMPLE_RATE)
    y = np.concatenate([tone, silence, tone])
    rms, frame_length = compute_frame_rms(y, SAMPLE_RATE)
    is_active = detect_active_frames(rms)
    assert is_active[0]  # 先頭はトーン
    mid_frame = len(is_active) // 2
    assert not is_active[mid_frame]  # 中央付近は無音


def test_frames_to_runs_merges_contiguous_frames():
    is_active = np.array([True, True, False, False, False, True])
    runs = frames_to_runs(is_active, frame_length=10, total_samples=60)
    assert runs == [(0, 20, True), (20, 50, False), (50, 60, True)]


def test_split_into_characters_separates_two_well_spaced_tones():
    """1文字ずつの短いトーン2つを長い無音で挟むと、2文字として分割される。"""
    char1 = _tone(0.1)
    gap = _silence(0.5, SAMPLE_RATE)  # 十分に長い無音(文字境界)
    char2 = _tone(0.1)
    y = np.concatenate([char1, gap, char2])

    ranges = split_into_characters(y, SAMPLE_RATE, min_char_gap_seconds=0.25)
    assert len(ranges) == 2
    for start, end in ranges:
        assert 0 <= start < end <= len(y)


def test_split_into_characters_keeps_short_internal_gap_as_one_character():
    """符号内の短い間隔(0.1秒)は文字境界のしきい値(0.25秒)未満なので、
    1文字として連結されたまま残る。"""
    dot = _tone(0.1)
    short_gap = _silence(0.1, SAMPLE_RATE)
    dash = _tone(0.3)
    y = np.concatenate([dot, short_gap, dash])

    ranges = split_into_characters(y, SAMPLE_RATE, min_char_gap_seconds=0.25)
    assert len(ranges) == 1


def test_split_into_characters_with_generated_sequence():
    """generate_morse_audio.sequence_to_waveformで生成した複数文字の連続音声が、
    文字数と同じ数に分割されることを確認する。"""
    rng = np.random.default_rng(0)
    text = "イロハニ"
    y = sequence_to_waveform(text, dot_duration=0.1, rng=rng)

    ranges = split_into_characters(y, SAMPLE_RATE, min_char_gap_seconds=0.25)
    assert len(ranges) == len(text)


def test_segment_audio_returns_waveform_chunks_matching_split_into_characters():
    rng = np.random.default_rng(1)
    text = "アイ"
    y = sequence_to_waveform(text, dot_duration=0.1, rng=rng)

    ranges = split_into_characters(y, SAMPLE_RATE)
    chunks = segment_audio(y, SAMPLE_RATE)

    assert len(chunks) == len(ranges)
    for chunk, (start, end) in zip(chunks, ranges):
        assert len(chunk) == end - start


def test_hysteresis_ignores_brief_dip_that_single_threshold_would_split():
    """トーン中に瞬間的な振幅の落ち込み(ノイズ・フェージングを想定)があっても、
    ヒステリシスなら1つの有音区間として扱われ、単一しきい値では2つに
    分割されてしまうケースを再現する。"""
    tone = _tone(0.3)
    # トーンの中央付近に、ピークの20%まで一瞬落ち込む区間(5ms)を作る
    dip_start = int(0.145 * SAMPLE_RATE)
    dip_end = int(0.150 * SAMPLE_RATE)
    tone[dip_start:dip_end] *= 0.2

    rms, frame_length = compute_frame_rms(tone, SAMPLE_RATE)

    single = detect_active_frames(rms, threshold_ratio=0.25)
    hysteresis = detect_active_frames_hysteresis(rms, high_ratio=0.25, low_ratio=0.125)

    single_runs = frames_to_runs(single, frame_length, len(tone))
    hysteresis_runs = frames_to_runs(hysteresis, frame_length, len(tone))

    single_on_runs = sum(1 for _, _, is_on in single_runs if is_on)
    hysteresis_on_runs = sum(1 for _, _, is_on in hysteresis_runs if is_on)

    assert single_on_runs >= 2  # 単一しきい値では落ち込みで分割されてしまう
    assert hysteresis_on_runs == 1  # ヒステリシスなら1つの有音区間のまま


def test_hysteresis_still_detects_genuine_silence_gap():
    """本物の(十分に振幅が下がる)無音区間はヒステリシスでも正しく検出できる。"""
    tone1 = _tone(0.1)
    silence = _silence(0.3, SAMPLE_RATE)
    tone2 = _tone(0.1)
    y = np.concatenate([tone1, silence, tone2])

    rms, frame_length = compute_frame_rms(y, SAMPLE_RATE)
    hysteresis = detect_active_frames_hysteresis(rms)
    runs = frames_to_runs(hysteresis, frame_length, len(y))
    on_runs = sum(1 for _, _, is_on in runs if is_on)
    assert on_runs == 2


def test_hysteresis_despikes_isolated_short_noise_blip_in_silence():
    """無音区間中に1フレーム分(5ms)だけ瞬間的にしきい値を超えるノイズスパイクが
    あっても、デスパイク処理でそのスパイクは無音として統合され、有音区間の
    数が増えない(過分割の原因となるダミー文字が生まれない)ことを確認する。"""
    tone1 = _tone(0.1)
    silence_a = _silence(0.2, SAMPLE_RATE)
    spike = _tone(0.005, amplitude=0.6)  # 5msの孤立したノイズスパイク
    silence_b = _silence(0.2, SAMPLE_RATE)
    tone2 = _tone(0.1)
    y = np.concatenate([tone1, silence_a, spike, silence_b, tone2])

    rms, frame_length = compute_frame_rms(y, SAMPLE_RATE)
    active = detect_active_frames_hysteresis(rms)
    runs = frames_to_runs(active, frame_length, len(y))
    on_runs = sum(1 for _, _, is_on in runs if is_on)
    assert on_runs == 2  # スパイクが3つ目のON区間として数えられない


def test_hysteresis_despike_preserves_fast_speed_dot():
    """最速(fast)速度の短点(0.05秒=10フレーム)はデスパイクのしきい値
    (3フレーム=15ms)より十分長いため、消えずに残ることを確認する。"""
    fast_dot = _tone(0.05)
    silence = _silence(0.3, SAMPLE_RATE)
    y = np.concatenate([silence, fast_dot, silence])

    rms, frame_length = compute_frame_rms(y, SAMPLE_RATE)
    active = detect_active_frames_hysteresis(rms)
    runs = frames_to_runs(active, frame_length, len(y))
    on_runs = [(s, e) for s, e, is_on in runs if is_on]
    assert len(on_runs) == 1
    duration = (on_runs[0][1] - on_runs[0][0]) / SAMPLE_RATE
    assert duration == pytest.approx(0.05, abs=0.006)


def test_despike_can_be_disabled():
    """despike_min_run_frames<=1で無効化すると、孤立したスパイクがそのまま残る。"""
    silence_a = _silence(0.2, SAMPLE_RATE)
    spike = _tone(0.005, amplitude=0.6)
    silence_b = _silence(0.2, SAMPLE_RATE)
    y = np.concatenate([silence_a, spike, silence_b])

    rms, frame_length = compute_frame_rms(y, SAMPLE_RATE)
    active = detect_active_frames_hysteresis(rms, despike_min_run_frames=1)
    runs = frames_to_runs(active, frame_length, len(y))
    on_runs = sum(1 for _, _, is_on in runs if is_on)
    assert on_runs == 1  # スパイクがそのままON区間として残る


def test_split_into_characters_empty_audio_returns_no_ranges():
    y = np.zeros(0, dtype=np.float32)
    ranges = split_into_characters(y, SAMPLE_RATE)
    assert ranges == []


def test_split_into_characters_silence_only_returns_no_ranges():
    y = _silence(1.0, SAMPLE_RATE)
    ranges = split_into_characters(y, SAMPLE_RATE)
    assert ranges == []


def test_default_segmentation_is_tight_around_tone_edges():
    """切り出し区間は音の開始・終了に密着しているべき(前後の余白は既定で付けない)。

    学習データは先頭・末尾に無音が無いため、余白付きで切り出すと認識精度が
    大きく低下する。フレーム分解能(5ms)を超えて外側に広がらないことを確認する。
    """
    lead, tone_seconds, trail = 0.5, 0.3, 0.5
    y = np.concatenate([_silence(lead, SAMPLE_RATE), _tone(tone_seconds), _silence(trail, SAMPLE_RATE)])

    ranges = split_into_characters(y, SAMPLE_RATE)
    assert len(ranges) == 1
    start, end = ranges[0]

    tolerance = int(0.006 * SAMPLE_RATE)
    assert abs(start - int(lead * SAMPLE_RATE)) <= tolerance
    assert abs(end - int((lead + tone_seconds) * SAMPLE_RATE)) <= tolerance

"""audio_preprocessing.py のユニットテスト。"""

import numpy as np
import pytest

from audio_preprocessing import (
    analyze_audio,
    bandpass_filter,
    estimate_center_frequency,
    estimate_char_gap_seconds,
    estimate_dot_seconds,
    normalize_amplitude,
    preprocess_for_inference,
    time_stretch_to_target_dot,
)
from generate_morse_audio import SAMPLE_RATE, _silence, sequence_to_waveform


def _tone(duration_seconds: float, frequency_hz: float, amplitude: float = 0.6, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n = int(round(duration_seconds * sample_rate))
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)


def _tone_with_harmonic(
    duration_seconds: float, frequency_hz: float, amplitude: float = 0.6, harmonic_ratio: float = 0.5
) -> np.ndarray:
    fundamental = _tone(duration_seconds, frequency_hz, amplitude)
    harmonic = _tone(duration_seconds, frequency_hz * 3, amplitude * harmonic_ratio)
    return (fundamental + harmonic).astype(np.float32)


def test_estimate_center_frequency_matches_known_tone():
    y = _tone(0.5, 700.0)
    freq = estimate_center_frequency(y, SAMPLE_RATE)
    assert abs(freq - 700.0) < 20.0


def test_estimate_center_frequency_fallback_on_silence():
    y = _silence(0.5, SAMPLE_RATE)
    freq = estimate_center_frequency(y, SAMPLE_RATE, fallback=700.0)
    assert freq == 700.0


def test_estimate_center_frequency_survives_one_corrupted_run():
    """有音区間のうち1つが別周波数(強いノイズ等を想定)に支配されていても、
    他の複数区間が示す本来の周波数へ引っ張られること(実データ検証で、
    最長1区間のみを使うと稀に誤ったバンドパス周波数に固定される問題が
    見つかった際の回帰テスト)。"""
    true_freq = 700.0
    interfering_freq = 1200.0
    silence = _silence(0.1, SAMPLE_RATE)
    # 最長区間だけを別周波数にすると、最長1区間方式では誤って1200Hzを返す
    corrupted_longest_run = _tone(0.6, interfering_freq)
    genuine_runs = [_tone(0.3, true_freq) for _ in range(4)]

    parts = [corrupted_longest_run]
    for run in genuine_runs:
        parts.append(silence)
        parts.append(run)
    y = np.concatenate(parts)

    freq = estimate_center_frequency(y, SAMPLE_RATE, max_runs=5)
    assert abs(freq - true_freq) < 20.0


def test_estimate_dot_seconds_matches_known_dot_duration():
    rng = np.random.default_rng(0)
    dot_duration = 0.08
    y = sequence_to_waveform("イロハニホ", dot_duration=dot_duration, rng=rng)
    estimate = estimate_dot_seconds(y, SAMPLE_RATE)
    assert abs(estimate - dot_duration) < 0.02


def test_estimate_char_gap_seconds_matches_known_gap():
    """CHARACTER_GAP_UNITS=7かつdot=0.08sなので、文字間隔は約0.56秒のはず。"""
    rng = np.random.default_rng(1)
    dot_duration = 0.08
    y = sequence_to_waveform("イロハ", dot_duration=dot_duration, rng=rng)
    gap = estimate_char_gap_seconds(y, SAMPLE_RATE)
    assert 0.4 < gap < 0.7


def test_analyze_audio_returns_reasonable_values():
    rng = np.random.default_rng(2)
    y = sequence_to_waveform("イロハニ", dot_duration=0.09, frequency_hz=650.0, rng=rng)
    analysis = analyze_audio(y, SAMPLE_RATE)
    assert abs(analysis.frequency_hz - 650.0) < 25.0
    assert abs(analysis.dot_duration_seconds - 0.09) < 0.02
    assert analysis.wpm > 0


def test_bandpass_filter_suppresses_harmonic_relative_to_fundamental():
    y = _tone_with_harmonic(0.5, 700.0, amplitude=0.6, harmonic_ratio=0.6)
    filtered = bandpass_filter(y, SAMPLE_RATE, center_freq_hz=700.0, halfwidth_hz=150.0)

    def harmonic_to_fundamental_power_ratio(signal: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / SAMPLE_RATE)
        fundamental_power = spectrum[(freqs > 650) & (freqs < 750)].max()
        harmonic_power = spectrum[(freqs > 2000) & (freqs < 2200)].max()
        return harmonic_power / fundamental_power

    ratio_before = harmonic_to_fundamental_power_ratio(y)
    ratio_after = harmonic_to_fundamental_power_ratio(filtered)
    assert ratio_after < ratio_before * 0.3  # 倍音が大幅に抑圧されること


def test_bandpass_filter_preserves_fundamental_reasonably():
    y = _tone(0.5, 700.0, amplitude=0.6)
    filtered = bandpass_filter(y, SAMPLE_RATE, center_freq_hz=700.0, halfwidth_hz=150.0)
    # フィルタ後も基本波の実効振幅がある程度保たれていること
    assert np.sqrt(np.mean(filtered**2)) > np.sqrt(np.mean(y**2)) * 0.5


def test_bandpass_filter_empty_input():
    y = np.zeros(0, dtype=np.float32)
    assert bandpass_filter(y, SAMPLE_RATE).size == 0


def test_normalize_amplitude_scales_to_target_peak():
    y = _tone(0.2, 700.0, amplitude=0.2)
    normalized = normalize_amplitude(y, target_peak=0.7)
    assert np.max(np.abs(normalized)) == pytest.approx(0.7, abs=1e-3)


def test_normalize_amplitude_handles_silence():
    y = np.zeros(100, dtype=np.float32)
    normalized = normalize_amplitude(y, target_peak=0.7)
    assert np.all(normalized == 0)


def test_time_stretch_to_target_dot_shortens_slow_audio():
    rng = np.random.default_rng(3)
    y = sequence_to_waveform("イロハ", dot_duration=0.16, rng=rng)
    stretched = time_stretch_to_target_dot(y, SAMPLE_RATE, current_dot_seconds=0.16, target_dot_seconds=0.08)
    # 短点長を半分にする(rate=2)ので、全体の長さも概ね半分になるはず
    assert len(stretched) < len(y) * 0.7


def test_time_stretch_to_target_dot_lengthens_fast_audio():
    rng = np.random.default_rng(4)
    y = sequence_to_waveform("イロハ", dot_duration=0.05, rng=rng)
    stretched = time_stretch_to_target_dot(y, SAMPLE_RATE, current_dot_seconds=0.05, target_dot_seconds=0.1)
    assert len(stretched) > len(y) * 1.3


def test_time_stretch_scales_total_duration_by_expected_rate():
    """位相ボコーダはモールス信号特有の急峻なオン/オフ切り替え付近で細かい
    アーティファクトを生むため、フレーム単位のオン/オフ長を個別に見ると
    正確ではない(estimate_dot_secondsでの再推定は信頼できない)。しかし
    全体の再生時間はrateの逆数倍に正しくスケールする。"""
    rng = np.random.default_rng(5)
    original_dot = 0.05
    target_dot = 0.1
    y = sequence_to_waveform("イロハニホヘト", dot_duration=original_dot, rng=rng)
    stretched = time_stretch_to_target_dot(y, SAMPLE_RATE, current_dot_seconds=original_dot, target_dot_seconds=target_dot)

    expected_rate = original_dot / target_dot  # 0.5 (遅く=長くなる)
    expected_length = len(y) / expected_rate
    assert abs(len(stretched) - expected_length) / expected_length < 0.05


def test_preprocess_for_inference_returns_waveform_and_analysis():
    rng = np.random.default_rng(6)
    y = sequence_to_waveform("イロハ", dot_duration=0.08, frequency_hz=650.0, rng=rng)
    processed, analysis = preprocess_for_inference(y, SAMPLE_RATE)

    assert isinstance(processed, np.ndarray)
    assert processed.size > 0
    assert abs(analysis.frequency_hz - 650.0) < 30.0
    assert np.max(np.abs(processed)) <= 0.71  # target_peak=0.7近辺


def test_preprocess_for_inference_can_disable_individual_steps():
    rng = np.random.default_rng(7)
    y = sequence_to_waveform("イロハ", dot_duration=0.08, rng=rng)
    processed, _ = preprocess_for_inference(
        y, SAMPLE_RATE, apply_bandpass=False, apply_time_stretch=False, apply_amplitude_normalize=False
    )
    assert len(processed) == len(y)
    assert np.allclose(processed, y, atol=1e-6)

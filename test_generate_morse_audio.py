"""generate_morse_audio.py のユニットテスト(Milestone 2: データ拡張)。"""

import csv

import numpy as np
import pytest
import soundfile as sf

from generate_morse_audio import (
    AMPLITUDE_RANGE,
    DEFAULT_AMPLITUDE,
    DEFAULT_DOT_DURATION,
    DEFAULT_FREQUENCY_HZ,
    FREQUENCY_RANGE,
    SPEED_RANGES,
    add_white_noise,
    append_metadata_row,
    code_to_waveform,
    generate_morse_wav,
    resolve_generation_params,
)


def test_resolve_generation_params_default_is_fixed_normal_speed():
    rng = np.random.default_rng(0)
    params = resolve_generation_params(rng)
    assert params["speed"] == "normal"
    assert params["dot_duration"] == DEFAULT_DOT_DURATION
    assert params["frequency_hz"] == DEFAULT_FREQUENCY_HZ
    assert params["amplitude"] == DEFAULT_AMPLITUDE


@pytest.mark.parametrize("speed", sorted(SPEED_RANGES))
def test_resolve_generation_params_speed_ranges(speed):
    rng = np.random.default_rng(42)
    low, high = SPEED_RANGES[speed]
    for _ in range(50):
        params = resolve_generation_params(rng, speed=speed)
        assert params["speed"] == speed
        assert low <= params["dot_duration"] <= high


def test_resolve_generation_params_explicit_dot_duration_overrides_speed_range():
    rng = np.random.default_rng(1)
    params = resolve_generation_params(rng, speed="slow", dot_duration=0.05)
    assert params["dot_duration"] == 0.05
    assert params["speed"] == "slow"


def test_resolve_generation_params_randomize_frequency_and_amplitude():
    rng = np.random.default_rng(7)
    for _ in range(50):
        params = resolve_generation_params(rng, randomize=True)
        assert FREQUENCY_RANGE[0] <= params["frequency_hz"] <= FREQUENCY_RANGE[1]
        assert AMPLITUDE_RANGE[0] <= params["amplitude"] <= AMPLITUDE_RANGE[1]


def test_resolve_generation_params_unknown_speed_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        resolve_generation_params(rng, speed="ultra-fast")


def test_timing_jitter_keeps_dot_shorter_than_dash():
    """±10%ジッターを与えても短点は必ず長点より短いままであることを確認する。"""
    rng = np.random.default_rng(123)
    dot_duration = 0.10
    jitter = 0.10
    max_dot_samples = 0
    min_dash_samples = float("inf")
    for _ in range(200):
        dot_wave = code_to_waveform(".", dot_duration=dot_duration, timing_jitter=jitter, rng=rng)
        dash_wave = code_to_waveform("-", dot_duration=dot_duration, timing_jitter=jitter, rng=rng)
        max_dot_samples = max(max_dot_samples, len(dot_wave))
        min_dash_samples = min(min_dash_samples, len(dash_wave))
    assert max_dot_samples < min_dash_samples


def test_timing_jitter_out_of_range_raises():
    with pytest.raises(ValueError):
        code_to_waveform(".", timing_jitter=1.5)
    with pytest.raises(ValueError):
        code_to_waveform(".", timing_jitter=-0.1)


def test_add_white_noise_matches_requested_snr_approximately():
    rng = np.random.default_rng(99)
    dot_duration = 0.1
    tone = code_to_waveform("-", dot_duration=dot_duration, frequency_hz=700.0, amplitude=0.6, rng=rng)
    snr_db_target = 10.0
    noisy = add_white_noise(tone, snr_db_target, rng=np.random.default_rng(5))

    noise_only = noisy - tone
    active_mask = tone != 0
    signal_power = float(np.mean(tone[active_mask] ** 2))
    noise_power = float(np.mean(noise_only[active_mask] ** 2))
    measured_snr_db = 10 * np.log10(signal_power / noise_power)

    assert measured_snr_db == pytest.approx(snr_db_target, abs=1.0)


def test_generate_morse_wav_clean_creates_expected_metadata(tmp_path):
    output_path = tmp_path / "イ.wav"
    path, metadata = generate_morse_wav("イ", output_path, seed=1)

    assert path.exists()
    data, sr = sf.read(path)
    assert sr == 16000
    assert len(data) > 0

    assert metadata["label"] == "イ"
    assert metadata["morse_code"] == ".-"
    assert metadata["speed"] == "normal"
    assert metadata["noise_type"] == "clean"
    assert metadata["snr_db"] == ""


def test_generate_morse_wav_with_augmentation(tmp_path):
    output_path = tmp_path / "ロ.wav"
    path, metadata = generate_morse_wav(
        "ロ",
        output_path,
        speed="fast",
        randomize=True,
        timing_jitter=0.10,
        noise_type="white",
        snr_db=5.0,
        seed=42,
    )

    assert path.exists()
    low, high = SPEED_RANGES["fast"]
    assert low <= metadata["dot_duration"] <= high
    assert FREQUENCY_RANGE[0] <= metadata["frequency_hz"] <= FREQUENCY_RANGE[1]
    assert AMPLITUDE_RANGE[0] <= metadata["amplitude"] <= AMPLITUDE_RANGE[1]
    assert metadata["noise_type"] == "white"
    assert metadata["snr_db"] == 5.0


def test_generate_morse_wav_normalizes_instead_of_hard_clipping(tmp_path):
    """極端な低SNRでも一様スケーリングで正規化され、飽和(ハードクリップ)しないことを確認する。"""
    output_path = tmp_path / "ロ_extreme.wav"
    path, _ = generate_morse_wav(
        "ロ",
        output_path,
        speed="normal",
        amplitude=0.9,
        noise_type="white",
        snr_db=-10.0,
        seed=123,
    )
    data, _ = sf.read(path)
    assert float(np.max(np.abs(data))) <= 1.0
    saturated_ratio = float(np.mean(np.abs(data) > 0.999))
    assert saturated_ratio < 0.01


def test_generate_morse_wav_white_noise_requires_snr(tmp_path):
    with pytest.raises(ValueError):
        generate_morse_wav("イ", tmp_path / "x.wav", noise_type="white")


def test_generate_morse_wav_invalid_noise_type_raises(tmp_path):
    with pytest.raises(ValueError):
        generate_morse_wav("イ", tmp_path / "x.wav", noise_type="pink")


def test_append_metadata_row_writes_header_once(tmp_path):
    csv_path = tmp_path / "metadata.csv"
    row1 = {
        "filepath": "a.wav", "label": "イ", "morse_code": ".-", "speed": "normal",
        "dot_duration": 0.1, "frequency_hz": 700.0, "amplitude": 0.6,
        "timing_jitter": 0.0, "noise_type": "clean", "snr_db": "",
    }
    row2 = dict(row1, filepath="b.wav", label="ロ", morse_code=".-.-")

    append_metadata_row(csv_path, row1)
    append_metadata_row(csv_path, row2)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["label"] == "イ"
    assert rows[1]["label"] == "ロ"

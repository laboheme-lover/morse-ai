"""mp3_augment.py のテスト。"""

import numpy as np
import pytest

from generate_morse_audio import SAMPLE_RATE, char_to_waveform
from mp3_augment import COMPRESSION_LEVELS, mp3_roundtrip


def _waveform():
    return char_to_waveform("ロ", dot_duration=0.09, frequency_hz=602.0, amplitude=0.6)


def test_roundtrip_preserves_length_and_dtype():
    y = _waveform()
    out = mp3_roundtrip(y, SAMPLE_RATE, compression_level=0.8)
    assert out.shape == y.shape
    assert out.dtype == np.float32


def test_roundtrip_preserves_onset_position():
    """符号化遅延が補正され、音の開始位置(先頭)がずれないこと。
    学習データは先頭に無音が無いため、ずれると入力の形が変わってしまう。"""
    y = _waveform()
    out = mp3_roundtrip(y, SAMPLE_RATE, compression_level=0.8)
    onset = int(np.argmax(np.abs(out) > 0.1 * np.abs(out).max()))
    assert onset < int(0.01 * SAMPLE_RATE)  # 10ms以内


def test_roundtrip_changes_waveform_but_keeps_tone_energy():
    y = _waveform()
    out = mp3_roundtrip(y, SAMPLE_RATE, compression_level=0.8)
    assert not np.allclose(out, y, atol=1e-4)  # 圧縮による変化がある
    rms_ratio = np.sqrt(np.mean(out**2)) / np.sqrt(np.mean(y**2))
    assert 0.8 < rms_ratio < 1.2


def test_roundtrip_adds_noise_floor_in_silence():
    """完全な無音だった区間に、MP3特有の微小な床ノイズが生じる(実録音と同様)。"""
    y = _waveform()
    out = mp3_roundtrip(y, SAMPLE_RATE, compression_level=0.95)
    silent = y == 0
    assert silent.any()
    assert np.max(np.abs(out[silent])) > 0.0


@pytest.mark.parametrize("level", COMPRESSION_LEVELS)
def test_all_configured_levels_work(level):
    out = mp3_roundtrip(_waveform(), SAMPLE_RATE, compression_level=level)
    assert np.isfinite(out).all()


def test_roundtrip_empty_input():
    assert mp3_roundtrip(np.zeros(0, dtype=np.float32), SAMPLE_RATE).size == 0

"""add_esc50_noise.py のユニットテスト。

実際のESC-50データセットには依存せず、conftest.pyのfake_esc50_root fixtureを
使用する。SNR計算やfold分離の正しさを検証する。
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from add_esc50_noise import _fit_noise_length, mix_wav_with_esc50, mix_with_noise
from generate_morse_audio import generate_morse_wav

REAL_ESC50_ROOT = Path("data/ESC-50-master")
_real_esc50_available = (REAL_ESC50_ROOT / "meta" / "esc50.csv").exists()


def test_fit_noise_length_crops_when_longer():
    rng = np.random.default_rng(0)
    noise = np.arange(100, dtype=np.float32)
    fitted = _fit_noise_length(noise, 10, rng)
    assert len(fitted) == 10


def test_fit_noise_length_loops_when_shorter():
    rng = np.random.default_rng(0)
    noise = np.arange(5, dtype=np.float32)
    fitted = _fit_noise_length(noise, 23, rng)
    assert len(fitted) == 23


def test_fit_noise_length_empty_noise_returns_zeros():
    rng = np.random.default_rng(0)
    fitted = _fit_noise_length(np.zeros(0, dtype=np.float32), 10, rng)
    assert len(fitted) == 10
    assert np.all(fitted == 0)


def test_mix_with_noise_matches_requested_snr_approximately():
    rng = np.random.default_rng(1)
    # トーン区間だけの単純な信号(全区間が"オン")を用意する
    signal = 0.6 * np.sin(2 * np.pi * 700 * np.arange(16000) / 16000).astype(np.float32)
    noise = rng.normal(0, 1.0, size=len(signal)).astype(np.float32)

    snr_target = 5.0
    mixed = mix_with_noise(signal, noise, snr_target, rng=np.random.default_rng(2))

    noise_component = mixed - signal
    signal_power = float(np.mean(signal**2))
    noise_power = float(np.mean(noise_component**2))
    measured_snr = 10 * np.log10(signal_power / noise_power)

    assert measured_snr == pytest.approx(snr_target, abs=1.0)


def test_mix_wav_with_esc50_train_split(tmp_path, fake_esc50_root):
    morse_path, _ = generate_morse_wav("イ", tmp_path / "イ.wav", seed=1)
    output_path = tmp_path / "イ_esc50.wav"

    path, metadata = mix_wav_with_esc50(
        morse_path, fake_esc50_root, split="train", snr_db=10.0, output_path=output_path, seed=42
    )

    assert path.exists()
    data, sr = sf.read(path)
    assert sr == 16000
    assert float(np.max(np.abs(data))) <= 1.0

    assert metadata["noise_type"] == "esc50"
    assert metadata["snr_db"] == 10.0
    assert metadata["esc50_fold"] in (1, 2, 3)
    assert metadata["esc50_filename"]
    assert metadata["esc50_category"]
    assert metadata["esc50_src_file"]


def test_mix_wav_with_esc50_respects_split_folds(tmp_path, fake_esc50_root):
    morse_path, _ = generate_morse_wav("ロ", tmp_path / "ロ.wav", seed=1)

    for split, expected_folds in (("train", (1, 2, 3)), ("validation", (4,)), ("test", (5,))):
        output_path = tmp_path / f"ロ_{split}.wav"
        _, metadata = mix_wav_with_esc50(
            morse_path, fake_esc50_root, split=split, snr_db=10.0, output_path=output_path, seed=7
        )
        assert metadata["esc50_fold"] in expected_folds


@pytest.mark.skipif(not _real_esc50_available, reason="実際のESC-50データセットが未ダウンロードです")
@pytest.mark.parametrize("split,snr_db", [("train", 20.0), ("validation", 5.0), ("test", -5.0)])
def test_mix_wav_with_real_esc50(tmp_path, split, snr_db):
    morse_path, _ = generate_morse_wav("ロ", tmp_path / "ロ.wav", speed="normal", seed=1)
    output_path = tmp_path / f"ロ_{split}.wav"

    path, metadata = mix_wav_with_esc50(
        morse_path, REAL_ESC50_ROOT, split=split, snr_db=snr_db, output_path=output_path, seed=10
    )

    assert path.exists()
    data, sr = sf.read(path)
    assert sr == 16000
    assert float(np.max(np.abs(data))) <= 1.0
    assert metadata["esc50_category"]
    assert metadata["snr_db"] == snr_db


def test_mix_wav_with_esc50_missing_dataset_raises(tmp_path):
    morse_path, _ = generate_morse_wav("イ", tmp_path / "イ.wav", seed=1)
    with pytest.raises(FileNotFoundError):
        mix_wav_with_esc50(
            morse_path,
            tmp_path / "no_such_esc50",
            split="train",
            snr_db=10.0,
            output_path=tmp_path / "out.wav",
        )

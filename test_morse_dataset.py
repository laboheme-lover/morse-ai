"""morse_dataset.py のユニットテスト(Milestone 5)。"""

import csv

import numpy as np
import pytest
import soundfile as sf
import torch

from generate_morse_audio import generate_morse_wav
from morse_dataset import (
    HOP_LENGTH,
    N_MELS,
    MorseCharacterDataset,
    audio_to_mel_tensor,
    build_dataloaders,
    compute_max_frames,
    estimate_mel_frames,
    fix_length,
    wav_to_mel,
)
from generate_dataset import METADATA_FIELDS


@pytest.mark.parametrize("speed", ["fast", "normal", "slow"])
def test_estimate_mel_frames_matches_actual_output(tmp_path, speed):
    wav_path, _ = generate_morse_wav("コ", tmp_path / "コ.wav", speed=speed, seed=1)
    info = sf.info(wav_path)
    mel = wav_to_mel(wav_path)
    assert mel.shape[0] == N_MELS
    assert mel.shape[1] == estimate_mel_frames(info.frames)


def test_fix_length_pads_short_spectrogram():
    mel = np.ones((N_MELS, 10), dtype=np.float32) * 5.0
    fixed = fix_length(mel, 20)
    assert fixed.shape == (N_MELS, 20)
    assert np.all(fixed[:, 10:] == mel.min())
    assert np.array_equal(fixed[:, :10], mel)


def test_fix_length_truncates_long_spectrogram():
    mel = np.arange(N_MELS * 30, dtype=np.float32).reshape(N_MELS, 30)
    fixed = fix_length(mel, 10)
    assert fixed.shape == (N_MELS, 10)
    assert np.array_equal(fixed, mel[:, :10])


def test_fix_length_no_change_when_equal():
    mel = np.random.default_rng(0).normal(size=(N_MELS, 15)).astype(np.float32)
    fixed = fix_length(mel, 15)
    assert fixed.shape == (N_MELS, 15)
    assert np.array_equal(fixed, mel)


def test_audio_to_mel_tensor_shape(tmp_path):
    wav_path, _ = generate_morse_wav("イ", tmp_path / "イ.wav", seed=1)
    tensor = audio_to_mel_tensor(wav_path, fixed_frames=100)
    assert tensor.shape == (1, N_MELS, 100)
    assert tensor.dtype == torch.float32


def _write_fake_metadata(tmp_path, rows):
    metadata_csv = tmp_path / "metadata.csv"
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return metadata_csv


def _make_row(wav_path, split, label, morse_code):
    return {
        "filepath": str(wav_path).replace("\\", "/"),
        "split": split,
        "label": label,
        "morse_code": morse_code,
        "speed": "normal",
        "dot_duration": 0.1,
        "frequency_hz": 700.0,
        "amplitude": 0.6,
        "noise_type": "clean",
        "snr_db": "",
        "esc50_filename": "",
        "esc50_category": "",
        "esc50_fold": "",
        "esc50_src_file": "",
    }


@pytest.fixture
def tiny_dataset(tmp_path):
    rows = []
    labels = [("イ", ".-"), ("ロ", ".-.-"), ("1", ".----")]
    for split in ("train", "validation", "test"):
        for label, code in labels:
            for i in range(3):
                wav_path = tmp_path / "wav" / split / label / f"{label}_{i}.wav"
                generate_morse_wav(label, wav_path, seed=i)
                rows.append(_make_row(wav_path, split, label, code))
    metadata_csv = _write_fake_metadata(tmp_path, rows)
    return metadata_csv


def test_compute_max_frames_matches_actual_max(tiny_dataset):
    import pandas as pd

    max_frames = compute_max_frames(tiny_dataset)
    df = pd.read_csv(tiny_dataset, keep_default_na=False)
    actual_max = max(wav_to_mel(p).shape[1] for p in df["filepath"])
    assert max_frames == actual_max


def test_morse_character_dataset_len_and_item_shape(tiny_dataset):
    dataset = MorseCharacterDataset(tiny_dataset, split="train")
    assert len(dataset) == 9  # 3ラベル x 3件

    x, y = dataset[0]
    assert x.shape == (1, N_MELS, dataset.fixed_frames)
    assert isinstance(y, int)
    assert 0 <= y < len(dataset.label_to_index)


def test_morse_character_dataset_shared_label_mapping_across_splits(tiny_dataset):
    train_ds = MorseCharacterDataset(tiny_dataset, split="train")
    label_to_index = train_ds.label_to_index
    test_ds = MorseCharacterDataset(tiny_dataset, split="test", label_to_index=label_to_index, fixed_frames=train_ds.fixed_frames)
    assert test_ds.label_to_index == label_to_index


def test_morse_character_dataset_unknown_split_raises(tiny_dataset):
    with pytest.raises(ValueError):
        MorseCharacterDataset(tiny_dataset, split="does_not_exist")


def test_build_dataloaders_batch_shapes(tiny_dataset):
    train_loader, val_loader, test_loader, label_to_index, fixed_frames = build_dataloaders(
        tiny_dataset, batch_size=4
    )
    assert len(label_to_index) == 3
    assert len(train_loader.dataset) == 9
    assert len(val_loader.dataset) == 9
    assert len(test_loader.dataset) == 9

    x, y = next(iter(train_loader))
    assert x.shape == (4, 1, N_MELS, fixed_frames)
    assert y.shape == (4,)
    assert y.dtype == torch.int64

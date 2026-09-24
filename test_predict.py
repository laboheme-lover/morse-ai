"""predict.py のユニットテスト(Milestone 8)。

Milestone 6/7のテストと同様、周波数の異なる3クラスの容易な分類タスクで
モデルを学習させ、推論結果(Top1予測・確信度・Top-K候補・メルスペクトログラム)
が正しく返ることを検証する。
"""

import csv

import numpy as np
import pytest
import torch

from generate_dataset import METADATA_FIELDS
from generate_morse_audio import generate_morse_wav
from morse_dataset import N_MELS
from predict import load_model, predict, predict_file
from train import train

LABELS = [("イ", 700.0), ("ロ", 750.0), ("1", 650.0)]


def _make_easy_metadata(tmp_path, samples_per_split=15):
    rows = []
    seed = 0
    for label, freq in LABELS:
        for split in ("train", "validation", "test"):
            for i in range(samples_per_split):
                wav_path = tmp_path / "wav" / split / label / f"{label}_{i}.wav"
                _, meta = generate_morse_wav(
                    label, wav_path, speed="normal", frequency_hz=freq, amplitude=0.6, seed=seed
                )
                seed += 1
                row = {field: meta.get(field, "") for field in METADATA_FIELDS}
                row["split"] = split
                for esc50_field in ("esc50_filename", "esc50_category", "esc50_fold", "esc50_src_file"):
                    row.setdefault(esc50_field, "")
                rows.append(row)

    metadata_csv = tmp_path / "metadata.csv"
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return metadata_csv


@pytest.fixture(scope="module")
def trained_models_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("predict_easy")
    metadata_csv = _make_easy_metadata(tmp_path)
    models_dir = tmp_path / "models"
    train(metadata_csv=metadata_csv, models_dir=models_dir, epochs=8, batch_size=8, seed=42)
    return models_dir, metadata_csv


@pytest.fixture(scope="module")
def sample_wav_per_label(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("predict_samples")
    paths = {}
    for label, freq in LABELS:
        wav_path = tmp_path / f"{label}_sample.wav"
        generate_morse_wav(label, wav_path, speed="normal", frequency_hz=freq, amplitude=0.6, seed=999)
        paths[label] = wav_path
    return paths


def test_load_model_returns_expected_structures(trained_models_dir):
    models_dir, _ = trained_models_dir
    model, label_to_index, index_to_label, fixed_frames = load_model(models_dir, device="cpu")
    assert set(label_to_index.keys()) == {label for label, _ in LABELS}
    assert set(index_to_label.values()) == {label for label, _ in LABELS}
    assert fixed_frames > 0
    assert isinstance(model, torch.nn.Module)


def test_predict_file_returns_top3_by_default(trained_models_dir, sample_wav_per_label):
    models_dir, _ = trained_models_dir
    wav_path = sample_wav_per_label["イ"]
    result = predict_file(wav_path, models_dir=models_dir, device="cpu")
    assert len(result.top_k) == 3


def test_predict_confidence_is_probability_and_sorted_descending(trained_models_dir, sample_wav_per_label):
    models_dir, _ = trained_models_dir
    wav_path = sample_wav_per_label["ロ"]
    result = predict_file(wav_path, models_dir=models_dir, device="cpu")

    confidences = [c.confidence for c in result.top_k]
    assert confidences == sorted(confidences, reverse=True)
    for c in confidences:
        assert 0.0 <= c <= 1.0
    assert result.confidence == confidences[0]
    assert result.prediction == result.top_k[0].label


def test_predict_achieves_correct_prediction_on_easy_task(trained_models_dir, sample_wav_per_label):
    """周波数が大きく異なる簡単なタスクなので、各クラスとも正しく予測できるはず。"""
    models_dir, _ = trained_models_dir
    for label, wav_path in sample_wav_per_label.items():
        result = predict_file(wav_path, models_dir=models_dir, device="cpu")
        assert result.prediction == label, f"{label} を正しく予測できなかった(予測: {result.prediction})"


def test_predict_mel_spectrogram_shape(trained_models_dir, sample_wav_per_label):
    models_dir, _ = trained_models_dir
    wav_path = sample_wav_per_label["1"]
    result = predict_file(wav_path, models_dir=models_dir, device="cpu")
    assert result.mel_spectrogram.ndim == 2
    assert result.mel_spectrogram.shape[0] == N_MELS


def test_predict_top_k_capped_at_num_classes(trained_models_dir, sample_wav_per_label):
    models_dir, _ = trained_models_dir
    wav_path = sample_wav_per_label["イ"]
    result = predict_file(wav_path, models_dir=models_dir, top_k=10, device="cpu")
    assert len(result.top_k) == len(LABELS)  # クラス数(3)を超えない


def test_predict_low_level_function_matches_predict_file(trained_models_dir, sample_wav_per_label):
    models_dir, _ = trained_models_dir
    model, _label_to_index, index_to_label, fixed_frames = load_model(models_dir, device="cpu")
    wav_path = sample_wav_per_label["ロ"]

    result_low_level = predict(wav_path, model, index_to_label, fixed_frames, device="cpu")
    result_file = predict_file(wav_path, models_dir=models_dir, device="cpu")

    assert result_low_level.prediction == result_file.prediction
    assert result_low_level.confidence == pytest.approx(result_file.confidence)

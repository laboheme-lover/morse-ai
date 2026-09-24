"""evaluate_model.py のユニットテスト(Milestone 7)。"""

import csv

import pandas as pd
import pytest

from evaluate_model import (
    compute_common_errors,
    compute_noise_type_metrics,
    compute_overall_metrics,
    compute_snr_metrics,
    evaluate,
)
from generate_dataset import METADATA_FIELDS
from generate_morse_audio import generate_morse_wav
from train import train


def _predictions_df(rows):
    return pd.DataFrame(rows)


def test_compute_overall_metrics_perfect_accuracy():
    df = _predictions_df(
        [
            {"label": "イ", "predicted_label": "イ", "correct": True},
            {"label": "ロ", "predicted_label": "ロ", "correct": True},
        ]
    )
    metrics = compute_overall_metrics(df)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["n_samples"] == 2


def test_compute_overall_metrics_with_errors():
    df = _predictions_df(
        [
            {"label": "イ", "predicted_label": "イ", "correct": True},
            {"label": "イ", "predicted_label": "ロ", "correct": False},
            {"label": "ロ", "predicted_label": "ロ", "correct": True},
            {"label": "ロ", "predicted_label": "ロ", "correct": True},
        ]
    )
    metrics = compute_overall_metrics(df)
    assert metrics["accuracy"] == 0.75


def test_compute_snr_metrics_groups_clean_and_snr_levels():
    df = _predictions_df(
        [
            {"label": "イ", "predicted_label": "イ", "correct": True, "noise_type": "clean", "snr_db": ""},
            {"label": "ロ", "predicted_label": "ロ", "correct": True, "noise_type": "clean", "snr_db": ""},
            {"label": "イ", "predicted_label": "イ", "correct": True, "noise_type": "white", "snr_db": "20.0"},
            {"label": "ロ", "predicted_label": "イ", "correct": False, "noise_type": "white", "snr_db": "20.0"},
            {"label": "イ", "predicted_label": "ロ", "correct": False, "noise_type": "esc50", "snr_db": "-5.0"},
        ]
    )
    snr_df = compute_snr_metrics(df)
    assert list(snr_df["snr_db"]) == ["clean", "-5.0", "20.0"]
    clean_row = snr_df[snr_df["snr_db"] == "clean"].iloc[0]
    assert clean_row["accuracy"] == 1.0
    assert clean_row["n_samples"] == 2

    snr20_row = snr_df[snr_df["snr_db"] == "20.0"].iloc[0]
    assert snr20_row["accuracy"] == 0.5
    assert snr20_row["n_samples"] == 2


def test_compute_noise_type_metrics():
    df = _predictions_df(
        [
            {"label": "イ", "predicted_label": "イ", "correct": True, "noise_type": "clean"},
            {"label": "イ", "predicted_label": "イ", "correct": True, "noise_type": "white"},
            {"label": "イ", "predicted_label": "ロ", "correct": False, "noise_type": "white"},
        ]
    )
    noise_df = compute_noise_type_metrics(df)
    assert set(noise_df["noise_type"]) == {"clean", "white"}
    white_row = noise_df[noise_df["noise_type"] == "white"].iloc[0]
    assert white_row["accuracy"] == 0.5
    assert white_row["n_samples"] == 2


def test_compute_common_errors_counts_and_sorts_descending():
    df = _predictions_df(
        [
            {"label": "イ", "predicted_label": "ロ", "correct": False},
            {"label": "イ", "predicted_label": "ロ", "correct": False},
            {"label": "イ", "predicted_label": "ハ", "correct": False},
            {"label": "ロ", "predicted_label": "ロ", "correct": True},
        ]
    )
    errors = compute_common_errors(df)
    assert list(errors.columns) == ["true_label", "predicted_label", "count"]
    assert errors.iloc[0]["true_label"] == "イ"
    assert errors.iloc[0]["predicted_label"] == "ロ"
    assert errors.iloc[0]["count"] == 2
    assert len(errors) == 2  # (イ,ロ)と(イ,ハ)の2ペア


def test_compute_common_errors_empty_when_all_correct():
    df = _predictions_df(
        [
            {"label": "イ", "predicted_label": "イ", "correct": True},
            {"label": "ロ", "predicted_label": "ロ", "correct": True},
        ]
    )
    errors = compute_common_errors(df)
    assert len(errors) == 0
    assert list(errors.columns) == ["true_label", "predicted_label", "count"]


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
def evaluated_easy_task(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("eval_easy")
    metadata_csv = _make_easy_metadata(tmp_path)
    models_dir = tmp_path / "models"
    train(metadata_csv=metadata_csv, models_dir=models_dir, epochs=8, batch_size=8, seed=42)

    output_dir = tmp_path / "evaluation"
    result = evaluate(metadata_csv=metadata_csv, models_dir=models_dir, output_dir=output_dir)
    return output_dir, result


def test_evaluate_creates_all_expected_output_files(evaluated_easy_task):
    output_dir, _ = evaluated_easy_task
    for filename in (
        "test_predictions.csv",
        "snr_metrics.csv",
        "noise_metrics.csv",
        "common_errors.csv",
        "confusion_matrix.png",
        "performance_by_snr.png",
    ):
        assert (output_dir / filename).exists(), f"{filename} が生成されていない"


def test_evaluate_achieves_good_accuracy_on_easy_task(evaluated_easy_task):
    _, result = evaluated_easy_task
    assert result["overall"]["accuracy"] > 0.8
    assert result["overall"]["macro_f1"] > 0.8


def test_evaluate_predictions_csv_has_expected_columns(evaluated_easy_task):
    output_dir, _ = evaluated_easy_task
    df = pd.read_csv(output_dir / "test_predictions.csv", keep_default_na=False)
    for col in ("filepath", "label", "predicted_label", "confidence", "correct", "noise_type", "snr_db"):
        assert col in df.columns
    assert len(df) == len(LABELS) * 15


def test_evaluate_snr_metrics_only_has_clean_for_easy_task(evaluated_easy_task):
    _, result = evaluated_easy_task
    snr_df = result["snr_metrics"]
    assert list(snr_df["snr_db"]) == ["clean"]

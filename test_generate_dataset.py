"""generate_dataset.py のユニットテスト(Milestone 4)。"""

import csv
from pathlib import Path

import numpy as np
import pytest

from esc50_manager import TEST_FOLDS, TRAIN_FOLDS, VAL_FOLDS
from generate_dataset import (
    METADATA_FIELDS,
    SPLIT_RATIOS,
    assign_splits,
    generate_dataset,
    target_labels,
)
from morse_table import MORSE_CODE_TABLE


def test_target_labels_covers_all_63_classes():
    labels = target_labels()
    assert len(labels) == 63
    assert "。" in labels
    assert set(labels) == set(MORSE_CODE_TABLE.keys())


def test_assign_splits_ratio_for_large_n():
    rng = np.random.default_rng(0)
    splits = assign_splits(1000, rng)
    assert len(splits) == 1000
    counts = {s: splits.count(s) for s in SPLIT_RATIOS}
    for split, ratio in SPLIT_RATIOS.items():
        assert counts[split] / 1000 == pytest.approx(ratio, abs=0.02)


def test_assign_splits_small_n_covers_all_splits():
    rng = np.random.default_rng(1)
    splits = assign_splits(3, rng)
    assert sorted(splits) == sorted(SPLIT_RATIOS.keys())


def test_assign_splits_total_count_matches_n():
    rng = np.random.default_rng(2)
    for n in (1, 2, 5, 10, 17):
        splits = assign_splits(n, rng)
        assert len(splits) == n


def test_assign_splits_tie_breaking_is_not_biased_to_one_split():
    """n=10だとvalidation/testの端数(1.5件)が同点になる。多数のクラスに対して
    繰り返し呼び出した際、常に同じ側へ余りが偏らないことを確認する(最大剰余法の
    タイブレークがランダム化されていることの検証)。"""
    rng = np.random.default_rng(123)
    val_counts = []
    test_counts = []
    for _ in range(200):
        splits = assign_splits(10, rng)
        val_counts.append(splits.count("validation"))
        test_counts.append(splits.count("test"))

    # 200回中、validationが2件・testが1件になる回数と、その逆になる回数の
    # 両方が一定数以上存在すること(常に同じパターンにならないこと)を確認する
    val_more = sum(1 for v, t in zip(val_counts, test_counts) if v > t)
    test_more = sum(1 for v, t in zip(val_counts, test_counts) if t > v)
    assert val_more > 30
    assert test_more > 30

    # 集計するとほぼ70/15/15に近づくこと
    assert sum(val_counts) / (200 * 10) == pytest.approx(0.15, abs=0.03)
    assert sum(test_counts) / (200 * 10) == pytest.approx(0.15, abs=0.03)


def test_generate_dataset_without_esc50_uses_only_clean_or_white(tmp_path):
    metadata_path = generate_dataset(
        output_root=tmp_path / "dataset",
        metadata_csv=tmp_path / "dataset" / "metadata.csv",
        samples_per_class=4,
        esc50_root=tmp_path / "no_such_esc50",
        seed=0,
        labels=["イ", "ロ"],
    )

    with open(metadata_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 8
    assert {row["noise_type"] for row in rows} <= {"clean", "white"}
    for row in rows:
        assert Path(row["filepath"]).exists()
        assert row["label"] in {"イ", "ロ"}
        assert row["split"] in SPLIT_RATIOS


def test_generate_dataset_metadata_has_expected_columns(tmp_path):
    metadata_path = generate_dataset(
        output_root=tmp_path / "dataset",
        metadata_csv=tmp_path / "dataset" / "metadata.csv",
        samples_per_class=2,
        esc50_root=tmp_path / "no_such_esc50",
        seed=0,
        labels=["1"],
    )
    with open(metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == METADATA_FIELDS


def test_generate_dataset_with_esc50_respects_split_folds(tmp_path, fake_esc50_root):
    metadata_path = generate_dataset(
        output_root=tmp_path / "dataset",
        metadata_csv=tmp_path / "dataset" / "metadata.csv",
        samples_per_class=20,
        esc50_root=fake_esc50_root,
        seed=1,
        labels=["イ"],
    )

    with open(metadata_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    esc50_rows = [row for row in rows if row["noise_type"] == "esc50"]
    assert len(esc50_rows) > 0, "esc50ノイズのサンプルが1件も生成されなかった"

    expected_folds = {"train": set(TRAIN_FOLDS), "validation": set(VAL_FOLDS), "test": set(TEST_FOLDS)}
    for row in esc50_rows:
        assert int(row["esc50_fold"]) in expected_folds[row["split"]]
        assert row["esc50_filename"]
        assert row["esc50_category"]
        assert Path(row["filepath"]).exists()


def test_generate_dataset_is_idempotent_on_rerun(tmp_path):
    """metadata.csvは実行のたびに上書きされ、前回分と混ざらないことを確認する。"""
    output_root = tmp_path / "dataset"
    metadata_csv = output_root / "metadata.csv"

    generate_dataset(output_root, metadata_csv, samples_per_class=3, esc50_root=tmp_path / "no_esc50", seed=0, labels=["イ"])
    generate_dataset(output_root, metadata_csv, samples_per_class=2, esc50_root=tmp_path / "no_esc50", seed=0, labels=["イ"])

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2

"""train.py のユニットテスト(Milestone 6)。

学習が実際に機能すること(Lossの低下・Accuracyの上昇・ベストモデルと
label_mappingの保存)を、少量かつ容易に分類できる合成データセットで検証する。
"""

import csv
import json

import pytest
import torch

from generate_dataset import METADATA_FIELDS
from generate_morse_audio import generate_morse_wav
from morse_cnn import MorseCNN
from train import train

LABELS = [("イ", 700.0), ("ロ", 750.0), ("1", 650.0)]


def _make_easy_metadata(tmp_path, samples_per_split=15):
    """クラスごとに周波数を変え、単純に判別しやすい合成データセットを作る。"""
    rows = []
    seed = 0
    for label, freq in LABELS:
        morse_code = None
        for split in ("train", "validation", "test"):
            for i in range(samples_per_split):
                wav_path = tmp_path / "wav" / split / label / f"{label}_{i}.wav"
                _, meta = generate_morse_wav(
                    label, wav_path, speed="normal", frequency_hz=freq, amplitude=0.6, seed=seed
                )
                seed += 1
                morse_code = meta["morse_code"]
                row = {field: meta.get(field, "") for field in METADATA_FIELDS}
                row["split"] = split
                row.setdefault("esc50_filename", "")
                row.setdefault("esc50_category", "")
                row.setdefault("esc50_fold", "")
                row.setdefault("esc50_src_file", "")
                rows.append(row)

    metadata_csv = tmp_path / "metadata.csv"
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return metadata_csv


@pytest.fixture(scope="module")
def easy_train_result(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("easy_dataset")
    metadata_csv = _make_easy_metadata(tmp_path)
    models_dir = tmp_path / "models"
    result = train(metadata_csv=metadata_csv, models_dir=models_dir, epochs=8, batch_size=8, seed=42)
    return result


def test_train_saves_best_model_and_label_mapping(easy_train_result):
    assert easy_train_result["best_model_path"].exists()
    assert easy_train_result["label_map_path"].exists()
    assert easy_train_result["history_path"].exists()


def test_train_history_has_expected_length(easy_train_result):
    assert len(easy_train_result["history"]) == 8
    for entry in easy_train_result["history"]:
        assert set(entry) == {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}


def test_train_loss_decreases_over_epochs(easy_train_result):
    history = easy_train_result["history"]
    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_train_accuracy_increases_over_epochs(easy_train_result):
    history = easy_train_result["history"]
    assert history[-1]["train_acc"] >= history[0]["train_acc"]


def test_train_achieves_good_val_accuracy_on_easy_task(easy_train_result):
    """周波数が大きく異なる3クラスの簡単な分類タスクなので高精度に達するはず。"""
    assert easy_train_result["best_val_acc"] > 0.8


def test_label_mapping_json_is_loadable_and_consistent(easy_train_result):
    with open(easy_train_result["label_map_path"], encoding="utf-8") as f:
        mapping = json.load(f)
    assert set(mapping["label_to_index"].keys()) == {label for label, _ in LABELS}
    assert isinstance(mapping["fixed_frames"], int)
    assert mapping["fixed_frames"] > 0


def test_best_model_checkpoint_loads_into_model(easy_train_result):
    checkpoint = torch.load(easy_train_result["best_model_path"], weights_only=True)
    model = MorseCNN(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    assert checkpoint["num_classes"] == len(LABELS)
    assert checkpoint["fixed_frames"] > 0


def test_train_saves_last_model_checkpoint_every_epoch(easy_train_result):
    assert easy_train_result["last_model_path"].exists()
    checkpoint = torch.load(easy_train_result["last_model_path"], weights_only=True)
    assert checkpoint["epoch"] == 8
    assert "optimizer_state_dict" in checkpoint


def test_train_saves_model_config_for_architecture_reconstruction(easy_train_result):
    """model_configが保存されていれば、学習時と全く同じ構造でMorseCNNを
    再構築できる(predict.py/evaluate_model.pyがこれに依存する)。"""
    checkpoint = torch.load(easy_train_result["best_model_path"], weights_only=True)
    assert "model_config" in checkpoint
    config = checkpoint["model_config"]
    assert config["num_classes"] == len(LABELS)

    model = MorseCNN.from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])  # 形状が一致すればエラーにならない


def test_train_resume_continues_epoch_numbering_and_history(tmp_path):
    metadata_csv = _make_easy_metadata(tmp_path)
    models_dir = tmp_path / "models_resume"

    first = train(metadata_csv=metadata_csv, models_dir=models_dir, epochs=3, batch_size=8, seed=42)
    assert [entry["epoch"] for entry in first["history"]] == [1, 2, 3]

    second = train(metadata_csv=metadata_csv, models_dir=models_dir, epochs=3, batch_size=8, seed=42, resume=True)
    assert [entry["epoch"] for entry in second["history"]] == [1, 2, 3, 4, 5, 6]

    with open(second["history_path"], encoding="utf-8") as f:
        saved_history = json.load(f)
    assert [entry["epoch"] for entry in saved_history] == [1, 2, 3, 4, 5, 6]


def test_train_resume_without_existing_checkpoint_raises(tmp_path):
    metadata_csv = _make_easy_metadata(tmp_path)
    empty_models_dir = tmp_path / "models_no_checkpoint"
    with pytest.raises(FileNotFoundError):
        train(metadata_csv=metadata_csv, models_dir=empty_models_dir, epochs=1, resume=True)


def test_train_resume_preserves_or_improves_best_val_acc(tmp_path):
    metadata_csv = _make_easy_metadata(tmp_path)
    models_dir = tmp_path / "models_resume_bestacc"

    first = train(metadata_csv=metadata_csv, models_dir=models_dir, epochs=5, batch_size=8, seed=42)
    second = train(metadata_csv=metadata_csv, models_dir=models_dir, epochs=5, batch_size=8, seed=42, resume=True)

    assert second["best_val_acc"] >= first["best_val_acc"]


def test_train_supports_crnn_architecture_and_saves_arch_in_config(tmp_path):
    metadata_csv = _make_easy_metadata(tmp_path)
    models_dir = tmp_path / "models_crnn"

    result = train(metadata_csv=metadata_csv, models_dir=models_dir, epochs=3, batch_size=8, seed=42, arch="crnn")
    checkpoint = torch.load(result["best_model_path"], weights_only=True)
    assert checkpoint["model_config"]["arch"] == "crnn"

    # 再開時は保存済みのarchで再構築される(archを指定し直す必要がない)
    resumed = train(metadata_csv=metadata_csv, models_dir=models_dir, epochs=1, batch_size=8, seed=42, resume=True)
    assert len(resumed["history"]) == 4
    assert torch.load(resumed["last_model_path"], weights_only=True)["model_config"]["arch"] == "crnn"

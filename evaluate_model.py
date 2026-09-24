"""学習済みCNNをtestデータで評価するモジュール(Milestone 7)。

Accuracy・Macro F1・混同行列に加え、SNR別・noise_type別の性能内訳と
主要な誤認識ペアを算出し、evaluation/ 以下に保存する。
testデータはハイパーパラメータ調整には用いず、最終評価にのみ使用する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ヘッドレス実行でも保存できるようにする
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from morse_cnn import MorseCNN, model_from_config
from morse_dataset import DEFAULT_METADATA_CSV, MorseCharacterDataset

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

for _font_name in ("Yu Gothic", "Meiryo", "MS Gothic"):
    if _font_name in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_MODELS_DIR = Path("models")
DEFAULT_EVAL_DIR = Path("evaluation")


def load_model(models_dir: str | Path, device: str) -> tuple[MorseCNN, dict, dict, int]:
    """models/best_model.pt と label_mapping.json を読み込む。"""
    models_dir = Path(models_dir)
    with open(models_dir / "label_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)
    label_to_index: dict[str, int] = mapping["label_to_index"]
    fixed_frames: int = mapping["fixed_frames"]
    index_to_label = {i: label for label, i in label_to_index.items()}

    checkpoint = torch.load(models_dir / "best_model.pt", map_location=device, weights_only=True)
    model_config = checkpoint.get("model_config", {"num_classes": checkpoint["num_classes"]})
    model = model_from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, label_to_index, index_to_label, fixed_frames


def run_inference(
    model: MorseCNN,
    dataset: MorseCharacterDataset,
    index_to_label: dict[int, str],
    device: str,
    batch_size: int = 32,
) -> pd.DataFrame:
    """datasetの全サンプルを推論し、metadataに予測結果・確信度・正誤を結合して返す。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    predicted_labels: list[str] = []
    confidences: list[float] = []
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)
            predicted_labels.extend(index_to_label[int(i)] for i in pred_idx.cpu().tolist())
            confidences.extend(conf.cpu().tolist())

    result = dataset.df.copy().reset_index(drop=True)
    result["predicted_label"] = predicted_labels
    result["confidence"] = confidences
    result["correct"] = result["label"] == result["predicted_label"]
    return result


def compute_overall_metrics(df: pd.DataFrame) -> dict:
    accuracy = float(df["correct"].mean())
    macro_f1 = float(f1_score(df["label"], df["predicted_label"], average="macro", zero_division=0))
    return {"accuracy": accuracy, "macro_f1": macro_f1, "n_samples": int(len(df))}


def compute_snr_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """SNR別(cleanも1カテゴリとして含む)のAccuracy・Macro F1を算出する。"""
    df = df.copy()
    df["snr_group"] = df.apply(lambda r: "clean" if r["noise_type"] == "clean" else r["snr_db"], axis=1)

    rows = []
    for snr_group, group in df.groupby("snr_group"):
        rows.append(
            {
                "snr_db": snr_group,
                "n_samples": len(group),
                "accuracy": float(group["correct"].mean()),
                "macro_f1": float(f1_score(group["label"], group["predicted_label"], average="macro", zero_division=0)),
            }
        )

    def sort_key(value: str) -> tuple[int, float]:
        return (-1, 0.0) if value == "clean" else (0, float(value))

    rows.sort(key=lambda r: sort_key(r["snr_db"]))
    return pd.DataFrame(rows)


def compute_noise_type_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """noise_type別のAccuracyを算出する。"""
    rows = [
        {"noise_type": noise_type, "n_samples": len(group), "accuracy": float(group["correct"].mean())}
        for noise_type, group in df.groupby("noise_type")
    ]
    return pd.DataFrame(rows).sort_values("noise_type").reset_index(drop=True)


def compute_common_errors(df: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """誤認識ペア(正解ラベル, 予測ラベル)を出現頻度順に並べる。"""
    errors = df[~df["correct"]]
    counts = (
        errors.groupby(["label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .rename(columns={"label": "true_label"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    if top_n is not None:
        counts = counts.head(top_n)
    return counts


def plot_confusion_matrix(df: pd.DataFrame, labels: list[str], output_path: str | Path) -> Path:
    cm = confusion_matrix(df["label"], df["predicted_label"], labels=labels)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("予測ラベル")
    ax.set_ylabel("正解ラベル")
    ax.set_title("混同行列(テストデータ)")
    fig.colorbar(im, ax=ax, label="件数")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_performance_by_snr(snr_df: pd.DataFrame, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = [str(v) for v in snr_df["snr_db"]]
    x = np.arange(len(x_labels))
    width = 0.35
    ax.bar(x - width / 2, snr_df["accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, snr_df["macro_f1"], width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("SNR [dB] (cleanはノイズなし)")
    ax.set_ylabel("スコア")
    ax.set_ylim(0, 1)
    ax.set_title("SNR別性能")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def evaluate(
    metadata_csv: str | Path = DEFAULT_METADATA_CSV,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    output_dir: str | Path = DEFAULT_EVAL_DIR,
    batch_size: int = 32,
    device: str | None = None,
) -> dict:
    """testデータで評価を実行し、指標とグラフをoutput_dirへ保存する。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, label_to_index, index_to_label, fixed_frames = load_model(models_dir, device)
    dataset = MorseCharacterDataset(
        metadata_csv, split="test", fixed_frames=fixed_frames, label_to_index=label_to_index
    )

    predictions_df = run_inference(model, dataset, index_to_label, device, batch_size)
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")

    overall = compute_overall_metrics(predictions_df)

    snr_df = compute_snr_metrics(predictions_df)
    snr_df.to_csv(output_dir / "snr_metrics.csv", index=False, encoding="utf-8")

    noise_df = compute_noise_type_metrics(predictions_df)
    noise_df.to_csv(output_dir / "noise_metrics.csv", index=False, encoding="utf-8")

    errors_df = compute_common_errors(predictions_df)
    errors_df.to_csv(output_dir / "common_errors.csv", index=False, encoding="utf-8")

    labels_sorted = sorted(label_to_index, key=lambda label: label_to_index[label])
    plot_confusion_matrix(predictions_df, labels_sorted, output_dir / "confusion_matrix.png")
    plot_performance_by_snr(snr_df, output_dir / "performance_by_snr.png")

    print(f"Accuracy: {overall['accuracy']:.4f}")
    print(f"Macro F1: {overall['macro_f1']:.4f}")
    print(f"評価件数: {overall['n_samples']}")
    print(f"評価結果を保存しました: {output_dir}")

    return {
        "overall": overall,
        "snr_metrics": snr_df,
        "noise_metrics": noise_df,
        "common_errors": errors_df,
        "predictions": predictions_df,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="学習済みCNNをtestデータで評価する")
    parser.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_EVAL_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    evaluate(args.metadata_csv, args.models_dir, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()

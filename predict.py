"""単一WAVファイルからモールス信号の文字を推論するモジュール(Milestone 8)。

WAV -> メルスペクトログラム(morse_dataset.pyの前処理関数を学習時と共有) ->
CNN -> Softmax -> Top-K候補、を返す。models/best_model.ptとlabel_mapping.json
を読み込んで使用する。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import librosa

from morse_cnn import MorseCNN, model_from_config
from morse_dataset import SAMPLE_RATE, fix_length, waveform_to_mel

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_MODELS_DIR = Path("models")
DEFAULT_TOP_K = 3


@dataclass
class Candidate:
    label: str
    confidence: float


@dataclass
class PredictionResult:
    prediction: str
    confidence: float
    top_k: list[Candidate]
    mel_spectrogram: np.ndarray  # 固定サイズ化前(実際の音声長)のメルスペクトログラム


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


def predict_waveform(
    y: np.ndarray,
    model: MorseCNN,
    index_to_label: dict[int, str],
    fixed_frames: int,
    sample_rate: int = SAMPLE_RATE,
    device: str = "cpu",
    top_k: int = DEFAULT_TOP_K,
) -> PredictionResult:
    """波形(numpy配列)から文字を推論する。

    連続音声認識(predict_sequence.py)で文字ごとに切り出したチャンクを、
    ファイルに書き出さず直接推論する際に使う。学習(MorseCharacterDataset)と
    同一のwaveform_to_mel/fix_length関数を使うことで前処理を確実に共有する。
    """
    mel = waveform_to_mel(y, sample_rate)
    mel_fixed = fix_length(mel, fixed_frames)
    x = torch.from_numpy(mel_fixed).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, n_mels, fixed_frames]

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(top_k, probs.shape[0])
    top_probs, top_indices = torch.topk(probs, k)
    candidates = [
        Candidate(label=index_to_label[int(idx)], confidence=float(p))
        for p, idx in zip(top_probs.tolist(), top_indices.tolist())
    ]

    return PredictionResult(
        prediction=candidates[0].label,
        confidence=candidates[0].confidence,
        top_k=candidates,
        mel_spectrogram=mel,
    )


def predict(
    wav_path: str | Path,
    model: MorseCNN,
    index_to_label: dict[int, str],
    fixed_frames: int,
    device: str = "cpu",
    top_k: int = DEFAULT_TOP_K,
) -> PredictionResult:
    """1つのWAVファイルから文字を推論する。"""
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    return predict_waveform(y, model, index_to_label, fixed_frames, sr, device, top_k)


def predict_file(
    wav_path: str | Path,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    top_k: int = DEFAULT_TOP_K,
    device: str | None = None,
) -> PredictionResult:
    """モデルの読み込みから推論までを一括で行う簡易エントリポイント(CLI等から利用)。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _label_to_index, index_to_label, fixed_frames = load_model(models_dir, device)
    return predict(wav_path, model, index_to_label, fixed_frames, device, top_k)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="モールス信号WAVから文字を推論する")
    parser.add_argument("wav", help="入力WAVファイル")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = predict_file(args.wav, args.models_dir, args.top_k)
    print(f"予測文字: {result.prediction} (確信度: {result.confidence:.4f})")
    print("Top候補:")
    for i, cand in enumerate(result.top_k, start=1):
        print(f"  {i}. {cand.label}  確信度={cand.confidence:.4f}")


if __name__ == "__main__":
    main()

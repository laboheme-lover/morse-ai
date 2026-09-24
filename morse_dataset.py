"""WAV -> メルスペクトログラム -> Tensor変換とDataLoaderを提供するモジュール(Milestone 5)。

学習(train.py)・推論(predict.py)で同一の前処理を共有できるよう、
audio_to_mel_tensor() を共通の前処理エントリポイントとする。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 128

DEFAULT_METADATA_CSV = Path("data/dataset/metadata.csv")


def estimate_mel_frames(n_samples: int, hop_length: int = HOP_LENGTH) -> int:
    """librosa.feature.melspectrogram(center=True, デフォルト)が生成するフレーム数を返す。"""
    return 1 + n_samples // hop_length


def compute_max_frames(metadata_csv: str | Path = DEFAULT_METADATA_CSV, sample_rate: int = SAMPLE_RATE) -> int:
    """データセット全体で最大となるメルスペクトログラムのフレーム数を求める。

    モールス信号の末尾を切り詰めないよう、学習前にこの値をfixed_framesとして
    採用する(WAVを全てデコードせず soundfile.info() で長さのみ取得するため高速)。
    """
    df = pd.read_csv(metadata_csv, keep_default_na=False)
    max_frames = 0
    for filepath in df["filepath"]:
        info = sf.info(filepath)
        n_samples = info.frames
        if info.samplerate != sample_rate:
            n_samples = round(n_samples * sample_rate / info.samplerate)
        max_frames = max(max_frames, estimate_mel_frames(n_samples))
    return max_frames


def waveform_to_mel(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """波形(numpy配列)からメルスペクトログラム(dB)を計算する。形状は[n_mels, time]。

    連続音声認識(predict_sequence.py)のように、ファイルに書き出さず
    切り出した波形チャンクを直接推論する場合に使う。
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def wav_to_mel(
    wav_path: str | Path,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """WAVファイルを読み込みメルスペクトログラム(dB)を計算する。形状は[n_mels, time]。"""
    y, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    return waveform_to_mel(y, sample_rate, n_mels, n_fft, hop_length)


def fix_length(mel: np.ndarray, fixed_frames: int) -> np.ndarray:
    """時間方向のフレーム数をfixed_framesに揃える。

    短い場合は右側を無音相当の値でパディングし、長い場合は切り詰める。
    パディング値には0ではなくそのスペクトログラム自身の最小値(=無音に近い
    dB値)を使うことで、無音区間との連続性を保つ。
    """
    current_frames = mel.shape[1]
    if current_frames == fixed_frames:
        return mel
    if current_frames < fixed_frames:
        pad_width = fixed_frames - current_frames
        pad_value = float(mel.min()) if mel.size > 0 else 0.0
        return np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=pad_value)
    return mel[:, :fixed_frames]


def audio_to_mel_tensor(
    wav_path: str | Path,
    fixed_frames: int,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    """WAVファイルを固定サイズのメルスペクトログラムTensor([1, n_mels, fixed_frames])に変換する。"""
    mel = wav_to_mel(wav_path, sample_rate, n_mels, n_fft, hop_length)
    mel = fix_length(mel, fixed_frames)
    return torch.from_numpy(mel).unsqueeze(0)


class MorseCharacterDataset(Dataset):
    """metadata.csvに基づく和文モールス1文字分類用データセット。"""

    def __init__(
        self,
        metadata_csv: str | Path = DEFAULT_METADATA_CSV,
        split: str | None = None,
        fixed_frames: int | None = None,
        label_to_index: dict[str, int] | None = None,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ):
        df = pd.read_csv(metadata_csv, keep_default_na=False)
        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"split={split!r} に該当するサンプルがありません: {metadata_csv}")

        self.df = df
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fixed_frames = fixed_frames if fixed_frames is not None else compute_max_frames(metadata_csv, sample_rate)

        if label_to_index is None:
            labels = sorted(df["label"].unique())
            label_to_index = {label: i for i, label in enumerate(labels)}
        self.label_to_index = label_to_index
        self.index_to_label = {i: label for label, i in label_to_index.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        mel_tensor = audio_to_mel_tensor(
            row["filepath"], self.fixed_frames, self.sample_rate, self.n_mels, self.n_fft, self.hop_length
        )
        label_index = self.label_to_index[row["label"]]
        return mel_tensor, label_index


def build_dataloaders(
    metadata_csv: str | Path = DEFAULT_METADATA_CSV,
    batch_size: int = 32,
    num_workers: int = 0,
    fixed_frames: int | None = None,
    label_to_index: dict[str, int] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int], int]:
    """train/validation/test用のDataLoaderを構築する。

    戻り値のlabel_to_indexとfixed_framesは学習・推論間で一致させる必要が
    あるため、train.py側でmodels/へ保存し、predict.py側で読み込む想定。
    学習を再開する場合は、保存済みのlabel_to_index/fixed_framesをそのまま
    渡すことで、クラスの対応付けがずれないようにする。
    """
    if fixed_frames is None:
        fixed_frames = compute_max_frames(metadata_csv)

    if label_to_index is None:
        full_df = pd.read_csv(metadata_csv, keep_default_na=False)
        labels = sorted(full_df["label"].unique())
        label_to_index = {label: i for i, label in enumerate(labels)}

    loaders = {}
    for split, shuffle in (("train", True), ("validation", False), ("test", False)):
        dataset = MorseCharacterDataset(
            metadata_csv, split=split, fixed_frames=fixed_frames, label_to_index=label_to_index
        )
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loaders["train"], loaders["validation"], loaders["test"], label_to_index, fixed_frames


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DataLoaderを構築しバッチ形状を確認する")
    parser.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_loader, val_loader, test_loader, label_to_index, fixed_frames = build_dataloaders(
        args.metadata_csv, batch_size=args.batch_size
    )
    print(f"クラス数: {len(label_to_index)}")
    print(f"fixed_frames: {fixed_frames}")
    print(f"train: {len(train_loader.dataset)}件 ({len(train_loader)}バッチ)")
    print(f"validation: {len(val_loader.dataset)}件 ({len(val_loader)}バッチ)")
    print(f"test: {len(test_loader.dataset)}件 ({len(test_loader)}バッチ)")

    x, y = next(iter(train_loader))
    print(f"batch x shape: {tuple(x.shape)}, dtype: {x.dtype}")
    print(f"batch y shape: {tuple(y.shape)}, dtype: {y.dtype}")


if __name__ == "__main__":
    main()

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

# ============================================================================
# このモジュールの役割
#   音声(波形)を、モデルが読める「音の絵」= メルスペクトログラムに変換し、
#   学習用にまとめて(バッチにして)取り出せるようにする。
#
# メルスペクトログラムとは
#   横軸=時間、縦軸=音の高さ(周波数)、各マスの値=その時刻・その高さの音の強さ(dB)を表す画像。
#   モールス信号なら「ある高さ(トーンの周波数)の横線が、短く/長く点滅している絵」になる。
#   縦軸は人間の聴覚に近い「メル尺度」で64段に区切る。
#
# 重要: 学習と推論で同じ変換を使う
#   学習時と推論時で変換条件(下の4つの定数)が1つでも違うと、モデルは見たことのない絵を
#   見せられることになり精度が落ちる。そのため変換はこのモジュールの関数に一本化し、
#   train.py・predict.py・predict_sequence.py のすべてがここを呼ぶ。
# ============================================================================

SAMPLE_RATE = 16000  # 音声を16kHzに揃えて読み込む
N_MELS = 64  # 縦方向(周波数)の段数
N_FFT = 512  # 1回の周波数分析に使うサンプル数(=32ミリ秒分の窓)
HOP_LENGTH = 128  # 窓をずらす幅(=8ミリ秒)。横方向の1マス = 8ミリ秒

DEFAULT_METADATA_CSV = Path("data/dataset/metadata.csv")


def estimate_mel_frames(n_samples: int, hop_length: int = HOP_LENGTH) -> int:
    """librosa.feature.melspectrogram(center=True, デフォルト)が生成するフレーム数を返す。"""
    return 1 + n_samples // hop_length


def compute_max_frames(metadata_csv: str | Path = DEFAULT_METADATA_CSV, sample_rate: int = SAMPLE_RATE) -> int:
    """データセット全体で最大となるメルスペクトログラムのフレーム数を求める。

    モールス信号の末尾を切り詰めないよう、学習前にこの値をfixed_framesとして
    採用する(WAVを全てデコードせず soundfile.info() で長さのみ取得するため高速)。
    """
    # モデルには同じ大きさの絵しか入れられないので、全音声の中で一番長いものに合わせて
    # 横幅(フレーム数)を決める。一番長いのは「slowの速度で符号が長い文字」で、現行データでは424フレーム。
    # keep_default_na=False: 空欄を NaN にせず空文字のまま読む(snr_db が空欄の行があるため)
    df = pd.read_csv(metadata_csv, keep_default_na=False)
    max_frames = 0
    for filepath in df["filepath"]:
        info = sf.info(filepath)  # ヘッダーだけ読んで長さ(サンプル数)を得る。中身は読まないので速い
        n_samples = info.frames
        if info.samplerate != sample_rate:
            # サンプリング周波数が違うファイルは、16kHzに変換した場合の長さに換算する
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
    # (1) 波形を8ミリ秒ずつずらしながら周波数分析し、64段のメル尺度にまとめる(値は電力)
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    # (2) 電力をデシベル(dB)に変換する。ref=np.max なので、その音声の一番強いマスが 0dB、
    #     それより弱いマスは負の値になる。音量の違う音声でも同じ尺度で比べられる(音量の正規化)。
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
    current_frames = mel.shape[1]  # 今の横幅(時間方向のマス数)
    if current_frames == fixed_frames:
        return mel
    if current_frames < fixed_frames:
        # 短い場合: 右側(時間の後ろ)に足りない分の列を足す。
        # 足す値は「この絵の中で一番弱い値」= 無音に相当する値。0dB(=一番強い音)で埋めると
        # 「大きな音が鳴っている」ように見えてしまうため。
        pad_width = fixed_frames - current_frames
        pad_value = float(mel.min()) if mel.size > 0 else 0.0
        # ((0, 0), (0, pad_width)) = 縦方向は足さない、横方向は右にだけ pad_width 列足す
        return np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=pad_value)
    # 長い場合: 右側を切り捨てる(連続音声から切り出した長すぎる区間など)
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
    mel = wav_to_mel(wav_path, sample_rate, n_mels, n_fft, hop_length)  # [64, 長さ]
    mel = fix_length(mel, fixed_frames)  # [64, fixed_frames] に揃える
    # PyTorchのTensorに変換し、先頭に「チャンネル」の次元を1つ足す → [1, 64, fixed_frames]
    # (画像の「白黒1枚」に相当。畳み込み層は [チャンネル, 縦, 横] の形を受け取るため)
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
        # PyTorch の Dataset: 「全部で何件あるか(__len__)」と「i番目を取り出す(__getitem__)」を
        # 定義しておくと、DataLoader が自動でバッチ(32件ずつ等)にまとめてくれる。
        # ここでは metadata.csv の一覧だけを持ち、音声の読み込みは取り出すときに1件ずつ行う
        # (18,900件の絵を全部メモリに載せずに済む)。
        df = pd.read_csv(metadata_csv, keep_default_na=False)
        if split is not None:
            # 指定された split(train/validation/test)の行だけに絞る
            df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"split={split!r} に該当するサンプルがありません: {metadata_csv}")

        self.df = df
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fixed_frames = fixed_frames if fixed_frames is not None else compute_max_frames(metadata_csv, sample_rate)

        # 文字 → 番号(0〜62)の対応表。モデルの出力は「何番のクラスか」という番号なので、
        # 文字との対応を固定しておく必要がある。指定が無ければ文字を並べ替えた順に番号を振る。
        if label_to_index is None:
            labels = sorted(df["label"].unique())
            label_to_index = {label: i for i, label in enumerate(labels)}
        self.label_to_index = label_to_index
        self.index_to_label = {i: label for label, i in label_to_index.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # idx 番目の行のWAVを読み込んで「音の絵」にし、正解の文字を番号にして返す
        # (入力 [1, 64, fixed_frames] のTensor, 正解のクラス番号) の組 = 学習の1問分
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

    # 3つの split それぞれの DataLoader を作る。
    # 学習用(train)だけは毎エポック順番をシャッフルする(同じ順だと並び順まで覚えてしまうため)。
    # 検証・テスト用は順番を固定し、毎回同じ条件で評価できるようにする。
    # 3つとも同じ fixed_frames と label_to_index を使う(ずれると比較できない)。
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

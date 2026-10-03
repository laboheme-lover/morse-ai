"""モールス信号WAVの波形とメルスペクトログラムを可視化するモジュール。

前処理パラメータはSection 9の初期値(SAMPLE_RATE=16000, N_MELS=64,
N_FFT=512, HOP_LENGTH=128)に合わせておき、学習パイプラインと同じ値を
後で再利用できるようにする。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

matplotlib.use("Agg")  # Windowsのヘッドレス実行でも保存できるようにする
import matplotlib.pyplot as plt
import numpy as np

# 確認用の補助ツール(学習・推論の処理には使われない)。生成した音声が意図どおりか目で確かめるためのもの。
# 値は morse_dataset.py の前処理と同じにしてあり、モデルが見るのとほぼ同じ「音の絵」を描く。
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 128

# 日本語(カタカナ)をタイトルに表示するためのフォント設定。
# 見つかったフォントを使い、無ければデフォルトのまま続行する。
for _font_name in ("Yu Gothic", "Meiryo", "MS Gothic"):
    if _font_name in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False


def load_audio(wav_path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    return y


def compute_mel_spectrogram(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
    )
    return librosa.power_to_db(mel, ref=np.max)


def plot_waveform_and_spectrogram(
    wav_path: str | Path,
    output_path: str | Path | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """WAVの波形とメルスペクトログラムを1枚の画像として保存する。"""
    wav_path = Path(wav_path)
    y = load_audio(wav_path, sample_rate)
    mel_db = compute_mel_spectrogram(y, sample_rate)

    # 縦に2段の図を作る: 上 = 波形(時間と振幅)、下 = メルスペクトログラム(時間と周波数)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    times = np.arange(len(y)) / sample_rate  # 各サンプルの時刻[秒]
    axes[0].plot(times, y, linewidth=0.8)
    axes[0].set_title(f"波形: {wav_path.name}")
    axes[0].set_xlabel("時間 [秒]")
    axes[0].set_ylabel("振幅")

    img = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axes[1],
    )
    axes[1].set_title("メルスペクトログラム")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    fig.tight_layout()

    # 保存先の指定が無ければ、WAVと同じ場所・同じ名前で拡張子だけ .png にする
    if output_path is None:
        output_path = wav_path.with_suffix(".png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="モールス信号WAVの波形・メルスペクトログラムを可視化する")
    parser.add_argument("wav", help="入力WAVファイルパス")
    parser.add_argument("-o", "--output", default=None, help="出力PNGパス")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = plot_waveform_and_spectrogram(args.wav, args.output)
    print(f"保存しました: {path}")


if __name__ == "__main__":
    main()

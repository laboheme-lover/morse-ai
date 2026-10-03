"""ESC-50環境音をモールス信号音声に付加するモジュール。

指定したSNR[dB]でESC-50環境音をモールス信号音声に混合する。
train/validation/testでノイズ素材のfoldを分離するため、esc50_manager.Esc50Manager
を通じて対象split内のクリップのみを使用する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from esc50_manager import Esc50Clip, Esc50Manager

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def load_esc50_clip(clip: Esc50Clip, sample_rate: int) -> np.ndarray:
    """ESC-50クリップを読み込み、指定サンプリング周波数のモノラル波形に変換する。"""
    y, _ = librosa.load(str(clip.path), sr=sample_rate, mono=True)
    return y.astype(np.float32)


def _fit_noise_length(noise: np.ndarray, target_length: int, rng: np.random.Generator) -> np.ndarray:
    """ノイズ波形の長さをtarget_lengthに合わせる。

    ノイズがモールス音声より長い場合はランダムな位置から切り出し、
    短い場合はループさせてから切り出す。
    """
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(noise) == 0:
        return np.zeros(target_length, dtype=np.float32)
    # 環境音のほうが長い場合(1文字の音声は1秒程度、ESC-50は5秒):
    # ランダムな開始位置から必要な長さだけ切り出す。毎回同じ部分にならないようにするため。
    if len(noise) >= target_length:
        start = int(rng.integers(0, len(noise) - target_length + 1))
        return noise[start : start + target_length]
    # 環境音のほうが短い場合(数分ある電文の連続音声など):
    # 環境音を何回か繰り返してつなげ(np.tile)、そこからランダムな位置で切り出す。
    repeats = int(np.ceil(target_length / len(noise)))  # 必要な繰り返し回数(切り上げ)
    tiled = np.tile(noise, repeats)
    start = int(rng.integers(0, len(tiled) - target_length + 1))
    return tiled[start : start + target_length]


def mix_with_noise(
    waveform: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """波形にESC-50ノイズをSNR指定で混合する。

    信号電力はホワイトノイズ付加(generate_morse_audio.add_white_noise)と同様、
    トーン区間(振幅が0でないサンプル)の平均パワーを基準とする。
    """
    rng = rng if rng is not None else np.random.default_rng()
    # (1) 環境音の長さをモールス音声と同じにそろえる
    noise = _fit_noise_length(noise, len(waveform), rng)

    # (2) 信号の電力(音が鳴っている部分だけの二乗平均)と、環境音の電力を測る
    active = waveform[waveform != 0]
    signal_power = float(np.mean(active**2)) if active.size > 0 else 1e-12
    noise_power = float(np.mean(noise**2))

    # (3) 指定SNRになるのに必要な環境音の電力を求め、環境音の音量を何倍すればよいかを計算する。
    #     電力は振幅の2乗に比例するので、振幅の倍率は電力の比の平方根になる。
    snr_linear = 10 ** (snr_db / 10)
    target_noise_power = signal_power / snr_linear
    scale = np.sqrt(target_noise_power / noise_power) if noise_power > 0 else 0.0
    # (4) 音量を合わせた環境音を足し合わせる(無音区間にも環境音が乗る)
    return waveform + noise * scale


def mix_wav_with_esc50(
    morse_wav_path: str | Path,
    esc50_root: str | Path,
    split: str,
    snr_db: float,
    output_path: str | Path,
    seed: int | None = None,
) -> tuple[Path, dict]:
    """モールス信号WAVにESC-50環境音を混合してWAVを書き出す。

    戻り値は (出力パス, 生成条件metadata辞書) のタプル。
    """
    rng = np.random.default_rng(seed)
    # (1) 生成済みのモールス音声(ノイズなし)を読み込む
    waveform, sample_rate = sf.read(morse_wav_path)
    waveform = waveform.astype(np.float32)

    # (2) 音声と同じ split の fold から環境音を1つ選び、モールス音声と同じサンプリング周波数で読み込む
    #     (ESC-50は44100Hz、モールス音声は16000Hzなので変換が必要)
    manager = Esc50Manager(esc50_root)
    clip = manager.random_clip(split, rng)
    noise = load_esc50_clip(clip, sample_rate)

    # (3) 指定SNRで混ぜる
    mixed = mix_with_noise(waveform, noise, snr_db, rng)

    # ホワイトノイズ付加時と同様、ハードクリップではなく一様スケーリングで正規化する。
    peak = float(np.max(np.abs(mixed))) if mixed.size > 0 else 0.0
    if peak > 1.0:
        mixed = mixed / peak

    # (4) 保存する(generate_dataset.py からは入力と同じパスを渡して上書きしている)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, mixed.astype(np.float32), sample_rate, subtype="PCM_16")

    # (5) どの環境音を使ったかを記録する(後から「環境音の種類別の正解率」などを調べられる)
    metadata = {
        "noise_type": "esc50",
        "snr_db": snr_db,
        "esc50_filename": clip.filename,
        "esc50_category": clip.category,
        "esc50_fold": clip.fold,
        "esc50_src_file": clip.src_file,
    }
    return output_path, metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="モールス信号WAVにESC-50環境音を混合する")
    parser.add_argument("morse_wav", help="入力のモールス信号WAV")
    parser.add_argument("-o", "--output", required=True, help="出力WAVパス")
    parser.add_argument("--esc50-root", default="data/ESC-50-master", help="ESC-50データセットのルートディレクトリ")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train", help="使用するfoldのsplit")
    parser.add_argument("--snr", type=float, required=True, help="SNR[dB]")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path, metadata = mix_wav_with_esc50(
        args.morse_wav,
        args.esc50_root,
        args.split,
        args.snr,
        args.output,
        seed=args.seed,
    )
    print(f"生成しました: {path}")
    print(f"生成条件: {metadata}")


if __name__ == "__main__":
    main()

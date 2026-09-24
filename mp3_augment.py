"""MP3圧縮の往復(符号化→復号)による音声データ拡張。

実録音(モールス練習音源)はMP3(44.1kHz)で、圧縮由来の床ノイズ(無音区間でピーク比
約-41〜-43dB)や立ち上がり付近のなまり(プリエコー)を含む。合成データはこれらを
持たない完全な無音・急峻な立ち上がりのため、学習時にMP3往復を通した音を混ぜて
実録音の性質に近づける。

soundfile(libsndfile 1.2以降)の標準機能だけでMP3の書き出し・読み込みができ、
往復後も長さと先頭位置がずれない(追加パッケージ不要)ことを確認している。
"""

from __future__ import annotations

import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# MP3の符号化サンプリング周波数。実録音は44.1kHzだったので合わせる。
CODEC_SAMPLE_RATE = 44100

# CONSTANT(固定ビットレート)モードでのcompression_levelと、実測したおおよそのビットレート。
# 値が大きいほど高圧縮(低ビットレート)。実録音のビットレートは57〜233kbps(中央値108)。
COMPRESSION_LEVELS: tuple[float, ...] = (0.4, 0.6, 0.8, 0.95)  # 約192/160/96/48 kbps


def _resample(y: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return y
    divisor = gcd(from_rate, to_rate)
    return resample_poly(y, to_rate // divisor, from_rate // divisor).astype(np.float32)


def mp3_roundtrip(
    waveform: np.ndarray,
    sample_rate: int,
    compression_level: float = 0.8,
    codec_sample_rate: int = CODEC_SAMPLE_RATE,
) -> np.ndarray:
    """波形をMP3(固定ビットレート)で符号化・復号し、元と同じ長さ・サンプリング周波数で返す。

    codec_sample_rateへリサンプリングしてから符号化し、復号後に元のサンプリング周波数へ
    戻す。復号後の長さは元の長さに揃える(MP3の符号化遅延はlibsndfileが補正する)。
    """
    if waveform.size == 0:
        return waveform.astype(np.float32)

    upsampled = _resample(waveform.astype(np.float32), sample_rate, codec_sample_rate)
    buffer = io.BytesIO()
    sf.write(
        buffer,
        upsampled,
        codec_sample_rate,
        format="MP3",
        subtype="MPEG_LAYER_III",
        bitrate_mode="CONSTANT",
        compression_level=compression_level,
    )
    buffer.seek(0)
    decoded, decoded_rate = sf.read(buffer, dtype="float32")
    if decoded.ndim > 1:
        decoded = decoded.mean(axis=1)

    restored = _resample(decoded, decoded_rate, sample_rate)
    if len(restored) >= len(waveform):
        return restored[: len(waveform)].astype(np.float32)
    return np.pad(restored, (0, len(waveform) - len(restored))).astype(np.float32)

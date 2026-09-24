"""実録音を推論に適した形へ変換する前処理モジュール。

実データ検証(evaluate_real_samples.py)の結果、実録音のモールス信号は
合成学習データと比べて倍音成分が強く、基本波(トーン周波数)以外の
エネルギーが目立つことが判明した。この倍音成分や送信速度の違いがCNNの
誤認識(少数クラスへの偏り)の一因と考えられるため、推論前に音響特性を
測定した上で以下の正規化を行う。

- analyze_audio(): トーン周波数・短点長・文字間隔・概算打鍵速度(WPM)を測定する
- bandpass_filter(): 検出したトーン周波数付近だけを通すバンドパスフィルタで
  倍音・帯域外ノイズを除去する
- normalize_amplitude(): 振幅を学習データに近い水準(ピーク基準)へ正規化する
- time_stretch_to_target_dot(): 短点長を学習データの中心値へ合わせて
  時間伸縮する(ピッチ=トーン周波数は維持する)
- preprocess_for_inference(): 上記を組み合わせた前処理パイプライン
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

from morse_dataset import SAMPLE_RATE
from morse_segmentation import (
    DEFAULT_FRAME_MS,
    DEFAULT_THRESHOLD_RATIO,
    compute_frame_rms,
    detect_active_frames_hysteresis,
    frames_to_runs,
)

# generate_morse_audio.FREQUENCY_RANGE / SPEED_RANGES の中心付近に合わせる
DEFAULT_TARGET_FREQUENCY_HZ = 700.0
DEFAULT_TARGET_DOT_SECONDS = 0.09
DEFAULT_BANDPASS_HALFWIDTH_HZ = 150.0
DEFAULT_TARGET_PEAK_AMPLITUDE = 0.7

DEFAULT_DOT_PERCENTILE = 15.0
DEFAULT_GAP_PERCENTILE = 85.0
DEFAULT_DOT_FALLBACK_SECONDS = 0.08
DEFAULT_GAP_FALLBACK_SECONDS = 0.2


@dataclass
class AudioAnalysis:
    """音声から測定した音響特性。"""

    frequency_hz: float
    dot_duration_seconds: float
    char_gap_seconds: float
    wpm: float


DEFAULT_FREQUENCY_ESTIMATE_MAX_RUNS = 5


def estimate_center_frequency(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: float = DEFAULT_FRAME_MS,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
    fallback: float = DEFAULT_TARGET_FREQUENCY_HZ,
    max_runs: int = DEFAULT_FREQUENCY_ESTIMATE_MAX_RUNS,
) -> float:
    """有音区間のうち長い方から最大max_runs個を使い、FFTでトーンの基本周波数を推定する。

    振幅がしきい値を超えるサンプルだけを単純に間引いて連結すると、正弦波の
    山付近だけが抽出されて波形そのものが壊れ、周波数情報が失われてしまう
    (サンプル単位の閾値化は不適切)。そのためcompute_frame_rms/
    detect_active_framesと同じフレーム単位の判定で連続した有音区間を求め、
    その中で長い区間(時間的に連続な波形)をFFTにかける。

    最長1区間のみを使うと、その区間がたまたまESC-50環境音などの強いノイズに
    支配されていた場合に周波数推定そのものが誤り、以降のバンドパスフィルタが
    本来のトーンを除去してしまう(実データ検証でSNR-5dBのESC-50ノイズ付き
    音声で発生を確認)。複数の長い区間のパワースペクトルを合算することで、
    ノイズは区間ごとに支配的な周波数が変わりやすく合算後は薄まる一方、
    本来のトーン成分はどの区間でも同じ周波数に現れるため相対的に強調され、
    頑健性が上がる。
    """
    if y.size == 0:
        return fallback
    rms, frame_length = compute_frame_rms(y, sample_rate, frame_ms)
    active = detect_active_frames_hysteresis(rms, high_ratio=threshold_ratio, low_ratio=threshold_ratio * 0.5)
    runs = frames_to_runs(active, frame_length, len(y))
    on_runs = [(start, end) for start, end, is_on in runs if is_on]
    on_runs = [(start, end) for start, end in on_runs if end - start >= 32]
    if not on_runs:
        return fallback

    on_runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    selected = on_runs[:max_runs]
    common_length = min(end - start for start, end in selected)
    window = np.hanning(common_length)

    aggregate_spectrum: np.ndarray | None = None
    for start, end in selected:
        segment = y[start : start + common_length]
        spectrum = np.abs(np.fft.rfft(segment * window))
        aggregate_spectrum = spectrum if aggregate_spectrum is None else aggregate_spectrum + spectrum

    freqs = np.fft.rfftfreq(common_length, d=1.0 / sample_rate)
    # 直流・低周波成分(0-100Hz)を除外して基本波のピークを探す
    valid = freqs > 100.0
    if not np.any(valid):
        return fallback
    peak_index = int(np.argmax(aggregate_spectrum[valid]))
    return float(freqs[valid][peak_index])


def estimate_dot_seconds(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: float = DEFAULT_FRAME_MS,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
    percentile: float = DEFAULT_DOT_PERCENTILE,
    fallback: float = DEFAULT_DOT_FALLBACK_SECONDS,
) -> float:
    """音声中の短い「オン」区間の分布(下位percentile)から短点の長さを推定する。

    固定しきい値では送信速度の異なる実録音に対応できないため、ファイルごとに
    実際の短点長を推定する。
    """
    rms, frame_length = compute_frame_rms(y, sample_rate, frame_ms)
    active = detect_active_frames_hysteresis(rms, high_ratio=threshold_ratio, low_ratio=threshold_ratio * 0.5)
    runs = frames_to_runs(active, frame_length, len(y))
    on_durations = [(end - start) / sample_rate for start, end, is_on in runs if is_on]
    if not on_durations:
        return fallback
    return float(np.percentile(on_durations, percentile))


def estimate_char_gap_seconds(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: float = DEFAULT_FRAME_MS,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
    percentile: float = DEFAULT_GAP_PERCENTILE,
    fallback: float = DEFAULT_GAP_FALLBACK_SECONDS,
) -> float:
    """無音(オフ)区間の分布(上位寄りpercentile)から文字境界の間隔を推定する。"""
    rms, frame_length = compute_frame_rms(y, sample_rate, frame_ms)
    active = detect_active_frames_hysteresis(rms, high_ratio=threshold_ratio, low_ratio=threshold_ratio * 0.5)
    runs = frames_to_runs(active, frame_length, len(y))
    off_durations = [(end - start) / sample_rate for start, end, is_on in runs if not is_on]
    if not off_durations:
        return fallback
    return float(np.percentile(off_durations, percentile))


def analyze_audio(y: np.ndarray, sample_rate: int = SAMPLE_RATE) -> AudioAnalysis:
    """音声からトーン周波数・短点長・文字間隔・概算打鍵速度(WPM)を測定する。"""
    frequency = estimate_center_frequency(y, sample_rate)
    dot_seconds = estimate_dot_seconds(y, sample_rate)
    char_gap = estimate_char_gap_seconds(y, sample_rate)
    # PARIS基準の概算(1分あたり打てる標準的な語数)。短点1単位の長さから概算する。
    wpm = 1.2 / dot_seconds if dot_seconds > 0 else 0.0
    return AudioAnalysis(
        frequency_hz=frequency, dot_duration_seconds=dot_seconds, char_gap_seconds=char_gap, wpm=wpm
    )


def bandpass_filter(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    center_freq_hz: float = DEFAULT_TARGET_FREQUENCY_HZ,
    halfwidth_hz: float = DEFAULT_BANDPASS_HALFWIDTH_HZ,
    order: int = 4,
) -> np.ndarray:
    """検出したトーン周波数付近だけを通すバンドパスフィルタ(倍音・帯域外ノイズを抑える)。"""
    if y.size == 0:
        return y.astype(np.float32)
    nyquist = sample_rate / 2.0
    low = max(1.0, center_freq_hz - halfwidth_hz) / nyquist
    high = min(nyquist - 1.0, center_freq_hz + halfwidth_hz) / nyquist
    if not (0 < low < high < 1):
        return y.astype(np.float32)
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, y)
    return filtered.astype(np.float32)


def normalize_amplitude(y: np.ndarray, target_peak: float = DEFAULT_TARGET_PEAK_AMPLITUDE) -> np.ndarray:
    """振幅を学習データに近い水準(ピーク基準)へ正規化する。"""
    if y.size == 0:
        return y.astype(np.float32)
    peak = float(np.max(np.abs(y)))
    if peak <= 1e-9:
        return y.astype(np.float32)
    return (y * (target_peak / peak)).astype(np.float32)


def time_stretch_to_target_dot(
    y: np.ndarray,
    sample_rate: int,
    current_dot_seconds: float,
    target_dot_seconds: float = DEFAULT_TARGET_DOT_SECONDS,
) -> np.ndarray:
    """短点長を学習データの中心値へ合わせて時間伸縮する(トーン周波数は維持する)。

    librosa.effects.time_stretchのrateは1より大きいと速く(短く)、
    1より小さいと遅く(長く)なる。極端な伸縮を避けるため[0.25, 4.0]に丸める。

    注意: 全体の再生時間は狙い通りに変化するが、位相ボコーダ方式のため
    モールス信号特有の急峻なオン/オフ切り替え(キー打鍵)付近で細かい
    アーティファクト(偽の短いブリップ)が生じることを確認している。
    そのためpreprocess_for_inference()ではデフォルトで無効(apply_time_stretch=False)
    としており、全体的な速度感を粗く揃えたい用途向けの補助関数として提供する。
    """
    if y.size == 0 or current_dot_seconds <= 0:
        return y.astype(np.float32)
    rate = float(np.clip(current_dot_seconds / target_dot_seconds, 0.25, 4.0))
    stretched = librosa.effects.time_stretch(y.astype(np.float32), rate=rate)
    return stretched.astype(np.float32)


def preprocess_for_inference(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target_frequency_hz: float = DEFAULT_TARGET_FREQUENCY_HZ,
    target_dot_seconds: float = DEFAULT_TARGET_DOT_SECONDS,
    bandpass_halfwidth_hz: float = DEFAULT_BANDPASS_HALFWIDTH_HZ,
    apply_bandpass: bool = True,
    apply_time_stretch: bool = False,
    apply_amplitude_normalize: bool = True,
) -> tuple[np.ndarray, AudioAnalysis]:
    """実録音を推論に適した形へ変換する一連の前処理。

    音響特性(周波数・短点長)を測定した上で、(1)トーン周波数帯のみを通す
    バンドパスフィルタで倍音・帯域外ノイズを抑え、(2)振幅を正規化する。
    短点長の時間伸縮(time_stretch_to_target_dot)は位相ボコーダのアーティファクトの
    影響が大きいためデフォルトでは無効。必要な場合のみapply_time_stretch=Trueで
    有効化できる。戻り値は (前処理後の波形, 測定した音響特性) のタプル。
    """
    analysis = analyze_audio(y, sample_rate)
    processed = y.astype(np.float32)

    if apply_bandpass:
        processed = bandpass_filter(processed, sample_rate, analysis.frequency_hz, bandpass_halfwidth_hz)
    if apply_time_stretch:
        processed = time_stretch_to_target_dot(
            processed, sample_rate, analysis.dot_duration_seconds, target_dot_seconds
        )
    if apply_amplitude_normalize:
        processed = normalize_amplitude(processed)

    return processed, analysis

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
    # (1) 音が鳴っている区間を探す(区間分割と同じ判定方法)。短すぎる区間(32サンプル=2ms未満)は除く。
    rms, frame_length = compute_frame_rms(y, sample_rate, frame_ms)
    active = detect_active_frames_hysteresis(rms, high_ratio=threshold_ratio, low_ratio=threshold_ratio * 0.5)
    runs = frames_to_runs(active, frame_length, len(y))
    on_runs = [(start, end) for start, end, is_on in runs if is_on]
    on_runs = [(start, end) for start, end in on_runs if end - start >= 32]
    if not on_runs:
        return fallback  # 音が見つからなければ既定値(700Hz)

    # (2) 長い順に最大5区間を選ぶ(長いほど周波数を細かく測れる)。
    #     スペクトルを足し合わせるには周波数の目盛りをそろえる必要があるので、一番短い区間の長さに切りそろえる。
    on_runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    selected = on_runs[:max_runs]
    common_length = min(end - start for start, end in selected)
    # ハニング窓: 区間の両端をなだらかに0へ近づける重み。区間を途中で切ったことによる
    # 偽の周波数成分(漏れ)を抑え、ピークを鋭くする。
    window = np.hanning(common_length)

    # (3) 各区間をFFT(フーリエ変換)して「どの周波数がどれだけ含まれるか」を求め、足し合わせる
    aggregate_spectrum: np.ndarray | None = None
    for start, end in selected:
        segment = y[start : start + common_length]
        spectrum = np.abs(np.fft.rfft(segment * window))  # 各周波数の強さ
        aggregate_spectrum = spectrum if aggregate_spectrum is None else aggregate_spectrum + spectrum

    # (4) 一番強い周波数 = トーンの周波数
    freqs = np.fft.rfftfreq(common_length, d=1.0 / sample_rate)  # 各要素が何Hzに当たるか
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
    # 音が鳴っている区間の長さを全部集める。中身は短点(1単位)と長点(3単位)が混ざっているので、
    # 短いほうから15%の位置の値(15パーセンタイル)を短点の長さとみなす。
    # 最小値を使わないのは、雑音による極端に短い区間に引きずられないようにするため。
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
    # 無音の区間の長さを集め、長いほうから15%の位置(85パーセンタイル)を文字の区切りの間隔とみなす。
    # 無音は「符号内の短い間隔」が大半で、「文字の区切りの長い間隔」はそれより少ないため。
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
    # 通す範囲 = トーン周波数 ± 150Hz(例 600Hz なら 450〜750Hz)。それ以外の周波数の音を弱める。
    # ホワイトノイズは全周波数に広がっているので、この範囲だけ通すと雑音の大部分を取り除ける。
    # 周波数は「ナイキスト周波数(サンプリング周波数の半分=8000Hz)に対する比率」で指定する決まり。
    nyquist = sample_rate / 2.0
    low = max(1.0, center_freq_hz - halfwidth_hz) / nyquist
    high = min(nyquist - 1.0, center_freq_hz + halfwidth_hz) / nyquist
    if not (0 < low < high < 1):
        return y.astype(np.float32)  # 範囲が不正なら何もしない
    # 4次のバターワース型フィルタ(通す範囲の中はできるだけ平らに通す型)を設計する。
    # sos 形式 = 2次のフィルタを直列につないだ表現。高次でも計算誤差が出にくい。
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    # sosfiltfilt: 前から1回・後ろから1回かけて、フィルタによる時間のずれを打ち消す。
    # 音の始まり・終わりの位置がずれると区間分割の位置もずれるため。
    filtered = sosfiltfilt(sos, y)
    return filtered.astype(np.float32)


def normalize_amplitude(y: np.ndarray, target_peak: float = DEFAULT_TARGET_PEAK_AMPLITUDE) -> np.ndarray:
    """振幅を学習データに近い水準(ピーク基準)へ正規化する。"""
    if y.size == 0:
        return y.astype(np.float32)
    # 一番大きい振幅が target_peak(0.7)になるよう、全体を同じ倍率で拡大・縮小する
    peak = float(np.max(np.abs(y)))
    if peak <= 1e-9:
        return y.astype(np.float32)  # ほぼ無音なら何もしない(0で割らないため)
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
    # (1) まず元の音声から、トーン周波数・短点の長さなどを測る
    analysis = analyze_audio(y, sample_rate)
    processed = y.astype(np.float32)

    # (2) 測ったトーン周波数の付近だけを通す(雑音を減らす)。
    #     雑音の多い音声では区間分割が成功するかどうかを左右するので、アプリでは既定でON。
    if apply_bandpass:
        processed = bandpass_filter(processed, sample_rate, analysis.frequency_hz, bandpass_halfwidth_hz)
    if apply_time_stretch:
        processed = time_stretch_to_target_dot(
            processed, sample_rate, analysis.dot_duration_seconds, target_dot_seconds
        )
    if apply_amplitude_normalize:
        processed = normalize_amplitude(processed)

    return processed, analysis

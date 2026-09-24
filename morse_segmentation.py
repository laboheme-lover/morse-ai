"""連続モールス信号音声を文字単位に分割するモジュール(Milestone 9)。

音声のON/OFF(有音/無音)区間をフレーム単位のRMSエネルギーで検出し、
一定時間(固定しきい値)以上続く無音を文字境界とみなして1文字ずつに分割する。

ミニマム版のため、文字間隔の判定は固定しきい値によるルールベースとする
(ドット長を推定して動的にしきい値を決める改善は将来課題とする)。
強いノイズ条件(0dB, -5dB等)では無音区間にもノイズが乗るため、分割精度が
下がることがある。
"""

from __future__ import annotations

import numpy as np

from morse_dataset import SAMPLE_RATE

DEFAULT_FRAME_MS = 5.0
DEFAULT_THRESHOLD_RATIO = 0.25
DEFAULT_MIN_CHAR_GAP_SECONDS = 0.25
# ノイズによる瞬間的な誤判定(スパイク)を除去するデスパイク処理の既定しきい値。
# 5ms/フレームで3フレーム=15msは、最速のfast速度の短点(0.05秒=10フレーム)より
# 明確に短く、正当な短点を誤って消さない安全マージンを持つ。
DEFAULT_DESPIKE_MIN_RUN_FRAMES = 3
# 切り出し区間の前後に足す余白。学習データ(1文字=1WAV)は先頭・末尾に無音が無いため、
# 余白を付けると学習時と入力の形が変わり、認識精度が大きく低下する(同一の300サンプルで
# 単体ファイル78.3%・余白0.02秒22.0%・余白なし78.0%と計測)。既定は余白なしとする。
DEFAULT_PADDING_SECONDS = 0.0


def compute_frame_rms(
    y: np.ndarray, sample_rate: int = SAMPLE_RATE, frame_ms: float = DEFAULT_FRAME_MS
) -> tuple[np.ndarray, int]:
    """フレーム単位のRMSエネルギー系列を計算する。戻り値は (RMS系列, フレーム長[サンプル])。"""
    frame_length = max(int(round(sample_rate * frame_ms / 1000)), 1)
    n_frames = int(np.ceil(len(y) / frame_length)) if len(y) > 0 else 0
    rms = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        chunk = y[i * frame_length : (i + 1) * frame_length].astype(np.float64)
        rms[i] = np.sqrt(np.mean(chunk**2)) if chunk.size > 0 else 0.0
    return rms, frame_length


def detect_active_frames(rms: np.ndarray, threshold_ratio: float = DEFAULT_THRESHOLD_RATIO) -> np.ndarray:
    """RMS系列から有音フレームの真偽配列を求める。

    しきい値は固定値ではなく、そのクリップ全体の最大RMSに対する比率
    (threshold_ratio)として適応的に決める。これにより振幅(音量)が
    サンプルごとに異なっていても同じ比率で扱える。
    """
    peak = float(rms.max()) if rms.size > 0 else 0.0
    threshold = peak * threshold_ratio
    return rms > threshold


def _despike_active_frames(active: np.ndarray, min_run_frames: int) -> np.ndarray:
    """min_run_frames未満の孤立した短い区間を直前の区間へ統合して除去する。

    ノイズを音声全体(符号間の無音部分も含む)に付加した長時間音声で、
    無音中にRMSが瞬間的にしきい値を超える(あるいは逆に有音中に瞬間的に
    下回る)スパイクが多数発生し、estimate_dot_seconds()の推定値が
    数msまで下がって以降の区間分割が過分割になることが実データ検証で
    判明した。フレーム単位で先頭から走査し、min_run_frames未満の区間を
    直前の状態で塗りつぶす(1パスのみ。連鎖的な統合も同じパス内で反映される)。
    先頭(直前が無い)区間はそのまま残す。
    """
    if min_run_frames <= 1 or active.size == 0:
        return active
    result = active.copy()
    n = len(result)
    i = 0
    while i < n:
        j = i
        while j < n and result[j] == result[i]:
            j += 1
        run_len = j - i
        if run_len < min_run_frames and i > 0:
            result[i:j] = result[i - 1]
        i = j
    return result


def detect_active_frames_hysteresis(
    rms: np.ndarray,
    high_ratio: float = DEFAULT_THRESHOLD_RATIO,
    low_ratio: float = DEFAULT_THRESHOLD_RATIO * 0.5,
    despike_min_run_frames: int = DEFAULT_DESPIKE_MIN_RUN_FRAMES,
) -> np.ndarray:
    """2段階のしきい値(ヒステリシス)で有音フレームを判定する。

    単一しきい値(detect_active_frames)だと、トーン中の瞬間的なノイズや
    フェージングによる振幅の落ち込みが誤って無音と判定され、本来1つの
    符号(短点・長点)が2つに分割されてしまう(過分割)ことがある。
    実データ検証で、認識文字数が正解より1〜2割ほど多くなる傾向が確認された
    原因の一つと考えられる。

    ヒステリシスでは「オンになるには高い方のしきい値(high_ratio)を超える
    必要があるが、いったんオンになったら低い方のしきい値(low_ratio)を
    下回るまではオンを維持する」という粘りを持たせることで、瞬間的な
    落ち込みによる過分割を抑える。

    その上でdespike_min_run_frames(既定3フレーム=15ms)未満の孤立した
    短い区間を_despike_active_framesで除去する。1(以下)を指定すると無効化できる。
    """
    peak = float(rms.max()) if rms.size > 0 else 0.0
    high = peak * high_ratio
    low = peak * low_ratio
    active = np.zeros(rms.shape, dtype=bool)
    state = False
    for i, value in enumerate(rms):
        state = (value > low) if state else (value > high)
        active[i] = state
    return _despike_active_frames(active, despike_min_run_frames)


def frames_to_runs(is_active: np.ndarray, frame_length: int, total_samples: int) -> list[tuple[int, int, bool]]:
    """フレーム単位の有音/無音フラグを、連続するサンプル区間(開始,終了,有音か)のリストへまとめる。"""
    if is_active.size == 0:
        return []
    runs: list[tuple[int, int, bool]] = []
    start_frame = 0
    current = bool(is_active[0])
    for i in range(1, len(is_active)):
        if bool(is_active[i]) != current:
            runs.append((start_frame * frame_length, i * frame_length, current))
            start_frame = i
            current = bool(is_active[i])
    runs.append((start_frame * frame_length, total_samples, current))
    return runs


def split_into_characters(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    min_char_gap_seconds: float = DEFAULT_MIN_CHAR_GAP_SECONDS,
    frame_ms: float = DEFAULT_FRAME_MS,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
    padding_seconds: float = DEFAULT_PADDING_SECONDS,
    use_hysteresis: bool = True,
    hysteresis_low_ratio: float | None = None,
) -> list[tuple[int, int]]:
    """連続音声を、min_char_gap_seconds以上の無音を文字境界として1文字ずつの
    (開始サンプル, 終了サンプル)区間リストへ分割する。

    符号内の短い間隔(短点・長点の間隔)はmin_char_gap_seconds未満なので、
    同一文字として連結されたまま残る。

    use_hysteresis=True(デフォルト)の場合、detect_active_frames_hysteresisで
    有音判定を行い、トーン中の瞬間的な落ち込みによる過分割を抑える。
    Falseにすると単一しきい値のdetect_active_framesを使う(従来の挙動)。
    """
    rms, frame_length = compute_frame_rms(y, sample_rate, frame_ms)
    if use_hysteresis:
        low_ratio = hysteresis_low_ratio if hysteresis_low_ratio is not None else threshold_ratio * 0.5
        is_active = detect_active_frames_hysteresis(rms, high_ratio=threshold_ratio, low_ratio=low_ratio)
    else:
        is_active = detect_active_frames(rms, threshold_ratio)
    runs = frames_to_runs(is_active, frame_length, len(y))

    min_gap_samples = int(round(min_char_gap_seconds * sample_rate))
    padding_samples = int(round(padding_seconds * sample_rate))

    char_ranges: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    for start, end, is_on in runs:
        if is_on:
            if current_start is None:
                current_start = start
            current_end = end
        elif current_start is not None and (end - start) >= min_gap_samples:
            char_ranges.append((current_start, current_end))
            current_start = None
            current_end = None
        # 短い無音(符号内の間隔)はそのまま同じ文字として継続する
    if current_start is not None:
        char_ranges.append((current_start, current_end))

    return [
        (max(0, start - padding_samples), min(len(y), end + padding_samples)) for start, end in char_ranges
    ]


def segment_audio(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    min_char_gap_seconds: float = DEFAULT_MIN_CHAR_GAP_SECONDS,
    frame_ms: float = DEFAULT_FRAME_MS,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
    padding_seconds: float = DEFAULT_PADDING_SECONDS,
    use_hysteresis: bool = True,
    hysteresis_low_ratio: float | None = None,
) -> list[np.ndarray]:
    """連続音声を1文字ずつの波形チャンクのリストへ分割する。"""
    ranges = split_into_characters(
        y,
        sample_rate,
        min_char_gap_seconds,
        frame_ms,
        threshold_ratio,
        padding_seconds,
        use_hysteresis,
        hysteresis_low_ratio,
    )
    return [y[start:end] for start, end in ranges]

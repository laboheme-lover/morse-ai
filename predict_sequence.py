"""連続モールス信号音声から複数文字を認識するモジュール(Milestone 9)。

音声を文字ごとに分割(morse_segmentation.py)し、各区間をCNNで推論
(predict.predict_waveform)して結果を連結し、日本語文字列として返す。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import librosa
import numpy as np
import torch

from morse_cnn import MorseCNN
from morse_dataset import SAMPLE_RATE
from morse_segmentation import (
    DEFAULT_FRAME_MS,
    DEFAULT_MIN_CHAR_GAP_SECONDS,
    DEFAULT_PADDING_SECONDS,
    DEFAULT_THRESHOLD_RATIO,
    split_into_characters,
)
from morse_table import compose_pair
from predict import DEFAULT_MODELS_DIR, DEFAULT_TOP_K, PredictionResult, load_model, predict_waveform

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# 符号が要素1つだけの文字。ヘ = 短点1つ(1単位)、ム = 長点1つ(3単位)。
# 1文字だけを見るモデルには「その音声の速度」が分からないため、速い送信の長点(ム)と、
# 遅い送信の短点(ヘ)が同じ長さになると区別できない(audio2の実録音でム→ヘの取り違えが22回発生)。
# そこで、音声全体から推定した短点の長さを基準に、音の長さで判定し直す。
SINGLE_DOT_CHAR = "ヘ"
SINGLE_DASH_CHAR = "ム"
# 短点(1単位)と長点(3単位)のちょうど中間。実測でもヘは1.0倍前後、ムは3.0倍前後に分かれた。
SINGLE_ELEMENT_RATIO_THRESHOLD = 2.0
# 短点の推定値がこの範囲外なら(雑音で推定が崩れた等)、判定し直さずモデルの結果をそのまま使う。
# 下限は学習データの最も短い短点(0.05秒、揺らぎ-10%で0.045秒)。audio2の加工音源では、雑音のない音源の
# 推定短点長は45〜150msに収まり、45ms未満の推定はSN比0dB以下などの雑音で推定が崩れた音源にだけ現れた
# (例: 実際は短点115ms以上の slow の音源で30msと推定)。上限は学習データの最長(約0.19秒)に余裕を持たせた値。
PLAUSIBLE_DOT_SECONDS = (0.045, 0.25)


@dataclass
class CharacterPrediction:
    """1文字分の区間情報と推論結果。"""

    start_sample: int
    end_sample: int
    sample_rate: int
    result: PredictionResult
    # 音の長さに基づいて ヘ⇔ム を判定し直した場合 True(correct_single_element_by_duration)
    duration_corrected: bool = False

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate


@dataclass
class SequencePrediction:
    """連続音声全体の認識結果。"""

    text: str
    characters: list[CharacterPrediction]


@dataclass
class DisplayUnit:
    """画面表示用の1文字分(濁音・半濁音は基本仮名+濁点/半濁点の2区間を1単位に結合する)。"""

    text: str
    confidence: float  # 構成する区間の確信度のうち最小のもの
    segment_indices: list[int]
    needs_review: bool


def build_display_units(sequence: SequencePrediction, low_confidence_threshold: float) -> list[DisplayUnit]:
    """認識結果を画面表示用の単位へ変換する。

    CNNは基本仮名と濁点/半濁点を別クラスとして認識するため、モールス符号上は
    2区間に分かれている。morse_table.compose_dakuten()と同じ規則で(基本仮名, 濁点/
    半濁点)の並びを1単位(ガ・プ等)へ結合し、確信度は2区間のうち低い方を採用する。
    確信度がlow_confidence_threshold未満の単位は要確認(needs_review)とする。
    """
    units: list[DisplayUnit] = []
    chars = sequence.characters
    i = 0
    # morse_table.compose_dakuten() と同じ「今と次の2区間が (基本仮名, 濁点) なら1文字に結合」の考え方。
    # 文字列だけでなく、各区間の位置(音声の何秒目か)と確信度も対応付けたまま結合するため、別に実装している。
    while i < len(chars):
        current = chars[i].result
        if i + 1 < len(chars):
            following = chars[i + 1].result
            composed = compose_pair(current.prediction, following.prediction)
            if composed is not None:
                # 「ガ」の確信度は「カ」と「゛」のうち低いほう(どちらかが怪しければ全体も怪しい)
                confidence = min(current.confidence, following.confidence)
                units.append(DisplayUnit(composed, confidence, [i, i + 1], confidence < low_confidence_threshold))
                i += 2
                continue
        units.append(
            DisplayUnit(current.prediction, current.confidence, [i], current.confidence < low_confidence_threshold)
        )
        i += 1
    return units


def correct_single_element_by_duration(sequence: SequencePrediction, dot_seconds: float | None) -> SequencePrediction:
    """ヘ/ムと認識された区間を、音の長さと推定短点長の比で判定し直す。

    区間分割は音の始まりと終わりに密着して切り出すため、要素1つの文字では区間の長さ ≒ 音の長さになる。
    長さが短点の2倍以上ならム(長点)、未満ならヘ(短点)とする。モデルが ヘ/ム 以外と認識した区間は変えない。
    確信度はモデルの値をそのまま残し、判定し直したことは duration_corrected で分かるようにする。
    """
    low, high = PLAUSIBLE_DOT_SECONDS
    if dot_seconds is None or not (low <= dot_seconds <= high):
        return sequence

    corrected: list[CharacterPrediction] = []
    for char in sequence.characters:
        label = char.result.prediction
        if label in (SINGLE_DOT_CHAR, SINGLE_DASH_CHAR):
            duration = (char.end_sample - char.start_sample) / char.sample_rate
            by_length = SINGLE_DASH_CHAR if duration >= dot_seconds * SINGLE_ELEMENT_RATIO_THRESHOLD else SINGLE_DOT_CHAR
            if by_length != label:
                char = replace(char, result=replace(char.result, prediction=by_length), duration_corrected=True)
        corrected.append(char)
    text = "".join(c.result.prediction for c in corrected)
    return SequencePrediction(text=text, characters=corrected)


def predict_sequence_from_waveform(
    y: np.ndarray,
    sr: int,
    model: MorseCNN,
    index_to_label: dict[int, str],
    fixed_frames: int,
    top_k: int = DEFAULT_TOP_K,
    min_char_gap_seconds: float = DEFAULT_MIN_CHAR_GAP_SECONDS,
    frame_ms: float = DEFAULT_FRAME_MS,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
    padding_seconds: float = DEFAULT_PADDING_SECONDS,
    device: str = "cpu",
    dot_seconds: float | None = None,
) -> SequencePrediction:
    """読み込み済みの波形を文字ごとに分割し、各文字を推論して連結する。

    モデルを読み込み済みの状態で繰り返し推論できるため、Streamlit UIのように
    しきい値等を変えながら何度も認識する用途に向く。
    dot_seconds(この音声の推定短点長)を渡すと、ヘ/ムを音の長さで判定し直す。
    """
    # 連続音声認識の2段構え:
    # (1) 区間分割: 無音を手掛かりに音声を1文字ずつの区間 [(開始, 終了), ...] に区切る
    ranges = split_into_characters(y, sr, min_char_gap_seconds, frame_ms, threshold_ratio, padding_seconds)

    # (2) 文字判定: 各区間を切り出してモデルで1文字ずつ推論する。
    #     学習データは「1文字=1つの音声」なので、同じ形に切り出してから渡す。
    characters: list[CharacterPrediction] = []
    for start, end in ranges:
        chunk = y[start:end]
        result = predict_waveform(chunk, model, index_to_label, fixed_frames, sr, device, top_k)
        characters.append(CharacterPrediction(start, end, sr, result))

    # 予測した文字をつなげて文字列にする(この時点では濁音は「カ゛」のまま。表示時に結合する)
    text = "".join(c.result.prediction for c in characters)
    # (3) 要素1つの文字(ヘ/ム)は、音声全体の速度を基準に音の長さで判定し直す
    return correct_single_element_by_duration(SequencePrediction(text=text, characters=characters), dot_seconds)


def predict_sequence(
    wav_path: str | Path,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    top_k: int = DEFAULT_TOP_K,
    min_char_gap_seconds: float = DEFAULT_MIN_CHAR_GAP_SECONDS,
    frame_ms: float = DEFAULT_FRAME_MS,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
    padding_seconds: float = DEFAULT_PADDING_SECONDS,
    device: str | None = None,
    dot_seconds: float | None = None,
) -> SequencePrediction:
    """連続音声WAVを文字ごとに分割し、各文字を推論して連結する。

    文字間の無音がmin_char_gap_seconds未満だと2文字が1文字として結合されて
    しまい、逆に短すぎる値だと1文字が誤って複数に分割されてしまう。
    このしきい値はStreamlit UI(Milestone 10)でユーザーが調整できるようにする。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _label_to_index, index_to_label, fixed_frames = load_model(models_dir, device)

    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    return predict_sequence_from_waveform(
        y, sr, model, index_to_label, fixed_frames, top_k,
        min_char_gap_seconds, frame_ms, threshold_ratio, padding_seconds, device, dot_seconds,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="連続モールス信号WAVから複数文字を認識する")
    parser.add_argument("wav", help="入力WAVファイル")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--min-char-gap", type=float, default=DEFAULT_MIN_CHAR_GAP_SECONDS, help="文字境界とみなす無音の最小長[秒]"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = predict_sequence(args.wav, args.models_dir, args.top_k, min_char_gap_seconds=args.min_char_gap)
    print(f"認識結果: {result.text}")
    for i, c in enumerate(result.characters, start=1):
        print(
            f"  {i}. {c.result.prediction} (確信度={c.result.confidence:.3f}, "
            f"区間={c.start_seconds:.2f}-{c.end_seconds:.2f}s)"
        )


if __name__ == "__main__":
    main()

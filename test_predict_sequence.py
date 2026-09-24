"""predict_sequence.py のユニットテスト(Milestone 9)。

Milestone 8のテストと同様、周波数の異なる3クラス(イ/ロ/1)で学習した
容易な分類モデルを用いる。連続音声のテスト用に、各文字ごとに個別の周波数を
指定した上で文字間隔(無音)を挟んで手動で連結した音声を作成する
(sequence_to_waveformは1つの周波数しか指定できないため、複数クラスを
確実に識別できるテスト用音声には向かないため)。
"""

import csv

import numpy as np
import pytest
import soundfile as sf

from generate_dataset import METADATA_FIELDS
from generate_morse_audio import DEFAULT_AMPLITUDE, SAMPLE_RATE, _silence, char_to_waveform, generate_morse_wav
from morse_segmentation import DEFAULT_MIN_CHAR_GAP_SECONDS
from predict_sequence import predict_sequence
from train import train

LABELS = [("イ", 700.0), ("ロ", 750.0), ("1", 650.0)]
FREQ_BY_LABEL = dict(LABELS)

# 文字境界の判定しきい値(デフォルト0.25秒)より明確に長い間隔を使う
CHARACTER_GAP_SECONDS = 0.4


def _make_easy_metadata(tmp_path, samples_per_split=15):
    rows = []
    seed = 0
    for label, freq in LABELS:
        for split in ("train", "validation", "test"):
            for i in range(samples_per_split):
                wav_path = tmp_path / "wav" / split / label / f"{label}_{i}.wav"
                _, meta = generate_morse_wav(
                    label, wav_path, speed="normal", frequency_hz=freq, amplitude=0.6, seed=seed
                )
                seed += 1
                row = {field: meta.get(field, "") for field in METADATA_FIELDS}
                row["split"] = split
                for esc50_field in ("esc50_filename", "esc50_category", "esc50_fold", "esc50_src_file"):
                    row.setdefault(esc50_field, "")
                rows.append(row)

    metadata_csv = tmp_path / "metadata.csv"
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return metadata_csv


@pytest.fixture(scope="module")
def trained_models_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("predict_sequence_easy")
    metadata_csv = _make_easy_metadata(tmp_path)
    models_dir = tmp_path / "models"
    train(metadata_csv=metadata_csv, models_dir=models_dir, epochs=8, batch_size=8, seed=42)
    return models_dir


def _build_sequence_wav(tmp_path, labels, dot_duration=0.1, amplitude=DEFAULT_AMPLITUDE, filename="sequence.wav"):
    segments = []
    for i, label in enumerate(labels):
        if i > 0:
            segments.append(_silence(CHARACTER_GAP_SECONDS, SAMPLE_RATE))
        segments.append(
            char_to_waveform(
                label, dot_duration=dot_duration, frequency_hz=FREQ_BY_LABEL[label], amplitude=amplitude
            )
        )
    waveform = np.concatenate(segments)
    wav_path = tmp_path / filename
    sf.write(wav_path, waveform, SAMPLE_RATE, subtype="PCM_16")
    return wav_path


def test_predict_sequence_splits_into_correct_number_of_characters(tmp_path, trained_models_dir):
    labels = ["イ", "ロ", "1"]
    wav_path = _build_sequence_wav(tmp_path, labels)

    result = predict_sequence(wav_path, models_dir=trained_models_dir, device="cpu")
    assert len(result.characters) == len(labels)


def test_predict_sequence_recognizes_correct_text(tmp_path, trained_models_dir):
    """周波数が大きく異なる簡単なタスクなので、連続音声でも正しく連結されるはず。"""
    labels = ["イ", "ロ", "1", "イ"]
    wav_path = _build_sequence_wav(tmp_path, labels)

    result = predict_sequence(wav_path, models_dir=trained_models_dir, device="cpu")
    assert result.text == "".join(labels)


def test_predict_sequence_character_timings_are_increasing_and_within_bounds(tmp_path, trained_models_dir):
    labels = ["ロ", "1"]
    wav_path = _build_sequence_wav(tmp_path, labels)

    result = predict_sequence(wav_path, models_dir=trained_models_dir, device="cpu")
    assert len(result.characters) == 2

    first, second = result.characters
    assert 0 <= first.start_seconds < first.end_seconds
    assert first.end_seconds <= second.start_seconds
    assert second.start_seconds < second.end_seconds


def test_predict_sequence_each_character_has_top_k_candidates(tmp_path, trained_models_dir):
    labels = ["イ", "1"]
    wav_path = _build_sequence_wav(tmp_path, labels)

    result = predict_sequence(wav_path, models_dir=trained_models_dir, top_k=2, device="cpu")
    for char_pred in result.characters:
        assert len(char_pred.result.top_k) == 2


def test_predict_sequence_too_short_gap_merges_characters(tmp_path, trained_models_dir):
    """文字境界しきい値より短い無音しか無いと、2文字が1文字として結合されてしまう
    (仕様書が指摘する固定しきい値方式の既知の限界)。"""
    dot_duration = 0.1
    segments = [
        char_to_waveform("イ", dot_duration=dot_duration, frequency_hz=FREQ_BY_LABEL["イ"], amplitude=0.6),
        _silence(0.05, SAMPLE_RATE),  # 文字境界しきい値(0.25s)よりずっと短い
        char_to_waveform("ロ", dot_duration=dot_duration, frequency_hz=FREQ_BY_LABEL["ロ"], amplitude=0.6),
    ]
    waveform = np.concatenate(segments)
    wav_path = tmp_path / "too_short_gap.wav"
    sf.write(wav_path, waveform, SAMPLE_RATE, subtype="PCM_16")

    result = predict_sequence(wav_path, models_dir=trained_models_dir, device="cpu")
    assert len(result.characters) == 1


def test_predict_sequence_empty_audio_returns_empty_text(tmp_path, trained_models_dir):
    wav_path = tmp_path / "silence_only.wav"
    sf.write(wav_path, _silence(1.0, SAMPLE_RATE), SAMPLE_RATE, subtype="PCM_16")

    result = predict_sequence(wav_path, models_dir=trained_models_dir, device="cpu")
    assert result.text == ""
    assert result.characters == []


# --- Milestone 10: UI向けの追加API --------------------------------------


def _fake_character(prediction: str, confidence: float, start: int = 0, end: int = 100):
    from predict import Candidate, PredictionResult
    from predict_sequence import CharacterPrediction

    result = PredictionResult(
        prediction=prediction,
        confidence=confidence,
        top_k=[Candidate(prediction, confidence)],
        mel_spectrogram=np.zeros((64, 4), dtype=np.float32),
    )
    return CharacterPrediction(start, end, SAMPLE_RATE, result)


def _fake_sequence(pairs):
    from predict_sequence import SequencePrediction

    characters = [_fake_character(p, c) for p, c in pairs]
    return SequencePrediction(text="".join(p for p, _ in pairs), characters=characters)


def test_build_display_units_merges_dakuten_pair_and_uses_min_confidence():
    from predict_sequence import build_display_units

    sequence = _fake_sequence([("カ", 0.9), ("゛", 0.4), ("イ", 0.8)])
    units = build_display_units(sequence, low_confidence_threshold=0.5)

    assert [u.text for u in units] == ["ガ", "イ"]
    assert units[0].confidence == 0.4
    assert units[0].segment_indices == [0, 1]
    assert units[0].needs_review is True
    assert units[1].needs_review is False


def test_build_display_units_matches_compose_dakuten_text():
    from morse_table import compose_dakuten
    from predict_sequence import build_display_units

    sequence = _fake_sequence([("ハ", 0.9), ("゜", 0.9), ("ハ", 0.9), ("゛", 0.9), ("ン", 0.9), ("゛", 0.9), ("ー", 0.9)])
    units = build_display_units(sequence, low_confidence_threshold=0.5)
    assert "".join(u.text for u in units) == compose_dakuten(sequence.text)


def test_build_display_units_flags_only_below_threshold():
    from predict_sequence import build_display_units

    sequence = _fake_sequence([("イ", 0.49), ("ロ", 0.5), ("ハ", 0.51)])
    units = build_display_units(sequence, low_confidence_threshold=0.5)
    assert [u.needs_review for u in units] == [True, False, False]


def test_build_display_units_empty_sequence():
    from predict_sequence import build_display_units

    assert build_display_units(_fake_sequence([]), 0.5) == []


def test_predict_sequence_from_waveform_matches_file_based_predict_sequence(tmp_path, trained_models_dir):
    import librosa

    from predict import load_model
    from predict_sequence import predict_sequence_from_waveform

    wav_path = _build_sequence_wav(tmp_path, ["イ", "ロ", "1"])
    from_file = predict_sequence(wav_path, models_dir=trained_models_dir, device="cpu")

    model, _label_to_index, index_to_label, fixed_frames = load_model(trained_models_dir, "cpu")
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    from_waveform = predict_sequence_from_waveform(y, sr, model, index_to_label, fixed_frames)

    assert from_waveform.text == from_file.text
    assert len(from_waveform.characters) == len(from_file.characters)

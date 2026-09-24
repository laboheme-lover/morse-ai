"""app.py(Streamlit UI, Milestone 10)のテスト。

処理部分(音声読込・しきい値決定・結果整形)は単体テストし、画面全体の動作は
Streamlit標準のAppTestでアップロードから結果表示まで自動操作して確認する。
"""

import io

import numpy as np
import pytest
import soundfile as sf

import app
import predict
from audio_preprocessing import AudioAnalysis
from generate_morse_audio import SAMPLE_RATE, _silence, char_to_waveform
from predict import Candidate, PredictionResult
from predict_sequence import CharacterPrediction, DisplayUnit, SequencePrediction, build_display_units


def _wav_bytes(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _sequence_waveform(labels=("イ", "ロ"), gap_seconds=0.4, dot_duration=0.1):
    segments = []
    for i, label in enumerate(labels):
        if i > 0:
            segments.append(_silence(gap_seconds, SAMPLE_RATE))
        segments.append(char_to_waveform(label, dot_duration=dot_duration, frequency_hz=700.0, amplitude=0.6))
    return np.concatenate(segments)


def _characters(pairs):
    chars = []
    for i, (label, confidence) in enumerate(pairs):
        result = PredictionResult(
            prediction=label,
            confidence=confidence,
            top_k=[Candidate(label, confidence), Candidate("ト", 0.1)],
            mel_spectrogram=np.zeros((64, 4), dtype=np.float32),
        )
        chars.append(CharacterPrediction(i * 1000, i * 1000 + 800, SAMPLE_RATE, result))
    return SequencePrediction(text="".join(p for p, _ in pairs), characters=chars)


# --- 処理部分 -----------------------------------------------------------------


def test_load_audio_bytes_returns_mono_waveform_at_target_rate():
    original = _sequence_waveform()
    y, sr = app.load_audio_bytes(_wav_bytes(original), ".wav")
    assert sr == SAMPLE_RATE
    assert y.ndim == 1
    assert len(y) == len(original)


def test_load_audio_bytes_resamples_to_target_rate():
    tone = np.sin(2 * np.pi * 700 * np.arange(8000) / 8000).astype(np.float32)  # 8kHz・1秒
    y, sr = app.load_audio_bytes(_wav_bytes(tone, sample_rate=8000), ".wav")
    assert sr == SAMPLE_RATE
    assert abs(len(y) - SAMPLE_RATE) <= 2


def test_load_audio_bytes_accepts_suffix_without_dot():
    y, _ = app.load_audio_bytes(_wav_bytes(_sequence_waveform()), "wav")
    assert len(y) > 0


def test_resolve_min_char_gap_uses_manual_value_when_auto_disabled():
    analysis = AudioAnalysis(700.0, 0.08, 0.25, 15.0)
    assert app.resolve_min_char_gap(analysis, 0.33, auto_estimate=False) == 0.33


def test_resolve_min_char_gap_scales_dot_duration_when_auto_enabled():
    analysis = AudioAnalysis(700.0, 0.075, 0.25, 16.0)
    gap = app.resolve_min_char_gap(analysis, 0.33, auto_estimate=True)
    assert gap == pytest.approx(0.075 * app.AUTO_GAP_MULTIPLIER)


def test_resolve_min_char_gap_respects_floor_for_very_fast_audio():
    analysis = AudioAnalysis(700.0, 0.01, 0.05, 120.0)
    assert app.resolve_min_char_gap(analysis, 0.33, auto_estimate=True) == app.AUTO_GAP_FLOOR_SECONDS


def test_build_summary_counts_and_averages():
    units = [
        DisplayUnit("イ", 0.9, [0], False),
        DisplayUnit("ロ", 0.3, [1], True),
        DisplayUnit("ハ", 0.6, [2], False),
    ]
    summary = app.build_summary(units)
    assert summary["count"] == 3
    assert summary["review_count"] == 1
    assert summary["mean_confidence"] == pytest.approx(0.6)


def test_build_summary_empty():
    assert app.build_summary([]) == {"count": 0, "mean_confidence": 0.0, "review_count": 0}


def test_build_details_table_merges_dakuten_into_one_row_with_time_range():
    sequence = _characters([("カ", 0.9), ("゛", 0.4), ("イ", 0.8)])
    units = build_display_units(sequence, 0.5)
    table = app.build_details_table(sequence, units)

    assert list(table["予測文字"]) == ["ガ", "イ"]
    assert list(table["要確認"]) == ["要確認", ""]
    # 「ガ」は2区間(0番・1番)にまたがるので、開始は0番の開始・終了は1番の終了
    assert table.iloc[0]["開始[秒]"] == pytest.approx(0.0)
    assert table.iloc[0]["終了[秒]"] == pytest.approx((1000 + 800) / SAMPLE_RATE, abs=0.01)
    assert "カ(0.90)" in table.iloc[0]["Top候補"]


def test_render_text_html_highlights_only_review_units_and_escapes():
    units = [DisplayUnit("イ", 0.9, [0], False), DisplayUnit("ロ", 0.2, [1], True)]
    rendered = app.render_text_html(units)
    assert rendered.count("background:rgba(255,170,0") == 1
    assert ">ロ</span>" in rendered
    assert "イ" in rendered

    unsafe = app.render_text_html([DisplayUnit("<b>", 0.1, [0], True)])
    assert "<b>" not in unsafe
    assert "&lt;b&gt;" in unsafe


def test_recognize_with_real_pipeline_runs_end_to_end():
    """学習済みモデルが無くても、ダミーの小さなモデルでrecognize()が動くことを確認する。"""
    from morse_cnn import MorseCNN

    model = MorseCNN(num_classes=3, channels=(4, 8)).eval()
    index_to_label = {0: "イ", 1: "ロ", 2: "1"}
    y = _sequence_waveform(("イ", "ロ", "1"))

    result = app.recognize(
        y, SAMPLE_RATE, model, index_to_label, fixed_frames=200,
        min_char_gap_seconds=0.25, auto_gap=False, top_k=2, preprocess=False,
    )
    assert len(result.sequence.characters) == 3
    assert result.min_char_gap_seconds == 0.25
    assert result.waveform is y

    with_preprocess = app.recognize(
        y, SAMPLE_RATE, model, index_to_label, fixed_frames=200,
        min_char_gap_seconds=0.25, auto_gap=True, top_k=2, preprocess=True,
    )
    assert len(with_preprocess.sequence.characters) == 3
    assert with_preprocess.min_char_gap_seconds >= app.AUTO_GAP_FLOOR_SECONDS


# --- 画面全体(AppTest) --------------------------------------------------------


@pytest.fixture
def app_test():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file("app.py", default_timeout=120)


@pytest.fixture
def tiny_models_dir(tmp_path, monkeypatch):
    """models/の代わりに使う、ごく小さな学習済み相当のモデル一式。"""
    import json

    import torch

    from morse_cnn import MorseCNN

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model = MorseCNN(num_classes=3, channels=(4, 8))
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": model.config(), "num_classes": 3,
         "fixed_frames": 200, "val_acc": 0.5, "epoch": 1},
        models_dir / "best_model.pt",
    )
    (models_dir / "label_mapping.json").write_text(
        json.dumps({"label_to_index": {"イ": 0, "ロ": 1, "1": 2}, "fixed_frames": 200}, ensure_ascii=False),
        encoding="utf-8",
    )
    # app.pyはAppTestで毎回スクリプトとして実行され、predictモジュールから
    # DEFAULT_MODELS_DIRを取り込むため、predict側の値を差し替える。
    monkeypatch.setattr(predict, "DEFAULT_MODELS_DIR", models_dir)
    return models_dir


def test_app_shows_upload_prompt_before_any_file(app_test, tiny_models_dir):
    app_test.run()
    assert not app_test.exception
    assert any("アップロード" in info.value for info in app_test.info)


def test_app_reports_missing_model(app_test, tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "DEFAULT_MODELS_DIR", tmp_path / "no_such_models")
    app_test.run()
    assert not app_test.exception
    assert any("学習済みモデルが見つかりません" in err.value for err in app_test.error)


def test_app_default_sidebar_settings(app_test, tiny_models_dir):
    app_test.run()
    sliders = {s.label: s.value for s in app_test.sidebar.slider}
    assert sliders["文字間隔しきい値 [秒]"] == pytest.approx(0.25)
    assert sliders["Top K候補数"] == 3
    assert sliders["低確信度しきい値"] == pytest.approx(0.5)


def test_app_recognizes_uploaded_wav_and_shows_results(app_test, tiny_models_dir):
    wav = _wav_bytes(_sequence_waveform(("イ", "ロ", "1")))
    app_test.run()
    app_test.file_uploader[0].set_value(("sample.wav", wav, "audio/wav"))
    app_test.run()

    assert not app_test.exception
    metrics = {m.label: m.value for m in app_test.metric}
    assert metrics["認識文字数"] == "3"
    assert "平均確信度" in metrics
    assert "要確認文字数" in metrics
    # 認識結果(HTML)と、文字ごとの詳細一覧が表示されていること
    assert any("認識結果" in sub.value for sub in app_test.subheader)
    assert any("文字ごとの詳細" in sub.value for sub in app_test.subheader)
    assert len(app_test.dataframe) == 1


def test_app_low_confidence_threshold_controls_review_count(app_test, tiny_models_dir):
    wav = _wav_bytes(_sequence_waveform(("イ", "ロ", "1")))
    app_test.run()
    app_test.file_uploader[0].set_value(("sample.wav", wav, "audio/wav"))

    # しきい値0.0なら確信度は必ずそれ以上なので要確認は0件
    low_conf = [s for s in app_test.sidebar.slider if s.label == "低確信度しきい値"][0]
    low_conf.set_value(0.0)
    app_test.run()
    zero = {m.label: m.value for m in app_test.metric}["要確認文字数"]
    assert zero == "0"

    # しきい値1.0なら確信度は必ず未満なので全件が要確認
    low_conf = [s for s in app_test.sidebar.slider if s.label == "低確信度しきい値"][0]
    low_conf.set_value(1.0)
    app_test.run()
    all_flagged = {m.label: m.value for m in app_test.metric}["要確認文字数"]
    assert all_flagged == "3"


def test_app_warns_when_no_sound_detected(app_test, tiny_models_dir):
    silence = _wav_bytes(_silence(1.0, SAMPLE_RATE))
    app_test.run()
    app_test.file_uploader[0].set_value(("silence.wav", silence, "audio/wav"))
    app_test.run()

    assert not app_test.exception
    assert any("音の区間が検出できません" in w.value for w in app_test.warning)


def test_app_shows_error_for_corrupted_audio(app_test, tiny_models_dir):
    app_test.run()
    app_test.file_uploader[0].set_value(("broken.wav", b"this is not audio", "audio/wav"))
    app_test.run()

    assert not app_test.exception
    assert any("失敗しました" in err.value for err in app_test.error)

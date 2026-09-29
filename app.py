"""和文モールス信号AI翻訳アプリ(Streamlit UI, Milestone 10)。

音声ファイルをアップロードすると、文字ごとに分割してCNNで認識し、
日本語文字列として表示する。確信度の低い文字は「要確認」として強調表示する。

起動:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from audio_preprocessing import AudioAnalysis, analyze_audio, preprocess_for_inference
from morse_dataset import SAMPLE_RATE
from morse_segmentation import DEFAULT_MIN_CHAR_GAP_SECONDS
from predict import DEFAULT_MODELS_DIR, load_model
from predict_sequence import (
    DisplayUnit,
    SequencePrediction,
    build_display_units,
    predict_sequence_from_waveform,
)

matplotlib.use("Agg")
for _font_name in ("Yu Gothic", "Meiryo", "MS Gothic"):
    if _font_name in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False

SUPPORTED_AUDIO_TYPES = ["wav", "mp3"]

# 相対パスのモデルディレクトリは、起動時のカレントディレクトリではなく
# このファイルの場所を基準に解決する(どこから起動しても動くようにするため)。
APP_DIR = Path(__file__).resolve().parent

# 「文字間隔しきい値を自動推定する」場合、推定した短点長のこの倍率を文字境界とする。
AUTO_GAP_MULTIPLIER = 1.8
AUTO_GAP_FLOOR_SECONDS = 0.08

DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.5


# --- 処理(UIから独立しており単体テストできる) ------------------------------


def load_audio_bytes(audio_bytes: bytes, suffix: str, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """アップロードされた音声(bytes)をモノラル・指定サンプリング周波数の波形へ変換する。

    mp3等のデコードにはファイルパスが必要なため、一時ファイルに書き出してから読み込む。
    """
    fd, tmp_name = tempfile.mkstemp(suffix=suffix if suffix.startswith(".") else f".{suffix}")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        y, sr = librosa.load(tmp_name, sr=sample_rate, mono=True)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return y, sr


def resolve_min_char_gap(
    analysis: AudioAnalysis,
    manual_gap_seconds: float,
    auto_estimate: bool,
) -> float:
    """文字境界とみなす無音の最小長を決める。自動推定時は測定した短点長の倍数とする。"""
    if not auto_estimate:
        return manual_gap_seconds
    return max(AUTO_GAP_FLOOR_SECONDS, analysis.dot_duration_seconds * AUTO_GAP_MULTIPLIER)


@dataclass
class RecognitionResult:
    """認識処理の結果一式(Streamlitのキャッシュ対象)。"""

    sequence: SequencePrediction
    waveform: np.ndarray  # 再生用の元波形(前処理前)
    sample_rate: int
    analysis: AudioAnalysis
    min_char_gap_seconds: float


def recognize(
    y: np.ndarray,
    sr: int,
    model,
    index_to_label: dict[int, str],
    fixed_frames: int,
    min_char_gap_seconds: float,
    auto_gap: bool,
    top_k: int,
    preprocess: bool,
    device: str = "cpu",
) -> RecognitionResult:
    """波形を認識する(音響特性の測定→任意の前処理→しきい値決定→分割・推論)。"""
    analysis = analyze_audio(y, sr)
    model_input = preprocess_for_inference(y, sr)[0] if preprocess else y
    gap = resolve_min_char_gap(analysis, min_char_gap_seconds, auto_gap)
    sequence = predict_sequence_from_waveform(
        model_input, sr, model, index_to_label, fixed_frames, top_k=top_k, min_char_gap_seconds=gap, device=device
    )
    return RecognitionResult(
        sequence=sequence, waveform=y, sample_rate=sr, analysis=analysis, min_char_gap_seconds=gap
    )


def build_summary(units: list[DisplayUnit]) -> dict:
    """概要(認識文字数・平均確信度・要確認文字数)を求める。"""
    if not units:
        return {"count": 0, "mean_confidence": 0.0, "review_count": 0}
    return {
        "count": len(units),
        "mean_confidence": float(np.mean([u.confidence for u in units])),
        "review_count": sum(1 for u in units if u.needs_review),
    }


def build_details_table(sequence: SequencePrediction, units: list[DisplayUnit]) -> pd.DataFrame:
    """文字ごとの詳細一覧(予測文字・確信度・Top候補・区間・要確認)をDataFrameにする。"""
    rows = []
    for number, unit in enumerate(units, start=1):
        segments = [sequence.characters[i] for i in unit.segment_indices]
        # 濁音・半濁音は2区間からなるため、Top候補は基本仮名側(最初の区間)を示す
        candidates = " / ".join(f"{c.label}({c.confidence:.2f})" for c in segments[0].result.top_k)
        rows.append(
            {
                "番号": number,
                "予測文字": unit.text,
                "確信度": round(unit.confidence, 3),
                "Top候補": candidates,
                "開始[秒]": round(segments[0].start_seconds, 2),
                "終了[秒]": round(segments[-1].end_seconds, 2),
                "要確認": "要確認" if unit.needs_review else "",
            }
        )
    return pd.DataFrame(rows)


def render_text_html(units: list[DisplayUnit]) -> str:
    """認識結果を、要確認の文字を強調したHTMLにする(暗色テーマでも読める半透明の背景色)。"""
    parts = []
    for unit in units:
        text = html.escape(unit.text)
        if unit.needs_review:
            parts.append(
                f'<span style="background:rgba(255,170,0,0.35);border-bottom:3px solid #e67e00;'
                f'border-radius:3px;padding:0 1px;" title="確信度 {unit.confidence:.2f}">{text}</span>'
            )
        else:
            parts.append(text)
    body = "".join(parts)
    return (
        '<div style="font-size:1.9rem;line-height:2.4rem;letter-spacing:0.08em;word-break:break-all;">'
        f"{body}</div>"
    )


def plot_mel(mel_db: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(6, 2.6))
    img = ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("フレーム")
    ax.set_ylabel("メルビン")
    fig.colorbar(img, ax=ax, label="dB")
    fig.tight_layout()
    return fig


# --- キャッシュ ------------------------------------------------------------


@st.cache_resource(show_spinner="モデルを読み込み中...")
def get_model_bundle(models_dir: str, model_mtime: float):
    """モデルはプロセス内で1回だけ読み込む(model_mtimeは再学習時にキャッシュを無効にするため)。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _label_to_index, index_to_label, fixed_frames = load_model(models_dir, device)
    return model, index_to_label, fixed_frames, device


@st.cache_data(show_spinner="認識中...(長い音声では数十秒かかることがあります)")
def run_recognition(
    audio_bytes: bytes,
    suffix: str,
    min_char_gap_seconds: float,
    auto_gap: bool,
    top_k: int,
    preprocess: bool,
    models_dir: str,
    model_mtime: float,
) -> RecognitionResult:
    model, index_to_label, fixed_frames, device = get_model_bundle(models_dir, model_mtime)
    y, sr = load_audio_bytes(audio_bytes, suffix)
    return recognize(
        y, sr, model, index_to_label, fixed_frames, min_char_gap_seconds, auto_gap, top_k, preprocess, device
    )


# --- 画面 ------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="和文モールス信号AI翻訳", page_icon="📻", layout="wide")
    st.title("📻 和文モールス信号AI翻訳")
    st.caption("音声ファイルをアップロードすると、文字ごとに分割してCNNで認識し、日本語文字列として表示します。")

    models_dir = str(APP_DIR / DEFAULT_MODELS_DIR)
    best_model_path = Path(models_dir) / "best_model.pt"
    if not best_model_path.exists():
        st.error(f"学習済みモデルが見つかりません: {best_model_path}。先に train.py で学習してください。")
        return
    model_mtime = best_model_path.stat().st_mtime

    with st.sidebar:
        st.header("認識設定")
        auto_gap = st.checkbox(
            "文字間隔しきい値を自動推定する",
            value=True,
            help="音声中の短点の長さを測定し、その約1.8倍を文字境界とみなします。送信速度が分からない実録音向けです。"
            "既定でON(固定しきい値は速度の合わない実録音では文字が大量に結合されてしまうため)。",
        )
        min_gap = st.slider(
            "文字間隔しきい値 [秒]",
            min_value=0.05,
            max_value=1.0,
            value=float(DEFAULT_MIN_CHAR_GAP_SECONDS),
            step=0.01,
            disabled=auto_gap,
            help="この長さ以上の無音を文字と文字の境界とみなします。",
        )
        top_k = st.slider("Top K候補数", min_value=1, max_value=5, value=3)
        low_conf = st.slider(
            "低確信度しきい値",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
            step=0.05,
            help="確信度がこの値未満の文字を「要確認」として強調表示します。",
        )
        preprocess = st.checkbox(
            "実録音向け前処理(バンドパス+振幅正規化)",
            value=True,
            help="検出したトーン周波数付近だけを通し、倍音などの帯域外成分を抑えてから認識します。"
            "ノイズの多い録音では区間分割の精度が大きく改善するため既定でON(ノイズの少ない録音では"
            "ほぼ差はない)。",
        )

    uploaded = st.file_uploader(
        "モールス信号の音声ファイルを選択(WAV / MP3)", type=SUPPORTED_AUDIO_TYPES, accept_multiple_files=False
    )
    if uploaded is None:
        st.info("音声ファイルをアップロードしてください。")
        return

    audio_bytes = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix.lower() or ".wav"

    st.subheader("アップロード音声")
    st.audio(audio_bytes)

    try:
        result = run_recognition(
            audio_bytes, suffix, float(min_gap), bool(auto_gap), int(top_k), bool(preprocess), models_dir, model_mtime
        )
    except Exception as exc:  # noqa: BLE001 - 読み込み失敗等をユーザーに分かりやすく示す
        st.error(f"音声の読み込みまたは認識に失敗しました: {exc}")
        return

    units = build_display_units(result.sequence, low_conf)
    if not units:
        st.warning("音の区間が検出できませんでした。無音のファイル、または文字間隔しきい値が合っていない可能性があります。")
        return

    st.subheader("認識結果")
    st.markdown(render_text_html(units), unsafe_allow_html=True)
    if any(u.needs_review for u in units):
        st.caption("背景が橙色の文字は確信度が低く、要確認です。")

    summary = build_summary(units)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("認識文字数", summary["count"])
    col2.metric("平均確信度", f"{summary['mean_confidence']:.2f}")
    col3.metric("要確認文字数", summary["review_count"])
    col4.metric("使用した文字間隔しきい値", f"{result.min_char_gap_seconds * 1000:.0f} ms")

    with st.expander("音響特性の測定結果"):
        a = result.analysis
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("トーン周波数", f"{a.frequency_hz:.0f} Hz")
        m2.metric("短点長", f"{a.dot_duration_seconds * 1000:.0f} ms")
        m3.metric("文字間の無音(目安)", f"{a.char_gap_seconds * 1000:.0f} ms")
        m4.metric("概算打鍵速度", f"{a.wpm:.0f} WPM")

    st.subheader("文字ごとの詳細")
    table = build_details_table(result.sequence, units)
    only_review = st.checkbox("要確認の文字のみ表示", value=False)
    shown = table[table["要確認"] != ""] if only_review else table
    st.dataframe(
        shown,
        hide_index=True,
        use_container_width=True,
        column_config={
            "確信度": st.column_config.ProgressColumn("確信度", min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )

    st.markdown("##### 文字を選んでメルスペクトログラムと該当音声を確認")
    options = shown["番号"].tolist()
    if not options:
        st.info("表示する文字がありません。")
        return
    selected_number = st.selectbox(
        "文字",
        options,
        format_func=lambda n: f"{n}: {units[n - 1].text}(確信度 {units[n - 1].confidence:.2f})"
        + ("  要確認" if units[n - 1].needs_review else ""),
    )
    unit = units[selected_number - 1]
    segments = [result.sequence.characters[i] for i in unit.segment_indices]
    start_sample, end_sample = segments[0].start_sample, segments[-1].end_sample
    st.audio(result.waveform[start_sample:end_sample], sample_rate=result.sample_rate)

    columns = st.columns(len(segments))
    for column, segment in zip(columns, segments):
        with column:
            prediction = segment.result
            st.markdown(
                f"**予測: {prediction.prediction}**(確信度 {prediction.confidence:.2f}、"
                f"{segment.start_seconds:.2f}〜{segment.end_seconds:.2f}秒)"
            )
            st.pyplot(plot_mel(prediction.mel_spectrogram, f"メルスペクトログラム({prediction.prediction})"))
            st.table(
                pd.DataFrame(
                    {"候補": [c.label for c in prediction.top_k], "確信度": [f"{c.confidence:.2f}" for c in prediction.top_k]}
                )
            )
            plt.close("all")

    audio_hash = hashlib.md5(audio_bytes).hexdigest()[:8]
    st.caption(f"ファイル: {uploaded.name}(md5先頭8桁: {audio_hash})")


if __name__ == "__main__":
    main()

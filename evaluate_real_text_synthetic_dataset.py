"""generate_real_text_synthetic_dataset.pyが作った合成テストデータを評価する。

evaluate_real_samples.pyと同じ比較ロジック(compare_texts, autojunk=False)を使うが、
正解テキストは合成時に既にmorse_table語彙へ絞り込み済みなので、
normalize_ground_truthは不要(生成時に使ったテキストそのものと比較する)。

前処理(audio_preprocessing.preprocess_for_inference、バンドパスフィルタ+振幅正規化)は
既定で有効にする。ホワイトノイズ/ESC-50ノイズを音声全体(符号間の無音部分も含む)に
付加した長時間音声では、区間分割(morse_segmentation)が使うRMSヒステリシスの
しきい値をノイズの床が上回ってしまい、無音を検出できず音声全体が1区間として
扱われてしまう(SNR20dB未満で顕著、文字一致率が数%まで崩壊)ことが判明した。
トーン周波数帯だけを通すバンドパスフィルタを適用すると帯域外のノイズ電力が
大きく減り、無音区間のRMSが再びしきい値を下回るようになって正しく分割できる
ようになる(--no-preprocessで無効化して比較することも可能)。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import librosa
import soundfile as sf

from audio_preprocessing import estimate_dot_seconds, preprocess_for_inference
from evaluate_real_samples import compare_texts
from morse_table import compose_dakuten
from predict_sequence import predict_sequence

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_METADATA_CSV = Path("data/real_text_synthetic_test/metadata.csv")
DEFAULT_OUTPUT_CSV = Path("evaluation/real_text_synthetic_report.csv")

REPORT_FIELDS = [
    "category",
    "message_no",
    "duration_sec",
    "speed",
    "noise_type",
    "snr_db",
    "preprocess",
    "measured_frequency_hz",
    "dot_estimate_ms",
    "min_gap_used_ms",
    "ground_truth_len",
    "predicted_len",
    "num_segments",
    "similarity",
    "char_match_rate",
    "char_substitution_rate",
    "char_insertion_rate",
    "char_deletion_rate",
]


def evaluate_row(
    row: dict,
    models_dir: str | Path,
    top_k: int,
    gap_multiplier: float,
    min_gap_floor: float,
    preprocess: bool,
    tmp_dir: Path,
) -> dict:
    # 処理の流れは evaluate_real_samples.evaluate_pair と同じ。違いは、正解テキストが
    # 合成時の本文そのもの(既に語彙で絞り込み済み)なので、正規化をしない点だけ。
    wav_path = Path(row["wav_path"])
    text_path = Path(row["text_path"])
    ground_truth = text_path.read_text(encoding="utf-8")

    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)

    # 前処理(トーン周波数付近だけを通すフィルタ)は既定でON。
    # 雑音が無音部分にも乗るこのデータでは、前処理なしだと区間分割が崩れる(平均27.9%→前処理ありで68.2%)。
    if preprocess:
        processed_y, analysis = preprocess_for_inference(y, sr)
        measured_freq = analysis.frequency_hz
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_wav = tmp_dir / f"{wav_path.stem}_preprocessed.wav"
        sf.write(tmp_wav, processed_y, sr, subtype="PCM_16")
        inference_path: Path | str = tmp_wav
        dot_estimate = estimate_dot_seconds(processed_y, sr)
    else:
        measured_freq = 0.0
        inference_path = wav_path
        dot_estimate = estimate_dot_seconds(y, sr)

    # 文字の区切りのしきい値 = 測った短点の長さ × 1.8(アプリの自動推定と同じ規則)
    min_gap = max(min_gap_floor, dot_estimate * gap_multiplier)

    result = predict_sequence(inference_path, models_dir=models_dir, top_k=top_k, min_char_gap_seconds=min_gap)
    predicted_text = compose_dakuten(result.text)

    # 前処理用に作った一時WAVを消す
    if preprocess and isinstance(inference_path, Path):
        inference_path.unlink(missing_ok=True)

    comparison = compare_texts(predicted_text, ground_truth)

    return {
        "category": row["category"],
        "message_no": row["message_no"],
        "duration_sec": row["duration_sec"],
        "speed": row["speed"],
        "noise_type": row["noise_type"],
        "snr_db": row["snr_db"],
        "preprocess": preprocess,
        "measured_frequency_hz": round(measured_freq, 1),
        "dot_estimate_ms": round(dot_estimate * 1000, 1),
        "min_gap_used_ms": round(min_gap * 1000, 1),
        "ground_truth_len": len(ground_truth),
        "predicted_len": len(predicted_text),
        "num_segments": len(result.characters),
        "similarity": round(comparison["similarity"], 4),
        "char_match_rate": round(comparison["char_match_rate"], 4),
        "char_substitution_rate": round(comparison["char_substitution_rate"], 4),
        "char_insertion_rate": round(comparison["char_insertion_rate"], 4),
        "char_deletion_rate": round(comparison["char_deletion_rate"], 4),
    }


def run_evaluation(
    metadata_csv: str | Path = DEFAULT_METADATA_CSV,
    models_dir: str | Path = "models",
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    top_k: int = 3,
    gap_multiplier: float = 1.8,
    min_gap_floor: float = 0.08,
    preprocess: bool = True,
) -> list[dict]:
    with open(metadata_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    tmp_dir = Path("evaluation/_real_text_synthetic_preprocessed_tmp")
    print(f"評価対象: {len(rows)}件 (preprocess={preprocess})")
    results: list[dict] = []
    for row in rows:
        print(f"[{row['category']} 電文{int(row['message_no']):02d}] 処理中...")
        try:
            result = evaluate_row(row, models_dir, top_k, gap_multiplier, min_gap_floor, preprocess, tmp_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"  エラー: {exc}")
            result = {field: "" for field in REPORT_FIELDS}
            result["category"] = row["category"]
            result["message_no"] = row["message_no"]
        results.append(result)
        print(
            f"  similarity={result.get('similarity')} char_match_rate={result.get('char_match_rate')} "
            f"predicted_len={result.get('predicted_len')} ground_truth_len={result.get('ground_truth_len')}"
        )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"レポートを保存しました: {output_csv}")

    sims = [r["similarity"] for r in results if isinstance(r.get("similarity"), (int, float))]
    if sims:
        print(f"平均similarity: {sum(sims) / len(sims):.4f} ({len(sims)}件)")
    match_rates = [r["char_match_rate"] for r in results if isinstance(r.get("char_match_rate"), (int, float))]
    if match_rates:
        print(f"平均 文字一致率: {sum(match_rates) / len(match_rates):.4f}")

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="実データテキスト由来の合成テストデータを評価する")
    parser.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--no-preprocess", dest="preprocess", action="store_false",
        help="推論前のバンドパスフィルタ・振幅正規化(audio_preprocessing)を無効化する",
    )
    parser.set_defaults(preprocess=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_evaluation(args.metadata_csv, args.models_dir, args.output_csv, args.top_k, preprocess=args.preprocess)


if __name__ == "__main__":
    main()

"""加工音源(速度・雑音・SN比の違い)ごとの連続音声認識の精度を測るスクリプト。

想定するフォルダ構成(C:\\Users\\daizo\\OneDrive\\audio):
    <root>/no041/no041_original.wav              原音
    <root>/no041/no041_fast_clean.wav            速度 fast・雑音なし
    <root>/no041/no041_fast_white_snrp10.wav     速度 fast・ホワイトノイズ・SN比 +10dB
    <root>/no041/no041_slow_esc50_snrm05.wav     速度 slow・環境音・SN比 -5dB
    ... 各WAVと同じ名前の .txt が正解テキスト

認識はアプリ(app.py)の recognize() をそのまま使う。最新のモデル・文字間隔しきい値の自動推定ON・
実録音向け前処理ON という、アプリの既定の設定で認識した結果を測ることになる。
正解テキストの正規化と比較は evaluate_real_samples.py と同じ関数を使う。

1件ごとの結果は終わり次第 CSV に追記するので、途中で止まっても同じコマンドで続きから再開できる。
複数のCPUコアで並列に処理する(--workers)。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_ROOT = Path(r"C:\Users\daizo\OneDrive\audio")
DEFAULT_DETAILS_CSV = Path("evaluation/audio_variants_details.csv")
DEFAULT_SUMMARY_CSV = Path("evaluation/audio_variants_summary.csv")

# ファイル名 "no041_fast_white_snrp10" を (番号, 速度, 雑音, SN比) に分解する
VARIANT_RE = re.compile(r"^(no\d+)_(?:(original)|(fast|normal|slow)_(clean|white|esc50)(?:_snr([mp])(\d+))?)$")

SPEED_ORDER = ["fast", "normal", "slow"]
NOISE_ORDER = ["clean", "white", "esc50"]
SNR_ORDER = [20, 10, 5, 0, -5]

DETAIL_FIELDS = [
    "folder",
    "file",
    "pattern",
    "speed",
    "noise",
    "snr_db",
    "duration_sec",
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
    "excluded_chars",
    "elapsed_sec",
    "error",
    "predicted_text",
    "ground_truth",
]


def parse_variant(stem: str) -> dict | None:
    """ファイル名(拡張子なし)から加工条件を取り出す。形式に合わなければ None。"""
    m = VARIANT_RE.match(stem)
    if not m:
        return None
    folder, original, speed, noise, sign, value = m.groups()
    if original:
        return {"folder": folder, "pattern": "original", "speed": "", "noise": "", "snr_db": ""}
    snr = ""
    if sign is not None:
        snr = int(value) * (-1 if sign == "m" else 1)
    pattern = f"{speed}_{noise}" + (f"_snr{snr:+d}" if snr != "" else "")
    return {"folder": folder, "pattern": pattern, "speed": speed, "noise": noise, "snr_db": snr}


def pattern_sort_key(speed: str, noise: str, snr_db) -> tuple:
    """集計表の並び順: 原音 → 速度(fast/normal/slow)→ 雑音(なし/white/esc50)→ SN比(高い順)。"""
    if speed == "":
        return (0, 0, 0, 0)  # 原音
    snr_index = SNR_ORDER.index(int(snr_db)) if snr_db != "" else -1  # 雑音なしはSN比なし
    return (1, SPEED_ORDER.index(speed), NOISE_ORDER.index(noise), snr_index)


def find_jobs(root: Path, which: str) -> list[dict]:
    """評価対象の (WAV, 正解テキスト) の組を集める。which = original / processed / all"""
    jobs: list[dict] = []
    for wav in sorted(root.glob("*/*.wav")):
        info = parse_variant(wav.stem)
        if info is None:
            continue
        is_original = info["pattern"] == "original"
        if which == "original" and not is_original:
            continue
        if which == "processed" and is_original:
            continue
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue  # 正解テキストの無い音源は評価しない
        jobs.append({**info, "wav": str(wav), "txt": str(txt), "file": wav.name})
    return jobs


# ---- 並列処理の各ワーカー(プロセス)で使う状態 ----
_MODEL = None


def _init_worker(models_dir: str) -> None:
    # 各ワーカーのPyTorch・数値計算は1スレッドに制限する(ワーカー数×スレッド数でCPUを取り合わないように)
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from predict import load_model

    global _MODEL
    model, _label_to_index, index_to_label, fixed_frames = load_model(models_dir, "cpu")
    _MODEL = (model, index_to_label, fixed_frames)


def _evaluate_job(job: dict) -> dict:
    """1ファイルを認識して正解と比べる(ワーカープロセス内で実行される)。"""
    import librosa

    from app import recognize
    from evaluate_real_samples import compare_texts, normalize_ground_truth
    from morse_dataset import SAMPLE_RATE
    from morse_segmentation import DEFAULT_MIN_CHAR_GAP_SECONDS
    from morse_table import compose_dakuten

    started = time.time()
    row = {field: "" for field in DETAIL_FIELDS}
    row.update({k: job[k] for k in ("folder", "file", "pattern", "speed", "noise", "snr_db")})
    try:
        model, index_to_label, fixed_frames = _MODEL
        raw_text = Path(job["txt"]).read_text(encoding="utf-8-sig")
        ground_truth, excluded = normalize_ground_truth(raw_text)

        y, sr = librosa.load(job["wav"], sr=SAMPLE_RATE, mono=True)
        # アプリと同じ処理: 自動推定ON・前処理ON(手動しきい値の値は自動推定ONなら使われない)
        result = recognize(
            y, sr, model, index_to_label, fixed_frames,
            min_char_gap_seconds=DEFAULT_MIN_CHAR_GAP_SECONDS, auto_gap=True, top_k=3, preprocess=True,
        )
        predicted = compose_dakuten(result.sequence.text)
        comparison = compare_texts(predicted, ground_truth)
        row.update(
            {
                "duration_sec": round(len(y) / sr, 1),
                "measured_frequency_hz": round(result.analysis.frequency_hz, 1),
                "dot_estimate_ms": round(result.analysis.dot_duration_seconds * 1000, 1),
                "min_gap_used_ms": round(result.min_char_gap_seconds * 1000, 1),
                "ground_truth_len": len(ground_truth),
                "predicted_len": len(predicted),
                "num_segments": len(result.sequence.characters),
                "similarity": round(comparison["similarity"], 4),
                "char_match_rate": round(comparison["char_match_rate"], 4),
                "char_substitution_rate": round(comparison["char_substitution_rate"], 4),
                "char_insertion_rate": round(comparison["char_insertion_rate"], 4),
                "char_deletion_rate": round(comparison["char_deletion_rate"], 4),
                "excluded_chars": "".join(excluded),
                "predicted_text": predicted,
                "ground_truth": ground_truth,
            }
        )
    except Exception as exc:  # noqa: BLE001 - 1件の失敗で全体を止めない
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["elapsed_sec"] = round(time.time() - started, 1)
    return row


def load_done(details_csv: Path) -> set[str]:
    """既に結果があるファイル名(エラーでないもの)。再開時はこれらを飛ばす。"""
    if not details_csv.exists():
        return set()
    with open(details_csv, encoding="utf-8", newline="") as f:
        return {r["file"] for r in csv.DictReader(f) if not r.get("error")}


def run(root: Path, which: str, details_csv: Path, models_dir: str, workers: int) -> None:
    from multiprocessing import Pool

    jobs = find_jobs(root, which)
    done = load_done(details_csv)
    todo = [j for j in jobs if j["file"] not in done]
    print(f"対象 {len(jobs)}件 / 処理済み {len(jobs) - len(todo)}件 / 今回 {len(todo)}件 (workers={workers})", flush=True)
    if not todo:
        return

    details_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not details_csv.exists()
    started = time.time()
    with open(details_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        if new_file:
            writer.writeheader()
        with Pool(processes=workers, initializer=_init_worker, initargs=(models_dir,)) as pool:
            # 長い音声から先に処理すると、最後に長いものが1つだけ残って待たされることを避けられる
            todo.sort(key=lambda j: os.path.getsize(j["wav"]), reverse=True)
            for n, row in enumerate(pool.imap_unordered(_evaluate_job, todo), start=1):
                writer.writerow(row)
                f.flush()
                status = row["error"] or f"一致率 {row['char_match_rate']}"
                elapsed = time.time() - started
                print(f"[{n}/{len(todo)}] {row['file']}  {status}  (経過 {elapsed / 60:.1f}分)", flush=True)


def summarize(details_csv: Path, summary_csv: Path) -> None:
    """明細をパターン(原音 / 速度×雑音×SN比)ごとに集計する。"""
    import pandas as pd

    df = pd.read_csv(details_csv, encoding="utf-8", keep_default_na=False)
    df = df[df["error"] == ""].copy()
    metrics = ["char_match_rate", "char_insertion_rate", "char_substitution_rate", "char_deletion_rate", "similarity"]
    for c in metrics + ["ground_truth_len", "predicted_len"]:
        df[c] = pd.to_numeric(df[c])
    # 一致した文字数(正解の文字数×一致率)。全件まとめた一致率(文字数で重み付け)の計算に使う
    df["matched_chars"] = df["char_match_rate"] * df["ground_truth_len"]

    rows = []
    for pattern, g in df.groupby("pattern"):
        first = g.iloc[0]
        rows.append(
            {
                "pattern": pattern,
                "speed": first["speed"],
                "noise": first["noise"],
                "snr_db": first["snr_db"],
                "files": len(g),
                "char_match_rate_mean": round(g["char_match_rate"].mean(), 4),
                "char_match_rate_min": round(g["char_match_rate"].min(), 4),
                "char_match_rate_overall": round(g["matched_chars"].sum() / g["ground_truth_len"].sum(), 4),
                "char_insertion_rate_mean": round(g["char_insertion_rate"].mean(), 4),
                "char_substitution_rate_mean": round(g["char_substitution_rate"].mean(), 4),
                "char_deletion_rate_mean": round(g["char_deletion_rate"].mean(), 4),
                "similarity_mean": round(g["similarity"].mean(), 4),
                "files_match_ge_95pct": int((g["char_match_rate"] >= 0.95).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["_order"] = [pattern_sort_key(s, n, r) for s, n, r in zip(summary["speed"], summary["noise"], summary["snr_db"])]
    summary = summary.sort_values("_order").drop(columns="_order")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False, encoding="utf-8")
    print(summary.to_string(index=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="加工音源ごとの連続音声認識の精度を測る")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--which", choices=["original", "processed", "all"], default="all")
    parser.add_argument("--details-csv", default=str(DEFAULT_DETAILS_CSV))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--summary-only", action="store_true", help="認識はせず、既存の明細から集計だけ行う")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    details_csv = Path(args.details_csv)
    if not args.summary_only:
        run(Path(args.root), args.which, details_csv, args.models_dir, args.workers)
    if details_csv.exists():
        summarize(details_csv, Path(args.summary_csv))


if __name__ == "__main__":
    main()

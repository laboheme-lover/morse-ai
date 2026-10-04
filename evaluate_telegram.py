"""電報形式の音源で、後処理(telegram_format)の前と後の精度を、前文・名あて・本文ごとに測る。

対象: <root>/<フォルダ>/<名前>.wav と同じ名前の .txt(audio2 と generate_telegram_synthetic_dataset.py の出力)。
認識はアプリ(app.recognize)の既定の設定(文字間隔の自動推定ON・前処理ON)で行い、その結果を
    後処理なし: モデルの出力そのまま
    後処理あり: telegram_format.read_telegram(mode="auto")
の2通りで採点する。正解テキストは telegram_scoring で前文・名あて・本文に分け、「」」は「。」として、
括弧は残して比べる。比較のため、従来の採点(evaluate_audio_variants.prepare_ground_truth、
「」」と括弧を正解から除く)での文字一致率も残す。

1件ごとの結果は終わり次第 CSV に追記する(途中で止まっても同じコマンドで再開できる)。
"""

from __future__ import annotations

import argparse
# ============================================================================
# このスクリプトの流れ
#   (1) find_jobs: 評価する (WAV, 正解テキスト) の組を集める
#   (2) run: 複数のプロセスで並列に _evaluate_job を実行し、1件終わるごとにCSVへ追記する
#       _evaluate_job = 認識(app.recognize)→ 音から符号を読む → 後処理 → 採点(score_row)
#   (3) summarize: パターン(原音・加工の種類)ごとに平均して表示する
#   後処理だけを直したときは --rescore で、CSVに保存した区間の列(tokens)に後処理をかけ直す。
#   認識(数十分〜数時間)をやり直さずに、後処理の変更の効果を数秒で確かめられる。
#
# CSVの列の読み方
#   raw_*  = 後処理なし(モデルの出力そのまま)、post_* = 後処理あり
#   *_all / *_header / *_address / *_body = 全体 / 前文 / 名あて / 本文
#   *_match = 文字一致率、*_insert = 挿入率(余分な文字の数 / 正解の文字数)
#   old_char_match_rate = 従来の採点(「」」と括弧を正解から除く)での後処理なしの文字一致率
# ============================================================================
import csv
import os
import re
import sys
import time
from pathlib import Path

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# 部分の名前(telegram_scoring と同じ)と、CSVの列名に使う英字の名前の対応
SECTIONS = ("前文", "名あて", "本文")
SECTION_KEYS = {"前文": "header", "名あて": "address", "本文": "body", "全体": "all"}

DETAIL_FIELDS = (
    ["folder", "file", "pattern", "speed", "noise", "snr_db", "dot_estimate_ms", "detected", "signals", "warnings",
     "declared_count", "body_len_read", "old_char_match_rate"]
    + [f"{stage}_{key}_{m}" for stage in ("raw", "post") for key in ("all", "header", "address", "body")
       for m in ("match", "insert")]
    + ["elapsed_sec", "error", "predicted_raw", "predicted_post", "ground_truth", "tokens"]
)


# <root>/<フォルダ>/*.wav のうち、同じ名前の .txt があるものを集める。
# ファイル名が audio2 の形(no001_fast_white_snrp10 など)なら加工条件を読み取り、
# それ以外(合成データ tg001_original など)は原音として扱う。
def find_jobs(root: Path, which: str, folder_regex: str | None) -> list[dict]:
    from evaluate_audio_variants import parse_variant

    jobs = []
    for wav in sorted(root.glob("*/*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        if folder_regex and not re.fullmatch(folder_regex, wav.parent.name):
            continue
        info = parse_variant(wav.stem) or {"folder": wav.parent.name, "pattern": "original", "speed": "",
                                           "noise": "", "snr_db": ""}
        is_original = info["pattern"] == "original"
        if (which == "original" and not is_original) or (which == "processed" and is_original):
            continue
        jobs.append({**info, "folder": wav.parent.name, "wav": str(wav), "txt": str(txt), "file": wav.name})
    return jobs


# 並列処理の各ワーカー(プロセス)で1回だけ読み込むモデル
_MODEL = None


# 各ワーカーのPyTorchは1スレッドに制限する(ワーカー数×スレッド数でCPUを取り合わないように)
def _init_worker(models_dir: str) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    from predict import load_model

    global _MODEL
    model, _label_to_index, index_to_label, fixed_frames = load_model(models_dir, "cpu")
    _MODEL = (model, index_to_label, fixed_frames)


def score_row(predicted_raw: str, reading, ground_truth_raw: str) -> dict:
    """後処理前・後の認識結果を、正解テキストと部分ごとに比べる(認識をしないので単体テストできる)。"""
    from evaluate_audio_variants import prepare_ground_truth
    from evaluate_real_samples import compare_texts
    from telegram_scoring import compare_by_section, split_telegram_ground_truth

    # 新しい採点の正解(前文・名あて・本文に分け、「」」→「。」、括弧は残す)と、比較用に従来の採点の正解
    gt = split_telegram_ground_truth(ground_truth_raw)
    old_gt, _ = prepare_ground_truth(ground_truth_raw)
    row = {
        "detected": reading.detected,
        "signals": " ".join(reading.signals),
        "warnings": " / ".join(reading.warnings),
        "declared_count": gt.declared_count if gt.declared_count is not None else "",
        "body_len_read": len(reading.body) if reading.detected else "",
        "old_char_match_rate": round(compare_texts(predicted_raw, old_gt)["char_match_rate"], 4),
        "predicted_raw": predicted_raw,
        "predicted_post": reading.text,
        "ground_truth": gt.text,
    }
    # 後処理なし・ありの両方を、同じ正解・同じ方法で部分ごとに採点する。
    # 正解の無い部分(本文だけのテキストの前文など)は空欄のままにする
    for stage, text in (("raw", predicted_raw), ("post", reading.text)):
        result = compare_by_section(text, gt.sections)
        for name, key in SECTION_KEYS.items():
            r = result[name]
            if r["length"] == 0:
                continue
            row[f"{stage}_{key}_match"] = round(r["char_match_rate"], 4)
            row[f"{stage}_{key}_insert"] = round(r["inserted"] / r["length"], 4)
    return row


# 例: 「、,.-.-.-,0.21;1,.----,0.2;…」(音の符号が読めなかった区間は符号が空、最後の区間は無音が空)
def encode_tokens(tokens) -> str:
    """区間の列を「文字,符号,無音[秒]」の並び(; 区切り)にして保存する。後処理だけをやり直すときに使う。"""
    return ";".join(f"{t.text},{t.pattern or ''},{'' if t.pause_after is None else round(t.pause_after, 3)}" for t in tokens)


def decode_tokens(text: str):
    from telegram_format import Token

    tokens = []
    for item in text.split(";") if text else []:
        label, pattern, pause = item.split(",")
        tokens.append(Token(label, pattern or None, float(pause) if pause else None))
    return tokens


def rescore(details_csv: Path, roots: list[Path]) -> None:
    """保存した区間の列に後処理をかけ直して採点し直す(認識はやり直さない)。"""
    import pandas as pd

    from telegram_format import interpret_telegram

    # すべて文字列として読み、書き戻すときに数値の表記(0.9800 → 0.98 など)が変わらないようにする
    df = pd.read_csv(details_csv, encoding="utf-8", keep_default_na=False, dtype=str)
    # ファイル名 → 正解テキストのパス(CSVには正解テキストの元の形が残っていないので、読み直す)
    txt_by_file = {wav.name: wav.with_suffix(".txt") for root in roots for wav in root.glob("*/*.wav")}
    rows = []
    for row in df.to_dict("records"):
        if not row["error"]:
            reading = interpret_telegram(decode_tokens(row["tokens"]), mode="auto")
            raw_text = txt_by_file[row["file"]].read_text(encoding="utf-8-sig")
            # 前回の後処理ありの値を消してから付け直す(今回空欄になる部分に古い値が残らないように)
            row.update({k: "" for k in DETAIL_FIELDS if k.startswith("post_")})
            row.update(score_row(row["predicted_raw"], reading, raw_text))
        rows.append(row)
    with open(details_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"採点し直しました: {len(rows)}件")


def _evaluate_job(job: dict) -> dict:
    import librosa

    from app import recognize
    from morse_dataset import SAMPLE_RATE
    from morse_segmentation import DEFAULT_MIN_CHAR_GAP_SECONDS
    from morse_table import compose_dakuten
    from telegram_format import interpret_telegram, read_element_patterns, tokens_from_sequence

    started = time.time()
    row = {f: "" for f in DETAIL_FIELDS}
    row.update({k: job[k] for k in ("folder", "file", "pattern", "speed", "noise", "snr_db")})
    try:
        model, index_to_label, fixed_frames = _MODEL
        y, sr = librosa.load(job["wav"], sr=SAMPLE_RATE, mono=True)
        # アプリと同じ設定で認識する(文字間隔の自動推定ON・前処理ON。手動のしきい値は自動推定ONなら使われない)
        result = recognize(
            y, sr, model, index_to_label, fixed_frames,
            min_char_gap_seconds=DEFAULT_MIN_CHAR_GAP_SECONDS, auto_gap=True, top_k=3, preprocess=True,
        )
        dot = result.analysis.dot_duration_seconds
        ranges = [(c.start_sample, c.end_sample) for c in result.sequence.characters]
        # 区間ごとに音から短点・長点の並びを読み、区間の列(tokens)にして後処理にかける(アプリの read_telegram と同じ手順。
        # 評価では区間の列をCSVに保存するため、ここでは手順を分けて書いている)
        patterns = read_element_patterns(result.model_input, sr, ranges, dot) if ranges else []
        tokens = tokens_from_sequence(result.sequence, sr, patterns)
        reading = interpret_telegram(tokens, mode="auto")
        raw_text = Path(job["txt"]).read_text(encoding="utf-8-sig")
        row.update(score_row(compose_dakuten(result.sequence.text), reading, raw_text))
        row["tokens"] = encode_tokens(tokens)
        row["dot_estimate_ms"] = round(dot * 1000, 1)
    except Exception as exc:  # noqa: BLE001 - 1件の失敗で全体を止めない
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["elapsed_sec"] = round(time.time() - started, 1)
    return row


def run(root: Path, which: str, details_csv: Path, models_dir: str, workers: int, folder_regex: str | None) -> None:
    from multiprocessing import Pool

    jobs = find_jobs(root, which, folder_regex)
    # 再開: 既にCSVにある(エラーでない)ファイルは飛ばす
    done = set()
    if details_csv.exists():
        with open(details_csv, encoding="utf-8", newline="") as f:
            done = {r["file"] for r in csv.DictReader(f) if not r.get("error")}
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
        with Pool(processes=min(workers, len(todo)), initializer=_init_worker, initargs=(models_dir,)) as pool:
            # 長い音声から先に処理すると、最後に長いものが1つだけ残って待たされることを避けられる
            todo.sort(key=lambda j: os.path.getsize(j["wav"]), reverse=True)
            for n, row in enumerate(pool.imap_unordered(_evaluate_job, todo), start=1):
                writer.writerow(row)
                f.flush()
                status = row["error"] or f"本文 {row['raw_body_match']}→{row['post_body_match']}"
                print(f"[{n}/{len(todo)}] {row['file']}  {status}  (経過 {(time.time() - started) / 60:.1f}分)", flush=True)


def summarize(details_csv: Path) -> None:
    """パターンごとに、後処理前→後の平均文字一致率(全体・部分ごと)と挿入率を表示する。"""
    import pandas as pd

    df = pd.read_csv(details_csv, encoding="utf-8", keep_default_na=False)
    df = df[df["error"] == ""].copy()
    cols = [c for c in DETAIL_FIELDS if c.startswith(("raw_", "post_")) or c == "old_char_match_rate"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # CSVから読むと True/False が文字列になることがあるので、真偽値にそろえる
    df["detected"] = df["detected"].astype(str) == "True"
    rows = []
    for pattern, g in df.groupby("pattern"):
        row = {"pattern": pattern, "files": len(g), "detected": int(g["detected"].sum()),
               "old_all": g["old_char_match_rate"].mean()}
        for key in ("all", "header", "address", "body"):
            row[f"{key}_raw"] = g[f"raw_{key}_match"].mean()
            row[f"{key}_post"] = g[f"post_{key}_match"].mean()
        row["insert_raw"] = g["raw_all_insert"].mean()
        row["insert_post"] = g["post_all_insert"].mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="電報形式の音源で後処理の前後の精度を測る")
    parser.add_argument("--root", required=True)
    parser.add_argument("--which", choices=["original", "processed", "all"], default="all")
    parser.add_argument("--details-csv", required=True)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--folder-regex", default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--rescore", action="store_true", help="認識はせず、保存した区間の列に後処理をかけ直して採点する")
    args = parser.parse_args()
    details_csv = Path(args.details_csv)
    if args.rescore:
        rescore(details_csv, [Path(args.root)])
    elif not args.summary_only:
        run(Path(args.root), args.which, details_csv, args.models_dir, args.workers, args.folder_regex)
    if details_csv.exists():
        summarize(details_csv)


if __name__ == "__main__":
    main()

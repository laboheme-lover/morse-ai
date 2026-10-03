"""実際の電気通信術受信練習用原信(和文)テキストから、合成テストデータを生成する。

`C:\\Users\\daizo\\社内研修\\電気通信術受信練習用原信（和文）` の実録音(mp3)には対応する
正解テキスト(修正版)が付属している。このテキストを種に、`generate_dataset.py`が
学習データ生成で使ったのと同じ手続き(速度カテゴリの抽選・周波数/振幅/立ち上がりの
ランダム化・タイミングジッター・ノイズ種別[clean/white/esc50]とSNRの抽選)で
連続音声(1電文=1WAV)を合成し、実録音の代わりに(あるいは実録音に加えて)
評価に使える「合成テストデータ」を作る。

学習データ生成(generate_dataset.py)は1文字=1サンプルごとに独立して速度・周波数
などを抽選するが、本スクリプトは1電文(複数文字の連続音声)=1サンプルとして扱う。
実際の無線電信は1回の送信中は速度・周波数がほぼ一定であるため(実録音でもそうで
あった)、電文単位で1回だけ抽選するのが実態に近い。ノイズ種別・SNRの抽選確率・
候補値(NOISE_TYPE_WEIGHTS, SNR_LEVELS_DB)はgenerate_dataset.pyと共通のものを
そのまま使う。

正解テキストの前処理:
- 見出し行(■...)・電文番号見出し(【電文NN】)・空行・末尾の「(以上)」を除去し、
  本文行を連結する(改行は表示上の折り返しに過ぎないため単純連結する)。
- 漢数字(〇一二三四五六七八九)は数字符号を表す表記のため、算用数字(0-9)へ変換する
  (このコーパス独自の慣行。本文は漢数字以外の漢字を含まないため、一律変換して問題ない)。
- NFKC正規化・空白除去(全角スペースは語群の区切りを示す表示上の空白で、実際の
  送信では文字間隔と地続きのため、evaluate_real_samples.normalize_ground_truthと
  同様に除去する)。
- morse_table.is_supported()で符号化できない文字(「」」「(」「)」等)を除外する
  (除外文字は metadata に記録する)。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from add_esc50_noise import mix_wav_with_esc50
from esc50_manager import Esc50Manager
from generate_dataset import NOISE_TYPE_WEIGHTS, SNR_LEVELS_DB, _sample_noise_type
from generate_morse_audio import (
    RAMP_SECONDS,
    SAMPLE_RATE,
    SPEED_RANGES,
    add_white_noise,
    resolve_generation_params,
    sequence_to_waveform,
)
from morse_table import is_supported

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_MASTER_TEXT = Path(
    r"C:\Users\daizo\社内研修\電気通信術受信練習用原信（和文）_修正版.txt"
)
DEFAULT_OUTPUT_ROOT = Path("data/real_text_synthetic_test")
DEFAULT_ESC50_ROOT = Path("data/ESC-50-master")

KANJI_DIGIT: dict[str, str] = {
    "〇": "0", "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}

HEADER_RE = re.compile(r"／音源\s*([0-9]+-[0-9]+)[)）]")
MESSAGE_MARKER_RE = re.compile(r"^【電文(\d+)】$")

METADATA_FIELDS = [
    "category",
    "message_no",
    "wav_path",
    "text_path",
    "raw_char_count",
    "synth_char_count",
    "excluded_chars",
    "duration_sec",
    "speed",
    "dot_duration",
    "frequency_hz",
    "amplitude",
    "ramp_ms",
    "timing_jitter",
    "noise_type",
    "snr_db",
    "esc50_filename",
    "esc50_category",
    "esc50_fold",
    "esc50_src_file",
    "seed",
]

TIMING_JITTER = 0.10  # generate_dataset.pyの1文字サンプル生成と同じ値


def parse_master_text(path: str | Path) -> list[tuple[str, int, str]]:
    """修正版テキストを (カテゴリ, 電文番号, 本文) のリストへ分割する。"""
    # 修正版テキストの書式(例):
    #   ■ 一・二総通：...(p.8〜9／音源 1-01)   ← カテゴリの見出し。「1-01」を取り出す
    #   【電文01】                                ← 電文の始まり
    #   一三三　アタミマルチヨウシムセン …         ← 本文(画面の幅で折り返されているだけ)
    #   (空行)
    #   【電文02】 …
    #   (以上)                                    ← 文書の終わり
    # 1行ずつ読み、見出し・電文番号・本文を振り分ける。utf-8-sig はファイル先頭のBOMを読み飛ばす指定。
    text = Path(path).read_text(encoding="utf-8-sig")
    category = ""
    messages: list[tuple[str, int, str]] = []
    current_lines: list[str] | None = None  # 今読んでいる電文の本文の行(None = 電文の外)
    current_number: int | None = None

    def flush() -> None:
        # 読みかけの電文があれば、本文の行をつなげて1件として確定する。
        # 改行は表示上の折り返しなので、間に空白を入れずにそのままつなげる。
        if current_number is not None and current_lines is not None:
            messages.append((category, current_number, "".join(current_lines)))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        header_match = HEADER_RE.search(line)
        if header_match:  # 新しいカテゴリの見出し
            flush()
            category = header_match.group(1)
            current_lines = None
            current_number = None
            continue
        marker_match = MESSAGE_MARKER_RE.match(line)
        if marker_match:  # 【電文NN】: 前の電文を確定し、新しい電文を始める
            flush()
            current_number = int(marker_match.group(1))
            current_lines = []
            continue
        if line == "" or line == "(以上)" or line == "（以上）":
            continue  # 空行と文書の終わりの印は本文に含めない
        if current_lines is not None:
            current_lines.append(line)  # 本文の行
    flush()  # 最後の電文を確定する
    return messages


def normalize_message_text(raw_text: str) -> tuple[str, list[str]]:
    """漢数字→算用数字変換・NFKC正規化・空白除去・語彙フィルタを行う。

    戻り値は (合成に使う本文, 除外された文字の一覧)。
    """
    import unicodedata

    # (1) 漢数字を算用数字に置き換える(例「一三三」→「133」)。表に無い文字はそのまま。
    converted = "".join(KANJI_DIGIT.get(ch, ch) for ch in raw_text)
    # (2) 全角の括弧「（）」などを半角にそろえる
    normalized = unicodedata.normalize("NFKC", converted)
    # (3) 語群の区切りの全角スペースを除く
    no_space = "".join(ch for ch in normalized if not ch.isspace())
    # (4) 符号で表せない文字(」・括弧)を除き、除いた文字は記録用に返す。
    #     ここで除いておかないと、音声生成(char_to_units)で未対応文字のエラーになる。
    filtered = "".join(ch for ch in no_space if is_supported(ch))
    excluded = sorted({ch for ch in no_space if not is_supported(ch)})
    return filtered, excluded


def generate_message_wav(
    text: str,
    output_path: Path,
    esc50_root: Path,
    esc50_available: bool,
    seed: int,
) -> dict:
    """1電文分の連続音声WAVを、学習データ生成と同じ手続きで合成する。"""
    rng = np.random.default_rng(seed)

    # (1) 学習データと同じ規則で生成条件を抽選する。ただし電文1つにつき1回だけ
    #     (1回の送信の中では速度・周波数・音量は一定)。
    speed = str(rng.choice(sorted(SPEED_RANGES)))
    params = resolve_generation_params(rng, speed=speed, randomize=True)

    # (2) 電文全体を1本の連続音声にする(文字の間は7単位の無音)
    waveform = sequence_to_waveform(
        text,
        dot_duration=params["dot_duration"],
        frequency_hz=params["frequency_hz"],
        amplitude=params["amplitude"],
        sample_rate=SAMPLE_RATE,
        timing_jitter=TIMING_JITTER,
        rng=rng,
        ramp_seconds=params["ramp_seconds"],
    )

    # (3) ノイズの種類を学習データと同じ比率で抽選する(なし20%・ホワイト40%・環境音40%)
    noise_type = _sample_noise_type(rng, esc50_available)
    snr_db: float | None = None
    esc50_fields = {"esc50_filename": "", "esc50_category": "", "esc50_fold": "", "esc50_src_file": ""}

    # (4) ホワイトノイズなら、ここで音声全体に足す
    if noise_type == "white":
        snr_db = float(rng.choice(SNR_LEVELS_DB))
        waveform = add_white_noise(waveform, snr_db, rng)

    # (5) -1〜+1 に収めて保存する(generate_morse_wav と同じ正規化)
    peak = float(np.max(np.abs(waveform))) if waveform.size > 0 else 0.0
    if peak > 1.0:
        waveform = waveform / peak
    waveform = waveform.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, SAMPLE_RATE, subtype="PCM_16")

    # (6) 環境音なら、保存した音声に環境音を混ぜて上書きする。
    #     評価専用のデータなので、学習では使っていないテスト用の fold5 の環境音だけを使う。
    if noise_type == "esc50":
        snr_db = float(rng.choice(SNR_LEVELS_DB))
        esc50_seed = int(rng.integers(0, 2**31 - 1))
        _, esc50_meta = mix_wav_with_esc50(
            output_path, esc50_root, split="test", snr_db=snr_db, output_path=output_path, seed=esc50_seed,
        )
        esc50_fields = {
            "esc50_filename": esc50_meta["esc50_filename"],
            "esc50_category": esc50_meta["esc50_category"],
            "esc50_fold": esc50_meta["esc50_fold"],
            "esc50_src_file": esc50_meta["esc50_src_file"],
        }
        waveform, _ = sf.read(output_path)

    duration_sec = len(waveform) / SAMPLE_RATE

    metadata = {
        "duration_sec": round(duration_sec, 2),
        "speed": speed,
        "dot_duration": round(float(params["dot_duration"]), 6),
        "frequency_hz": round(float(params["frequency_hz"]), 3),
        "amplitude": round(float(params["amplitude"]), 4),
        "ramp_ms": round(float(params["ramp_seconds"]) * 1000, 2),
        "timing_jitter": TIMING_JITTER,
        "noise_type": noise_type,
        "snr_db": snr_db if snr_db is not None else "",
        **esc50_fields,
    }
    return metadata


def generate_real_text_synthetic_dataset(
    master_text: str | Path = DEFAULT_MASTER_TEXT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    esc50_root: str | Path = DEFAULT_ESC50_ROOT,
    seed: int | None = 42,
) -> Path:
    output_root = Path(output_root)
    esc50_root = Path(esc50_root)
    esc50_available = (esc50_root / "meta" / "esc50.csv").exists()

    # (1) 修正版テキストを20電文に分ける
    messages = parse_master_text(master_text)
    print(f"電文数: {len(messages)}")

    # 全体のシード(既定42)から電文ごとのシードを順に引く。同じシードなら毎回同じデータになる。
    master_rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for category, number, raw_text in messages:
        # (2) 本文を音声にできる文字だけにそろえる
        synth_text, excluded = normalize_message_text(raw_text)
        wav_path = output_root / "wav" / category / f"{category}_電文{number:02d}.wav"
        text_path = output_root / "text" / category / f"{category}_電文{number:02d}.txt"

        # (3) 電文1つ分の音声を合成する
        seed_i = int(master_rng.integers(0, 2**31 - 1))
        meta = generate_message_wav(synth_text, wav_path, esc50_root, esc50_available, seed_i)

        # (4) 合成に使った本文をそのまま正解テキストとして保存する(音声と1対1で対応する)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(synth_text, encoding="utf-8")

        row = {
            "category": category,
            "message_no": number,
            "wav_path": str(wav_path).replace("\\", "/"),
            "text_path": str(text_path).replace("\\", "/"),
            "raw_char_count": len(raw_text),
            "synth_char_count": len(synth_text),
            "excluded_chars": "".join(excluded),
            "seed": seed_i,
            **meta,
        }
        rows.append(row)
        print(
            f"[{category} 電文{number:02d}] chars={len(synth_text)} "
            f"speed={meta['speed']} noise={meta['noise_type']} snr={meta['snr_db']} "
            f"duration={meta['duration_sec']}s excluded={''.join(excluded) or '-'}"
        )

    metadata_csv = output_root / "metadata.csv"
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"metadata.csvを書き出しました: {metadata_csv}")

    return metadata_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="実データのテキストから合成テストデータを生成する")
    parser.add_argument("--master-text", default=str(DEFAULT_MASTER_TEXT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--esc50-root", default=str(DEFAULT_ESC50_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_real_text_synthetic_dataset(args.master_text, args.output_root, args.esc50_root, args.seed)


if __name__ == "__main__":
    main()

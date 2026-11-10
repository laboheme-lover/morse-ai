"""電報形式(和文電報形式)の合成テストデータを作る。telegram_format.py の後処理を確かめる開発用データ。

audio2(電気通信術受信練習用原信)の20件は最終確認に使うため、後処理の確認はまずこのデータで行う。
本文と名あての文字は audio(実録音44件、本文だけの送信)の正解テキストから切り出し、前文の字数・局・
発信番号・受付時刻は乱数で決める。送信の順番と信号は「電気通信術の試験の方法」の和文電報形式に従う。
    [HR HR] 開始(、) → 字数 → ハツ+局番号+タナ または 局名 → 発信番号 → コ/セ+時(略体)、分(略体) → 、
    → 名あて(」で区切る)→ 本文の始まり(-..---)→ 本文(60字ごとに ..--.. と約5秒の無音)
    → 終了(...-.、まれに ...-.- が2回続く)。前後に1〜3秒の無音を付ける(実際の録音と同じく、雑音は無音にも乗る)
速度・周波数・音量・揺らぎ・雑音の抽選は、学習データと同じ規則(generate_real_text_synthetic_dataset と共通)。

出力(audio2 と同じ形。evaluate_telegram.py でそのまま評価できる):
    <output_root>/tg001/tg001_original.wav, tg001_original.txt(1行目 前文、2行目 名あて、3行目以降 本文)
    <output_root>/metadata.csv
"""

from __future__ import annotations

# ============================================================================
# このスクリプトの流れ
#   generate: 電報ごとに乱数のシードを決め、次を繰り返す
#     (1) make_telegram: 本文・名あて・前文を決め、「送る符号の列」と「正解テキスト」を作る
#     (2) 学習データと同じ規則で速度・周波数・音量などを抽選し(resolve_generation_params)、
#         sequence_waveform で符号の列を音にする。前後に1〜3秒の無音を付ける
#     (3) add_noise_and_save: 学習データと同じ比率で雑音(なし・白色雑音・環境音)を加えて保存する
#   同じシードなら毎回同じデータになる(新PCでも作り直せる)。
#
# なぜ合成データを作るのか
#   後処理(telegram_format.py)の規則を、最終確認に使う audio2 の20件に合わせ込まないため。
#   規則は「試験の方法」の形式に従って作り、まずこのデータで確かめてから audio2 で最終確認する。
# ============================================================================
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from evaluate_real_samples import normalize_ground_truth
from generate_morse_audio import (
    CHARACTER_GAP_UNITS,
    INTER_CODE_GAP_UNITS,
    SAMPLE_RATE,
    SPEED_RANGES,
    _jittered_duration,
    _silence,
    code_to_waveform,
    resolve_generation_params,
)
from generate_real_text_synthetic_dataset import TIMING_JITTER, add_noise_and_save
from morse_table import MORSE_CODE_TABLE, char_to_units
from telegram_format import (
    BODY_START_CODE,
    CLOSE_BRACKET_CODE,
    END_CODES,
    EVERY_60_CODE,
    OPEN_BRACKET_CODE,
    RYAKUTAI_DIGITS,
)

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_TEXT_ROOT = Path.home() / "OneDrive" / "audio"
DEFAULT_OUTPUT_ROOT = Path("data/telegram_synthetic_test")
DEFAULT_ESC50_ROOT = Path("data/ESC-50-master")

# 架空の局名(局名で送る電報用)
STATION_NAMES = [
    "ナハムセン", "クシロムセン", "ハコダテマルチヨウシムセン", "サガミマルナガサキムセン",
    "ミヤコマルコウベムセン", "ツルガムセン", "ヒタチマルオタルムセン", "アワジマルモジムセン",
]
# 略体の数字の符号(数字 → 符号)。RYAKUTAI_DIGITS(仮名 → 数字)の逆。5は通常の数字の符号で送る。
# 例: RYAKUTAI_DIGITS の「イ」→「1」から、数字「1」→ イ の符号 ".-" を作る
RYAKUTAI_CODE = {d: MORSE_CODE_TABLE[k] for k, d in RYAKUTAI_DIGITS.items() if k != "ヘ"}
for _d in "456":
    RYAKUTAI_CODE[_d] = MORSE_CODE_TABLE[_d]
SEPARATOR_CODE = MORSE_CODE_TABLE["。"]  # 区切り「」」(句点と同じ符号)
START_CODE = MORSE_CODE_TABLE["、"]
HR_CODES = [MORSE_CODE_TABLE["ヌ"], MORSE_CODE_TABLE["ナ"]]
EVERY_60_PAUSE_SECONDS = 5.0
# 実際の録音は、最初の信号の前と最後の信号の後に無音がある(audio2の原音では最初の信号まで約2.8秒)。
# 雑音はこの無音にも乗るため、冒頭の雑音が短点(ヘ)と読まれることがある。その状況を再現するため、
# 前後に1〜3秒の無音を付ける。
LEAD_TRAIL_SILENCE_SECONDS = (1.0, 3.0)
BODY_LINE_WIDTH = 32  # 正解テキストの本文の折り返し幅(表示上のもの)

METADATA_FIELDS = [
    "folder", "wav_path", "text_path", "body_chars", "numbered_station", "with_hr", "with_extra_end",
    "duration_sec", "speed", "dot_duration", "frequency_hz", "noise_type", "snr_db", "esc50_filename", "seed",
]


def load_source_texts(text_root: Path) -> list[str]:
    """本文・名あての材料にする文章(audio の原音の正解テキスト)。"""
    # 正規化した正解テキスト(カタカナ・数字・記号のみ)のうち、本文と名あてを切り出せる長さのものを使う
    texts = []
    for txt in sorted(text_root.glob("no*/no*_original.txt")):
        text, _ = normalize_ground_truth(txt.read_text(encoding="utf-8-sig"))
        if len(text) >= 200:
            texts.append(text)
    return texts


def make_telegram(rng: np.random.Generator, source: str) -> dict:
    """1通分の電報(送信する符号の列と、正解テキストの各行)を作る。"""
    # 本文: 90〜160字。半分の電報には括弧で囲んだ語を1つ入れる
    n = int(rng.integers(90, 161))
    # 本文の後ろに名あて用の文字(最大3語×8字)が残るよう、切り出す位置の上限を下げている
    start = int(rng.integers(0, len(source) - n - 40))
    body = list(source[start : start + n])
    if rng.random() < 0.5:
        a = int(rng.integers(5, n - 15))
        b = a + int(rng.integers(3, 9))
        body = body[:a] + ["（"] + body[a:b] + ["）"] + body[b:]
    # 名あて: 本文の後ろから2〜3語を取り、「」」で区切る
    pos = start + n + 1
    parts = []
    for _ in range(int(rng.integers(2, 4))):
        k = int(rng.integers(4, 9))
        parts.append(source[pos : pos + k])
        pos += k
    address = "」".join(parts)

    numbered = bool(rng.random() < 0.5)
    # 前文の各欄: 発信局(半分は番号「01〜99」、半分は局名)・発信番号・午前/午後(コ/セ)・時・分。字数 = 本文の文字数
    station = f"{int(rng.integers(1, 100)):02d}" if numbered else str(rng.choice(STATION_NAMES))
    serial = str(int(rng.integers(1, 1000)))
    ampm = str(rng.choice(["コ", "セ"]))
    hour, minute = str(int(rng.integers(1, 13))), str(int(rng.integers(0, 60)))
    count = str(len(body))

    # 通常の文字を符号のリストの列にする(濁音は「基本仮名+濁点」の2つの符号になる)
    def chars(text: str) -> list[list[str]]:
        return [char_to_units(c) for c in text]

    sequence: list[list[str]] = []  # 1文字 = 符号のリスト(濁音は2つ)。"PAUSE" は約5秒の無音
    with_hr = bool(rng.random() < 0.2)
    if with_hr:
        sequence += [[c] for c in HR_CODES * 2]
    # ここから送信の順番どおりに符号を並べる: [HR HR] → 開始(、)→ 前文 → 、 → 名あて → 本文の始まり → 本文 → 終了
    sequence.append([START_CODE])
    sequence += chars(count)
    if numbered:
        sequence += chars("ハツ") + chars(station) + chars("タナ")
    else:
        sequence += chars(station)
    sequence += chars(serial) + chars(ampm)
    # 受付時刻は略体の数字で送り、時と分の間は「、」で区切る(例 5時39分 → ..... 、 ...- -.)
    sequence += [[RYAKUTAI_CODE[d]] for d in hour] + [[START_CODE]] + [[RYAKUTAI_CODE[d]] for d in minute]
    sequence.append([START_CODE])  # 受付時刻のあとの区切り
    # 名あての区切り「」」は句点と同じ符号 .-.-.. で送る
    sequence += [[SEPARATOR_CODE] if c == "」" else char_to_units(c) for c in address]
    sequence.append([BODY_START_CODE])
    for i, c in enumerate(body, start=1):
        if c == "（":
            sequence.append([OPEN_BRACKET_CODE])
        elif c == "）":
            sequence.append([CLOSE_BRACKET_CODE])
        else:
            sequence.append(char_to_units(c))
        # 60字目ごとに信号と約5秒の無音(最後の字のあとには付けない)
        if i % 60 == 0 and i < len(body):
            sequence += [[EVERY_60_CODE], ["PAUSE"]]
    with_extra_end = bool(rng.random() < 0.2)
    sequence.append([END_CODES[0]])
    if with_extra_end:
        sequence += [[END_CODES[1]], [END_CODES[1]]]

    body_text = "".join(body)
    # 正解テキストは audio2 と同じ形: 1行目 前文(欄の間は全角空白、ハツ・タナは書かない)、2行目 名あて、
    # 3行目以降 本文(32字ずつ折り返す)。telegram_scoring は前文の字数から本文の始まる行を見つける
    header_line = "　".join([count, station, serial, f"{ampm}{hour}、{minute}"])
    body_lines = [body_text[i : i + BODY_LINE_WIDTH] for i in range(0, len(body_text), BODY_LINE_WIDTH)]
    return {
        "sequence": sequence,
        "text": "\n".join([header_line, address, *body_lines]) + "\n",
        "body_chars": len(body),
        "numbered_station": numbered,
        "with_hr": with_hr,
        "with_extra_end": with_extra_end,
    }


def sequence_waveform(sequence: list[list[str]], params: dict, rng: np.random.Generator) -> np.ndarray:
    """符号の列を連続音声にする(generate_morse_audio.sequence_to_waveform と同じ間隔の規則)。"""
    dot = params["dot_duration"]
    pieces: list[np.ndarray] = []
    for i, units in enumerate(sequence):
        # 60字ごとの信号のあとの約5秒の無音(文字の間の無音の代わりに入れる)
        if units == ["PAUSE"]:
            pieces.append(_silence(EVERY_60_PAUSE_SECONDS, SAMPLE_RATE))
            continue
        # 文字と文字の間は7単位、濁音の「基本仮名」と「濁点」の間は3単位の無音(どちらも±10%揺らす)
        if i > 0 and sequence[i - 1] != ["PAUSE"]:
            pieces.append(_silence(_jittered_duration(dot * CHARACTER_GAP_UNITS, TIMING_JITTER, rng), SAMPLE_RATE))
        for k, code in enumerate(units):
            if k > 0:
                pieces.append(_silence(_jittered_duration(dot * INTER_CODE_GAP_UNITS, TIMING_JITTER, rng), SAMPLE_RATE))
            pieces.append(
                code_to_waveform(
                    code, dot, params["frequency_hz"], params["amplitude"], SAMPLE_RATE, TIMING_JITTER, rng,
                    params["ramp_seconds"],
                )
            )
    return np.concatenate(pieces)


def generate(count: int, text_root: Path, output_root: Path, esc50_root: Path, seed: int) -> Path:
    sources = load_source_texts(text_root)
    esc50_available = (esc50_root / "meta" / "esc50.csv").exists()
    master = np.random.default_rng(seed)
    rows = []
    for n in range(1, count + 1):
        seed_i = int(master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(seed_i)
        # 1通ごとに材料の文章を選び、速度などの条件も1通につき1回だけ抽選する(1回の送信の中では一定)
        telegram = make_telegram(rng, sources[int(rng.integers(0, len(sources)))])
        speed = str(rng.choice(sorted(SPEED_RANGES)))
        params = resolve_generation_params(rng, speed=speed, randomize=True)
        waveform = sequence_waveform(telegram["sequence"], params, rng)
        lead, trail = (float(rng.uniform(*LEAD_TRAIL_SILENCE_SECONDS)) for _ in range(2))
        waveform = np.concatenate([_silence(lead, SAMPLE_RATE), waveform, _silence(trail, SAMPLE_RATE)])
        folder = f"tg{n:03d}"
        wav_path = output_root / folder / f"{folder}_original.wav"
        waveform, noise_type, snr_db, esc50 = add_noise_and_save(waveform, wav_path, rng, esc50_root, esc50_available)
        text_path = wav_path.with_suffix(".txt")
        text_path.write_text(telegram["text"], encoding="utf-8")
        rows.append(
            {
                "folder": folder, "wav_path": str(wav_path), "text_path": str(text_path),
                "body_chars": telegram["body_chars"], "numbered_station": telegram["numbered_station"],
                "with_hr": telegram["with_hr"], "with_extra_end": telegram["with_extra_end"],
                "duration_sec": round(len(waveform) / SAMPLE_RATE, 1), "speed": speed,
                "dot_duration": round(float(params["dot_duration"]), 4),
                "frequency_hz": round(float(params["frequency_hz"]), 1), "noise_type": noise_type,
                "snr_db": "" if snr_db is None else snr_db, "esc50_filename": esc50["esc50_filename"], "seed": seed_i,
            }
        )
        print(f"[{folder}] 本文{telegram['body_chars']}字 speed={speed} noise={noise_type} snr={snr_db}", flush=True)
    metadata = output_root / "metadata.csv"
    with open(metadata, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="電報形式の合成テストデータを作る")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--text-root", default=str(DEFAULT_TEXT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--esc50-root", default=str(DEFAULT_ESC50_ROOT))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    generate(args.count, Path(args.text_root), Path(args.output_root), Path(args.esc50_root), args.seed)


if __name__ == "__main__":
    main()

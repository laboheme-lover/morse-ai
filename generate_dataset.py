"""和文モールス信号のデータセットを一括生成するモジュール(Milestone 4)。

morse_table.MORSE_CODE_TABLE の全63クラス(和文カタカナ48・主要記号5・数字10)
について、指定サンプル数ずつWAVを生成し、train/validation/testに分割して
metadata.csvへ記録する。

ノイズ種別はclean/white/esc50からランダムに選び、white/esc50の場合はSNRも
SNR_LEVELS_DBからランダムに選ぶ。ESC-50ノイズを使う場合は、そのサンプルの
split(train/validation/test)と同じ名前のESC-50 foldからクリップを選ぶことで、
音源のリークを防ぐ(仕様書 セクション7の「ESC-50ノイズについてはfold分離を
優先する」に対応)。
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

from add_esc50_noise import mix_wav_with_esc50
from esc50_manager import Esc50Manager
from generate_morse_audio import SPEED_RANGES, generate_morse_wav
from morse_table import MORSE_CODE_TABLE

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SAMPLES_PER_CLASS = 10  # 初期テスト値。疎通確認後300程度へ増量する。

# train/validation/testへのサンプル配分比率
SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "validation": 0.15, "test": 0.15}

# 環境音・ホワイトノイズを付加する際のSNR候補[dB]
SNR_LEVELS_DB: list[float] = [20.0, 10.0, 5.0, 0.0, -5.0]

# ノイズ種別の抽選比率(clean, white, esc50)
NOISE_TYPE_WEIGHTS: dict[str, float] = {"clean": 0.2, "white": 0.4, "esc50": 0.4}

DEFAULT_OUTPUT_ROOT = Path("data/dataset")
DEFAULT_METADATA_CSV = DEFAULT_OUTPUT_ROOT / "metadata.csv"
DEFAULT_ESC50_ROOT = Path("data/ESC-50-master")

METADATA_FIELDS = [
    "filepath",
    "split",
    "label",
    "morse_code",
    "speed",
    "dot_duration",
    "frequency_hz",
    "amplitude",
    "noise_type",
    "snr_db",
    "esc50_filename",
    "esc50_category",
    "esc50_fold",
    "esc50_src_file",
]

ESC50_EMPTY_FIELDS = {"esc50_filename": "", "esc50_category": "", "esc50_fold": "", "esc50_src_file": ""}


def target_labels() -> list[str]:
    """データセット生成対象の全クラスを返す(和文カタカナ48+主要記号5+数字10=63)。"""
    return sorted(MORSE_CODE_TABLE.keys())


def assign_splits(n_samples: int, rng: np.random.Generator) -> list[str]:
    """SPLIT_RATIOSに従いn_samples件分のsplitラベルを決め、シャッフルして返す。

    端数は最大剰余法(largest remainder method)で配分する。端数が同点の
    splitが複数ある場合はrngでシャッフルしてから優先順位を決めることで、
    (例えばvalidationとtestの比率が同じ場合に)常に同じsplitへ端数が
    偏らないようにする。
    """
    splits = list(SPLIT_RATIOS.keys())
    # (1) 比率どおりの理想の件数(小数)を計算し、まず切り捨てた件数を割り当てる。
    #     例 n=300: train 210.0 / validation 45.0 / test 45.0 → ちょうど割り切れる
    #     例 n=10 : train 7.0 / validation 1.5 / test 1.5 → 切り捨てで 7/1/1、1件余る
    exact = {s: n_samples * SPLIT_RATIOS[s] for s in splits}
    counts = {s: int(np.floor(exact[s])) for s in splits}
    remainder = n_samples - sum(counts.values())  # 切り捨てで余った件数

    # (2) 余りは「小数部分が大きい split」から順に1件ずつ配る(最大剰余法)。
    #     小数部分が同じ split 同士(上の例の validation と test)は、順番を乱数で決める。
    #     固定順だと全クラスで余りが常に同じ split に行き、件数が偏るため。
    fractional = {s: round(exact[s] - counts[s], 9) for s in splits}
    ordered = sorted(splits, key=lambda s: fractional[s], reverse=True)
    priority: list[str] = []
    for _, group_iter in itertools.groupby(ordered, key=lambda s: fractional[s]):
        group = list(group_iter)
        rng.shuffle(group)
        priority.extend(group)
    for split in priority[:remainder]:
        counts[split] += 1

    # サンプル数が少なく一部splitが0件になる場合、最も多いsplitから1件融通する
    if n_samples >= len(splits):
        for split in splits:
            if counts[split] == 0:
                donor = max(counts, key=lambda s: counts[s])
                counts[donor] -= 1
                counts[split] += 1

    # (3) ["train","train",...,"validation",...,"test",...] の並びを作ってシャッフルする。
    #     i番目のサンプルがどの split に入るかを表すラベルの列になる。
    labels: list[str] = []
    for split, count in counts.items():
        labels.extend([split] * count)
    rng.shuffle(labels)
    return labels


def _sample_noise_type(rng: np.random.Generator, esc50_available: bool) -> str:
    # ノイズの種類を NOISE_TYPE_WEIGHTS の比率(なし20%・ホワイト40%・環境音40%)で抽選する。
    # ESC-50 が無い環境では環境音を候補から外し、残りの比率を合計1になるよう割り直す(なし1/3・ホワイト2/3)。
    weights = dict(NOISE_TYPE_WEIGHTS)
    if not esc50_available:
        weights.pop("esc50", None)
    types = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=float)
    probs /= probs.sum()  # 合計が1になるよう正規化
    return str(rng.choice(types, p=probs))


def _generate_one_sample(
    label: str,
    split: str,
    wav_path: Path,
    speed: str,
    noise_type: str,
    esc50_root: Path,
    rng: np.random.Generator,
) -> dict:
    # このサンプル専用のシードを、全体の乱数生成器から1つ引く。
    # 全体のシード(例 42)が同じなら、全サンプルのシードも同じ順に決まり、データセット全体が再現できる。
    seed_i = int(rng.integers(0, 2**31 - 1))

    # どのノイズでも、速度は指定のカテゴリ・周波数/音量/立ち上がりはランダム(randomize=True)・
    # タイミングの揺らぎ±10%(timing_jitter=0.10)で生成する。
    if noise_type == "clean":
        _, meta = generate_morse_wav(label, wav_path, speed=speed, randomize=True, timing_jitter=0.10, seed=seed_i)
        meta.update(ESC50_EMPTY_FIELDS)
    elif noise_type == "white":
        snr_db = float(rng.choice(SNR_LEVELS_DB))
        _, meta = generate_morse_wav(
            label, wav_path, speed=speed, randomize=True, timing_jitter=0.10,
            noise_type="white", snr_db=snr_db, seed=seed_i,
        )
        meta.update(ESC50_EMPTY_FIELDS)
    elif noise_type == "esc50":
        # 環境音は2段階: まずノイズなしの音声を作って保存し、
        # それを読み直して同じ split の fold の環境音を混ぜ、同じパスに上書きする。
        _, clean_meta = generate_morse_wav(label, wav_path, speed=speed, randomize=True, timing_jitter=0.10, seed=seed_i)
        snr_db = float(rng.choice(SNR_LEVELS_DB))
        _, esc50_meta = mix_wav_with_esc50(
            wav_path, esc50_root, split=split, snr_db=snr_db, output_path=wav_path, seed=seed_i,
        )
        meta = dict(clean_meta)
        meta.update(esc50_meta)
    else:
        raise ValueError(f"未対応のnoise_typeです: {noise_type!r}")

    meta["split"] = split
    return meta


def generate_dataset(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    metadata_csv: str | Path = DEFAULT_METADATA_CSV,
    samples_per_class: int = SAMPLES_PER_CLASS,
    esc50_root: str | Path = DEFAULT_ESC50_ROOT,
    seed: int | None = None,
    labels: list[str] | None = None,
) -> Path:
    """全クラスについてsamples_per_class件ずつWAVを生成し、metadata.csvへ記録する。

    ESC-50データセットが esc50_root に存在しない場合は noise_type='esc50' の
    抽選を除外し、代わりに clean/white のみを用いる。
    """
    rng = np.random.default_rng(seed)
    output_root = Path(output_root)
    metadata_csv = Path(metadata_csv)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    esc50_root = Path(esc50_root)
    esc50_available = (esc50_root / "meta" / "esc50.csv").exists()

    target = labels if labels is not None else target_labels()
    rows: list[dict] = []

    # 全63文字それぞれについて samples_per_class 件(本番は300件)ずつ作る。
    for label in target:
        # この文字の各サンプルを train/validation/test のどれに入れるかを先に決める
        # (文字ごとに70/15/15になるので、どの split にも全文字が同じ比率で入る)
        splits = assign_splits(samples_per_class, rng)
        for i, split in enumerate(splits):
            # 1サンプルごとに速度カテゴリ(slow/normal/fastを等確率)とノイズの種類を抽選する
            speed = str(rng.choice(sorted(SPEED_RANGES)))
            noise_type = _sample_noise_type(rng, esc50_available)

            # 保存先: data/dataset/<split>/<文字>/<文字>_<split>_<番号>.wav
            wav_path = output_root / split / label / f"{label}_{split}_{i:03d}.wav"
            meta = _generate_one_sample(label, split, wav_path, speed, noise_type, esc50_root, rng)
            # metadata.csv の列だけを取り出して1行にする(無い列は空欄)
            rows.append({field: meta.get(field, "") for field in METADATA_FIELDS})

    # 全サンプルの生成条件を metadata.csv に書く。学習(morse_dataset.py)はこのCSVを読んで
    # ファイルの場所・正解の文字・split を知る。毎回上書きするので古い行は残らない。
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return metadata_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="和文モールス信号データセットを一括生成する")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="WAV出力先ルートディレクトリ")
    parser.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV), help="metadata.csv出力パス")
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS, help="クラスごとの生成サンプル数")
    parser.add_argument("--esc50-root", default=str(DEFAULT_ESC50_ROOT), help="ESC-50データセットのルートディレクトリ")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    labels = target_labels()
    print(f"対象クラス数: {len(labels)}")
    print(f"クラスあたりサンプル数: {args.samples_per_class}")
    print(f"想定生成サンプル総数: {len(labels) * args.samples_per_class}")

    metadata_path = generate_dataset(
        output_root=args.output_root,
        metadata_csv=args.metadata_csv,
        samples_per_class=args.samples_per_class,
        esc50_root=args.esc50_root,
        seed=args.seed,
    )
    print(f"metadata.csvを書き出しました: {metadata_path}")


if __name__ == "__main__":
    main()

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
    exact = {s: n_samples * SPLIT_RATIOS[s] for s in splits}
    counts = {s: int(np.floor(exact[s])) for s in splits}
    remainder = n_samples - sum(counts.values())

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

    labels: list[str] = []
    for split, count in counts.items():
        labels.extend([split] * count)
    rng.shuffle(labels)
    return labels


def _sample_noise_type(rng: np.random.Generator, esc50_available: bool) -> str:
    weights = dict(NOISE_TYPE_WEIGHTS)
    if not esc50_available:
        weights.pop("esc50", None)
    types = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=float)
    probs /= probs.sum()
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
    seed_i = int(rng.integers(0, 2**31 - 1))

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

    for label in target:
        splits = assign_splits(samples_per_class, rng)
        for i, split in enumerate(splits):
            speed = str(rng.choice(sorted(SPEED_RANGES)))
            noise_type = _sample_noise_type(rng, esc50_available)

            wav_path = output_root / split / label / f"{label}_{split}_{i:03d}.wav"
            meta = _generate_one_sample(label, split, wav_path, speed, noise_type, esc50_root, rng)
            rows.append({field: meta.get(field, "") for field in METADATA_FIELDS})

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

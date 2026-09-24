"""ESC-50環境音データセットの読み込み・fold分割管理を行うモジュール。

ESC-50は50カテゴリ×40クリップ=2000個の5秒・44100Hz・モノラルWAVから成り、
meta/esc50.csv にクリップごとの fold(1-5)・category・src_file が記録されている。
fold は同じ元音源(src_file)由来のクリップが複数foldへ分散しないよう
あらかじめ層化されているため、fold単位で分割すれば train/validation/test 間の
音源リークを避けられる。

- train      : fold 1, 2, 3
- validation : fold 4
- test       : fold 5
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_FOLDS: tuple[int, ...] = (1, 2, 3)
VAL_FOLDS: tuple[int, ...] = (4,)
TEST_FOLDS: tuple[int, ...] = (5,)

SPLIT_FOLDS: dict[str, tuple[int, ...]] = {
    "train": TRAIN_FOLDS,
    "validation": VAL_FOLDS,
    "test": TEST_FOLDS,
}


@dataclass(frozen=True)
class Esc50Clip:
    """ESC-50の1クリップ分のメタデータ。"""

    filename: str
    fold: int
    category: str
    src_file: str
    path: Path


class Esc50Manager:
    """ESC-50データセットのメタデータを読み込み、split別にクリップを取得する。"""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        meta_path = self.root_dir / "meta" / "esc50.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"ESC-50のmetaファイルが見つかりません: {meta_path}\n"
                "https://github.com/karolpiczak/ESC-50 からダウンロードし、"
                f"{self.root_dir} に展開してください。"
            )
        self.meta = pd.read_csv(meta_path)
        self.audio_dir = self.root_dir / "audio"

    def clips_for_split(self, split: str) -> list[Esc50Clip]:
        """指定したsplit(train/validation/test)に属する全クリップを返す。"""
        if split not in SPLIT_FOLDS:
            raise ValueError(f"未対応のsplitです: {split!r} (選択肢: {sorted(SPLIT_FOLDS)})")
        folds = SPLIT_FOLDS[split]
        rows = self.meta[self.meta["fold"].isin(folds)]
        return [
            Esc50Clip(
                filename=str(row["filename"]),
                fold=int(row["fold"]),
                category=str(row["category"]),
                src_file=str(row["src_file"]),
                path=self.audio_dir / str(row["filename"]),
            )
            for _, row in rows.iterrows()
        ]

    def random_clip(self, split: str, rng: np.random.Generator | None = None) -> Esc50Clip:
        """指定したsplitからランダムに1クリップを選ぶ。"""
        rng = rng if rng is not None else np.random.default_rng()
        clips = self.clips_for_split(split)
        if not clips:
            raise ValueError(f"split={split!r} に該当するESC-50クリップがありません")
        index = int(rng.integers(0, len(clips)))
        return clips[index]

    def assert_folds_are_disjoint(self) -> None:
        """train/validation/testのfoldが重複していないことを確認する(健全性チェック)。"""
        train_set, val_set, test_set = set(TRAIN_FOLDS), set(VAL_FOLDS), set(TEST_FOLDS)
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise AssertionError("SPLIT_FOLDSのfoldがsplit間で重複しています")

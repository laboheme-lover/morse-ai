"""テスト共通のfixture定義。

ESC-50関連のテストは実データセットのダウンロードに依存せず動作するよう、
foldごとに数個ずつのダミークリップを持つ擬似ESC-50データセットをtmp_pathに
生成して利用する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

FAKE_ESC50_SAMPLE_RATE = 44100


def _make_fake_esc50_dataset(root, num_per_fold: int = 2, sample_rate: int = FAKE_ESC50_SAMPLE_RATE):
    audio_dir = root / "audio"
    meta_dir = root / "meta"
    audio_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    rows = []
    rng = np.random.default_rng(0)
    for fold in range(1, 6):
        for i in range(num_per_fold):
            filename = f"fold{fold}-clip{i}.wav"
            category = f"category_{fold}"
            src_file = f"src-{fold}-{i}"
            waveform = rng.normal(0, 0.1, size=sample_rate).astype(np.float32)  # 1秒分のノイズ
            sf.write(audio_dir / filename, waveform, sample_rate)
            rows.append(
                {
                    "filename": filename,
                    "fold": fold,
                    "target": fold,
                    "category": category,
                    "esc10": False,
                    "src_file": src_file,
                    "take": "A",
                }
            )
    pd.DataFrame(rows).to_csv(meta_dir / "esc50.csv", index=False)
    return root


@pytest.fixture
def fake_esc50_root(tmp_path):
    return _make_fake_esc50_dataset(tmp_path)

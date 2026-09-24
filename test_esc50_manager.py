"""esc50_manager.py のユニットテスト。

実際のESC-50データセットのダウンロードに依存しないよう、conftest.pyの
fake_esc50_root fixtureが生成する擬似データセットを用いる。
"""

from pathlib import Path

import numpy as np
import pytest

from esc50_manager import Esc50Manager, TEST_FOLDS, TRAIN_FOLDS, VAL_FOLDS

REAL_ESC50_ROOT = Path("data/ESC-50-master")
_real_esc50_available = (REAL_ESC50_ROOT / "meta" / "esc50.csv").exists()


def test_split_folds_are_disjoint():
    assert set(TRAIN_FOLDS).isdisjoint(VAL_FOLDS)
    assert set(TRAIN_FOLDS).isdisjoint(TEST_FOLDS)
    assert set(VAL_FOLDS).isdisjoint(TEST_FOLDS)


def test_split_folds_cover_all_five_folds():
    assert set(TRAIN_FOLDS) | set(VAL_FOLDS) | set(TEST_FOLDS) == {1, 2, 3, 4, 5}


def test_missing_meta_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Esc50Manager(tmp_path / "does_not_exist")


def test_clips_for_split_train(fake_esc50_root):
    manager = Esc50Manager(fake_esc50_root)
    clips = manager.clips_for_split("train")
    assert len(clips) == 2 * len(TRAIN_FOLDS)
    assert all(clip.fold in TRAIN_FOLDS for clip in clips)


def test_clips_for_split_validation_and_test(fake_esc50_root):
    manager = Esc50Manager(fake_esc50_root)
    val_clips = manager.clips_for_split("validation")
    test_clips = manager.clips_for_split("test")
    assert len(val_clips) == 2 * len(VAL_FOLDS)
    assert len(test_clips) == 2 * len(TEST_FOLDS)
    assert all(clip.fold in VAL_FOLDS for clip in val_clips)
    assert all(clip.fold in TEST_FOLDS for clip in test_clips)


def test_clips_for_split_unknown_raises(fake_esc50_root):
    manager = Esc50Manager(fake_esc50_root)
    with pytest.raises(ValueError):
        manager.clips_for_split("unknown")


def test_no_src_file_overlap_between_splits(fake_esc50_root):
    """train/validation/testで元音源(src_file)が重複しないことを確認する。"""
    manager = Esc50Manager(fake_esc50_root)
    train_src = {c.src_file for c in manager.clips_for_split("train")}
    val_src = {c.src_file for c in manager.clips_for_split("validation")}
    test_src = {c.src_file for c in manager.clips_for_split("test")}
    assert train_src.isdisjoint(val_src)
    assert train_src.isdisjoint(test_src)
    assert val_src.isdisjoint(test_src)


def test_random_clip_reproducible_with_seed(fake_esc50_root):
    manager = Esc50Manager(fake_esc50_root)
    clip1 = manager.random_clip("train", rng=np.random.default_rng(123))
    clip2 = manager.random_clip("train", rng=np.random.default_rng(123))
    assert clip1 == clip2


def test_random_clip_path_exists(fake_esc50_root):
    manager = Esc50Manager(fake_esc50_root)
    clip = manager.random_clip("test", rng=np.random.default_rng(0))
    assert clip.path.exists()


def test_random_clip_empty_split_raises(fake_esc50_root, monkeypatch):
    manager = Esc50Manager(fake_esc50_root)
    monkeypatch.setitem(__import__("esc50_manager").SPLIT_FOLDS, "train", (99,))
    with pytest.raises(ValueError):
        manager.random_clip("train", rng=np.random.default_rng(0))


def test_assert_folds_are_disjoint_does_not_raise(fake_esc50_root):
    manager = Esc50Manager(fake_esc50_root)
    manager.assert_folds_are_disjoint()


# --- 実データセットを用いた統合テスト ---------------------------------
# data/ESC-50-master が未配置の環境ではスキップする。


@pytest.mark.skipif(not _real_esc50_available, reason="実際のESC-50データセットが未ダウンロードです")
def test_real_esc50_split_sizes():
    manager = Esc50Manager(REAL_ESC50_ROOT)
    assert len(manager.clips_for_split("train")) == 1200
    assert len(manager.clips_for_split("validation")) == 400
    assert len(manager.clips_for_split("test")) == 400


@pytest.mark.skipif(not _real_esc50_available, reason="実際のESC-50データセットが未ダウンロードです")
def test_real_esc50_train_has_no_source_leakage_with_val_or_test():
    """train(fold1-3)とvalidation/test(fold4,5)の間で元音源(src_file)の
    リークが無いことを実データで確認する。

    公式のfold分割はクラスバランスを優先しており、val/test間(fold4-5)に
    ごく僅か(2件)のsrc_file重複が存在することが分かっているが、
    train側とは一切重ならない。
    """
    manager = Esc50Manager(REAL_ESC50_ROOT)
    train_src = {c.src_file for c in manager.clips_for_split("train")}
    val_src = {c.src_file for c in manager.clips_for_split("validation")}
    test_src = {c.src_file for c in manager.clips_for_split("test")}

    assert train_src.isdisjoint(val_src)
    assert train_src.isdisjoint(test_src)
    assert len(val_src & test_src) <= 5  # 既知の僅かな重複の許容範囲


@pytest.mark.skipif(not _real_esc50_available, reason="実際のESC-50データセットが未ダウンロードです")
def test_real_esc50_random_clip_audio_is_loadable():
    import librosa

    manager = Esc50Manager(REAL_ESC50_ROOT)
    clip = manager.random_clip("test", rng=np.random.default_rng(0))
    y, sr = librosa.load(str(clip.path), sr=16000, mono=True)
    assert sr == 16000
    assert len(y) > 0

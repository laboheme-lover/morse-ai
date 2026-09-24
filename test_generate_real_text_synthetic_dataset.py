"""generate_real_text_synthetic_dataset.py のユニットテスト。"""

import csv

import soundfile as sf

from generate_real_text_synthetic_dataset import (
    METADATA_FIELDS,
    generate_message_wav,
    generate_real_text_synthetic_dataset,
    normalize_message_text,
    parse_master_text,
)

FAKE_MASTER_TEXT = (
    "■ 一・二総通：電気通信術受信練習用原信（和文）（p.8〜9／音源 1-01）\n"
    "\n"
    "【電文01】\n"
    "一三三　アタミマルチヨウシムセン　六七　コ二、八\n"
    "センダイシカワウチコウムインシユクシヤ」カワムラマモル\n"
    "\n"
    "【電文02】\n"
    "アイウエオ（テスト）\n"
    "\n"
    "■ 三総通：電気通信術受信練習用原信（和文）（p.12〜13／音源 2-01）\n"
    "\n"
    "【電文01】\n"
    "パンダガイル\n"
    "\n"
    "（以上）\n"
)


def test_parse_master_text_splits_by_category_and_message_number():
    messages = parse_master_text_from_string(FAKE_MASTER_TEXT)
    assert [(cat, num) for cat, num, _ in messages] == [
        ("1-01", 1),
        ("1-01", 2),
        ("2-01", 1),
    ]


def test_parse_master_text_joins_body_lines_without_inserting_spaces():
    messages = parse_master_text_from_string(FAKE_MASTER_TEXT)
    body = dict(((cat, num), text) for cat, num, text in messages)
    assert body[("1-01", 1)] == (
        "一三三　アタミマルチヨウシムセン　六七　コ二、八"
        "センダイシカワウチコウムインシユクシヤ」カワムラマモル"
    )


def test_parse_master_text_excludes_header_marker_and_closing_lines():
    messages = parse_master_text_from_string(FAKE_MASTER_TEXT)
    for _cat, _num, text in messages:
        assert "■" not in text
        assert "【" not in text
        assert "以上" not in text


def parse_master_text_from_string(text, tmp_path_factory=None):
    # parse_master_textはファイルパスを受け取るため、簡易にPathへ書き出して呼ぶ。
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "master.txt"
        p.write_text(text, encoding="utf-8")
        return parse_master_text(p)


def test_normalize_message_text_converts_kanji_digits_to_ascii():
    filtered, excluded = normalize_message_text("一三三　コ二、八")
    assert filtered == "133コ2、8"
    assert excluded == []


def test_normalize_message_text_excludes_unsupported_bracket_and_parens():
    filtered, excluded = normalize_message_text("アイウ」（エオ）")
    assert filtered == "アイウエオ"
    assert set(excluded) == {"」", "(", ")"}


def test_normalize_message_text_keeps_dakuten_and_kuten():
    filtered, excluded = normalize_message_text("ガパヴ。")
    assert filtered == "ガパヴ。"
    assert excluded == []


def test_generate_message_wav_is_reproducible_with_same_seed(tmp_path):
    out1 = tmp_path / "a.wav"
    out2 = tmp_path / "b.wav"
    meta1 = generate_message_wav("アイウエオ", out1, tmp_path, esc50_available=False, seed=123)
    meta2 = generate_message_wav("アイウエオ", out2, tmp_path, esc50_available=False, seed=123)

    assert meta1 == meta2
    y1, sr1 = sf.read(out1)
    y2, sr2 = sf.read(out2)
    assert sr1 == sr2
    assert (y1 == y2).all()
    assert meta1["duration_sec"] > 0


def test_generate_message_wav_different_seed_changes_speed_or_frequency(tmp_path):
    metas = [
        generate_message_wav("アイウエオ", tmp_path / f"{i}.wav", tmp_path, esc50_available=False, seed=i)
        for i in range(8)
    ]
    # 8回抽選すればspeed(3カテゴリ)や周波数がばらつくはず
    speeds = {m["speed"] for m in metas}
    assert len(speeds) > 1


def test_generate_real_text_synthetic_dataset_end_to_end(tmp_path):
    master_path = tmp_path / "master.txt"
    master_path.write_text(FAKE_MASTER_TEXT, encoding="utf-8")
    output_root = tmp_path / "out"

    metadata_csv = generate_real_text_synthetic_dataset(
        master_text=master_path, output_root=output_root, esc50_root=tmp_path / "no_esc50", seed=1,
    )

    assert metadata_csv.exists()
    with open(metadata_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == set(METADATA_FIELDS)
    assert len(rows) == 3

    from pathlib import Path

    for row in rows:
        wp = Path(row["wav_path"])
        tp = Path(row["text_path"])
        assert wp.exists()
        assert tp.exists()
        # noise_typeがesc50抽選でもesc50_root不在なのでclean/whiteのいずれかのはず
        assert row["noise_type"] in ("clean", "white")


def test_generate_real_text_synthetic_dataset_is_reproducible(tmp_path):
    master_path = tmp_path / "master.txt"
    master_path.write_text(FAKE_MASTER_TEXT, encoding="utf-8")

    csv1 = generate_real_text_synthetic_dataset(
        master_text=master_path, output_root=tmp_path / "out1", esc50_root=tmp_path / "no_esc50", seed=7,
    )
    csv2 = generate_real_text_synthetic_dataset(
        master_text=master_path, output_root=tmp_path / "out2", esc50_root=tmp_path / "no_esc50", seed=7,
    )

    with open(csv1, encoding="utf-8") as f:
        rows1 = list(csv.DictReader(f))
    with open(csv2, encoding="utf-8") as f:
        rows2 = list(csv.DictReader(f))

    for r1, r2 in zip(rows1, rows2):
        for key in ("speed", "dot_duration", "frequency_hz", "amplitude", "noise_type", "snr_db"):
            assert r1[key] == r2[key]

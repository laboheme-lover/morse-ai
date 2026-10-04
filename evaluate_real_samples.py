"""実際のモールス信号音声サンプルで連続音声認識(predict_sequence.py)の
精度を検証するスクリプト。

以下のように番号で対応するSound(音声)/TEXT(正解テキスト)のペアを対象とする。
    <sound_dir>/Taman_<番号> [...].mp3
    <text_dir>/<番号>.txt
正解テキストが存在しない音声ファイルは評価対象から除外する。

このスクリプトはMilestone 9の本体ではなく、実データでの精度検証用の
補助スクリプトである。実録音は本プロジェクトの合成データ(固定の文字間隔・
一定の送信速度)とは大きく条件が異なるため、以下の工夫を行う:

- 正解テキストはひらがなをカタカナへ変換し(和文モールスはカタカナで送るため、
  ひらがなで書かれた正解もカタカナと同義)、NFKC正規化(全角数字等を統一)した上で、本システムの語彙
  (morse_table.MORSE_CODE_TABLE の63クラス。句点「。」を含む)に含まれる文字のみに
  絞り込む。「「」」等の未対応記号は除外した上で比較する。
- 音声中の短い「オン」区間の分布から実際の短点の長さを推定し、それに応じて
  文字境界の無音しきい値を動的に決める(固定しきい値では送信速度の異なる
  録音に対応できないため)。
- 推論前にaudio_preprocessing.preprocess_for_inference()で、検出したトーン
  周波数帯のみを通すバンドパスフィルタ(倍音・帯域外ノイズの抑圧)と振幅正規化を
  適用する(--no-preprocessで無効化可能)。実データのメルスペクトログラムを
  合成データと比較したところ、実録音は倍音成分が強く、これがCNNの誤認識
  (少数クラスへの偏り)の一因と考えられるため。
- CNNは基本仮名と濁点/半濁点を別クラスとして認識するため、
  morse_table.compose_dakuten()で通常表記(ガ、プ等)へ復元してから比較する。
- 認識結果と正解テキストの比較には、セグメンテーションのずれ(挿入・脱落)を
  許容できるdifflib.SequenceMatcherの類似度(0-1)を用いる。
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import sys
import unicodedata
from pathlib import Path

import librosa
import soundfile as sf

from audio_preprocessing import estimate_dot_seconds, preprocess_for_inference
from morse_table import compose_dakuten, is_supported
from predict_sequence import predict_sequence

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

SOUND_FILENAME_RE = re.compile(r"^Taman_(\d+)\s")

# 和文モールスには小書き仮名(拗音・促音)専用の符号が無いため、通常サイズの
# 仮名として送出・受信される。正解テキスト側もこれに合わせて正規化する。
SMALL_KANA_TO_NORMAL = str.maketrans(
    {
        "ッ": "ツ", "ャ": "ヤ", "ュ": "ユ", "ョ": "ヨ",
        "ァ": "ア", "ィ": "イ", "ゥ": "ウ", "ェ": "エ", "ォ": "オ",
    }
)

DEFAULT_SOUND_DIR = Path(r"C:\Users\daizo\OneDrive\デスクトップ\和文モールス信号サンプル\Sound")
DEFAULT_TEXT_DIR = Path(r"C:\Users\daizo\OneDrive\デスクトップ\和文モールス信号サンプル\TEXT")
DEFAULT_OUTPUT_CSV = Path("evaluation/real_samples_report.csv")


def find_matched_pairs(sound_dir: Path, text_dir: Path) -> list[tuple[int, Path, Path]]:
    """Sound側とTEXT側の両方に存在する番号のペアのみを返す(片方しか無いものは除外)。"""
    # (1) Soundフォルダの音声を「番号 → ファイル」の辞書にする。
    #     ファイル名 "Taman_41 [xxxx].mp3" から正規表現で番号 41 を取り出す。
    sound_by_number: dict[int, Path] = {}
    for path in sound_dir.iterdir():
        m = SOUND_FILENAME_RE.match(path.name)
        if m:
            sound_by_number[int(m.group(1))] = path

    # (2) TEXTフォルダの "41.txt" などを番号順に見て、同じ番号の音声があるものだけを組にする
    #     (正解テキストの無い音声・音声の無いテキストは評価から外す)
    pairs: list[tuple[int, Path, Path]] = []
    for text_path in sorted(text_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1):
        if not text_path.stem.isdigit():
            continue
        number = int(text_path.stem)
        sound_path = sound_by_number.get(number)
        if sound_path is not None:
            pairs.append((number, sound_path, text_path))
    return pairs


def hiragana_to_katakana(text: str) -> str:
    """ひらがな(ぁ〜ゖ)をカタカナへ変換する。それ以外の文字はそのまま返す。

    和文モールスはカタカナで送出するため、ひらがなで書かれた正解テキスト
    (Taman_49の49.txtが該当)を語彙外として捨ててしまうと、正解が大幅に
    欠けて評価が誤る(実際に295字のひらがなが除外され、84字しか残らなかった)。
    """
    # Unicodeではひらがな(ぁ U+3041 〜)とカタカナ(ァ U+30A1 〜)が同じ並び順で0x60離れているので、
    # 文字コードに0x60を足せば対応するカタカナになる(例 あ U+3042 → ア U+30A2)
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def normalize_ground_truth(raw_text: str) -> tuple[str, list[str]]:
    """ひらがなをカタカナへ、全角/半角を統一(NFKC)し、小書き仮名を通常サイズへ変換し、
    空白を除去した上で、本システムの語彙(濁音・半濁音の合成仮名を含む)に含まれる
    文字のみ残す。

    語彙判定にはmorse_table.is_supported()を使う。MORSE_CODE_TABLEのキーだけで
    判定すると、ガ・プ等の合成仮名(基本仮名+濁点/半濁点の2符号で送出される)が
    誤って未対応と判定されてしまうため注意する。

    戻り値は (絞り込んだ正解テキスト, 除外された文字の一覧) のタプル。
    """
    # (1) NFKC正規化: 全角数字「１」→「1」、半角カナ「ｱ」→「ア」など、見た目が同じ文字の表記を1つにそろえる
    # (2) ひらがな → カタカナ
    text = hiragana_to_katakana(unicodedata.normalize("NFKC", raw_text))
    # (3) 小書き仮名 → 通常の仮名(ッ→ツ 等)。和文モールスに小書き専用の符号は無い
    text = text.translate(SMALL_KANA_TO_NORMAL)
    # (4) 空白・改行を除く(送信には現れない)
    text = "".join(ch for ch in text if not ch.isspace())
    # (5) 符号で表せる文字だけを残し、除いた文字は一覧にして返す(何が除かれたかを後で確認するため)
    filtered = "".join(ch for ch in text if is_supported(ch))
    excluded = sorted({ch for ch in text if not is_supported(ch)})
    return filtered, excluded


def compare_texts(predicted: str, ground_truth: str) -> dict:
    """認識結果と正解テキストを文字単位で対応付け、類似度と一致・置換・挿入・脱落の内訳を返す。

    difflib.SequenceMatcherは、標準のautojunk=Trueだと200文字以上の文字列で
    「頻出する文字」を照合の対象から外してしまい、実際には8〜9割一致している
    文章でも類似度が数%にまで過小評価される。そのためautojunk=Falseで照合する。
    (実際に400文字・一致率86%の文章が、標準設定では4.3%と評価された。)
    """
    # 正解と認識結果を、文字の並びとして最もよく重なるように対応付ける(文字のずれ・抜け・余分を許容する)
    matcher = difflib.SequenceMatcher(None, ground_truth, predicted, autojunk=False)
    matched = substituted = inserted = deleted = 0
    # get_opcodes() は「正解の i1〜i2 文字目」と「認識結果の j1〜j2 文字目」がどういう関係かを順に返す
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        gt_len, pred_len = i2 - i1, j2 - j1
        if tag == "equal":  # 一致: 同じ文字が並んでいる
            matched += gt_len
        elif tag == "replace":  # 置換: 別の文字に間違えた
            # 長さが違う場合、対になる分は置換、正解側の余りは脱落、認識側の余りは挿入として数える
            paired = min(gt_len, pred_len)
            substituted += paired
            deleted += gt_len - paired
            inserted += pred_len - paired
        elif tag == "delete":  # 脱落: 正解にある文字を認識できなかった
            deleted += gt_len
        elif tag == "insert":  # 挿入: 正解に無い文字を余分に出した
            inserted += pred_len
    # すべて「正解の文字数」に対する割合にする(一致率+置換率+脱落率 = 1。挿入率は別枠)
    n = max(len(ground_truth), 1)
    return {
        "similarity": matcher.ratio(),
        "char_match_rate": matched / n,
        "char_substitution_rate": substituted / n,
        "char_insertion_rate": inserted / n,
        "char_deletion_rate": deleted / n,
    }


def evaluate_pair(
    sound_path: Path,
    text_path: Path,
    models_dir: str | Path = "models",
    top_k: int = 3,
    gap_multiplier: float = 1.8,
    min_gap_floor: float = 0.08,
    preprocess: bool = True,
    tmp_dir: Path | None = None,
) -> dict:
    """1組のSound/TEXTペアを評価する。"""
    # (1) 正解テキストを読み、認識結果と比べられる形にそろえる
    raw_text = text_path.read_text(encoding="utf-8")
    ground_truth, excluded_chars = normalize_ground_truth(raw_text)

    # (2) 音声を16kHz・モノラルで読み込む(MP3 44.1kHzステレオからの変換を含む)
    y, sr = librosa.load(str(sound_path), sr=16000, mono=True)

    # (3) 前処理ありなら、前処理後の音声を一時WAVに書いて認識に使う。短点の長さもその音声で測る
    if preprocess:
        processed_y, analysis = preprocess_for_inference(y, sr)
        measured_freq = analysis.frequency_hz
        tmp_dir = tmp_dir or Path("evaluation/_preprocessed_tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_wav = tmp_dir / f"{sound_path.stem}_preprocessed.wav"
        sf.write(tmp_wav, processed_y, sr, subtype="PCM_16")
        inference_path: Path | str = tmp_wav
        dot_estimate = estimate_dot_seconds(processed_y, sr)
    else:
        measured_freq = 0.0
        inference_path = sound_path
        dot_estimate = estimate_dot_seconds(y, sr)

    # (4) 文字の区切りのしきい値 = 短点の長さ × 1.8(アプリの自動推定と同じ規則)
    min_gap = max(min_gap_floor, dot_estimate * gap_multiplier)

    # (5) 連続音声認識を行い、「カ゛」を「ガ」のように結合して正解テキストと同じ表記にする
    result = predict_sequence(
        inference_path, models_dir=models_dir, top_k=top_k, min_char_gap_seconds=min_gap, dot_seconds=dot_estimate
    )
    predicted_text = compose_dakuten(result.text)

    if preprocess and isinstance(inference_path, Path):
        inference_path.unlink(missing_ok=True)

    # (6) 正解と比べて、一致・置換・挿入・脱落の率を求め、1行分の結果としてまとめる
    comparison = compare_texts(predicted_text, ground_truth)

    return {
        "sound_file": sound_path.name,
        "text_file": text_path.name,
        "duration_sec": round(len(y) / sr, 1),
        "preprocess": preprocess,
        "measured_frequency_hz": round(measured_freq, 1),
        "dot_estimate_ms": round(dot_estimate * 1000, 1),
        "min_gap_used_ms": round(min_gap * 1000, 1),
        "ground_truth_len": len(ground_truth),
        "predicted_len": len(predicted_text),
        "num_segments": len(result.characters),
        "similarity": round(comparison["similarity"], 4),
        "char_match_rate": round(comparison["char_match_rate"], 4),
        "char_substitution_rate": round(comparison["char_substitution_rate"], 4),
        "char_insertion_rate": round(comparison["char_insertion_rate"], 4),
        "char_deletion_rate": round(comparison["char_deletion_rate"], 4),
        "excluded_chars": "".join(excluded_chars),
        "predicted_text": predicted_text,
        "ground_truth": ground_truth,
    }


REPORT_FIELDS = [
    "sound_file",
    "text_file",
    "duration_sec",
    "preprocess",
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
    "predicted_text",
    "ground_truth",
]


def run_evaluation(
    sound_dir: str | Path = DEFAULT_SOUND_DIR,
    text_dir: str | Path = DEFAULT_TEXT_DIR,
    models_dir: str | Path = "models",
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    top_k: int = 3,
    limit: int | None = None,
    preprocess: bool = True,
) -> list[dict]:
    sound_dir = Path(sound_dir)
    text_dir = Path(text_dir)
    pairs = find_matched_pairs(sound_dir, text_dir)
    if limit is not None:
        pairs = pairs[:limit]

    print(f"評価対象ペア数: {len(pairs)} (preprocess={preprocess})")

    rows: list[dict] = []
    for number, sound_path, text_path in pairs:
        print(f"[{number}] {sound_path.name} を処理中...")
        # 1件でエラーが起きても全体を止めず、その行を空欄で記録して次に進む(数十分かかる処理のため)
        try:
            row = evaluate_pair(sound_path, text_path, models_dir=models_dir, top_k=top_k, preprocess=preprocess)
        except Exception as exc:  # noqa: BLE001 - バッチ処理を止めず記録して続行する
            print(f"  エラー: {exc}")
            row = {field: "" for field in REPORT_FIELDS}
            row["sound_file"] = sound_path.name
            row["text_file"] = text_path.name
            row["similarity"] = ""
        rows.append(row)
        sim = row.get("similarity", "")
        print(f"  similarity={sim} predicted_len={row.get('predicted_len')} ground_truth_len={row.get('ground_truth_len')}")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"レポートを保存しました: {output_csv}")

    valid = [r["similarity"] for r in rows if isinstance(r.get("similarity"), (int, float))]
    if valid:
        print(f"平均similarity: {sum(valid) / len(valid):.4f} ({len(valid)}件)")
    match_rates = [r["char_match_rate"] for r in rows if isinstance(r.get("char_match_rate"), (int, float))]
    if match_rates:
        print(f"平均 文字一致率: {sum(match_rates) / len(match_rates):.4f}")

    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="実音声サンプルで連続音声認識の精度を検証する")
    parser.add_argument("--sound-dir", default=str(DEFAULT_SOUND_DIR))
    parser.add_argument("--text-dir", default=str(DEFAULT_TEXT_DIR))
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="処理するペア数の上限(動作確認用)")
    parser.add_argument(
        "--no-preprocess", dest="preprocess", action="store_false",
        help="推論前のバンドパスフィルタ・振幅正規化(audio_preprocessing)を無効化する",
    )
    parser.set_defaults(preprocess=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_evaluation(
        args.sound_dir, args.text_dir, args.models_dir, args.output_csv, args.top_k, args.limit, args.preprocess
    )


if __name__ == "__main__":
    main()

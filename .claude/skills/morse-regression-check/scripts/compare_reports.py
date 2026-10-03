"""評価レポートCSVを前回版と比較し、回帰の有無をMarkdownで出力する。

対応するレポート:
- 実録音: evaluate_real_samples.py の出力(キー列 sound_file)
- 最終確認データ20電文: evaluate_real_text_synthetic_dataset.py の出力(キー列 category, message_no)

使い方:
    python compare_reports.py <前回のCSV> <今回のCSV>

終了コード: 0 = 全行が完全一致(回帰なし)、1 = 差分あり、2 = 入力エラー
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# 比較する指標(存在するものだけ使う)。文字一致率は挿入を含まないため、挿入率も必ず見る。
METRICS = [
    "char_match_rate",
    "similarity",
    "char_insertion_rate",
    "char_substitution_rate",
    "char_deletion_rate",
    "predicted_len",
    "ground_truth_len",
]


def detect_keys(df: pd.DataFrame) -> list[str]:
    if "sound_file" in df.columns:
        return ["sound_file"]
    if {"category", "message_no"} <= set(df.columns):
        return ["category", "message_no"]
    raise ValueError(f"キー列が見つかりません(列: {list(df.columns)})")


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    base_path, new_path = Path(sys.argv[1]), Path(sys.argv[2])
    base = pd.read_csv(base_path)
    new = pd.read_csv(new_path)
    keys = detect_keys(new)
    if detect_keys(base) != keys:
        print("前回と今回でレポートの種類が違います。")
        return 2

    metrics = [m for m in METRICS if m in base.columns and m in new.columns]
    merged = base[keys + metrics].merge(new[keys + metrics], on=keys, how="outer",
                                        suffixes=("_before", "_after"), indicator=True)

    only_before = merged[merged["_merge"] == "left_only"]
    only_after = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"].copy()

    changed_mask = pd.Series(False, index=both.index)
    for m in metrics:
        changed_mask |= both[f"{m}_before"].ne(both[f"{m}_after"])
    changed = both[changed_mask]

    print(f"# 評価レポートの比較\n")
    print(f"- 前回: `{base_path.name}`({len(base)}件)")
    print(f"- 今回: `{new_path.name}`({len(new)}件)")
    print(f"- 共通の件数: {len(both)}件 / 値が変わった件数: {len(changed)}件")
    if len(only_before) or len(only_after):
        print(f"- 片方にしか無い件数: 前回のみ {len(only_before)}件、今回のみ {len(only_after)}件"
              "(--limit の違いなどで対象が揃っていない可能性)")
    print()

    print("## 平均値(共通の件のみ)\n")
    print("| 指標 | 前回 | 今回 | 差 |")
    print("|---|---|---|---|")
    for m in metrics:
        before = both[f"{m}_before"].mean()
        after = both[f"{m}_after"].mean()
        print(f"| {m} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")
    print()

    group_cols = [c for c in ("noise_type", "snr_db") if c in new.columns]
    if group_cols and "char_match_rate" in metrics:
        print("## 条件別の文字一致率(共通の件のみ)\n")
        extra = new[keys + group_cols]
        grouped = both.merge(extra, on=keys, how="left")
        if "snr_db" in grouped:
            grouped["snr_db"] = grouped["snr_db"].astype(object).where(grouped["snr_db"].notna(), "clean")
        print("| " + " | ".join(group_cols) + " | 件数 | 前回 | 今回 | 差 |")
        print("|" + "---|" * (len(group_cols) + 4))
        for name, g in grouped.groupby(group_cols, dropna=False):
            name = name if isinstance(name, tuple) else (name,)
            before = g["char_match_rate_before"].mean()
            after = g["char_match_rate_after"].mean()
            print("| " + " | ".join(str(n) for n in name) + f" | {len(g)} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")
        print()

    if len(changed):
        print("## 値が変わった行\n")
        show = [m for m in ("char_match_rate", "char_insertion_rate", "predicted_len") if m in metrics]
        header = keys + [f"{m}(前→後)" for m in show]
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for _, row in changed.iterrows():
            cells = [str(row[k]) for k in keys]
            cells += [f"{fmt(row[f'{m}_before'])} → {fmt(row[f'{m}_after'])}" for m in show]
            print("| " + " | ".join(cells) + " |")
        print()

    same_coverage = len(only_before) == 0 and len(only_after) == 0
    print("## 判定\n")
    if len(changed) == 0 and same_coverage:
        print("**完全一致**: 全件の値が前回と同じ。この評価データの範囲では既存動作は変わっていない。")
        return 0
    if len(changed) == 0:
        print(f"**共通の{len(both)}件は完全一致**: ただし対象件数が前回と異なるため部分的な確認。"
              "最終確認には全件で再評価すること。")
        return 0
    print("**差分あり**: 上の「値が変わった行」を確認し、改善か回帰かを判断すること。")
    return 1


if __name__ == "__main__":
    sys.exit(main())

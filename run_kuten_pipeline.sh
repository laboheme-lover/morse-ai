#!/bin/bash
# 句点(。)追加に伴う63クラス版の再生成・再学習・評価を順に実行する。
set -e
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8

echo "=== [1/4] データセット生成 $(date) ==="
$PY generate_dataset.py --samples-per-class 300 --seed 42

echo "=== [2/4] 学習(CRNN 60エポック) $(date) ==="
$PY train.py --arch crnn --epochs 60 --models-dir models --weight-decay 1e-4

echo "=== [3/4] 合成データtest評価 $(date) ==="
$PY evaluate_model.py --models-dir models --output-dir evaluation

echo "=== [4/4] 実録音評価 $(date) ==="
$PY evaluate_real_samples.py --no-preprocess --models-dir models --output-csv evaluation/real_samples_report_v11_kuten.csv

echo "=== 完了 $(date) ==="

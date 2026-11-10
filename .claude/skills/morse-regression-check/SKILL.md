---
name: morse-regression-check
description: 和文モールスAI(morse-ai)で、認識に関わるコードやモデルを変えた後に、実録音44件・最終確認データ20電文・合成テストデータで再評価し、前回の評価レポートと1件ずつ比較して回帰の有無を報告する手順。morse_segmentation.py / audio_preprocessing.py / predict*.py / morse_table.py / morse_cnn.py / app.py の認識処理を変えたとき、モデルを再学習したとき、または「再評価して」「回帰確認」「前回と比べて」「精度は変わった?」「既存の動作を壊していないか」と言われたときは必ずこのスキルを使う。
---

# morse-ai 回帰チェック

認識処理やモデルを変えたら、「どこが良くなり、どこも悪くなっていないか」を数字で示す。
このプロジェクトでは、評価側のバグ(比較関数の設定、正解テキストの正規化漏れ等)で精度を
大きく見誤ったことが何度もあるため、平均値だけで判断せず、必ず前回版と1件ずつ比べる。

プロジェクトのルートは `morse-ai/`。Pythonは `./.venv/Scripts/python.exe` を使う。

## 1. どの評価を回すか決める

変更内容に応じて選ぶ。迷ったら多めに回す。

| 変更したもの | 回す評価 |
|---|---|
| 区間分割・前処理(`morse_segmentation.py`, `audio_preprocessing.py`, `predict_sequence.py`) | 実録音 + 最終確認データ |
| アプリの設定や表示だけ(`app.py`) | テスト(`test_app.py`)のみで足りることが多い |
| 符号表(`morse_table.py`)やクラス数 | データ再生成 → 再学習 → 全評価 |
| モデル構造・再学習(`morse_cnn.py`, `train.py`, `models/`) | 合成テスト + 実録音 + 最終確認データ |

どの場合も、最初に全テストを流す:

```bash
./.venv/Scripts/python.exe -m pytest -q
```

## 2. 評価を実行する

出力ファイル名には今回の変更が分かるタグを付ける(例 `v14_xxx`)。前回のファイルを上書きしない。
評価CSVは `.gitignore` 対象なので、ファイル名が履歴代わりになる。

**実録音44件**(ノイズほぼ無し。既存動作が壊れていないかの確認に最適):

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe evaluate_real_samples.py --no-preprocess --output-csv evaluation/real_samples_report_<タグ>.csv
```

- 素早く確認するなら `--limit 10`(数分)。最終確認は全44件(十数分〜)。
- 正解テキストと音声は既定で `ホーム\OneDrive\デスクトップ\和文モールス信号サンプル\` の Sound/TEXT を読む。別PCでは `--sound-dir` `--text-dir` を指定する。

**最終確認データ20電文**(ノイズ条件あり。区間分割の頑健性の確認):

```bash
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe evaluate_real_text_synthetic_dataset.py --output-csv evaluation/real_text_synthetic_report_<タグ>.csv
```

- データが無ければ先に `generate_real_text_synthetic_dataset.py --seed 42` で生成する(同じseedなら同じデータになる)。

**合成テストデータ2,835件**(モデルを再学習したときのみ):

```bash
./.venv/Scripts/python.exe evaluate_model.py --models-dir models --output-dir evaluation
```

### 長時間の評価の回し方

- 10分を超えることが多いので、Bashツールの `run_in_background` で実行し、完了通知を待つ。
  コマンドの末尾に自分で `&` を付けない(ハーネスが追跡できなくなり、完了通知が来ない)。
- 出力はリダイレクト先に書かれるが、Pythonの標準出力はバッファされるため途中経過がすぐには
  見えない。進み具合を見たい場合は `python -u` を付ける。

## 3. 前回と比較する

同梱のスクリプトで、前回のレポートと今回のレポートを比べる:

```bash
./.venv/Scripts/python.exe .claude/skills/morse-regression-check/scripts/compare_reports.py evaluation/<前回>.csv evaluation/<今回>.csv
```

- 実録音(キー: `sound_file`)と最終確認データ(キー: `category`,`message_no`)の両方に対応。
- 平均値、ノイズ種別・SNR別の文字一致率、値が変わった行、判定をMarkdownで出す。
- 終了コード 0 = 共通の件がすべて一致、1 = 差分あり。

**前回のレポートの選び方**: 同じ種類のレポートで、タグの番号が一番新しいもの。
`ls evaluation/*.csv` で確認する。2026-09時点の基準は次のとおり:
- 実録音: `real_samples_report_v11_kuten.csv`(全44件、文字一致率99.66%)
- 最終確認データ: `real_text_synthetic_report_v4_freqfix.csv`(文字一致率79.0%)

## 4. 結果を読む

- **文字一致率だけで判断しない。** 文字一致率は「正解の文字のうち当たった割合」で、正解に無い
  文字を余分に出す「挿入」は含まれない。句点「。」が「オ」として309回も余分に出ていたときも、
  文字一致率はほぼ変わらず、挿入率にだけ表れた。必ず挿入率と `predicted_len` も見る。
- **実録音が完全一致なら、ノイズの無い音声での既存動作は壊れていない**と言える。
  区間分割の改良では、実録音は完全一致・最終確認データだけ改善、が理想の形。
- **最終確認データは件数が20と少ない。** 1件の変化で平均が数%動くので、条件別(ノイズ種別×SNR)と
  変わった行を見て、どの条件で良く/悪くなったかを説明する。
- **テストデータで調整しない。** 評価結果を見てパラメータを詰めると、その評価データに過剰に
  合わせることになる。しきい値などの調整は原因を理解した上で、理屈で決める。

## 5. 報告の形

ユーザーへの報告は次の順にまとめる:

1. 何を変えたか(1〜2行)
2. テスト結果(何件中何件パス)
3. 評価ごとの前回→今回の比較表(文字一致率・挿入率。最終確認データは条件別も)
4. 値が変わった行の説明(良くなった理由/悪くなった理由)
5. 判定: 回帰なし / 回帰あり(あればその内容と次の対応案)

数値は比較スクリプトの出力から転記し、推測で書かない。一部しか評価していない場合
(`--limit` 等)はそのことを明記する。

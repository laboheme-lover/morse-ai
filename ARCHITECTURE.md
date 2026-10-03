# プログラムの全体構造

和文モールス信号AI翻訳アプリ(morse-ai)を構成するプログラムの全体像をまとめる。
各処理の細かい内容は、それぞれのソースコード中のコメントに書いてある。
開発の経緯(Milestoneごとの作業・評価結果)は [README.md](README.md) を参照。

## 1. 全体像: 4つの流れ

プログラムは、次の4つの流れに分かれる。

| 流れ | 何をするか | 主なプログラム | 主な成果物 |
|---|---|---|---|
| ① データを作る | 学習用のモールス音声をプログラムで合成する | `generate_morse_audio.py`, `generate_dataset.py` | `data/dataset/`(18,900件)と `metadata.csv` |
| ② 学習する | 音声を「音の絵」に変換し、モデルに学習させる | `morse_dataset.py`, `morse_cnn.py`, `train.py` | `models/best_model.pt`, `label_mapping.json` |
| ③ 評価する | テストデータ・実録音・最終確認データで精度を測る | `evaluate_model.py`, `evaluate_real_samples.py`, `evaluate_real_text_synthetic_dataset.py` | `evaluation/*.csv`, `*.png` |
| ④ 使う | 音声ファイルから文字列を認識し、画面に表示する | `predict.py`, `morse_segmentation.py`, `predict_sequence.py`, `audio_preprocessing.py`, `app.py` | 認識結果(画面表示) |

全体の土台として、文字と符号の対応を定義する `morse_table.py` がある。

## 2. 処理の流れ

```text
                         ┌──────────────────────┐
                         │ morse_table.py       │ 文字⇔符号の表(63文字)。全体の土台
                         └──────────┬───────────┘
                                    │
 ① データを作る                     ▼
   generate_morse_audio.py ── 1文字の音声を合成(速度・周波数・音量・揺らぎ・ホワイトノイズ)
   esc50_manager.py ───────┐
   add_esc50_noise.py ─────┴ 環境音(ESC-50)を混ぜる。学習/検証/テストで音源を分ける
   generate_dataset.py ──── 63文字×300件を一括生成 → data/dataset/ と metadata.csv
                                    │
 ② 学習する                         ▼
   morse_dataset.py ─────── WAV → メルスペクトログラム(64×424の「音の絵」)→ Tensor
   morse_cnn.py ─────────── モデルの定義(現行: MorseCRNN = CNN 4段 + 双方向GRU)
   train.py ─────────────── 学習ループ → models/best_model.pt, label_mapping.json
                                    │
            ┌───────────────────────┴───────────────────────┐
 ③ 評価する ▼                                               ▼ ④ 使う
   evaluate_model.py                              predict.py ─────────── 1文字を推論
     合成テストデータ2,835件で正解率・F1           morse_segmentation.py ─ 連続音声を1文字ずつに区切る
   evaluate_real_samples.py                       audio_preprocessing.py  周波数・速度の測定、雑音除去
     実録音44件で連続音声認識の文字一致率         predict_sequence.py ─── 区切る→1文字ずつ推論→文字列
   generate_real_text_synthetic_dataset.py         app.py ─────────────── ブラウザ画面(Streamlit)
   evaluate_real_text_synthetic_dataset.py
     最終確認データ20電文の生成と評価
```

## 3. 音声ファイルを認識するまでの流れ(アプリ実行時)

アプリに音声ファイルをアップロードしたときの処理の順番。

| 順番 | 処理 | 担当(関数) |
|---|---|---|
| 1 | ファイルを16kHz・モノラルの波形として読み込む | `app.load_audio_bytes` |
| 2 | トーン周波数・短点の長さを測る | `audio_preprocessing.analyze_audio` |
| 3 | トーン周波数±150Hzだけを通すフィルタで雑音を減らし、音量をそろえる(既定ON) | `audio_preprocessing.preprocess_for_inference` |
| 4 | 文字の区切りとみなす無音の長さを決める(短点の長さ×1.8、既定ON) | `app.resolve_min_char_gap` |
| 5 | 5msごとの音の強さから有音/無音を判定し、長い無音で1文字ずつに区切る | `morse_segmentation.split_into_characters` |
| 6 | 区切った1文字ごとに「音の絵」にしてモデルで推論する | `predict.predict_waveform` |
| 7 | 「カ」+「゛」を「ガ」に結合し、確信度の低い文字に印を付ける | `predict_sequence.build_display_units` |
| 8 | 文字列・確信度・文字ごとの詳細を表示する | `app.main` |

モデル(手順6)は「1文字分の音声」しか扱えない。連続した音声を認識できるのは、
その前の手順2〜5で音声を1文字ずつに区切っているからである。精度が落ちたときは、
「区切り方(手順2〜5)」と「1文字の判定(手順6)」のどちらの問題かを分けて考える。

## 4. プログラム一覧

### 土台

| ファイル | 役割 | 主な関数・定数 |
|---|---|---|
| `morse_table.py` | 文字⇔符号の表。カタカナ48・記号5・数字10の63文字。濁音は「基本仮名+濁点」の2符号に分解する | `MORSE_CODE_TABLE`, `char_to_units`, `is_supported`, `compose_dakuten` |

### ① データを作る

| ファイル | 役割 | 主な関数 |
|---|---|---|
| `generate_morse_audio.py` | 符号の規則どおりにトーンをオン・オフして音声を合成する(ジェネレートモデル) | `char_to_waveform`, `sequence_to_waveform`, `generate_morse_wav`, `add_white_noise` |
| `esc50_manager.py` | 環境音データセットESC-50の一覧を読み、fold(1〜3学習/4検証/5テスト)で分ける | `Esc50Manager.random_clip` |
| `add_esc50_noise.py` | 環境音を指定したSNRでモールス音声に混ぜる | `mix_with_noise`, `mix_wav_with_esc50` |
| `generate_dataset.py` | 63文字×300件を、条件をランダムに変えて一括生成し、`metadata.csv` に記録する | `generate_dataset`, `assign_splits` |
| `mp3_augment.py` | MP3圧縮の往復で音を劣化させる部品(現在はデータ生成に組み込んでいない) | `mp3_roundtrip` |

### ② 学習する

| ファイル | 役割 | 主な関数・クラス |
|---|---|---|
| `morse_dataset.py` | 音声をメルスペクトログラムに変換し、学習用にバッチで取り出せるようにする。学習・推論の前処理を一本化 | `waveform_to_mel`, `fix_length`, `MorseCharacterDataset`, `build_dataloaders` |
| `morse_cnn.py` | モデルの定義。現行は `MorseCRNN`(CNN 4段+双方向GRU、約59万パラメータ)。旧モデル `MorseCNN` も残す | `MorseCRNN`, `MorseCNN`, `model_from_config` |
| `train.py` | 学習ループ。検証正解率が最高のエポックのモデルを保存。停電時は `--resume` で再開 | `train`, `train_one_epoch` |

### ③ 評価する

| ファイル | 役割 | 対象データ |
|---|---|---|
| `evaluate_model.py` | 1文字ずつの正解率・Macro F1・混同行列・SNR別/ノイズ種別の成績 | 合成テストデータ 2,835件 |
| `evaluate_real_samples.py` | 連続音声認識の文字一致率・置換率・挿入率・脱落率。正解テキストの正規化(ひらがな→カタカナ等)も担う | 実録音 44件 |
| `generate_real_text_synthetic_dataset.py` | 電気通信術受信練習用原信(修正版テキスト)から、①と同じ手順で連続音声を合成する | 最終確認データ 20電文(生成) |
| `evaluate_real_text_synthetic_dataset.py` | 上で作ったデータで連続音声認識を評価する | 最終確認データ 20電文(評価) |

### ④ 使う

| ファイル | 役割 | 主な関数 |
|---|---|---|
| `predict.py` | 学習済みモデルを読み込み、1文字分の音声を推論して確信度・候補を返す | `load_model`, `predict_waveform` |
| `morse_segmentation.py` | 連続音声を、有音/無音の判定(ヒステリシス+デスパイク)と無音の長さで1文字ずつに区切る | `split_into_characters`, `detect_active_frames_hysteresis` |
| `audio_preprocessing.py` | トーン周波数・短点の長さの測定、バンドパスフィルタ、音量の正規化 | `analyze_audio`, `preprocess_for_inference` |
| `predict_sequence.py` | 区切る→1文字ずつ推論→文字列、の連続音声認識。濁音の結合 | `predict_sequence_from_waveform`, `build_display_units` |
| `app.py` | Streamlitのブラウザ画面。アップロード→認識→表示 | `recognize`, `main` |
| `visualize_audio.py` | 音声の波形とメルスペクトログラムを画像にする確認用ツール | `plot_waveform_and_spectrogram` |

## 5. プログラム同士の依存関係

各プログラムが、プロジェクト内のどのプログラムを読み込んで(import して)いるか。
下にあるものほど、多くの部品を組み合わせて使っている。

| プログラム | 読み込んでいるプログラム |
|---|---|
| `morse_table`, `morse_dataset`, `morse_cnn`, `esc50_manager`, `mp3_augment`, `visualize_audio` | (なし。他から使われる部品) |
| `generate_morse_audio` | `morse_table` |
| `add_esc50_noise` | `esc50_manager` |
| `morse_segmentation` | `morse_dataset` |
| `train`, `evaluate_model`, `predict` | `morse_cnn`, `morse_dataset` |
| `audio_preprocessing` | `morse_dataset`, `morse_segmentation` |
| `generate_dataset` | `generate_morse_audio`, `add_esc50_noise`, `esc50_manager`, `morse_table` |
| `predict_sequence` | `predict`, `morse_segmentation`, `morse_cnn`, `morse_dataset`, `morse_table` |
| `evaluate_real_samples` | `predict_sequence`, `audio_preprocessing`, `morse_table` |
| `generate_real_text_synthetic_dataset` | `generate_dataset`, `generate_morse_audio`, `add_esc50_noise`, `esc50_manager`, `morse_table` |
| `evaluate_real_text_synthetic_dataset` | `evaluate_real_samples`, `predict_sequence`, `audio_preprocessing`, `morse_table` |
| `app` | `predict_sequence`, `predict`, `audio_preprocessing`, `morse_segmentation`, `morse_dataset` |

学習(`train.py`)と推論(`predict.py`)が同じ `morse_dataset.py` の変換関数を使っているのが
重要な点である。変換条件が1つでも違うと、モデルは学習時と違う形の入力を受け取って精度が落ちる。

## 6. 主なデータ・成果物ファイル

| ファイル | 作るプログラム | 使うプログラム | 内容 |
|---|---|---|---|
| `data/dataset/<split>/<文字>/*.wav` | `generate_dataset.py` | `morse_dataset.py` | 学習用の合成音声 18,900件 |
| `data/dataset/metadata.csv` | `generate_dataset.py` | `morse_dataset.py`, `evaluate_model.py` | 各音声の正解・split・速度・周波数・ノイズ・SNR |
| `data/ESC-50-master/` | (外部から入手) | `esc50_manager.py` | 環境音2,000クリップ |
| `models/best_model.pt` | `train.py` | `predict.py` ほか | 学習済みの重みとモデル構成 |
| `models/label_mapping.json` | `train.py` | `predict.py` ほか | 文字⇔番号の対応、入力の長さ(424) |
| `models/last_model.pt`, `train_history.json` | `train.py` | `train.py --resume` | 学習の途中経過(再開用) |
| `data/real_text_synthetic_test/` | `generate_real_text_synthetic_dataset.py` | `evaluate_real_text_synthetic_dataset.py` | 最終確認データ20電文の音声と正解 |
| `evaluation/*.csv`, `*.png` | 評価プログラム各種 | (人が読む) | 評価結果 |

大きなデータ・モデルの重み・評価結果のCSVは `.gitignore` でgitの管理対象から外している。

## 7. テスト

`test_<プログラム名>.py` が、それぞれのプログラムの自動テストになっている(合計221件)。
`conftest.py` は、ESC-50が無い環境でもテストできるよう、小さな偽のESC-50データを用意する。

```powershell
python -m pytest -q
```

## 8. 補助ファイル

| ファイル | 役割 |
|---|---|
| `run_kuten_pipeline.sh` | 句点対応時に使った「データ生成→学習→評価」の一括実行スクリプト |
| `package_for_migration.ps1`, `verify_migration_zip.ps1` | 新PCへの移行用ZIPの作成と中身の確認 |
| `MIGRATION.md` | 新PC(GPU搭載)への移行手順 |
| `.claude/skills/morse-regression-check/` | 変更後の再評価・前回比較の手順(Claude Code用スキル)と比較スクリプト |

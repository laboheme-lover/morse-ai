# 和文モールス信号AI翻訳アプリ

このリポジトリのコードと文書は、Claude Code(Anthropic の AI)に日本語で指示して作成した。

和文モールス信号の音声ファイルをPyTorchの小型CNNで文字認識し、日本語文字列として
表示するStreamlitアプリ。ノイズ・速度・音量・タイミングの揺らぎに対するモデルの
頑健性を検証することを目的とする。

## 開発環境

- OS: Windows
- Python: 3.11
- 仮想環境: `.venv`

## 新しい PC で使う(GitHub から)

1. Python 3.11 を入れる(https://www.python.org/)
2. `git clone https://github.com/laboheme-lover/morse-ai.git` で取ってくる(または GitHub のページの Code → Download ZIP)
3. `setup.bat` をダブルクリック: `.venv` を作ってパッケージを入れ、学習済みモデル(`models/best_model.pt`)を Releases から取ってくる
4. `start_app.bat` をダブルクリック: ブラウザでアプリが開く(http://localhost:8502)

モデルを取ってこられないときは、https://github.com/laboheme-lover/morse-ai/releases から `best_model.pt` を落として `models` に置く。
学習データ・評価に使った録音・評価の結果はリポジトリに含めていない(学習データは下の手順で作り直せる)。

`main` への push と Pull Request のたびに、GitHub Actions でテストが動く(`.github/workflows/test.yml`。モデルや録音がなくても動くテストだけ)。

## セットアップ (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Milestone 1: データ生成の最小構成

### モールス符号表のテスト

```powershell
python -m pytest test_morse_table.py -v
```

### 1文字分のモールス信号WAVを生成

```powershell
python generate_morse_audio.py イ -o data/clean/イ.wav
```

### 波形・メルスペクトログラムの可視化

```powershell
python visualize_audio.py data/clean/イ.wav -o data/clean/イ.png
```

## Milestone 2: データ拡張

速度カテゴリ(slow/normal/fast)・周波数/音量のランダム化・タイミングジッター・
ホワイトノイズ+SNR制御に対応した。生成条件は`--metadata-csv`でCSVへ追記できる。

```powershell
# 速度slow、周波数・音量ランダム化、ジッター±10%
python generate_morse_audio.py イ -o data/clean/イ_slow.wav `
    --speed slow --randomize --jitter 0.10 --seed 1 `
    --metadata-csv data/clean/metadata_demo.csv

# SNR 0dBのホワイトノイズ付加
python generate_morse_audio.py ロ -o data/clean/ロ_snr0.wav `
    --speed normal --randomize --jitter 0.10 `
    --noise white --snr 0 --seed 2 `
    --metadata-csv data/clean/metadata_demo.csv
```

拡張機能のテスト:

```powershell
python -m pytest test_generate_morse_audio.py -v
```

## Milestone 3: ESC-50環境音ノイズ

[ESC-50](https://github.com/karolpiczak/ESC-50) (Piczak, 2015, CC BY-NC 3.0) を
環境音ノイズ素材として利用する。`data/ESC-50-master/`に展開して使用する
(リポジトリには含まれないため各自ダウンロードが必要)。

```powershell
# ESC-50データセットのダウンロード・展開
Invoke-WebRequest -Uri "https://github.com/karolpiczak/ESC-50/archive/master.zip" -OutFile "$env:TEMP\ESC-50-master.zip"
Expand-Archive -Path "$env:TEMP\ESC-50-master.zip" -DestinationPath "$env:TEMP\esc50_extract"
Move-Item "$env:TEMP\esc50_extract\ESC-50-master\*" "data\ESC-50-master\"
```

fold 1-3をtrain、fold 4をvalidation、fold 5をtestとして扱い、`esc50_manager.py`が
foldに基づく分割とクリップ選択を担当する。`add_esc50_noise.py`は指定したsplit・SNRで
モールス信号WAVにESC-50環境音を混合する。

```powershell
# train foldからSNR 10dBでESC-50ノイズを混合
python add_esc50_noise.py data/clean/ロ.wav -o data/noisy/ロ_esc50.wav `
    --esc50-root data/ESC-50-master --split train --snr 10 --seed 1
```

テスト(実データセットが無い環境では実データ依存分は自動的にスキップされる):

```powershell
python -m pytest test_esc50_manager.py test_add_esc50_noise.py -v
```

**注意**: ESC-50公式のfold分割はクラスバランスを優先しており、train(fold1-3)と
validation/test(fold4,5)の間で元音源(src_file)のリークは無いことを確認済みだが、
validation/test間(fold4-5)にのみごく僅か(1524件中2件)の重複が存在する。
また環境音は白色雑音と異なり非定常(突発音を含む)なため、SNRは信号のトーン区間と
ノイズ全体の平均パワー比として定義しており、瞬間的な聴感上の歪みやすさは
ノイズ素材によって変動する。

## Milestone 4: データセット生成

`generate_dataset.py`が全63クラス(和文カタカナ48・主要記号5[濁点・半濁点・長音・読点・句点]・数字10)について
指定サンプル数ずつWAVを生成し、`data/dataset/<split>/<label>/`以下に配置、
`data/dataset/metadata.csv`へ生成条件を記録する。速度・周波数・音量は
サンプルごとにランダム化し、ノイズ種別(clean/white/esc50)とSNRもランダムに
選ぶ。ESC-50ノイズを使うサンプルは、そのサンプルのsplitと同じfoldグループから
クリップを選び、音源のリークを防ぐ。ESC-50データセットが無い環境ではesc50は
自動的に除外され、clean/whiteのみで生成される。

```powershell
# 初期テスト(1クラスあたり10サンプル、全63クラスで630ファイル)
python generate_dataset.py --samples-per-class 10 --seed 42

# 本番相当(1クラスあたり300サンプル)
python generate_dataset.py --samples-per-class 300 --seed 42
```

テスト:

```powershell
python -m pytest test_generate_dataset.py -v
```

## Milestone 5: PyTorch入力

`morse_dataset.py`がWAV→メルスペクトログラム→Tensor変換とDataLoader構築を担う。
学習・推論で同じ前処理(`audio_to_mel_tensor()`)を共有する。時間方向のフレーム数は
データセット全体の最大長(`compute_max_frames()`)に合わせて自動的に決定し、
短い符号は無音相当の値で右側パディング、長い符号は切り詰める(通常は
fixed_framesをデータ全体の最大値に設定するため切り詰めは発生しない)。

```powershell
# DataLoaderを構築しバッチ形状を確認する
python morse_dataset.py --batch-size 32
```

テスト:

```powershell
python -m pytest test_morse_dataset.py -v
```

## Milestone 6: CNN学習

`morse_cnn.py`が仕様書通りの小型CNN(Conv-BN-ReLU-MaxPoolを3段、
AdaptiveAvgPool、全結合2層、チャネル数1→16→32→64)を定義する。
`train.py`がCrossEntropyLoss・Adam(lr=1e-3)で学習し、Validation Accuracyが
最も高いエポックのモデルを`models/best_model.pt`へ、クラス対応と
前処理条件(fixed_frames)を`models/label_mapping.json`へ保存する。

```powershell
# 学習(デフォルト20エポック)
python train.py --epochs 20

# エポック数・学習率などを指定
python train.py --epochs 20 --lr 1e-3 --batch-size 32
```

テスト(合成データによる少量動作確認: Lossの低下・Accuracyの上昇・
モデル/label_mapping保存を検証):

```powershell
python -m pytest test_morse_cnn.py test_train.py -v
```

### 現在の既定モデル(畳み込み+双方向GRU、`--arch crnn`)

当初の小型CNN(3段)から、容量拡大・正則化を経て、最終的に**時間方向の並びを保持する
畳み込み+双方向GRU(`MorseCRNN`)**を既定モデルとした。同一条件(dropout0.5・weight
decay1e-4・同一データ)で30エポック学習した比較結果:

| | CNN(150エポック) | **CRNN(30エポック)** |
|---|---|---|
| 検証Accuracy | 76.2% | **98.6%** |
| 合成テスト(clean) | 78.0% | **99.8%** |
| 合成テスト SNR 20dB / 10dB | 79.6% / 80.0% | **100% / 99.5%** |
| 合成テスト SNR -5dB | 66.3% | **94.6%** |
| ル・ヒ・エ(cleanの正解率) | 0% | **100%** |
| 実録音44件 文字一致率 | 81.6% | **99.6%** |

仕様書の目標(SNR 20〜10dBでAccuracy 90%以上)を満たしている。実録音の誤りは
置換0.4%・脱落0.04%で、残りは正解テキストに含まれない音声冒頭の符号などの挿入
(認識文字数が正解の約1.12倍)である。旧CNNのモデル・評価結果は
`models_cnn_v5_backup/`、`evaluation_cnn_v5_backup/`に残してある。

**句点「。」への対応(63クラス化)**: 実録音では文末に句点の符号(`.-.-..`)が
頻繁に現れるが、符号表に無いためCNNが最も近い「オ」等へ誤認識し、実際「オ」の
挿入だけで309回発生していた。符号表・データセット・モデルを63クラス(和文カタカナ48・
主要記号5[濁点・半濁点・長音・読点・句点]・数字10)へ拡張し、CRNNを再学習した
(60エポック、最良epoch 47でval_acc 98.84%)。

| | 62クラス(句点非対応) | **63クラス(句点対応)** |
|---|---|---|
| 合成テスト Accuracy | - | **99.08%**(Macro F1 99.08%) |
| 合成テスト SNR 20dB / -5dB | - | **100% / 95.3%** |
| 実録音44件 平均挿入率 | 3.73% | **2.07%** |
| 実録音44件「オ」誤挿入(合計) | 309回 | **0回** |

62クラス版のモデル・評価結果・データセットは`models_crnn62_backup/`、
`evaluation_crnn62_backup/`、`data/dataset_crnn62_backup/`に残してある。

```powershell
# CRNNで学習(既定は従来のCNN)
python train.py --arch crnn --epochs 30

# 再開時は保存済みの構成(arch)を自動で使う
python train.py --resume --epochs 30
```

## Milestone 7: 評価

`evaluate_model.py`が`models/best_model.pt`をtestデータで評価し、Accuracy・
Macro F1・混同行列・SNR別/noise_type別性能・誤認識ペア一覧を`evaluation/`へ
保存する(testデータはチューニングに使わず最終評価のみに用いる)。

```powershell
python evaluate_model.py
```

出力:

```text
evaluation/
├── test_predictions.csv   # 1件ごとの予測結果・確信度・正誤
├── snr_metrics.csv         # SNR別(cleanを含む)Accuracy・Macro F1
├── noise_metrics.csv       # noise_type別Accuracy
├── common_errors.csv       # 誤認識ペア(正解, 予測, 件数)を頻度順に列挙
├── confusion_matrix.png
└── performance_by_snr.png
```

テスト:

```powershell
python -m pytest test_evaluate_model.py -v
```

## Milestone 8: 推論

`predict.py`が1つのWAVファイルから文字を推論する。WAV→メルスペクトログラム
(morse_dataset.pyと共通の前処理)→CNN→Softmax→Top-K候補、という流れで、
予測文字・確信度・Top-K候補・(固定サイズ化前の)メルスペクトログラムを返す。

```powershell
python predict.py data/dataset/test/イ/イ_test_003.wav --top-k 3
```

Pythonから直接使う場合:

```python
from predict import predict_file
result = predict_file("path/to.wav", models_dir="models", top_k=3)
result.prediction        # 予測文字
result.confidence        # Top1の確信度
result.top_k              # [Candidate(label, confidence), ...]
result.mel_spectrogram    # numpy配列 [n_mels, time]
```

テスト:

```powershell
python -m pytest test_predict.py -v
```

## Milestone 9: 連続音声認識

`morse_segmentation.py`が連続音声のON/OFF(有音/無音)区間をフレーム単位の
RMSエネルギーで検出し、固定しきい値(デフォルト0.25秒)以上続く無音を
文字境界とみなして1文字ずつに分割する(符号内の短い間隔は同一文字として
連結される)。`predict_sequence.py`が分割された各区間をCNNで推論し、
結果を日本語文字列として連結する。

```powershell
python predict_sequence.py path/to/sequence.wav --min-char-gap 0.25 --top-k 3
```

複数文字の連続音声はテスト・デモ用に`generate_morse_audio.generate_sequence_wav()`
で生成できる(文字間隔は文字内の最大間隔より明確に長い7単位を使用)。

**既知の限界**(仕様書が明記する初期版の制約): 文字間隔の判定は固定しきい値の
ルールベースであり、ドット長を推定した動的しきい値ではない。しきい値より
短い無音しか無い場合は2文字が1文字として結合される。また強いノイズ条件
(0dB, -5dB等)では無音区間にもノイズが乗るため分割精度が下がることがある。

テスト:

```powershell
python -m pytest test_morse_segmentation.py test_predict_sequence.py -v
```

## 実録音向け前処理(補助モジュール)

`audio_preprocessing.py`は、音声のトーン周波数・短点長・文字間隔・概算打鍵速度を
測定し(`analyze_audio()`)、バンドパスフィルタ・振幅正規化・時間伸縮などの
前処理(`preprocess_for_inference()`)を提供する。UIの「文字間隔しきい値の自動推定」
は、ここで測定した短点長を使う。

**前処理の効果は確認できなかった**: 実録音44ファイルで前処理あり/なしを比較したところ、
文字一致率の差は+0.3pt(標準誤差0.4pt)で、有意な差は無かった。ただしノイズを含む音声では
区間分割の精度を大きく左右することが後に判明したため(「実データのテキストから合成テストデータを
追加生成」の節を参照)、アプリ(`app.py`)では既定でONにしている。
(当初は「実録音は倍音が強い」と考え前処理を作ったが、実録音のスペクトルを測定すると
倍音は存在せず、メルスペクトログラムで見えた広帯域の縦縞は、立ち上がりの急峻さ
(実録音は約2.3ms)によるものだった。)

- `time_stretch_to_target_dot()`は位相ボコーダの性質上、キー打鍵特有の急峻な
  オン/オフでアーティファクトが生じるため、`preprocess_for_inference()`では
  既定で無効

テスト:

```powershell
python -m pytest test_audio_preprocessing.py -v
```

## 実音声サンプルでの精度検証(補助スクリプト)

`evaluate_real_samples.py`は、実際のモールス信号録音(Sound: mp3)と正解テキスト
(TEXT: txt)が番号で対応するペアを用いて、連続音声認識(predict_sequence.py)の
精度を検証する。テキストが存在しない音声は自動的に評価対象から除外される。

```powershell
python evaluate_real_samples.py --sound-dir "<Soundフォルダ>" --text-dir "<TEXTフォルダ>"
```

実録音は本プロジェクトの合成データ(固定の文字間隔・speedカテゴリ)とは条件が
大きく異なるため、以下を行っている。

- 正解テキストはひらがなをカタカナへ変換(和文モールスはカタカナで送るため。これを
  怠ると正解の大半が「未対応文字」として落ち、評価が成り立たない)・NFKC正規化
  (全角数字等の統一)・小書き仮名の通常化(拗音・促音は和文モールスに専用符号が無い
  ため通常サイズの仮名として扱う)を行った上で、本システムの語彙(濁音・半濁音の
  合成仮名と句点「。」を含む)に含まれる文字のみに絞り込む
- 音声中の短い「オン」区間の分布から実際の短点長を推定し、文字境界の無音
  しきい値を動的に決める(固定しきい値では送信速度の異なる録音に対応できない)
- CNNは基本仮名と濁点/半濁点を別クラスとして認識するため、
  `morse_table.compose_dakuten()`で通常表記(ガ・プ等)へ復元してから比較する
- 認識結果と正解テキストの比較には、セグメンテーションのずれを許容できる
  `difflib.SequenceMatcher`の類似度(0-1)を用いる

**評価指標について**: 認識結果と正解テキストの比較は`difflib.SequenceMatcher`を
`autojunk=False`で使う。標準設定(autojunk=True)だと200文字以上の文字列で頻出文字が
照合から外れ、実際に一致率86%の400文字が4.3%と過小評価される(過去にこれで実録音の
精度を大きく誤って見積もった)。レポートには類似度に加え、文字単位の
一致率・置換率・挿入率・脱落率(`char_*_rate`)を出力する。

**これまでの主な発見(実録音44ファイル、文字一致率)**:

| 段階 | 文字一致率 |
|---|---|
| 切り出し区間の前後に余白あり(v1〜v5) | 7.7〜15.7% |
| **余白なし(修正後)** | **81.6%** |
| 余白なし + 前処理 | 81.9%(差なし) |

切り出し区間に前後20msの余白を付けると認識精度が大きく低下していた(同一の合成
クリーン音300件で、単体78.3%・余白付き22.0%・余白なし78.0%)。学習データは
1文字=1WAVで先頭・末尾に無音が無いためで、`DEFAULT_PADDING_SECONDS`は0にした。
余白を外した後の実録音の誤りは、ス→ア、ル→シ、ヒ→ユ、エ→アなど、**短点・長点の
並び順だけが違う符号の取り違え**に集中しており、合成テストでも同じ組が誤認識される
(ル・ヒ・エの正解率0%)。実録音固有のズレではなく、モデルが要素の順序を捉えにくい
ことが原因と考えられる。合成音の周波数(602Hz)・立ち上がり(2.9ms)・MP3圧縮
(96〜192kbps)を実録音並みにしても現行モデルの精度はほぼ変わらない(約78%→約80%)。

**時間方向の順序を扱うモデル**として、畳み込み+双方向GRUの`MorseCRNN`
(`morse_cnn.py`)を追加した。`python train.py --arch crnn`で学習できる
(既定は従来の`MorseCNN`)。チェックポイントには構成(`model_config.arch`)を保存して
おり、`predict.py`等は自動でその構成を復元する。

テスト:

```powershell
python -m pytest test_evaluate_real_samples.py -v
```

## 実データのテキストから合成テストデータを追加生成

プロジェクトの1つ上のフォルダの`電気通信術受信練習用原信（和文）`にある実際の電気通信術
受信練習用原信(修正・統合版テキスト、20電文)を種に、`generate_dataset.py`(学習
データ生成)と同じ手続き(速度カテゴリ抽選・周波数/振幅/立ち上がりのランダム化・
タイミングジッター10%・ノイズ種別[clean/white/esc50]とSNRの抽選)で電文単位の
連続合成音声を生成する[generate_real_text_synthetic_dataset.py](generate_real_text_synthetic_dataset.py)を追加した。
学習データ生成が1文字=1サンプルごとに独立抽選するのに対し、本スクリプトは
1電文(複数文字の連続音声)=1サンプルとして電文単位で1回だけ抽選する(実際の
無線電信は1回の送信中は速度・周波数がほぼ一定であるため)。

正解テキストの前処理では、このコーパス特有の表記(漢数字が数字符号を表す)に
対応するため漢数字(〇一二三四五六七八九)→算用数字の変換を追加している。
ESC-50は`split="test"`(fold5)のみを使い、学習データとの音源リークを防ぐ。

```powershell
python generate_real_text_synthetic_dataset.py --seed 42
```

生成データを[evaluate_real_text_synthetic_dataset.py](evaluate_real_text_synthetic_dataset.py)で評価したところ、
ノイズなし(clean)は100%だったが、SNR20dB未満のノイズ付きファイルで文字一致率が
数%まで崩壊する現象が見つかった。原因を調べたところ、**区間分割(`morse_segmentation.py`)
がノイズを含む長時間音声に弱いという、これまで表面化していなかった限界**だった:

- ホワイトノイズ/ESC-50ノイズを音声全体(符号間の無音部分も含む)に付加すると、
  無音区間のRMSエネルギーがヒステリシスしきい値(ピークの12.5%)を下回らなくなり、
  無音を検出できず音声全体が1区間として扱われてしまう(SNR20dB未満で顕著)。
  これは1文字ずつ切り出し済みの音声を分類するモデル自体の弱さではない
  (学習時のSNR -5dBでもモデル単体の分類精度は95.9%ある)。実録音・従来の合成
  デモ音声はいずれもノイズなし(または符号部分のみ)だったため、この問題は
  これまで表面化していなかった。
- 対策として`audio_preprocessing.preprocess_for_inference()`(トーン周波数帯の
  バンドパスフィルタ)を評価前に適用したところ、帯域外ノイズが大きく減り
  無音検出が復活した(文字一致率 平均27.9%→68.2%)。
- さらに、ノイズがバンドパス後も残す瞬間的なスパイクが`estimate_dot_seconds()`の
  推定値を数msまで押し下げ、過分割(1電文が数百区間に分割される)を招いていたため、
  `morse_segmentation.detect_active_frames_hysteresis()`に**デスパイク処理**
  (3フレーム=15ms未満の孤立した短い区間を直前の状態へ統合する。最速[fast]速度の
  短点0.05秒[10フレーム]には十分な安全マージンがあり誤って消さない)を追加した
  (文字一致率 平均68.2%→79.0%)。実録音10件で修正前後の結果が完全に一致することを
  確認しており、ノイズなしの既存動作への影響はない。

| 段階 | 文字一致率(全20件平均) |
|---|---|
| 前処理なし | 27.9% |
| +バンドパス前処理 | 68.2% |
| +デスパイク処理 | 79.0% |
| +周波数推定の頑健化(後述) | 79.0%(変化なし) |

**SNR -5dB向けの追加調査**: 残った最悪ケース4件を個別に調べたところ、原因は
1つではなかった。

- `estimate_center_frequency()`が最長の有音区間1つだけをFFTにかけていたため、
  たまたまその区間がノイズに支配されていると誤った周波数を掴み、以降の
  バンドパスフィルタが本来のトーンを除去してしまうケースを発見した
  (ESC-50ノイズ`clock_alarm`、SNR-5dBの1件で885Hzと誤検出、真値765Hz)。
  長い有音区間を複数(既定5個)集めてパワースペクトルを合算するよう改善し、
  単発的な汚染に対する頑健性を上げた(回帰テスト
  `test_estimate_center_frequency_survives_one_corrupted_run`を追加)。
- ただし上記のclock_alarmケースでは、長い有音区間の大半(15件中14件)が
  ノイズ側の周波数に支配されており、複数区間を合算しても改善しなかった
  (全20件の平均に変化なし)。持続的に鳴り続ける妨害音は周波数ベースの
  手法では原理的に分離が難しい。
- デスパイクのしきい値をさらに強める(3→5〜8フレーム)ことも検証したが、
  周波数推定を誤っているファイルでは逆に無音検出が完全崩壊する(有音区間が
  0になる)副作用があり、安全に強化できないと判断して見送った。

SNR -5dB(信号よりノイズの方が3倍強い)は非常に厳しい条件であり、学習時の
モデル単体の分類精度も-5dBで95.9%(100%ではない)である。この水準での
完全な区間分割は、単純なエネルギーベースの手法では限界に近いと考えられる。
文字一致率が10%台まで崩れる最悪ケースは、SNR -5dBの3件(白色ノイズ1件・
ESC-50ノイズ2件)とESC-50ノイズSNR0dBの1件の計4件のみで、それ以外
(SNR0dB以上の白色ノイズ・SNR5dB以上のESC-50ノイズ)は平均91.1%と良好である。

テスト:

```powershell
python -m pytest test_generate_real_text_synthetic_dataset.py test_morse_segmentation.py test_audio_preprocessing.py -v
```

## Milestone 10: Streamlit UI

`app.py`が音声ファイルのアップロードから認識結果の表示までを行うブラウザアプリ。

```powershell
streamlit run app.py
```

- 入力: WAV/MP3のアップロード、アップロード音声の再生、文字間隔しきい値
  (「自動推定」で音声から測定した短点長の約1.8倍を使う。既定ON。OFFにすると手動の
  固定しきい値を使う)、Top K候補数、低確信度しきい値、実録音向け前処理
  (バンドパス+振幅正規化、既定ON)のON/OFF、電報形式の解釈(自動/オフ/オン、既定は自動。
  後述の「電報形式の読み取り」を参照)
- 既定値について: 当初は自動推定・前処理とも既定OFFだったが、打鍵の速い実録音
  (`1-01_電文11.mp3`、短点約65ms)で固定しきい値0.25秒が文字間隔より長く、164文字の
  電文が3文字にまとまってしまう事象があった(自動推定ONでは0.117秒になり164文字を
  平均確信度0.99で認識)。ファイルを入れるだけで音源に合った設定になるよう、両方を既定ONにした
- 出力(上から順): 認識された日本語文字列(確信度が低い文字は橙色で強調)、認識文字数・
  平均確信度・要確認文字数・使用した文字間隔しきい値、音響特性の測定結果
  (トーン周波数・短点長・概算打鍵速度)、文字ごとの詳細一覧(予測文字・確信度・Top候補・
  該当区間・要確認)、選んだ文字のメルスペクトログラムと該当区間の音声再生
- 濁音・半濁音(ガ・パ等)は、CNNが認識した「基本仮名+濁点/半濁点」の2区間を1文字に
  結合して表示する(確信度は2区間のうち低い方)
- 認識はしきい値等を変えるたびに再計算するため、長い音声では数十秒かかることがある
  (同じ条件での再表示はキャッシュされる)

テスト(処理部分の単体テストと、AppTestによるアップロード〜結果表示の自動操作):

```powershell
python -m pytest test_app.py -v
```

**切り出し区間の余白について**: 連続音声から1文字ずつ切り出す際、区間の前後に余白
(無音)を付けると認識精度が大きく低下することが判明した(同一の合成クリーン音300件で、
単体ファイル78.3%・前後0.02秒の余白付き22.0%・余白なし78.0%)。学習データは
1文字=1WAVで先頭・末尾に無音が無いためで、`DEFAULT_PADDING_SECONDS`は0にしている。

## 加工音源での評価とヘ/ム補正

`evaluate_audio_variants.py`は、原音とその加工音源(速度 fast/normal/slow × 雑音なし/白色雑音/環境音
ESC-50 × SN比 +20〜-5dB、1音源あたり34件)の文字一致率を、アプリと同じ設定(`app.recognize`、
文字間隔の自動推定ON・前処理ON)で測る。結果はパターンごとに集計する。

```powershell
python evaluate_audio_variants.py --root <加工音源のフォルダ> --which all --workers 24 `
    --details-csv evaluation\audio2_variants_details.csv --summary-csv evaluation\audio2_variants_summary.csv
```

- 正解テキストは、送信されない見出し行を除き、漢数字を数字に直してから正規化する(`prepare_ground_truth`)
- 1件ごとに結果をCSVへ追記するので、止まっても同じコマンドで続きから再開できる

**ヘ/ム補正**(`predict_sequence.correct_single_element_by_duration`): 要素が1つだけの文字 ヘ(短点)と
ム(長点)は、1文字だけを見るモデルには速度が分からないため取り違えやすい(audio2の原音でム→ヘが22回)。
ヘ/ムと認識した区間は、音の長さが推定した短点の2倍以上ならム、未満ならヘと判定し直す。
短点の推定値が0.045〜0.25秒の範囲外(雑音で推定が崩れた場合)なら補正しない。

| データ | 補正前 | 補正後 | 改善/悪化 |
|---|---|---|---|
| audio2 原音20件 | 98.11% | 98.79% | ム→ヘの誤り22回がすべて解消 |
| audio2 原音+加工音源680件 | 81.48% | 81.71% | 139 / 1 |
| audio 原音+加工音源1,496件 | 84.08% | 84.12% | 142 / 10(悪化はSN比0dB前後以下の強い雑音のみ) |

## 電報形式の読み取り(和文電報形式)

無線の電報は「電気通信術の試験の方法」の和文電報形式で送られ、本文の前後に前文と信号が付く。

```text
開始の信号 → 前文(字数・発信局・発信番号・受付時刻)→ 区切り → 名あて → 本文の始まりの信号
→ 本文(60字ごとに信号を送り、約5秒あける)→ 終了の信号
```

モデルは符号表の63文字のどれかとして読むため、符号表に無い信号は最も近い仮名として出てくる。
audio2の原音20件で、該当する区間の短点・長点の並びを音から読み取り、実際の符号を確かめた。

| 信号 | 実際の符号 | モデルの出力 |
|---|---|---|
| 開始 | `.-.-.-`(読点と同じ)。前に HR HR が付くことがある | 、 |
| 本文の始まり | `-..---` | エ |
| 60字ごと | `..--..`(このあと約5秒の無音) | ト |
| 終了 | `...-.`(このあとに `...-.-` が続くことがある) | チ |
| 括弧（ ） | `-.--.-` / `.-..-.` | サ / ン |

また、発信局を番号で表すときは「ハツ」、発信番号の前には「タナ」が付き、受付時刻の数字は略体
(1 = `.-` など)で送られるため仮名(イ など)として出てくる。

**後処理**(`telegram_format.py`、再学習なし):

- 前文の形(数字 → 局 → 数字 → コ/セ+時刻)と、ほかの信号が1つ以上見つかったときだけ電報形式とみなす。
  無線以外の送信(本文だけ)では何もしない(本文だけの録音1,496件で、電報と判定されたものは0件)
- 前文を字数・発信局・発信番号・受付時刻に分け、略体の数字は受付時刻の欄の中だけ数字に戻す
- 信号と括弧は、音から読み取った短点・長点の並びで見分ける。60字ごとの信号は約5秒の無音が続くことも
  条件にし、括弧は開きと閉じが対になっているものだけ戻す。雑音で生じた短点(ヘ)は、前文の中と、
  最初の信号の前・最後の信号の後では飛ばす
- 本文の文字数が前文の字数と合わなければ「要確認」の警告を出す
- アプリでは「電報として読んだ結果」として、前文の各欄・名あて・本文を分けて表示する

**採点**(`telegram_scoring.py`): 「」」は句点と同じ符号なので「。」として扱い、括弧は正解に残す。
正解テキストを前文・名あて・本文に分け(本文の始まりは前文の字数から決める)、全体を1回だけ対応付けて
部分ごとに数える。

```powershell
# 開発用の電報形式の合成データ(60通)を作る。本文は実録音audioの正解テキストから切り出す
python generate_telegram_synthetic_dataset.py
# 後処理の前後を、前文・名あて・本文ごとに評価する
python evaluate_telegram.py --root data\telegram_synthetic_test --details-csv evaluation\telegram_synthetic_details.csv
python evaluate_telegram.py --root <加工音源のフォルダ> --which original --details-csv evaluation\telegram_audio2_details.csv
# 後処理だけを変えたときは、保存した区間の列に後処理をかけ直す(認識はやり直さない)
python evaluate_telegram.py --root data\telegram_synthetic_test --details-csv evaluation\telegram_synthetic_details.csv --rescore
```

結果(文字一致率、後処理なし → あり。どのデータでも悪化した音源は0件):

| データ | 全体 | 前文 | 本文 | 余分な文字(挿入率) |
|---|---|---|---|---|
| 電報形式の合成データ60通(開発用) | 80.30% → 81.82% | 72.6% → 83.1% | 80.8% → 81.4% | 44.1% → 40.1% |
| audio2 原音20件(最終確認) | 97.89% → 99.71% | 88.8% → 100% | 98.66% → 99.70% | 4.95% → 0.06% |
| audio2 加工音源660件 | 80.45% → 81.71% | — | — | — |
| ↳ 電報と判定された469件 | 96.9% → 98.7% | 88.7% → 99.7% | 97.6% → 98.6% | 5.5% → 0.3% |

- 開発は合成データで行い、audio2は最終確認に使った。audio2で見つかった問題(冒頭・末尾の雑音、
  60字ごとの信号の誤検出)は、合成データでも同じことが起きるのを確かめてから直した
- 判定されないのは、主に認識そのものが崩れる強い雑音(SN比-5〜+5dB)の音源である
- 訂正の信号(短点8つ)と、和文・欧文の切替は、試せるデータが無いため未対応

テスト:

```powershell
python -m pytest test_telegram_format.py test_telegram_scoring.py test_app.py -v
```

## プロジェクト構成

```text
morse-ai/
├── morse_table.py              # 和文モールス符号表
├── test_morse_table.py         # 符号表のユニットテスト
├── generate_morse_audio.py     # 1文字分のモールス信号WAV生成(速度/周波数/音量ランダム化・ジッター・ホワイトノイズ対応)
├── test_generate_morse_audio.py # generate_morse_audio.pyのユニットテスト
├── esc50_manager.py             # ESC-50データセットのfold分割管理
├── test_esc50_manager.py        # esc50_manager.pyのユニットテスト
├── add_esc50_noise.py           # ESC-50環境音のモールス信号WAVへの混合
├── test_add_esc50_noise.py      # add_esc50_noise.pyのユニットテスト
├── generate_dataset.py          # 全クラス一括データセット生成・metadata.csv出力
├── test_generate_dataset.py     # generate_dataset.pyのユニットテスト
├── morse_dataset.py             # WAV->メルスペクトログラム->Tensor変換・DataLoader
├── test_morse_dataset.py        # morse_dataset.pyのユニットテスト
├── morse_cnn.py                 # 小型CNNモデル定義
├── test_morse_cnn.py            # morse_cnn.pyのユニットテスト
├── train.py                     # 学習ループ・ベストモデル/label_mapping保存
├── test_train.py                # train.pyのユニットテスト
├── evaluate_model.py             # testデータでの評価(Accuracy/F1/混同行列/SNR別・noise別/誤認識ペア)
├── test_evaluate_model.py        # evaluate_model.pyのユニットテスト
├── predict.py                    # 単一WAVからの推論(予測文字/確信度/Top-K/メルスペクトログラム)
├── test_predict.py               # predict.pyのユニットテスト
├── morse_segmentation.py         # 連続音声のON/OFF検出・文字境界での分割
├── test_morse_segmentation.py    # morse_segmentation.pyのユニットテスト
├── predict_sequence.py           # 連続音声認識(分割+各文字推論+文字列連結)
├── test_predict_sequence.py      # predict_sequence.pyのユニットテスト
├── audio_preprocessing.py        # 実録音の周波数/速度測定・バンドパス/振幅正規化(補助)
├── test_audio_preprocessing.py   # audio_preprocessing.pyのユニットテスト
├── mp3_augment.py                # MP3圧縮往復による音声拡張(実録音のMP3特性の再現用)
├── test_mp3_augment.py           # mp3_augment.pyのテスト
├── app.py                        # Streamlit UI(Milestone 10)
├── test_app.py                   # app.pyのテスト(処理部分+AppTestによる画面操作)
├── evaluate_real_samples.py      # 実音声サンプルでの連続音声認識精度の検証(補助)
├── test_evaluate_real_samples.py # evaluate_real_samples.pyのユニットテスト
├── generate_real_text_synthetic_dataset.py   # 実データのテキストから合成テストデータを生成
├── evaluate_real_text_synthetic_dataset.py   # 上の合成テストデータの評価
├── evaluate_audio_variants.py    # 加工音源(速度・雑音・SN比)ごとの精度の評価
├── telegram_format.py            # 電報形式の読み取り(前文・名あて・本文に分ける後処理)
├── test_telegram_format.py       # telegram_format.pyのユニットテスト
├── telegram_scoring.py           # 電報形式の正解テキストの分割と部分ごとの採点
├── test_telegram_scoring.py      # telegram_scoring.pyのユニットテスト
├── generate_telegram_synthetic_dataset.py    # 電報形式の合成データ(開発用)の生成
├── evaluate_telegram.py          # 電報形式の後処理の前後の評価
├── ARCHITECTURE.md               # プログラムの全体構造の説明
├── conftest.py                  # テスト共通fixture(擬似ESC-50データセット)
├── visualize_audio.py           # 波形・メルスペクトログラム可視化
├── requirements.txt
├── pytest.ini
├── .gitignore
├── data/
│   ├── clean/               # ノイズなしの生成音声
│   ├── noisy/                # ノイズ付加後の音声
│   ├── dataset/              # 学習用データセット
│   └── ESC-50-master/        # 環境音ノイズ素材(ESC-50、要ダウンロード)
├── models/                   # 学習済みモデル
└── evaluation/                # 評価結果
```

仕様書のMilestone 1〜10は以上で一通り完了している。

## ライセンス

MIT ライセンス(`LICENSE`)。
学習済みモデルは、雑音の素材に ESC-50(Piczak, 2015, CC BY-NC 3.0)を使って学習した。

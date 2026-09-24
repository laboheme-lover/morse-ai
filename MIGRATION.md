# 新PC(GPU搭載)への移行手順

現在の学習パイプライン(63クラス化・CRNN 60エポック学習・評価)完了後、本プロジェクトを
GPU搭載の別PCへ移行するための手順。移行方法は **OneDriveなどのクラウドストレージ経由**、
データ(`data/`)・過去の実験バックアップ(`models_*_backup/`, `evaluation_*_backup/`)は
**そのままコピーして転送**する方針とする。

## 移行対象 / 対象外

| 項目 | 扱い | 理由 |
|---|---|---|
| `*.py`, `*.json`, `README.md`, `requirements.txt`, `pytest.ini`, `conftest.py`, `MIGRATION.md` | 転送する | ソースコード本体 |
| `models/`, `models_*_backup/` | 転送する | 学習済みチェックポイント(再学習すると変わってしまうため) |
| `evaluation/`, `evaluation_*_backup/` | 転送する | 評価結果(再現には時間がかかる) |
| `data/`(ESC-50・生成済みデータセット含む) | 転送する | ユーザー指定によりそのままコピー(再生成なし) |
| `.venv/` | **転送しない** | Windows依存のバイナリを含み、GPU版PyTorchを新PCで入れ直す必要があるため |
| `__pycache__/`, `.pytest_cache/` | **転送しない** | 再生成されるキャッシュ |
| `C:\Users\daizo\社内研修\.claude\launch.json` | 転送しない(手動で再作成) | プロジェクト外(`社内研修`直下)にあるため。内容は下記参照 |

## 手順

### 1. (旧PC) 学習パイプライン完了を待つ

現在実行中の `run_kuten_pipeline.sh`(データ生成→CRNN60エポック学習→評価→実録音評価)が
完了するまで待つ。完了は `models/train_history.json` のエポック数が60になっていること、
`evaluation/` と `evaluation/real_samples_report_v11_kuten.csv` が生成されていることで確認する。

### 2. (旧PC) 転送用ZIPを作成する

PowerShellで実行する(`.venv`と`__pycache__`系を除外):

```bash
powershell -NoProfile -Command "cd 'C:\Users\daizo\社内研修\morse-ai'; $items = Get-ChildItem -Path . -Force -Exclude '.venv','__pycache__','.pytest_cache'; Compress-Archive -Path $items.FullName -DestinationPath '..\morse-ai_transfer.zip' -CompressionLevel Optimal -Force"
```

- 出力先は `C:\Users\daizo\社内研修\morse-ai_transfer.zip`(プロジェクトの一つ上の階層)。
- 対象は約2.4GB(dataフォルダが大半)のため、圧縮に数十分かかる可能性がある。
- 完了後、念のためZIP内のファイル数を確認する:

```bash
powershell -NoProfile -Command "(Get-ChildItem '..\morse-ai_transfer.zip').Length / 1GB"
```

### 3. (旧PC) OneDriveへアップロード

作成したZIPをOneDriveの同期フォルダへ移動(またはOneDriveの共有フォルダへ手動アップロード)する。
この操作はブラウザ/エクスプローラでの手動操作を想定しているため、こちらでは行わない。

### 4. (新PC) ZIPをダウンロードして展開

OneDriveから `morse-ai_transfer.zip` をダウンロードし、任意の作業フォルダ(例: 旧PCと同じ
`社内研修\morse-ai`)へ展開する。

### 5. (新PC) Python環境を作る

Python 3.11系(旧PCは3.11.9)をインストール済みであることを確認し、プロジェクト直下で:

```bash
python -m venv .venv
```

### 6. (新PC) GPU版PyTorchを入れる

まずGPU・ドライバのCUDAバージョンを確認する:

```bash
nvidia-smi
```

出力の "CUDA Version" に応じて、[PyTorch公式](https://pytorch.org/get-started/locally/)の
案内するインデックスURLを使う。例(CUDA 12.4系の場合):

```bash
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

その後、残りの依存関係を入れる(torchは既に入っているため上書きされない):

```bash
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 7. (新PC) GPU認識とテストを確認する

```bash
./.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
./.venv/Scripts/python.exe -m pytest -q
```

`torch.cuda.is_available()` が `True` になり、テストが旧PCと同じ件数(全パス)で通ることを確認する。
`train.py`・`evaluate_model.py`・`predict.py`・`app.py` はいずれも
`"cuda" if torch.cuda.is_available() else "cpu"` で自動的にGPUを使うため、コード変更は不要。

### 8. (新PC) Streamlitプレビュー設定を再作成する

プロジェクトの一つ上の階層(例 `C:\Users\daizo\社内研修\.claude\launch.json`)に以下を作成する
(パスは新PCでの実際の配置に合わせて調整する):

```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "morse-app",
      "runtimeExecutable": "morse-ai/.venv/Scripts/python.exe",
      "runtimeArgs": ["-m", "streamlit", "run", "morse-ai/app.py", "--server.port", "8502", "--server.headless", "true", "--browser.gatherUsageStats", "false"],
      "port": 8502
    }
  ]
}
```

### 9. (新PC) 移行後の動作確認

- `models/train_history.json` のエポック数・`best_val_acc` が旧PCの完了時点と一致することを確認する。
- `evaluation/real_samples_report_v11_kuten.csv` が存在し、旧PCでの結果と一致することを確認する。
- 必要なら少量のデータで学習を1〜2エポック走らせ(`python train.py --epochs 1 --resume` 等)、
  GPU使用時のエポック当たり秒数が旧PC(CPU、約185〜235秒/エポック)より大幅に短いことを確認する。

### 10. GPU移行後に見直すとよい点(任意)

- `train.py --num-workers` を0より大きくする(Windowsでも`if __name__ == "__main__":`で
  保護されているため動作する)。CPU側の前処理(メルスペクトログラム変換)がGPU学習のボトルネックに
  なりやすいため。
- `--batch-size` を増やす(GPUメモリに余裕があれば64・128等)。ただしハイパーパラメータを
  変えると過去の結果と単純比較できなくなる点に注意。
- 今後の変更履歴を追いやすくするため、新PC移行を機にこのプロジェクトをgit管理下に置くことも
  検討できる(必須ではない)。

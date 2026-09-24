# 新PC(GPU搭載)への移行手順

本プロジェクトをGPU搭載の別PCへ移行するための手順。移行方法は **OneDriveなどの
クラウドストレージ経由**、データ(`data/`)・過去の実験バックアップ(`models_*_backup/`,
`evaluation_*_backup/`)は **そのままコピーして転送**する方針とする。

**2026-09-24更新**: プロジェクトを`git init`した(コミット `ff9f856`, `a8f5191`)ため、
コード部分は`git bundle`で転送する方式に切り替えた。データ(`data/`)・モデル
(`models/`, `models_*_backup/`)は引き続きZIPでそのままコピーする。OneDriveの
`morse-ai_migration`フォルダに以下の3ファイルを配置済み:

| ファイル | 内容 | サイズ |
|---|---|---|
| `morse-ai_repo.bundle` | コード一式(gitの全履歴) | 0.14MB |
| `morse-ai_transfer.zip` | 2026-09-22時点のデータ・モデル・過去バックアップ一式(`.venv`除く) | 1.72GB |
| `morse-ai_data_delta.zip` | `morse-ai_transfer.zip`作成後に追加した`data/real_text_synthetic_test/`のみ | 163MB |

`morse-ai_transfer.zip`作成後に変更されたのはコード(→bundleでカバー)と
`data/real_text_synthetic_test/`の追加(→delta zipでカバー)のみで、
`data/dataset`・`data/ESC-50-master`・`models/`(63クラスCRNN)の中身は
変更されていないため、`morse-ai_transfer.zip`はそのまま使える。

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

## 手順(2026-09-24版: git bundle + 差分ZIP方式)

旧PCでの準備(パッケージ作成・OneDriveへの配置)は完了済み。以下は新PC側の手順。

### 1. (新PC) OneDriveから3ファイルをダウンロードする

OneDriveの `morse-ai_migration` フォルダから、以下3ファイルを任意の作業フォルダ
(例: `C:\Users\daizo\社内研修\`)へダウンロードする。

- `morse-ai_repo.bundle`(コード一式、gitの全履歴)
- `morse-ai_transfer.zip`(データ・モデル・過去バックアップ一式。`.venv`は含まない)
- `morse-ai_data_delta.zip`(上記zip作成後に追加した`data/real_text_synthetic_test/`)

### 2. (新PC) コードをgit bundleから復元する

```bash
git clone morse-ai_repo.bundle morse-ai
cd morse-ai
```

これで`ff9f856`・`a8f5191`の2コミットを含む履歴付きのコードが復元される。
以後、通常のgitリポジトリとして扱える(旧PCへの`git remote`を張れば以後は
git経由で差分同期もできる)。

### 3. (新PC) データ・モデルを展開する

**注意**: `morse-ai_transfer.zip`は2026-09-22時点のスナップショットで、`*.py`等の
ソースファイルも含むが、これは手順2でgitから復元した現在のコード(2026-09-24
時点、句点対応後の区間分割修正等を含む)より**古い**。直接`morse-ai/`直下へ
展開すると新しいコードが古い内容で上書きされてしまうため、**一旦別フォルダへ
展開してから、データ・モデル関連のフォルダだけを移す**。

```powershell
# 一旦ステージングフォルダへ展開する(morse-aiフォルダの外)
Expand-Archive -Path morse-ai_transfer.zip -DestinationPath morse-ai_transfer_staging -Force
Expand-Archive -Path morse-ai_data_delta.zip -DestinationPath morse-ai_data_delta_staging -Force

# データ・モデル関連のフォルダだけをmorse-ai/直下へ移す(*.py等のコードは移さない)
$dataDirs = 'data','models','models_cnn_v5_backup','models_crnn30_backup','models_crnn62_backup',
            'models_v1_before_speed_fix','models_v2_before_capacity_expansion','models_v3_before_regularization',
            'evaluation','evaluation_cnn_v5_backup','evaluation_crnn30_backup','evaluation_crnn62_backup',
            'evaluation_v1_before_speed_fix'
foreach ($d in $dataDirs) {
    $src = Join-Path 'morse-ai_transfer_staging' $d
    if (Test-Path $src) { Move-Item $src (Join-Path 'morse-ai' $d) -Force }
}
Move-Item 'morse-ai_data_delta_staging\data\real_text_synthetic_test' 'morse-ai\data\real_text_synthetic_test' -Force

Remove-Item morse-ai_transfer_staging, morse-ai_data_delta_staging -Recurse -Force
```

移行後、`git status`(morse-aiフォルダ内で実行)で`.py`等のソースファイルに
変更が出ていないことを確認する(データ・モデルは`.gitignore`対象なので
`git status`には出ない)。

### 4. (新PC) Python環境を作る

Python 3.11系(旧PCは3.11.9)をインストール済みであることを確認し、プロジェクト直下で:

```bash
python -m venv .venv
```

### 5. (新PC) GPU版PyTorchを入れる

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

### 6. (新PC) GPU認識とテストを確認する

```bash
./.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
./.venv/Scripts/python.exe -m pytest -q
```

`torch.cuda.is_available()` が `True` になり、テストが旧PCと同じ件数(全パス)で通ることを確認する。
`train.py`・`evaluate_model.py`・`predict.py`・`app.py` はいずれも
`"cuda" if torch.cuda.is_available() else "cpu"` で自動的にGPUを使うため、コード変更は不要。

### 7. (新PC) Streamlitプレビュー設定を再作成する

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

### 8. (新PC) 移行後の動作確認

- `models/train_history.json` のエポック数・`best_val_acc` が旧PCの完了時点と一致することを確認する。
- `evaluation/real_samples_report_v11_kuten.csv` が存在し、旧PCでの結果と一致することを確認する。
- 必要なら少量のデータで学習を1〜2エポック走らせ(`python train.py --epochs 1 --resume` 等)、
  GPU使用時のエポック当たり秒数が旧PC(CPU、約185〜235秒/エポック)より大幅に短いことを確認する。

### 9. GPU移行後に見直すとよい点(任意)

- `train.py --num-workers` を0より大きくする(Windowsでも`if __name__ == "__main__":`で
  保護されているため動作する)。CPU側の前処理(メルスペクトログラム変換)がGPU学習のボトルネックに
  なりやすいため。
- `--batch-size` を増やす(GPUメモリに余裕があれば64・128等)。ただしハイパーパラメータを
  変えると過去の結果と単純比較できなくなる点に注意。
- 新PC側でも旧PCと同じgit履歴を引き継いでいるので、以後の変更はそのまま`git commit`で
  記録できる。旧PCへ戻す/同期する場合は`git remote add old-pc <旧PCへのパス>`等で
  つなぐか、同様に`git bundle`で差分を作って受け渡す。

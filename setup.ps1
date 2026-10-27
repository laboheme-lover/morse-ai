# 新しい PC で使えるようにする: 仮想環境(.venv)を作ってパッケージを入れ、学習済みモデルを Releases から取ってくる。
# 使い方(このフォルダで): powershell -ExecutionPolicy Bypass -File setup.ps1
# モデルは非公開リポジトリにあるので、GitHub CLI(gh)でログインしておくか、Releases のページから手で置く。
$ErrorActionPreference = 'Stop'
$repo = 'laboheme-lover/morse-ai'
Set-Location $PSScriptRoot

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    Write-Host '仮想環境を作ります(Python 3.11)'
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 が見つかりません。https://www.python.org/ から 3.11 を入れてください' }
}
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

Write-Host 'パッケージを入れます(torch は CPU 版。数分かかります)'
& $python -m pip install --upgrade pip --quiet
& $python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'パッケージを入れられませんでした' }

$model = 'models\best_model.pt'
if (Test-Path $model) {
    Write-Host "モデルはもうあります: $model"
} else {
    $gh = Get-Command gh -ErrorAction SilentlyContinue
    if (-not $gh -and (Test-Path 'C:\Program Files\GitHub CLI\gh.exe')) { $gh = Get-Command 'C:\Program Files\GitHub CLI\gh.exe' }
    if ($gh) {
        Write-Host '学習済みモデルを Releases から取ってきます'
        & $gh release download -R $repo -p best_model.pt -D models
    }
    if (-not (Test-Path $model)) {
        Write-Host "モデルを取ってこられませんでした。https://github.com/$repo/releases から best_model.pt を落として models フォルダに置いてください"
        exit 1
    }
}
Write-Host '準備ができました。start_app.bat をダブルクリックするとアプリが開きます'

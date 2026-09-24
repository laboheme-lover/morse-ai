# 新PC(GPU)移行用ZIPを作成する。.venv/__pycache__/.pytest_cache は除外する。
Set-Location -Path 'C:\Users\daizo\社内研修\morse-ai'
$items = Get-ChildItem -Path . -Force -Exclude '.venv','__pycache__','.pytest_cache'
Compress-Archive -Path $items.FullName -DestinationPath '..\morse-ai_transfer.zip' -CompressionLevel Optimal -Force
$f = Get-Item '..\morse-ai_transfer.zip'
Write-Output ("SizeGB=" + [math]::Round($f.Length/1GB, 2))
Write-Output ("Path=" + $f.FullName)

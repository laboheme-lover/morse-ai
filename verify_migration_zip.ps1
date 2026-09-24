# Avoid embedding literal Japanese path text (PowerShell 5.1 misreads non-BOM
# script files under the system codepage). Derive the zip path relatively
# from the current directory instead, since the caller already cd's here.
$projectDir = Get-Location
$parentDir = Split-Path $projectDir -Parent
$zip = Join-Path $parentDir 'morse-ai_transfer.zip'

Write-Output ("ZipPath=" + $zip)
Write-Output ("Exists=" + (Test-Path $zip))
if (Test-Path $zip) {
    $f = Get-Item $zip
    Write-Output ("SizeGB=" + [math]::Round($f.Length/1GB, 3))
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($zip)
    Write-Output ("EntryCount=" + $archive.Entries.Count)
    $topDirs = $archive.Entries | ForEach-Object { ($_.FullName -split '[\\/]')[0] } | Sort-Object -Unique
    Write-Output ("TopLevelCount=" + $topDirs.Count)
    Write-Output ("TopLevel=" + ($topDirs -join ', '))
    Write-Output ("VenvEntryCount=" + ($archive.Entries | Where-Object { $_.FullName -like '.venv/*' -or $_.FullName -like '.venv\*' }).Count)
    Write-Output ("PycacheEntryCount=" + ($archive.Entries | Where-Object { $_.FullName -like '*__pycache__*' }).Count)
    Write-Output ("HasReadme=" + (($archive.Entries | Where-Object { $_.FullName -eq 'README.md' }).Count -gt 0))
    Write-Output ("HasMigrationDoc=" + (($archive.Entries | Where-Object { $_.FullName -eq 'MIGRATION.md' }).Count -gt 0))
    $archive.Dispose()
}

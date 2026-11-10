# 議事録自動生成アプリ インストーラー（Windows）
# 使い方: install.bat をダブルクリック（または PowerShell で .\install.ps1）

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    OK: $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    注意: $msg" -ForegroundColor Yellow }

Write-Host "=============================================="
Write-Host "  議事録自動生成アプリ インストーラー"
Write-Host "=============================================="

# ── 1. Python ─────────────────────────────────────────────
Step "Python を確認"
$python = $null
foreach ($cmd in @("python", "py")) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.(\d+)" -and [int]$Matches[1] -ge 10) { $python = $cmd; break }
    }
}
if (-not $python) {
    Warn "Python 3.10 以上が見つかりません。winget でインストールします..."
    winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
    Write-Host "Python をインストールしました。PowerShell を開き直してから install.bat を再実行してください。" -ForegroundColor Yellow
    Read-Host "Enter で終了"; exit 1
}
Ok (& $python --version)

# ── 2. FFmpeg ─────────────────────────────────────────────
Step "FFmpeg を確認"
$ffmpegOk = [bool](Get-Command ffmpeg -ErrorAction SilentlyContinue)
if (-not $ffmpegOk) {
    $wingetPkgs = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path $wingetPkgs) {
        $ffmpegOk = [bool](Get-ChildItem $wingetPkgs -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1)
    }
}
if ($ffmpegOk) {
    Ok "FFmpeg は導入済み"
} else {
    Warn "FFmpeg が見つかりません。winget でインストールします..."
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    Ok "FFmpeg をインストールしました"
}

# ── 3. 仮想環境 ───────────────────────────────────────────
Step "仮想環境 (.venv) を作成"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $python -m venv .venv
    Ok "作成しました"
} else {
    Ok "既に存在します"
}
$venvPy = ".\.venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip --quiet

# ── 4. PyTorch（GPU の有無で切り替え）──────────────────────
Step "PyTorch をインストール"
$hasNvidia = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($hasNvidia) {
    Ok "NVIDIA GPU を検出 → CUDA 版をインストール"
    & $venvPy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
} else {
    Warn "NVIDIA GPU が見つかりません → CPU 版をインストール（文字起こしは遅くなります）"
    & $venvPy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# ── 5. 依存パッケージ ─────────────────────────────────────
Step "依存パッケージをインストール（数分かかります）"
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "パッケージのインストールに失敗しました" }
Ok "完了"

# ── 6. .env ───────────────────────────────────────────────
Step "設定ファイル (.env) を準備"
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Ok ".env を作成しました"
    Write-Host "    メモ帳が開くので、使う AI の API キーを1つ記入して保存してください。" -ForegroundColor Yellow
    Start-Process notepad.exe ".env" -Wait
} else {
    Ok ".env は既にあります（上書きしません）"
}

Write-Host "`n=============================================="
Write-Host "  インストール完了！" -ForegroundColor Green
Write-Host "  start.bat をダブルクリックで起動します。"
Write-Host "=============================================="
Read-Host "Enter で終了"

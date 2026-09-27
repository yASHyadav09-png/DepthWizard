<#
DepthWizard one-command demo start (Windows PowerShell 5.1+).

    .\start_demo.ps1              # backend + frontend, opens the browser
    .\start_demo.ps1 -Offline     # no internet: cached DEMs and model files only
    .\start_demo.ps1 -NoBrowser

Starts two windows (backend on http://127.0.0.1:8000, frontend on http://127.0.0.1:5173),
waits until the model is loaded, then opens the app. Close the two windows to stop.
Double-click alternative: start_demo.bat
#>
param(
    [switch]$Offline,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $root '.venv\Scripts\python.exe'
$runDir = if ($env:DW_RUN_DIR) { $env:DW_RUN_DIR } else { Join-Path $root 'runs\20260926-003630_phase2b_partial' }
$ckpt = Join-Path $runDir 'checkpoints\best.pt'

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

# ---- checks
if (-not (Test-Path $py)) { Fail "Python environment not found at $py. See README.md (Setup)." }
if (-not (Test-Path $ckpt)) { Fail "Model checkpoint not found: $ckpt. Copy the Phase 2b best.pt there (README.md, Model weights)." }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { Fail "npm (Node.js) not found on PATH." }
if (-not (Test-Path (Join-Path $root 'frontend\node_modules'))) {
    Write-Host 'Installing frontend packages (first run only)...'
    & npm --prefix (Join-Path $root 'frontend') ci
    if ($LASTEXITCODE -ne 0) { Fail 'npm ci failed.' }
}
foreach ($port in 8000, 5173) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        Fail "Port $port is already in use (is the demo already running?)."
    }
}

# ---- environment for the backend (inherited by the new window)
if ($Offline) {
    $env:DW_DEM_ALLOW_FETCH = '0'      # only cached GLO-30 DEMs (data/geo, data/dem)
    $env:HF_HUB_OFFLINE = '1'          # model files from the local Hugging Face cache
    $env:TRANSFORMERS_OFFLINE = '1'
    Write-Host 'Offline mode: no DEM downloads, no Hugging Face network access.'
}

# ---- start
$backendArgs = "-m uvicorn app.main:app --app-dir `"$(Join-Path $root 'backend')`" --host 127.0.0.1 --port 8000"
Start-Process -FilePath $py -ArgumentList $backendArgs -WorkingDirectory $root | Out-Null
Start-Process -FilePath 'cmd.exe' -ArgumentList "/k title DepthWizard frontend && npm --prefix `"$(Join-Path $root 'frontend')`" run dev -- --host 127.0.0.1 --port 5173" -WorkingDirectory $root | Out-Null

# ---- wait for the model to load (first start can take ~30 s)
Write-Host -NoNewline 'Waiting for the backend and the model'
$deadline = (Get-Date).AddSeconds(180)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 3
        if ($h.model_loaded) { $ready = $true; break }
    } catch { }
    Write-Host -NoNewline '.'
    Start-Sleep -Seconds 2
}
Write-Host ''
if (-not $ready) { Fail 'Backend did not become ready within 180 s; see its window for the error.' }
Write-Host "Backend ready: $($h.model_checkpoint) on $($h.device)" -ForegroundColor Green

$deadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $deadline) {
    try { Invoke-WebRequest -Uri 'http://127.0.0.1:5173' -UseBasicParsing -TimeoutSec 3 | Out-Null; break } catch { Start-Sleep -Seconds 1 }
}
Write-Host 'DepthWizard is running at http://127.0.0.1:5173' -ForegroundColor Green
Write-Host 'Demo inputs: sample_data\gamus_val (PNG) and sample_data\geo (GeoTIFF). Close the two windows to stop.'
if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:5173' }

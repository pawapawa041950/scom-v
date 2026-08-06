# Dev launcher for scom.
# 1. Creates/activates a local venv (for the GUI only)
# 2. Installs GUI deps (PySide6, websocket-client)
# 3. Runs the app from source
#
# The generation backend (PyTorch + ComfyUI) is NOT installed here — the app
# provisions it on first run via the bootstrap (uv-managed venv under
# userdata/backend), with progress shown in the setup dialog.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $here

if (-not (Test-Path ".\.venv")) {
    Write-Host "creating venv..." -ForegroundColor Cyan
    py -3.10 -m venv .venv
}
& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python scom.py

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
py -3.11 -m venv --clear .venv-core
.\.venv-core\Scripts\python.exe -m pip install --upgrade pip
.\.venv-core\Scripts\python.exe -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
.\.venv-core\Scripts\python.exe -m pip install -e ".[dev,tuning,data]"

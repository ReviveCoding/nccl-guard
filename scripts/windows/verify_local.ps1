$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
.\.venv-core\Scripts\python.exe -m ruff check src tests
.\.venv-core\Scripts\python.exe -m mypy src
.\.venv-core\Scripts\python.exe -m pytest
.\.venv-core\Scripts\python.exe -m build
.\.venv-core\Scripts\nccl-guard.exe doctor

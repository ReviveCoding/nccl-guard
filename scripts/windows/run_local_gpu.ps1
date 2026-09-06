$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
.\.venv-core\Scripts\nccl-guard.exe run-all --profile local-gpu --require-gpu --resume

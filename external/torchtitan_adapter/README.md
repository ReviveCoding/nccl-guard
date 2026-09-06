# TorchTitan adapter

This directory is an isolation boundary. A cluster operator creates `.venv-torchtitan`, pins an
exact TorchTitan revision compatible with its registered workload, and writes parsed metrics into
the NCCL-Guard evidence schema. Core never imports TorchTitan.

`adapter.py` writes deterministic TOML overlays, launches the pinned checkout as a subprocess, and
extracts JSON-line metrics. Example (after recording the checkout commit and preparing its isolated
environment):

```bash
.venv-torchtitan/bin/python external/torchtitan_adapter/adapter.py \
  --python .venv-torchtitan/bin/python --checkout external/torchtitan \
  --module torchtitan.train --config configs/torchtitan/frozen.toml \
  --output-log artifacts/torchtitan/run.jsonl
```

The adapter does not choose a replacement workload when model/tokenizer credentials are missing;
E11 remains `SKIP_NEEDS_CREDENTIALS` in that case.

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from nccl_guard.experiments.runner import run_experiment
from nccl_guard.parallel.strategies import (
    apply_fsdp2_bottom_up,
    apply_fsdp2_root_only,
    enable_separate_reduce_scatter_group,
    optimizer_after_sharding,
)
from nccl_guard.planner_freeze import SCIENTIFIC_FILES, freeze_planner
from nccl_guard.recovery.state import ApplicationState
from nccl_guard.status import Status


class Blocks:
    def __init__(self) -> None:
        self.blocks = [object(), object()]

    def parameters(self) -> list[torch.nn.Parameter]:
        return [torch.nn.Parameter(torch.ones(1))]


def test_fsdp_bottom_up_root_last() -> None:
    model = Blocks()
    seen = []
    with patch(
        "torch.distributed.fsdp.fully_shard", side_effect=lambda item, **kwargs: seen.append(item)
    ):
        apply_fsdp2_bottom_up(model, "mesh")
    assert seen == [*model.blocks, model]


def test_fsdp_root_only_diagnostic() -> None:
    model = Blocks()
    seen = []
    with patch(
        "torch.distributed.fsdp.fully_shard", side_effect=lambda item, **kwargs: seen.append(item)
    ):
        apply_fsdp2_root_only(model, "mesh")
    assert seen == [model]


def test_optimizer_requires_sharding_marker() -> None:
    model = Blocks()
    with pytest.raises(RuntimeError):
        optimizer_after_sharding(model, torch.optim.SGD, lr=0.1)
    model._nccl_guard_sharded = True
    assert isinstance(optimizer_after_sharding(model, torch.optim.SGD, lr=0.1), torch.optim.SGD)


def test_separate_reduce_scatter_capability_gate() -> None:
    with pytest.raises(NotImplementedError):
        enable_separate_reduce_scatter_group(object())


def test_application_state_captures_required_fields() -> None:
    model = torch.nn.Linear(2, 2)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 1)
    state = ApplicationState.capture(4, 20, "datahash", "confighash", scheduler)
    assert state.global_step == 4 and state.data_cursor == 20
    assert state.python_rng and state.numpy_rng and state.torch_rng.numel() > 0
    expected_python = random.Random()
    expected_python.setstate(state.python_rng)
    expected_python_value = expected_python.random()
    expected_numpy = np.random.RandomState()
    expected_numpy.set_state(state.numpy_rng)
    expected_numpy_value = expected_numpy.random_sample()
    optim.step()
    scheduler.step()
    random.random()
    np.random.random()
    torch.rand(1)
    state.restore_runtime(scheduler)
    assert random.random() == expected_python_value
    assert np.random.random() == expected_numpy_value
    assert scheduler.state_dict() == state.scheduler_state


def test_windows_distributed_experiments_skip_not_pass(tmp_path: Path) -> None:
    with (
        patch("platform.system", return_value="Windows"),
        patch(
            "nccl_guard.experiments.runner.collect_capabilities", return_value={"gpu": {"count": 1}}
        ),
    ):
        status, path = run_experiment(tmp_path, "E03")
    assert status is Status.SKIP_UNSUPPORTED
    assert '"SKIP_UNSUPPORTED"' in (path / "verdict.json").read_text(encoding="utf-8")


def test_capable_metadata_gate_still_cannot_claim_distributed_pass(tmp_path: Path) -> None:
    with (
        patch("platform.system", return_value="Linux"),
        patch(
            "nccl_guard.experiments.runner.collect_capabilities",
            return_value={"gpu": {"count": 8}, "torch": "test", "torch_cuda_build": "test"},
        ),
    ):
        status, path = run_experiment(tmp_path, "E03")
    assert status is Status.FAIL
    verdict = (path / "verdict.json").read_text(encoding="utf-8")
    assert "metadata gate cannot qualify" in verdict


def test_final_freeze_cannot_be_overwritten_after_start(tmp_path: Path) -> None:
    for name in SCIENTIFIC_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    manifest = freeze_planner(tmp_path)
    text = manifest.read_text(encoding="utf-8").replace('"CREATED"', '"STARTED"')
    manifest.write_text(text, encoding="utf-8")
    with pytest.raises(RuntimeError):
        freeze_planner(tmp_path)

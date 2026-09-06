from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from nccl_guard.analysis.statistics import summarize_launches
from nccl_guard.collectives.nccl_tests import command, parse_json, safe_max_message_bytes
from nccl_guard.final_transaction import FinalTransaction, TransactionState
from nccl_guard.models.sparse.model import SparseConfig, SyntheticSparseModel
from nccl_guard.tuning.simulation import candidate_space, run_nested_simulation


def test_nccl_message_bound_and_command_json_primary(tmp_path: Path) -> None:
    bound = safe_max_message_bytes(16 << 30, 8)
    assert 0 < bound <= 1 << 30
    args = command("all_reduce", tmp_path / "out.json", 16 << 30, 8)
    assert "-J" in args and "-T" in args and "-c" in args
    output = tmp_path / "out.json"
    output.write_text(json.dumps({"results": [{"size": 8, "busbw": 1.2}]}), encoding="utf-8")
    assert parse_json(output)[0]["size"] == 8


def test_nccl_invalid_gpu_count() -> None:
    with pytest.raises(ValueError):
        command("all_reduce", Path("x.json"), 1 << 30, 1)


def test_statistics_use_launch_count() -> None:
    summary = summarize_launches([10, 11, 12], samples=100)
    assert summary["n_launches"] == 3
    assert summary["mean"] == pytest.approx(11)


def test_final_transaction_state_machine() -> None:
    transaction = FinalTransaction("v1")
    transaction.transition(TransactionState.PREFLIGHT_PASSED)
    transaction.transition(TransactionState.STARTED)
    transaction.transition(TransactionState.COMPLETED)
    with pytest.raises(ValueError):
        transaction.transition(TransactionState.STARTED)


def test_sparse_schema_defaults() -> None:
    config = SparseConfig()
    assert config.dense_features == 13 and config.categorical_features == 26


def test_sparse_control_model_shape() -> None:
    config = SparseConfig(categorical_features=2, buckets=16, embedding_dim=4)
    model = SyntheticSparseModel(config)
    dense = torch.randn(3, config.dense_features)
    values = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    offsets = [torch.arange(3), torch.arange(3)]
    assert model(dense, values, offsets).shape == (3,)


def test_nested_simulation_keeps_evaluation_out_of_selection(tmp_path: Path) -> None:
    output = run_nested_simulation(tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["performance_claim"] is False
    assert payload["candidate_count"] == len(candidate_space()) == 12
    assert "SEARCH only" in payload["isolation"]
    assert set(payload["evaluation_pool_selected_only"]) == set(payload["selected"])
    assert set(payload["evaluation_configs_materialized"]) == set(payload["selected"].values())


def test_all_yaml_configs_parse() -> None:
    paths = list(Path("configs").rglob("*.yaml"))
    assert len(paths) >= 20
    for path in paths:
        assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)

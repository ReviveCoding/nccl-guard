from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nccl_guard.config.core import (
    ModelConfig,
    ParallelConfig,
    padded_vocab_size,
    scaling_efficiency,
    validate_tp,
    weak_scaling_metrics,
)
from nccl_guard.data.core import (
    SampleSchedule,
    prepare_synthetic,
    stable_bucket,
    verify_manifest,
    verify_no_duplicate_or_skip,
)
from nccl_guard.diagnosis.tracelens import diagnose
from nccl_guard.evidence.writer import RunWriter, verify_run
from nccl_guard.faults.core import FaultEvent, robust_straggler
from nccl_guard.metrics.core import (
    CollectiveRegime,
    ecr,
    exposed_communication,
    interval_union,
    rank_imbalance,
)
from nccl_guard.models.dense.transformer import (
    DecoderLM,
    RMSNorm,
    SwiGLU,
    apply_rope,
    canonical_model,
    estimate_activation_bytes,
    fits_memory_budget,
    state_fingerprint,
)
from nccl_guard.status import Status
from nccl_guard.tuning.protocol import (
    Candidate,
    GuardPlanner,
    Observation,
    SearchView,
    exhaustive_reference,
    random_search,
    tpe_search,
)


def tiny() -> ModelConfig:
    return ModelConfig("test", 2, 32, 4, 64, vocab_size=128, sequence_length=16)


def test_status_skip_is_not_success() -> None:
    assert Status.PASS.is_success
    assert not Status.SKIP_UNSUPPORTED.is_success


def test_parallel_batch_uses_dp_not_world() -> None:
    cfg = ParallelConfig(world_size=8, tp_degree=4, microbatch_per_dp_rank=3, grad_accumulation=2)
    assert cfg.dp_degree == 2
    assert cfg.global_batch == 12
    assert cfg.global_tokens(512) == 6144
    ddp = ParallelConfig(8, 1, 1, 1)
    tp_fsdp = ParallelConfig(8, 4, 1, 4)
    assert ddp.global_batch == tp_fsdp.global_batch == 8
    practical = [
        ParallelConfig(8, 1, 4, 1),
        ParallelConfig(8, 1, 2, 2),
        ParallelConfig(8, 2, 2, 4),
        ParallelConfig(8, 4, 2, 8),
    ]
    assert {config.global_batch for config in practical} == {32}


def test_tp_divisibility_and_invalid_world() -> None:
    validate_tp(tiny(), ParallelConfig(4, 2))
    with pytest.raises(ValueError):
        validate_tp(tiny(), ParallelConfig(3, 2))
    with pytest.raises(ValueError):
        validate_tp(ModelConfig("bad", 1, 30, 5, 63, 127), ParallelConfig(2, 2))


def test_strong_scaling_formula_and_infeasible_reference() -> None:
    assert scaling_efficiency(180, 2, 100) == pytest.approx(0.9)
    assert scaling_efficiency(180, 2, None) is None


def test_weak_scaling_metrics_are_not_strong_efficiency() -> None:
    result = weak_scaling_metrics(0.5, 100.0, 8)
    assert result == {
        "update_time_s": 0.5,
        "throughput_per_gpu": 100.0,
        "aggregate_throughput": 800.0,
    }
    assert "scaling_efficiency" not in result


def test_padded_vocab_and_memory_preflight() -> None:
    assert padded_vocab_size(50257, (2, 4)) == 50260
    assert padded_vocab_size(50304, (2, 4)) == 50304
    assert estimate_activation_bytes(tiny(), 2) > 0
    assert fits_memory_budget(tiny(), 2, 1 << 30)
    assert not fits_memory_budget(tiny(), 2, 1)


def test_model_shapes_loss_and_gradients() -> None:
    model = DecoderLM(tiny())
    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 128) and loss is not None and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_rope_position_zero_and_changes_later_positions() -> None:
    x = torch.randn(1, 2, 4, 8)
    out = apply_rope(x)
    assert torch.allclose(out[:, :, 0], x[:, :, 0])
    assert not torch.allclose(out[:, :, 1], x[:, :, 1])


def test_causal_mask_prevents_future_influence() -> None:
    model = DecoderLM(tiny()).eval()
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 99, 100]])
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    assert torch.allclose(la[:, :2], lb[:, :2], atol=1e-5)


def test_swigu_and_rmsnorm() -> None:
    assert SwiGLU(8, 16)(torch.randn(2, 3, 8)).shape == (2, 3, 8)
    out = RMSNorm(8)(torch.ones(2, 8))
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)


def test_canonical_fingerprint_is_stable() -> None:
    assert state_fingerprint(canonical_model(tiny(), 7)) == state_fingerprint(
        canonical_model(tiny(), 7)
    )
    assert state_fingerprint(canonical_model(tiny(), 7)) != state_fingerprint(
        canonical_model(tiny(), 8)
    )


def test_stable_hashing() -> None:
    assert stable_bucket("abc", 1000) == stable_bucket("abc", 1000)
    assert 0 <= stable_bucket("abc", 7) < 7


def test_dataset_manifest_and_hash(tmp_path: Path) -> None:
    manifest = prepare_synthetic(tmp_path, rows=8, seq=4, vocab=100)
    path = tmp_path / "data/manifests/synthetic_dense.json"
    assert manifest.rows == 8 and verify_manifest(tmp_path, path)
    (tmp_path / "data/processed/synthetic_dense_v1.bin").write_bytes(b"bad")
    assert not verify_manifest(tmp_path, path)


def test_sample_schedule_cross_dp_no_duplicates_or_skips() -> None:
    schedule = SampleSchedule(seed=2, updates=5, global_batch=12)
    assert verify_no_duplicate_or_skip(schedule, 1)
    assert verify_no_duplicate_or_skip(schedule, 3)
    assert schedule.fingerprint() == SampleSchedule(2, 5, 12).fingerprint()


def test_ecr_interval_union_no_double_counting() -> None:
    assert interval_union([(0, 3), (2, 5), (8, 9)]) == [(0, 5), (8, 9)]
    assert exposed_communication([(0, 5), (2, 4)], [(1, 3)]) == pytest.approx(3)
    assert ecr([(0, 5)], [(1, 3)], 10) == pytest.approx(0.3)


def test_rank_imbalance() -> None:
    assert rank_imbalance([1, 1, 2]) == pytest.approx(1)


def test_collective_comparability_is_exact() -> None:
    a = CollectiveRegime("AllReduce", 8, "bf16", 1024, 4096, 1, "NVLink")
    assert a.comparable(a)
    assert not a.comparable(CollectiveRegime("AllGather", 8, "bf16", 1024, 4096, 1, "NVLink"))


def observations(pool: str = "SEARCH") -> list[Observation]:
    return [
        Observation("a", 10, pool),
        Observation("b", 12, pool),
        Observation("c", 50, pool, oom=True),
    ]


def test_search_view_rejects_evaluation_leakage() -> None:
    with pytest.raises(ValueError):
        SearchView(observations("EVALUATION"))
    assert exhaustive_reference(SearchView(observations())) == "b"


def test_guard_visibility_budget_and_gates() -> None:
    candidates = [
        Candidate("a", "single", 2, 1),
        Candidate("b", "single", 1, 2),
        Candidate("c", "ddp", 4, 1),
    ]
    planner = GuardPlanner(3)
    proposed = planner.propose(candidates, {"native_windows": True}, "COMPUTE_BOUND")
    assert len(proposed) == 2 and all(c.strategy == "single" for c in proposed)
    assert planner.select(proposed, SearchView(observations())) == "b"
    assert random_search(candidates, SearchView(observations()), 3, 17) == "b"
    assert tpe_search(candidates, SearchView(observations()), 5, 17) in {"a", "b"}


def test_trace_lens_labels() -> None:
    assert (
        diagnose(ecr=0.4, hbm_headroom=0.5, rank_imbalance=0.01, message_bytes=512, p95_ratio=1.1)
        == "COMM_LATENCY_BOUND"
    )
    assert (
        diagnose(ecr=0.05, hbm_headroom=0.5, rank_imbalance=0.01, message_bytes=512, p95_ratio=1.1)
        == "COMPUTE_BOUND"
    )


def test_fault_validation_and_robust_detection() -> None:
    FaultEvent(3, 1, "compute_delay", 50).validate()
    with pytest.raises(ValueError):
        FaultEvent(-1, 0, "bad").validate()
    assert robust_straggler([1, 1.01, 0.99, 1, 1.02], 2)


def test_immutable_evidence_and_hashes(tmp_path: Path) -> None:
    with RunWriter(tmp_path, "test") as writer:
        writer.json("environment.json", {"x": 1})
        writer.finalize({"status": "PASS"})
        path = writer.path
        with pytest.raises(RuntimeError):
            writer.json("late.json", {})
    assert verify_run(path)
    (path / "environment.json").write_text("{}", encoding="utf-8")
    assert not verify_run(path)

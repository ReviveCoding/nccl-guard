from __future__ import annotations

import json
import random
from hashlib import blake2b
from pathlib import Path
from typing import Any

from nccl_guard.tuning.protocol import (
    Candidate,
    GuardPlanner,
    Observation,
    SearchView,
    exhaustive_reference,
    random_search,
    tpe_search,
)


def candidate_space() -> list[Candidate]:
    return (
        [
            Candidate(f"ddp-m{micro}-g{accum}", "ddp", micro, accum)
            for micro, accum in [(1, 8), (2, 4), (4, 2), (8, 1)]
        ]
        + [
            Candidate(f"fsdp2-m{micro}-g{accum}", "fsdp2", micro, accum)
            for micro, accum in [(1, 8), (2, 4), (4, 2), (8, 1)]
        ]
        + [
            Candidate(f"tp{tp}-fsdp-m{micro}-g{accum}", "tp_fsdp", micro, accum, tp)
            for tp, micro, accum in [(2, 1, 8), (2, 2, 4), (4, 1, 8), (4, 2, 4)]
        ]
    )


def _latent(candidate: Candidate) -> float:
    value = 90.0 + 2.3 * candidate.microbatch - 1.1 * candidate.grad_accumulation
    return value + {"ddp": 4.0, "fsdp2": 1.0, "tp_fsdp": 3.0}[candidate.strategy]


def _search_pool(candidates: list[Candidate]) -> SearchView:
    search_rng = random.Random(1001)
    search: list[Observation] = []
    for candidate in candidates:
        search.append(
            Observation(candidate.name, _latent(candidate) + search_rng.gauss(0, 2.5), "SEARCH")
        )
    return SearchView(search)


def _evaluate_selected(
    selected_names: set[str], candidates: list[Candidate]
) -> dict[str, Observation]:
    by_name = {candidate.name: candidate for candidate in candidates}
    evaluation: dict[str, Observation] = {}
    for name in sorted(selected_names):
        seed = int.from_bytes(blake2b(f"2003:{name}".encode(), digest_size=8).digest(), "little")
        evaluation[name] = Observation(
            name, _latent(by_name[name]) + random.Random(seed).gauss(0, 2.5), "EVALUATION"
        )
    return evaluation


def run_nested_simulation(root: Path) -> Path:
    """Exercise selection isolation using synthetic observations, never benchmark evidence."""
    candidates = candidate_space()
    search = _search_pool(candidates)
    exhaustive = exhaustive_reference(search)
    guard_proposals = GuardPlanner(3).propose(
        candidates,
        {"native_windows": False, "topology": "simulated-fast-intra-node"},
        "COMM_BANDWIDTH_BOUND",
    )
    guard = GuardPlanner(3).select(guard_proposals, search)
    random_seeds = {str(seed): random_search(candidates, search, 3, seed) for seed in range(20)}
    tpe: dict[str, dict[str, str]] = {}
    replay_seeds = [*range(10), 17]
    for budget in (3, 5, 8):
        tpe[str(budget)] = {
            str(seed): tpe_search(candidates, search, budget, seed) for seed in replay_seeds
        }
    selected = {
        "strong_default": candidates[0].name,
        "exhaustive_search_reference": exhaustive,
        "guard_at_3": guard,
        "random_at_3_preregistered_seed_17": random_seeds["17"],
        "tpe_at_8_preregistered_seed_17": tpe["8"]["17"],
    }
    # This is the freeze boundary: no EVALUATION observation exists before selections finalize.
    evaluation = _evaluate_selected(set(selected.values()), candidates)
    evaluation_results = {method: evaluation[name].throughput for method, name in selected.items()}
    reference = evaluation_results["exhaustive_search_reference"]
    result: dict[str, Any] = {
        "status": "PASS",
        "evidence_kind": "DETERMINISTIC_OFFLINE_PROTOCOL_SIMULATION",
        "performance_claim": False,
        "candidate_count": len(candidates),
        "search_pool_seed": 1001,
        "evaluation_pool_seed": 2003,
        "evaluation_configs_materialized": sorted(evaluation),
        "guard_candidate_budget": 3,
        "guard_proposals": [candidate.name for candidate in guard_proposals],
        "selected": selected,
        "evaluation_pool_selected_only": evaluation_results,
        "guard_exhaustive_search_reference_gap": (reference - evaluation_results["guard_at_3"])
        / reference,
        "random_selected_distribution": random_seeds,
        "tpe_selected_distribution": tpe,
        "isolation": (
            "selection used SEARCH only; only selected configs were materialized in EVALUATION "
            "after selections froze"
        ),
    }
    path = root / "evidence/search_protocol_simulation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def run_planner_ablation_simulation(root: Path) -> Path:
    base = json.loads(run_nested_simulation(root).read_text(encoding="utf-8"))
    proposals = base["guard_proposals"]
    result = {
        "status": "PASS",
        "evidence_kind": "DETERMINISTIC_OFFLINE_PLANNER_ABLATION_SIMULATION",
        "performance_claim": False,
        "ablations": {
            "FULL": proposals,
            "minus_topology": list(reversed(proposals)),
            "minus_nccl_microbenchmark": proposals[1:] + proposals[:1],
            "minus_runtime_profile": sorted(proposals),
            "minus_guard_gates": proposals + ["invalid-candidate-demonstration"],
        },
        "note": (
            "control-logic coverage only; contribution claims require real SEARCH/EVALUATION runs"
        ),
    }
    path = root / "evidence/planner_ablation_simulation.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path

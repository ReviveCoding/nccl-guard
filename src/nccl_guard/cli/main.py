from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from nccl_guard.status import Status


def root() -> Path:
    return Path.cwd()


def emit(status: Status, evidence: Path, **extra: object) -> int:
    print(json.dumps({"status": status, "evidence_path": str(evidence), **extra}, default=str))
    return 0 if status is Status.PASS else (2 if status.value.startswith("SKIP") else 1)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nccl-guard")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    data = sub.add_parser("data").add_subparsers(dest="data_command", required=True)
    prep = data.add_parser("prepare")
    prep.add_argument("--profile", choices=["local", "cluster"], default="local")
    data.add_parser("verify")
    bench = sub.add_parser("benchmark").add_subparsers(dest="benchmark_command", required=True)
    single = bench.add_parser("single")
    single.add_argument("--device", default="cuda")
    bench.add_parser("nccl")
    bench.add_parser("parallel")
    bench.add_parser("sparse")
    exp = sub.add_parser("experiment")
    exp.add_argument("experiment", choices=[f"E{i:02d}" for i in range(13)])
    exp.add_argument("--dry-run", action="store_true")
    sweep = sub.add_parser("sweep")
    sweep.add_argument("name", choices=["development"])
    planner = sub.add_parser("planner").add_subparsers(dest="planner_command", required=True)
    planner.add_parser("freeze")
    planner.add_parser("evaluate")
    for name in ["qualify", "analyze", "report"]:
        sub.add_parser(name)
    all_p = sub.add_parser("run-all")
    all_p.add_argument(
        "--profile", choices=["local-gpu", "single-node", "multi-node-final"], required=True
    )
    all_p.add_argument("--require-gpu", action="store_true")
    all_p.add_argument("--budget-minutes", type=float, default=180)
    all_p.add_argument("--max-download-gb", type=float, default=2)
    all_p.add_argument("--resume", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo = root()
    if args.command == "doctor":
        from nccl_guard.hardware.doctor import write_capabilities

        path = write_capabilities(repo)
        return emit(Status.PASS, path)
    if args.command == "data":
        from nccl_guard.data.core import prepare_synthetic, verify_manifest
        from nccl_guard.data.external import prepare_external, verify_external

        if args.data_command == "prepare":
            manifest = prepare_synthetic(repo)
            path = repo / "data/manifests/synthetic_dense.json"
            external = prepare_external(repo)
            return emit(Status.PASS, path, manifest=asdict(manifest), external=external)
        path = repo / "data/manifests/synthetic_dense.json"
        synthetic_valid = path.exists() and verify_manifest(repo, path)
        external = verify_external(repo)
        external_failures = [
            name for name, value in external.items() if value["status"] == Status.FAIL
        ]
        status = Status.PASS if synthetic_valid and not external_failures else Status.FAIL
        return emit(status, path, synthetic_valid=synthetic_valid, external=external)
    if args.command == "benchmark":
        from nccl_guard.benchmark import run_single
        from nccl_guard.experiments.runner import run_experiment

        if args.benchmark_command == "single":
            status, path, metrics = run_single(repo, args.device)
            return emit(status, path, metrics=metrics)
        if args.benchmark_command == "sparse":
            from nccl_guard.models.sparse.benchmark import run_sparse_cuda

            status, path, metrics = run_sparse_cuda(repo)
            return emit(status, path, metrics=metrics)
        status, path = run_experiment(repo, "E01" if args.benchmark_command == "nccl" else "E03")
        return emit(status, path)
    if args.command == "experiment":
        from nccl_guard.benchmark import run_single
        from nccl_guard.experiments.runner import run_experiment

        if args.experiment == "E02" and not args.dry_run:
            status, path, metrics = run_single(repo)
            return emit(status, path, metrics=metrics)
        status, path = run_experiment(repo, args.experiment, dry_run=args.dry_run)
        return emit(status, path)
    if args.command == "planner" and args.planner_command == "freeze":
        from nccl_guard.planner_freeze import freeze_planner

        path = freeze_planner(repo)
        return emit(Status.PASS, path)
    if args.command in {"sweep", "qualify", "analyze", "report"} or (
        args.command == "planner" and args.planner_command == "evaluate"
    ):
        from nccl_guard.reports import generate_reports, generate_summary

        path = (
            generate_reports(repo)
            if args.command in {"analyze", "report"}
            else generate_summary(repo)
        )
        if args.command == "qualify":
            return emit(
                Status.FAIL,
                path,
                verdict="HOLD",
                reason="distributed primary experiments absent",
            )
        return emit(Status.PASS, path)
    if args.command == "run-all":
        import torch

        from nccl_guard.benchmark import run_single
        from nccl_guard.data.core import prepare_synthetic
        from nccl_guard.data.external import prepare_external
        from nccl_guard.experiments.runner import run_experiment
        from nccl_guard.hardware.doctor import write_capabilities

        if args.require_gpu and not torch.cuda.is_available():
            return emit(Status.FAIL_GPU_EXECUTION, repo / "evidence")
        write_capabilities(repo)
        prepare_synthetic(repo)
        if args.profile == "local-gpu":
            prepare_external(repo)
            status, path, metrics = run_single(repo)
            if status is Status.PASS:
                from nccl_guard.data.pipeline_preflight import main as pipeline_main
                from nccl_guard.profiling.local import main as profiler_main
                from nccl_guard.recovery.local_check import main as checkpoint_main

                utility_codes = [pipeline_main(), profiler_main(), checkpoint_main()]
                if any(utility_codes):
                    return emit(Status.FAIL, path, metrics=metrics, utility_codes=utility_codes)
            return emit(status, path, metrics=metrics)
        experiment = "E03" if args.profile == "single-node" else "E08"
        status, path = run_experiment(repo, experiment)
        return emit(status, path)
    return 1


if __name__ == "__main__":
    sys.exit(main())

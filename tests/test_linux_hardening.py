from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "scripts" / "linux"
PIN = "b4d5beebca8a76cf01335f724d154b9b9d394d96"


def bash_executable() -> str:
    if shutil.which("bash") and not Path("C:/Windows/System32/bash.exe").exists():
        return str(shutil.which("bash"))
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    if git_bash.exists():
        return str(git_bash)
    pytest.skip("Git Bash or native Linux Bash is unavailable")


def run_preflight_function(body: str) -> subprocess.CompletedProcess[str]:
    preflight = (LINUX / "preflight.sh").as_posix()
    result = subprocess.run(
        [bash_executable(), "-c", f"set -euo pipefail; source '{preflight}'; {body}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if "couldn't create signal pipe" in result.stderr:
        pytest.skip("Git Bash cannot start in this restricted Windows process context")
    return result


def test_preflight_contract_static() -> None:
    preflight = (LINUX / "preflight.sh").read_text(encoding="utf-8")
    bootstrap = (LINUX / "bootstrap_single_node.sh").read_text(encoding="utf-8")
    assert "candidates=(python3.12 python3.11 python3 python)" in preflight
    assert "sys.version_info < (3, 11)" in preflight
    assert "SKIP_UNSUPPORTED" in preflight and "NON_AUTHORITATIVE_PREFLIGHT" in preflight
    assert "visible GPU count" in preflight
    assert bootstrap.index("nccl_guard_select_python") < bootstrap.index("-m venv")
    assert bootstrap.index("nccl_guard_select_python") < bootstrap.index("pip install")


def test_python_310_rejected_before_dependency_installation() -> None:
    result = run_preflight_function(
        "python3.12(){ return 127; }; "
        "python3.11(){ return 127; }; "
        "python3(){ if [[ $1 == -c ]]; then return 1; fi; printf 'Python 3.10.14\\n'; }; "
        "python(){ return 127; }; nccl_guard_select_python"
    )
    assert result.returncode == 2
    assert "Python 3.10.14" in result.stderr
    assert "no compatible Python found" in result.stderr

    bootstrap = (LINUX / "bootstrap_single_node.sh").read_text(encoding="utf-8")
    assert bootstrap.index("nccl_guard_select_python") < bootstrap.index("-m venv")
    assert bootstrap.index("nccl_guard_select_python") < bootstrap.index("pip install")


def test_python_311_or_newer_is_accepted() -> None:
    result = run_preflight_function(
        "python3.12(){ return 127; }; "
        "python3.11(){ if [[ $1 == -c ]]; then printf '3.11.9\\n'; return 0; fi; }; "
        "nccl_guard_select_python"
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "python3.11"
    assert "accepted python3.11 (3.11.9)" in result.stderr


def test_wsl_rejected_for_authoritative_qualification() -> None:
    result = run_preflight_function(
        "uname(){ if [[ $1 == -s ]]; then printf 'Linux\\n'; else "
        "printf '6.6.0-microsoft-standard-WSL2\\n'; fi; }; "
        "nccl_guard_require_authoritative_linux"
    )
    assert result.returncode == 2
    assert "SKIP_UNSUPPORTED" in result.stderr
    assert "NON_AUTHORITATIVE_PREFLIGHT" in result.stderr


def test_gpu_count_mismatch_is_rejected() -> None:
    result = run_preflight_function(
        "nvidia-smi(){ printf 'GPU 0: test\\nGPU 1: test\\n'; }; nccl_guard_validate_gpu_request 4"
    )
    assert result.returncode == 2
    assert "visible GPU count (2) is smaller than GPUS_PER_NODE (4)" in result.stderr


def test_linux_venv_is_separate_from_windows_venv() -> None:
    scripts = list(LINUX.glob("*.sh"))
    combined = "\n".join(path.read_text(encoding="utf-8") for path in scripts)
    assert 'NCCL_GUARD_LINUX_VENV=".venv-linux-core"' in combined
    assert ".venv-core" not in combined
    for script in scripts:
        if script.name != "preflight.sh":
            assert 'preflight.sh"' in script.read_text(encoding="utf-8")


def test_shell_runbook_and_exact_revision_consistency() -> None:
    single = (ROOT / "docs" / "SINGLE_NODE_GPU_RUNBOOK.md").read_text(encoding="utf-8")
    multi = (ROOT / "docs" / "MULTI_NODE_GPU_RUNBOOK.md").read_text(encoding="utf-8")
    preflight = (LINUX / "preflight.sh").read_text(encoding="utf-8")
    heading = "RUN THESE ONLY INSIDE THE TARGET LINUX HOST'S BASH SHELL"
    for runbook in (single, multi):
        assert heading in runbook
        assert "```powershell" in runbook
        assert "```bash" in runbook
        assert "/path/to/nccl-guard" not in runbook
        assert ".venv-core" not in runbook
    assert "GPUS_PER_NODE=8 bash" in single
    assert PIN in single and PIN in multi and PIN in preflight

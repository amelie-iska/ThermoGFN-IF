"""Conda environment discovery and command dispatch helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .constants import REQUIRED_ENVS, OPTIONAL_ENVS


def list_conda_envs() -> dict[str, str]:
    cmd = ["conda", "env", "list", "--json"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception:
        # Alternate parser from plain text output.
        out = subprocess.check_output(["conda", "env", "list"], stderr=subprocess.STDOUT, text=True)
        envs: dict[str, str] = {}
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                if parts[0] in {"*", "+"}:
                    continue
                envs[parts[0]] = parts[-1]
        return envs
    payload = json.loads(out)
    env_paths = payload.get("envs", [])
    envs: dict[str, str] = {}
    for p in env_paths:
        name = Path(p).name
        envs[name] = p
    return envs


def check_envs(run_health_checks: bool = False) -> dict[str, Any]:
    envs = list_conda_envs()
    report: dict[str, Any] = {"required": {}, "optional": {}}

    for name in REQUIRED_ENVS:
        status = "ready" if name in envs else "missing"
        report["required"][name] = {
            "status": status,
            "path": envs.get(name),
        }
    for name in OPTIONAL_ENVS:
        status = "ready" if name in envs else "missing"
        report["optional"][name] = {
            "status": status,
            "path": envs.get(name),
        }

    if run_health_checks:
        for name in REQUIRED_ENVS:
            if name not in envs:
                continue
            ok, details = run_health_check(name)
            report["required"][name]["health_ok"] = ok
            report["required"][name]["health_details"] = details
            report["required"][name]["status"] = "ready" if ok else "exists_unchecked"

    return report


def run_health_check(env_name: str) -> tuple[bool, dict[str, Any]]:
    checks = {
        "ligandmpnn_env": (
            "python -c \"import tqdm; print(tqdm.__version__)\" >/dev/null && "
            "export PYTHONPATH=\"$(pwd)/scripts/vendor/pycompat:${PYTHONPATH:-}\" && "
            "cd models/LigandMPNN && python run.py --help"
        ),
        "ADFLIP": "cd models/ADFLIP && PYTHONPATH=. python test/design.py --help",
        "KcatNet": (
            "cd models/KcatNet && python - <<'PY'\n"
            "import torch\n"
            "import esm\n"
            "import transformers\n"
            "import torch_scatter\n"
            "from torch_geometric.data import Data\n"
            "from rdkit import Chem\n"
            "from models.model_kcat import KcatNet\n"
            "from utils.protein_init import T5Tokenizer\n"
            "print('ok')\n"
            "PY"
        ),
        # Backward-compatible alias.
        "MMKcat": (
            "cd models/KcatNet && python - <<'PY'\n"
            "import torch\n"
            "import esm\n"
            "import transformers\n"
            "import torch_scatter\n"
            "from torch_geometric.data import Data\n"
            "from rdkit import Chem\n"
            "from models.model_kcat import KcatNet\n"
            "from utils.protein_init import T5Tokenizer\n"
            "print('ok')\n"
            "PY"
        ),
        # Lowercase alias for convenience.
        "kcatnet": (
            "cd models/KcatNet && python - <<'PY'\n"
            "import torch\n"
            "import esm\n"
            "import transformers\n"
            "import torch_scatter\n"
            "from torch_geometric.data import Data\n"
            "from rdkit import Chem\n"
            "from models.model_kcat import KcatNet\n"
            "from utils.protein_init import T5Tokenizer\n"
            "print('ok')\n"
            "PY"
        ),
        "apodock": (
            "if [[ -f \"$CONDA_PREFIX/lib/libLLVM-15.so\" ]]; then "
            "export LD_PRELOAD=\"$CONDA_PREFIX/lib/libLLVM-15.so${LD_PRELOAD:+:$LD_PRELOAD}\"; "
            "fi && cd models/GraphKcat && python predict.py --help"
        ),
        "graphkcat": (
            "if [[ -f \"$CONDA_PREFIX/lib/libLLVM-15.so\" ]]; then "
            "export LD_PRELOAD=\"$CONDA_PREFIX/lib/libLLVM-15.so${LD_PRELOAD:+:$LD_PRELOAD}\"; "
            "fi && cd models/GraphKcat && python predict.py --help"
        ),
        "spurs": (
            "python - <<'PY'\n"
            "import contextlib\n"
            "import io\n"
            "buf = io.StringIO()\n"
            "with contextlib.redirect_stdout(buf):\n"
            "    import spurs.inference as s\n"
            "print('ok')\n"
            "PY"
        ),
        "bioemu": (
            "python -m bioemu.sample --help >/dev/null && "
            "./scripts/env/setup_bioemu_colabfold_runtime.sh --bioemu-env bioemu --check-only >/dev/null"
        ),
        "uma-qc": "python -c \"import fairchem.core as f; from ase.md.langevin import Langevin; print('ok')\"",
    }
    cmd = checks.get(env_name)
    if cmd is None:
        return False, {"error": "unknown env for health check"}
    try:
        proc = subprocess.run(
            ["conda", "run", "-n", env_name, "bash", "-lc", cmd],
            capture_output=True,
            text=True,
            check=True,
        )
        return True, {"stdout_tail": proc.stdout[-500:], "stderr_tail": proc.stderr[-500:]}
    except subprocess.CalledProcessError as exc:
        return False, {
            "returncode": exc.returncode,
            "stdout_tail": (exc.stdout or "")[-500:],
            "stderr_tail": (exc.stderr or "")[-500:],
        }


def dispatch_conda(env_name: str, cmd: str, timeout_s: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["conda", "run", "-n", env_name, "bash", "-lc", cmd],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

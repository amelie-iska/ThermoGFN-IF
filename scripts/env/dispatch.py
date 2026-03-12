#!/usr/bin/env python3
"""Unified command dispatcher that runs commands inside conda environments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "scripts").exists() and (parent / "train").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _conda_root() -> Path | None:
    raw = os.environ.get("CONDA_ROOT")
    if raw:
        p = Path(raw)
        if p.exists():
            return p
    raw = os.environ.get("CONDA_EXE")
    if raw:
        p = Path(raw).resolve()
        for parent in (p.parent, p.parent.parent):
            if (parent / "envs").exists():
                return parent
    which_conda = shutil.which("conda")
    if which_conda:
        p = Path(which_conda).resolve()
        for parent in p.parents:
            if (parent / "envs").exists():
                return parent
    try:
        result = subprocess.run(
            ["conda", "info", "--base"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        result = None
    if result is not None:
        p = Path(result.stdout.strip())
        if p.exists() and (p / "envs").exists():
            return p
    exe = Path(sys.executable).resolve()
    if exe.parent.name == "bin":
        candidate = exe.parent.parent
        if (candidate / "envs").exists():
            return candidate
    return None


def _env_prefix(env_name: str) -> Path | None:
    conda_root = _conda_root()
    if conda_root is None:
        return None
    prefix = conda_root / "envs" / env_name
    if prefix.exists():
        return prefix
    return None


def _env_site_packages(env_name: str) -> Path | None:
    prefix = _env_prefix(env_name)
    if prefix is None:
        return None
    python_bin = prefix / "bin" / "python"
    if python_bin.exists():
        try:
            result = subprocess.run(
                [str(python_bin), "-c", "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')"],
                check=True,
                capture_output=True,
                text=True,
            )
            version = result.stdout.strip()
            candidate = prefix / "lib" / version / "site-packages"
            if candidate.exists():
                return candidate
        except Exception:  # noqa: BLE001
            pass
    candidates = sorted(prefix.glob("lib/python*/site-packages"))
    if not candidates:
        return None
    return candidates[-1]


def _stream_pipe(src, dst, sink_chunks: list[str]) -> None:
    """Forward subprocess output to terminal in real time while collecting logs."""
    try:
        while True:
            chunk = src.read(1)
            if not chunk:
                break
            dst.write(chunk)
            dst.flush()
            sink_chunks.append(chunk)
    finally:
        try:
            src.close()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--stdout-log", default=None)
    parser.add_argument("--stderr-log", default=None)
    parser.add_argument("--metadata-log", default=None)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--env-status-json", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()

    if args.require_ready:
        if not args.env_status_json:
            print("--require-ready requires --env-status-json", file=sys.stderr)
            return 2
        status = json.loads(Path(args.env_status_json).read_text())
        required = status.get("required", {})
        optional = status.get("optional", {})
        info = required.get(args.env_name) or optional.get(args.env_name)
        if info is None:
            print(f"env {args.env_name} not found in env status report", file=sys.stderr)
            return 2
        if info.get("status") != "ready":
            print(f"env {args.env_name} not ready: {info}", file=sys.stderr)
            return 2

    repo_root = _repo_root()
    hf_home = repo_root / ".cache" / "huggingface"
    hub_cache = hf_home / "hub"
    tx_cache = hf_home / "transformers"
    torch_home = hf_home / "torch"
    xdg_cache_home = repo_root / ".cache" / "xdg"
    uv_cache_dir = repo_root / ".cache" / "uv"
    pip_cache_dir = repo_root / ".cache" / "pip"
    bioemu_colabfold_home = repo_root / ".cache" / "bioemu" / "colabfold"
    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    tx_cache.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)
    xdg_cache_home.mkdir(parents=True, exist_ok=True)
    uv_cache_dir.mkdir(parents=True, exist_ok=True)
    pip_cache_dir.mkdir(parents=True, exist_ok=True)
    bioemu_colabfold_home.mkdir(parents=True, exist_ok=True)
    env_prefix = _env_prefix(args.env_name)
    env_site = _env_site_packages(args.env_name)

    path_export = ""
    if env_prefix is not None:
        path_export = f'export PATH="{env_prefix / "bin"}:$PATH"; '
    site_export = ""
    if env_site is not None:
        site_export = f'export PYTHONPATH="{env_site}"; '

    wrapped_cmd = (
        'unset PYTHONHOME PYTHONPATH VIRTUAL_ENV __PYVENV_LAUNCHER__; '
        'export PYTHONNOUSERSITE=1; '
        f"{path_export}"
        f"{site_export}"
        f'export HF_HOME="${{HF_HOME:-{hf_home}}}"; '
        f'export HUGGINGFACE_HUB_CACHE="${{HUGGINGFACE_HUB_CACHE:-{hub_cache}}}"; '
        f'export TRANSFORMERS_CACHE="${{TRANSFORMERS_CACHE:-{tx_cache}}}"; '
        f'export TORCH_HOME="${{TORCH_HOME:-{torch_home}}}"; '
        f'export XDG_CACHE_HOME="${{XDG_CACHE_HOME:-{xdg_cache_home}}}"; '
        f'export UV_CACHE_DIR="${{UV_CACHE_DIR:-{uv_cache_dir}}}"; '
        f'export PIP_CACHE_DIR="${{PIP_CACHE_DIR:-{pip_cache_dir}}}"; '
        f'export BIOEMU_COLABFOLD_DIR="${{BIOEMU_COLABFOLD_DIR:-{bioemu_colabfold_home}}}"; '
        f'export HF_HUB_ENABLE_HF_TRANSFER="${{HF_HUB_ENABLE_HF_TRANSFER:-0}}"; '
        f'mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$BIOEMU_COLABFOLD_DIR"; '
        f"{args.cmd}"
    )
    cmd = ["conda", "run", "--no-capture-output", "-n", args.env_name, "bash", "-c", wrapped_cmd]
    child_env = os.environ.copy()
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "__PYVENV_LAUNCHER__",
    ):
        child_env.pop(key, None)
    child_env["PYTHONNOUSERSITE"] = "1"
    start = datetime.now(timezone.utc)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=child_env,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    t_out = threading.Thread(target=_stream_pipe, args=(proc.stdout, sys.stdout, stdout_chunks), daemon=True)
    t_err = threading.Thread(target=_stream_pipe, args=(proc.stderr, sys.stderr, stderr_chunks), daemon=True)
    t_out.start()
    t_err.start()

    timed_out = False
    try:
        proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()

    t_out.join()
    t_err.join()

    stdout_text = "".join(stdout_chunks)
    stderr_text = "".join(stderr_chunks)

    if args.stdout_log:
        p = Path(args.stdout_log)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(stdout_text)

    if args.stderr_log:
        p = Path(args.stderr_log)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(stderr_text)

    if timed_out:
        if args.stderr_log:
            p = Path(args.stderr_log)
            p.write_text(p.read_text() + f"\nTimeout after {args.timeout}s\n")
        return 4

    if args.metadata_log:
        meta = {
            "timestamp_start_utc": start.isoformat(),
            "timestamp_end_utc": datetime.now(timezone.utc).isoformat(),
            "env_name": args.env_name,
            "cmd": args.cmd,
            "wrapped_cmd": wrapped_cmd,
            "returncode": proc.returncode,
            "timed_out": timed_out,
        }
        p = Path(args.metadata_log)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, indent=2, sort_keys=True))

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

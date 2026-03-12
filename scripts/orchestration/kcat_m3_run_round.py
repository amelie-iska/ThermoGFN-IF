#!/usr/bin/env python3
"""Run one Kcat-only Method III active-learning round end-to-end."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import time


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _run(
    cmd: list[str],
    logger,
    *,
    dry_run: bool = False,
    heartbeat_sec: float = 30.0,
    step_name: str = "",
) -> tuple[int, float]:
    logger.info("CMD: %s", " ".join(cmd))
    t0 = time.perf_counter()
    if dry_run:
        return 0, 0.0
    hb = max(1.0, float(heartbeat_sec))
    proc = subprocess.Popen(cmd)  # noqa: S603
    while True:
        try:
            rc = proc.wait(timeout=hb)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            logger.info("STEP %s still running elapsed=%.1fs", step_name or "<unnamed>", elapsed)
    return rc, time.perf_counter() - t0


def _gate_report(round_dir: Path, strict_gates: bool) -> tuple[bool, dict]:
    metrics_path = round_dir / "metrics" / "round_metrics.json"
    student_path = round_dir / "metrics" / "student_metrics.json"
    summary_path = round_dir / "manifests" / "append_summary.json"
    report = {"pass": True, "checks": {}, "strict": strict_gates}

    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
    else:
        m = {}
    if student_path.exists():
        s = json.loads(student_path.read_text())
    else:
        s = {}
    if summary_path.exists():
        a = json.loads(summary_path.read_text())
    else:
        a = {}

    n_next = int(a.get("n_next", 0))
    report["checks"]["gate_a_n_next_positive"] = n_next > 0

    kl = float(s.get("teacher_student_kl", 0.0))
    report["checks"]["gate_b_kl"] = kl <= 0.15

    unique_fraction = float(m.get("unique_fraction", 0.0))
    report["checks"]["gate_c_unique_fraction"] = unique_fraction >= 0.5

    best_reward = float(m.get("best_reward", 0.0))
    report["checks"]["gate_d_best_reward"] = best_reward > 0.0

    report["pass"] = all(report["checks"].values())
    if strict_gates and not report["pass"]:
        return False, report
    return True, report


def _q(v: str | Path) -> str:
    return shlex.quote(str(v))


def _pick_substrate_smiles(rec: dict) -> str | None:
    for key in ("substrate_smiles", "Smiles", "smiles", "ligand_smiles"):
        val = rec.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            for item in val:
                if item is not None and str(item).strip():
                    return str(item).strip()
            continue
        s = str(val).strip()
        if s:
            return s
    return None


def _validate_kcat_dataset(dataset_path: Path, logger) -> None:
    from train.thermogfn.io_utils import read_records

    rows = read_records(dataset_path)
    missing = [rec for rec in rows if _pick_substrate_smiles(rec) is None]
    if not missing:
        return
    labels = []
    for rec in missing[:10]:
        for key in ("candidate_id", "backbone_id", "stem", "example_id", "spec_path"):
            if rec.get(key):
                labels.append(f"{key}={rec[key]}")
                break
        else:
            labels.append("<unknown>")
    raise RuntimeError(
        f"Kcat dataset missing substrate metadata for {len(missing)}/{len(rows)} records in {dataset_path}. "
        "Examples: "
        + ", ".join(labels)
        + ". Provide substrate_smiles/Smiles/smiles/ligand_smiles or build the index with --metadata-overlay."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/kcat_m3_default.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--kcatnet-budget", type=int, default=None)
    parser.add_argument("--graphkcat-budget", type=int, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--surrogate-ensemble-size", type=int, default=None)
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--teacher-gamma-off", type=float, default=None)
    parser.add_argument("--student-steps", type=int, default=None)

    parser.add_argument("--kcatnet-env-name", default=None)
    parser.add_argument("--graphkcat-env-name", default=None)

    parser.add_argument("--kcatnet-model-root", default=None)
    parser.add_argument("--kcatnet-checkpoint", default=None)
    parser.add_argument("--kcatnet-config-path", default=None)
    parser.add_argument("--kcatnet-degree-path", default=None)
    parser.add_argument("--kcatnet-device", default=None)
    parser.add_argument("--kcatnet-batch-size", type=int, default=None)
    parser.add_argument("--kcatnet-std-default", type=float, default=None)
    parser.add_argument("--kcatnet-prott5-model", default=None)
    parser.add_argument("--kcatnet-prott5-dir", default=None)

    parser.add_argument("--graphkcat-model-root", default=None)
    parser.add_argument("--graphkcat-checkpoint", default=None)
    parser.add_argument("--graphkcat-cfg", default=None)
    parser.add_argument("--graphkcat-batch-size", type=int, default=None)
    parser.add_argument("--graphkcat-device", default=None)
    parser.add_argument("--graphkcat-distance-cutoff-a", type=float, default=None)
    parser.add_argument("--graphkcat-std-default", type=float, default=None)
    parser.add_argument("--graphkcat-heartbeat-sec", type=float, default=None)

    parser.add_argument("--fuse-w-kcatnet", type=float, default=None)
    parser.add_argument("--fuse-w-graphkcat", type=float, default=None)
    parser.add_argument("--fuse-w-agreement", type=float, default=None)
    parser.add_argument("--fuse-kappa-kcatnet", type=float, default=None)
    parser.add_argument("--fuse-kappa-graphkcat", type=float, default=None)

    parser.add_argument("--strict-gates", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env-status-json", default=None)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--step-heartbeat-sec", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    wall_t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.progress import configure_logging
    from train.thermogfn.config_utils import load_yaml_config, cfg_get

    logger = configure_logging("orchestrate.kcat_round", level=args.log_level)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_yaml_config(cfg_path)

    args.pool_size = int(args.pool_size if args.pool_size is not None else cfg_get(cfg, "round.pool_size", 50000))
    args.kcatnet_budget = int(
        args.kcatnet_budget if args.kcatnet_budget is not None else cfg_get(cfg, "round.kcatnet_budget", 1024)
    )
    args.graphkcat_budget = int(
        args.graphkcat_budget if args.graphkcat_budget is not None else cfg_get(cfg, "round.graphkcat_budget", 256)
    )
    args.max_checkpoints = int(
        args.max_checkpoints if args.max_checkpoints is not None else cfg_get(cfg, "round.max_checkpoints", 5)
    )
    args.step_heartbeat_sec = float(
        args.step_heartbeat_sec if args.step_heartbeat_sec is not None else cfg_get(cfg, "round.step_heartbeat_sec", 30.0)
    )
    args.surrogate_ensemble_size = int(
        args.surrogate_ensemble_size
        if args.surrogate_ensemble_size is not None
        else cfg_get(cfg, "method3.surrogate_ensemble_size", 8)
    )
    args.teacher_steps = int(args.teacher_steps if args.teacher_steps is not None else cfg_get(cfg, "method3.teacher_steps", 30000))
    args.teacher_gamma_off = float(
        args.teacher_gamma_off if args.teacher_gamma_off is not None else cfg_get(cfg, "method3.teacher_gamma_off", 0.5)
    )
    args.student_steps = int(args.student_steps if args.student_steps is not None else cfg_get(cfg, "method3.student_steps", 15000))

    args.kcatnet_env_name = str(args.kcatnet_env_name or cfg_get(cfg, "oracles.envs.kcatnet", "KcatNet"))
    args.graphkcat_env_name = str(args.graphkcat_env_name or cfg_get(cfg, "oracles.envs.graphkcat", "apodock"))

    args.kcatnet_model_root = str(args.kcatnet_model_root or cfg_get(cfg, "oracles.kcatnet.model_root", "models/KcatNet"))
    args.kcatnet_checkpoint = str(
        args.kcatnet_checkpoint or cfg_get(cfg, "oracles.kcatnet.checkpoint", "models/KcatNet/RESULT/model_KcatNet.pt")
    )
    args.kcatnet_config_path = str(
        args.kcatnet_config_path or cfg_get(cfg, "oracles.kcatnet.config_path", "models/KcatNet/config_KcatNet.json")
    )
    args.kcatnet_degree_path = str(
        args.kcatnet_degree_path or cfg_get(cfg, "oracles.kcatnet.degree_path", "models/KcatNet/Dataset/degree.pt")
    )
    args.kcatnet_device = str(args.kcatnet_device or cfg_get(cfg, "oracles.kcatnet.device", "cuda:0"))
    args.kcatnet_batch_size = int(
        args.kcatnet_batch_size if args.kcatnet_batch_size is not None else cfg_get(cfg, "oracles.kcatnet.batch_size", 8)
    )
    args.kcatnet_std_default = float(
        args.kcatnet_std_default if args.kcatnet_std_default is not None else cfg_get(cfg, "oracles.kcatnet.std_default", 0.25)
    )
    args.kcatnet_prott5_model = str(
        args.kcatnet_prott5_model or cfg_get(cfg, "oracles.kcatnet.prott5_model", "Rostlab/prot_t5_xl_uniref50")
    )
    args.kcatnet_prott5_dir = str(args.kcatnet_prott5_dir or cfg_get(cfg, "oracles.kcatnet.prott5_dir", ""))

    args.graphkcat_model_root = str(
        args.graphkcat_model_root or cfg_get(cfg, "oracles.graphkcat.model_root", "models/GraphKcat")
    )
    args.graphkcat_checkpoint = str(
        args.graphkcat_checkpoint or cfg_get(cfg, "oracles.graphkcat.checkpoint", "models/GraphKcat/checkpoint/paper.pt")
    )
    args.graphkcat_cfg = str(args.graphkcat_cfg or cfg_get(cfg, "oracles.graphkcat.cfg", "TrainConfig_kcat_enz"))
    args.graphkcat_batch_size = int(
        args.graphkcat_batch_size if args.graphkcat_batch_size is not None else cfg_get(cfg, "oracles.graphkcat.batch_size", 2)
    )
    args.graphkcat_device = str(args.graphkcat_device or cfg_get(cfg, "oracles.graphkcat.device", "cuda:0"))
    args.graphkcat_distance_cutoff_a = float(
        args.graphkcat_distance_cutoff_a
        if args.graphkcat_distance_cutoff_a is not None
        else cfg_get(cfg, "oracles.graphkcat.distance_cutoff_a", 8.0)
    )
    args.graphkcat_std_default = float(
        args.graphkcat_std_default
        if args.graphkcat_std_default is not None
        else cfg_get(cfg, "oracles.graphkcat.std_default", 0.25)
    )
    args.graphkcat_heartbeat_sec = float(
        args.graphkcat_heartbeat_sec
        if args.graphkcat_heartbeat_sec is not None
        else cfg_get(cfg, "oracles.graphkcat.heartbeat_sec", 30.0)
    )

    args.fuse_w_kcatnet = float(
        args.fuse_w_kcatnet if args.fuse_w_kcatnet is not None else cfg_get(cfg, "oracles.fusion.w_kcatnet", 0.65)
    )
    args.fuse_w_graphkcat = float(
        args.fuse_w_graphkcat
        if args.fuse_w_graphkcat is not None
        else cfg_get(cfg, "oracles.fusion.w_graphkcat", 0.45)
    )
    args.fuse_w_agreement = float(
        args.fuse_w_agreement
        if args.fuse_w_agreement is not None
        else cfg_get(cfg, "oracles.fusion.w_agreement", 0.15)
    )
    args.fuse_kappa_kcatnet = float(
        args.fuse_kappa_kcatnet
        if args.fuse_kappa_kcatnet is not None
        else cfg_get(cfg, "oracles.fusion.kappa_kcatnet", 1.0)
    )
    args.fuse_kappa_graphkcat = float(
        args.fuse_kappa_graphkcat
        if args.fuse_kappa_graphkcat is not None
        else cfg_get(cfg, "oracles.fusion.kappa_graphkcat", 1.0)
    )
    if not args.strict_gates and bool(cfg_get(cfg, "run.strict_gates", False)):
        args.strict_gates = True

    round_dir = root / args.output_dir
    (round_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (round_dir / "data").mkdir(parents=True, exist_ok=True)
    (round_dir / "models").mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    _validate_kcat_dataset(dataset_path, logger)
    (round_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (round_dir / "logs").mkdir(parents=True, exist_ok=True)

    logger.info(
        "Kcat round start run_id=%s round_id=%d dataset=%s pool=%d kcatnet_budget=%d graphkcat_budget=%d strict=%s dry_run=%s cfg=%s",
        args.run_id,
        args.round_id,
        args.dataset_path,
        args.pool_size,
        args.kcatnet_budget,
        args.graphkcat_budget,
        args.strict_gates,
        args.dry_run,
        cfg_path,
    )

    manifest = {
        "run_id": args.run_id,
        "round_id": args.round_id,
        "dataset_path": args.dataset_path,
        "mode": "kcat_only",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    refresh_env_status = bool(args.require_ready and args.env_status_json)
    total_steps = 12 + (1 if refresh_env_status else 0)
    step_counter = {"value": 0}

    def run_step(name: str, cmd: list[str]) -> int:
        step_counter["value"] += 1
        idx = step_counter["value"]
        started = datetime.now(timezone.utc).isoformat()
        logger.info("STEP [%d/%d] start %s", idx, total_steps, name)
        rc, dt = _run(
            cmd,
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=name,
        )
        ended = datetime.now(timezone.utc).isoformat()
        logger.info("STEP [%d/%d] end %s rc=%d duration=%.2fs", idx, total_steps, name, rc, dt)
        manifest["steps"].append(
            {
                "name": name,
                "cmd": cmd,
                "returncode": rc,
                "started_utc": started,
                "completed_utc": ended,
                "duration_s": dt,
            }
        )
        return rc

    env_status_path = None
    if args.env_status_json:
        env_status = Path(args.env_status_json)
        env_status_path = env_status if env_status.is_absolute() else (root / env_status)

    teacher_ckpt = round_dir / "models" / f"teacher_round_{args.round_id}.ckpt"
    student_ckpt = round_dir / "models" / f"student_round_{args.round_id}.ckpt"
    surrogate_ckpt = round_dir / "models" / f"surrogate_round_{args.round_id}.ckpt"
    pool = round_dir / "data" / f"candidate_pool_round_{args.round_id}.jsonl"
    kcatnet_sel = round_dir / "data" / f"kcatnet_selected_round_{args.round_id}.jsonl"
    kcatnet_scored = round_dir / "data" / f"kcatnet_scored_round_{args.round_id}.jsonl"
    graph_sel = round_dir / "data" / f"graphkcat_selected_round_{args.round_id}.jsonl"
    graph_scored = round_dir / "data" / f"graphkcat_scored_round_{args.round_id}.jsonl"
    fused = round_dir / "data" / f"fused_kcat_round_{args.round_id}.jsonl"
    dr_next = round_dir / "data" / f"D_{args.round_id + 1}.jsonl"

    common = ["--round-id", str(args.round_id)]

    pre_oracle_steps = [
        (
            "fit_surrogate",
            [
                "python",
                str(root / "scripts/train/m3_fit_surrogate.py"),
                "--input-dr",
                str(root / args.dataset_path),
                "--output-dir",
                str(round_dir / "models"),
                *common,
                "--ensemble-size",
                str(args.surrogate_ensemble_size),
                "--max-checkpoints",
                str(args.max_checkpoints),
                "--seed",
                str(args.seed),
                *( ["--no-progress"] if args.no_progress else [] ),
            ],
        ),
        (
            "train_teacher",
            [
                "python",
                str(root / "scripts/train/m3_train_teacher_gfn.py"),
                "--input-dr",
                str(root / args.dataset_path),
                "--output-dir",
                str(round_dir / "models"),
                *common,
                "--steps",
                str(args.teacher_steps),
                "--gamma-off",
                str(args.teacher_gamma_off),
                "--surrogate-ckpt",
                str(surrogate_ckpt),
                "--max-checkpoints",
                str(args.max_checkpoints),
                "--seed",
                str(args.seed),
                *( ["--no-progress"] if args.no_progress else [] ),
            ],
        ),
        (
            "distill_student",
            [
                "python",
                str(root / "scripts/train/m3_distill_student.py"),
                "--teacher-ckpt",
                str(teacher_ckpt),
                "--input-dr",
                str(root / args.dataset_path),
                "--output-dir",
                str(round_dir / "models"),
                *common,
                "--steps",
                str(args.student_steps),
                "--max-checkpoints",
                str(args.max_checkpoints),
                "--seed",
                str(args.seed),
                *( ["--no-progress"] if args.no_progress else [] ),
            ],
        ),
        (
            "generate_pool",
            [
                "python",
                str(root / "scripts/train/m3_generate_student_pool.py"),
                "--student-ckpt",
                str(student_ckpt),
                "--input-dr",
                str(root / args.dataset_path),
                "--output-path",
                str(pool),
                "--run-id",
                args.run_id,
                *common,
                "--pool-size",
                str(args.pool_size),
                "--seed",
                str(args.seed),
                *( ["--no-progress"] if args.no_progress else [] ),
            ],
        ),
        (
            "select_kcatnet",
            [
                "python",
                str(root / "scripts/train/m3_select_kcatnet_batch.py"),
                "--input-path",
                str(pool),
                "--output-path",
                str(kcatnet_sel),
                "--budget",
                str(args.kcatnet_budget),
                *( ["--no-progress"] if args.no_progress else [] ),
            ],
        ),
    ]

    for name, cmd in pre_oracle_steps:
        rc = run_step(name, cmd)
        if rc != 0:
            (round_dir / "manifests" / "round_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
            return rc

    dispatch = root / "scripts/env/dispatch.py"

    kcatnet_cmd = (
        f"python {_q(root / 'scripts/prep/oracles/kcatnet_score.py')} "
        f"--candidate-path {_q(kcatnet_sel)} --output-path {_q(kcatnet_scored)} "
        f"--model-root {_q(args.kcatnet_model_root)} --checkpoint {_q(args.kcatnet_checkpoint)} "
        f"--config-path {_q(args.kcatnet_config_path)} --degree-path {_q(args.kcatnet_degree_path)} "
        f"--device {_q(args.kcatnet_device)} --batch-size {args.kcatnet_batch_size} "
        f"--std-default {args.kcatnet_std_default} --prott5-model {_q(args.kcatnet_prott5_model)} "
        f"{f'--prott5-dir {_q(args.kcatnet_prott5_dir)} ' if args.kcatnet_prott5_dir else ''}"
        f"--log-level {_q(args.log_level)} "
        f"{'--no-progress' if args.no_progress else ''}"
    )
    rc = run_step(
        "kcatnet_score",
        [
            "python",
            str(dispatch),
            "--env-name",
            args.kcatnet_env_name,
            "--cmd",
            kcatnet_cmd,
            *( ["--require-ready", "--env-status-json", args.env_status_json] if args.require_ready and args.env_status_json else [] ),
        ],
    )
    if rc != 0:
        return 4

    rc = run_step(
        "select_graphkcat",
        [
            "python",
            str(root / "scripts/train/m3_select_graphkcat_batch.py"),
            "--input-path",
            str(kcatnet_scored),
            "--output-path",
            str(graph_sel),
            "--budget",
            str(args.graphkcat_budget),
            *( ["--no-progress"] if args.no_progress else [] ),
        ],
    )
    if rc != 0:
        return rc

    if refresh_env_status and env_status_path is not None:
        rc = run_step(
            "refresh_env_status",
            [
                "bash",
                "-lc",
                (
                    f"RUN_HEALTH_CHECKS=1 {root / 'scripts/env/check_kcat_envs.sh'} {env_status_path} "
                    f"{args.kcatnet_env_name} {args.graphkcat_env_name}"
                ),
            ],
        )
        if rc != 0:
            return 4

    graphkcat_cmd = (
        f"python {_q(root / 'scripts/prep/oracles/graphkcat_score.py')} "
        f"--candidate-path {_q(graph_sel)} --output-path {_q(graph_scored)} "
        f"--model-root {_q(args.graphkcat_model_root)} --checkpoint {_q(args.graphkcat_checkpoint)} "
        f"--cfg {_q(args.graphkcat_cfg)} --batch-size {args.graphkcat_batch_size} --device {_q(args.graphkcat_device)} "
        f"--distance-cutoff-a {args.graphkcat_distance_cutoff_a} --std-default {args.graphkcat_std_default} "
        f"--heartbeat-sec {args.graphkcat_heartbeat_sec} --work-dir {_q(round_dir / 'data' / 'graphkcat_work')} "
        f"--log-level {_q(args.log_level)} "
        f"{'--no-progress' if args.no_progress else ''}"
    )
    rc = run_step(
        "graphkcat_score",
        [
            "python",
            str(dispatch),
            "--env-name",
            args.graphkcat_env_name,
            "--cmd",
            graphkcat_cmd,
            *( ["--require-ready", "--env-status-json", args.env_status_json] if args.require_ready and args.env_status_json else [] ),
        ],
    )
    if rc != 0:
        return 4

    rc = run_step(
        "fuse_kcat_scores",
        [
            "python",
            str(root / "scripts/prep/oracles/fuse_kcat_scores.py"),
            "--candidate-path",
            str(graph_scored),
            "--output-path",
            str(fused),
            "--w-kcatnet",
            str(args.fuse_w_kcatnet),
            "--w-graphkcat",
            str(args.fuse_w_graphkcat),
            "--w-agreement",
            str(args.fuse_w_agreement),
            "--kappa-kcatnet",
            str(args.fuse_kappa_kcatnet),
            "--kappa-graphkcat",
            str(args.fuse_kappa_graphkcat),
            *( ["--no-progress"] if args.no_progress else [] ),
        ],
    )
    if rc != 0:
        return rc

    rc = run_step(
        "append_labels",
        [
            "python",
            str(root / "scripts/train/m3_append_labels.py"),
            "--input-dr",
            str(root / args.dataset_path),
            "--labeled-path",
            str(fused),
            "--output-dr-next",
            str(dr_next),
            "--summary-path",
            str(round_dir / "manifests" / "append_summary.json"),
            *( ["--no-progress"] if args.no_progress else [] ),
        ],
    )
    if rc != 0:
        return rc

    rc = run_step(
        "eval_design_metrics",
        [
            "python",
            str(root / "scripts/eval/eval_design_metrics.py"),
            "--input-path",
            str(fused),
            "--output",
            str(round_dir / "metrics" / "round_metrics.json"),
        ],
    )
    if rc != 0:
        return rc

    rc = run_step(
        "eval_teacher_student",
        [
            "python",
            str(root / "scripts/eval/eval_m3_teacher_student.py"),
            "--teacher-ckpt",
            str(teacher_ckpt),
            "--student-ckpt",
            str(student_ckpt),
            "--output",
            str(round_dir / "metrics" / "teacher_student_eval.json"),
        ],
    )
    if rc != 0:
        return rc

    if args.dry_run:
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["pass"] = True
        (round_dir / "manifests" / "round_gate_report.json").write_text(
            json.dumps({"pass": True, "checks": {}, "strict": args.strict_gates, "dry_run": True}, indent=2, sort_keys=True)
        )
        (round_dir / "manifests" / "round_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        logger.info("Kcat round dry-run complete elapsed=%.2fs", time.perf_counter() - wall_t0)
        return 0

    ok, gate = _gate_report(round_dir, strict_gates=args.strict_gates)
    (round_dir / "manifests" / "round_gate_report.json").write_text(json.dumps(gate, indent=2, sort_keys=True))
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["pass"] = ok
    (round_dir / "manifests" / "round_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("Kcat round complete pass=%s strict=%s elapsed=%.2fs", ok, args.strict_gates, time.perf_counter() - wall_t0)

    if args.strict_gates and not ok:
        logger.error("Strict gates enabled and round failed gates")
        return 6

    logger.info("Next dataset path: %s", dr_next)
    print(dr_next)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run one Method III active-learning round end-to-end."""

from __future__ import annotations

import argparse
import json
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
            logger.info(
                "STEP %s still running elapsed=%.1fs",
                step_name or "<unnamed>",
                elapsed,
            )
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

    # Gate A: data integrity
    n_next = int(a.get("n_next", 0))
    report["checks"]["gate_a_n_next_positive"] = n_next > 0

    # Gate B: teacher-student KL
    kl = float(s.get("teacher_student_kl", 0.0))
    report["checks"]["gate_b_kl"] = kl <= 0.15

    # Gate C: diversity
    unique_fraction = float(m.get("unique_fraction", 0.0))
    report["checks"]["gate_c_unique_fraction"] = unique_fraction >= 0.5

    # Gate D: reward
    best_reward = float(m.get("best_reward", 0.0))
    report["checks"]["gate_d_best_reward"] = best_reward > 0.0

    report["pass"] = all(report["checks"].values())
    if strict_gates and not report["pass"]:
        return False, report
    return True, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/m3_default.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--bioemu-budget", type=int, default=None)
    parser.add_argument("--uma-budget", type=int, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--surrogate-ensemble-size", type=int, default=None)
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--teacher-gamma-off", type=float, default=None)
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument("--spurs-env-name", default=None)
    parser.add_argument("--bioemu-env-name", default=None)
    parser.add_argument("--uma-env-name", default=None)
    parser.add_argument("--bioemu-runtime-env-name", default=None)
    parser.add_argument("--spurs-repo-id", default=None)
    parser.add_argument("--spurs-chain", default=None)
    parser.add_argument("--bioemu-model-name", default=None)
    parser.add_argument("--bioemu-num-samples", type=int, default=None)
    parser.add_argument("--bioemu-tiered-enabled", dest="bioemu_tiered_enabled", action="store_true")
    parser.add_argument("--no-bioemu-tiered-enabled", dest="bioemu_tiered_enabled", action="store_false")
    parser.add_argument("--bioemu-tier1-num-samples", type=int, default=None)
    parser.add_argument("--bioemu-tier2-num-samples", type=int, default=None)
    parser.add_argument("--bioemu-tier3-num-samples", type=int, default=None)
    parser.add_argument("--bioemu-tier2-top-frac", type=float, default=None)
    parser.add_argument("--bioemu-tier2-top-min", type=int, default=None)
    parser.add_argument("--bioemu-tier3-top-frac-original", type=float, default=None)
    parser.add_argument("--bioemu-tier3-top-min", type=int, default=None)
    parser.add_argument("--bioemu-tier-rank-metric", choices=["risk_adjusted", "calibrated"], default=None)
    parser.add_argument("--bioemu-tier-risk-kappa", type=float, default=None)
    parser.add_argument("--bioemu-batch-size-100", type=int, default=None)
    parser.add_argument("--bioemu-target-vram-frac", type=float, default=None)
    parser.add_argument("--bioemu-batch-size-100-min", type=int, default=None)
    parser.add_argument("--bioemu-batch-size-100-max", type=int, default=None)
    parser.add_argument("--bioemu-max-proteins-per-step", type=int, default=None)
    parser.add_argument("--bioemu-vram-control-metric", choices=["allocated", "reserved"], default=None)
    parser.add_argument("--bioemu-low-utilization-mult", type=float, default=None)
    parser.add_argument("--bioemu-batch-size-100-max-growth-factor", type=float, default=None)
    parser.add_argument("--bioemu-batch-size-100-max-shrink-factor", type=float, default=None)
    parser.add_argument("--bioemu-sort-by-length-desc", dest="bioemu_sort_by_length_desc", action="store_true")
    parser.add_argument("--no-bioemu-sort-by-length-desc", dest="bioemu_sort_by_length_desc", action="store_false")
    parser.add_argument("--bioemu-filter-samples", dest="bioemu_filter_samples", action="store_true")
    parser.add_argument("--no-bioemu-filter-samples", dest="bioemu_filter_samples", action="store_false")
    parser.add_argument("--bioemu-auto-batch-from-vram", dest="bioemu_auto_batch_from_vram", action="store_true")
    parser.add_argument("--no-bioemu-auto-batch-from-vram", dest="bioemu_auto_batch_from_vram", action="store_false")
    parser.add_argument("--uma-model-name", default=None)
    parser.add_argument("--uma-workers", type=int, default=None)
    parser.add_argument("--uma-replicates", type=int, default=None)
    parser.add_argument("--uma-atom-budget", type=int, default=None)
    parser.add_argument("--uma-atom-budget-min", type=int, default=None)
    parser.add_argument("--uma-atom-budget-max", type=int, default=None)
    parser.add_argument("--uma-max-candidates-per-step", type=int, default=None)
    parser.add_argument("--uma-target-vram-frac", type=float, default=None)
    parser.add_argument("--uma-vram-control-metric", choices=["allocated", "reserved"], default=None)
    parser.add_argument("--uma-atom-budget-max-growth-factor", type=float, default=None)
    parser.add_argument("--uma-atom-budget-max-shrink-factor", type=float, default=None)
    parser.add_argument("--uma-auto-atom-budget-from-vram", dest="uma_auto_atom_budget_from_vram", action="store_true")
    parser.add_argument("--no-uma-auto-atom-budget-from-vram", dest="uma_auto_atom_budget_from_vram", action="store_false")
    parser.add_argument("--uma-estimate-prepared-atoms", dest="uma_estimate_prepared_atoms", action="store_true")
    parser.add_argument("--no-uma-estimate-prepared-atoms", dest="uma_estimate_prepared_atoms", action="store_false")
    parser.add_argument("--uma-strict-prepared-atom-estimation", action="store_true")
    parser.add_argument("--uma-hydration-shell-ang", type=float, default=None)
    parser.add_argument("--uma-hydration-ph", type=float, default=None)
    parser.add_argument("--uma-require-torch-cuda-vram", action="store_true")
    parser.add_argument("--strict-gates", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env-status-json", default=None)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--step-heartbeat-sec", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.set_defaults(
        bioemu_tiered_enabled=None,
        bioemu_auto_batch_from_vram=None,
        bioemu_sort_by_length_desc=None,
        bioemu_filter_samples=None,
        uma_auto_atom_budget_from_vram=None,
        uma_estimate_prepared_atoms=None,
    )
    args = parser.parse_args()

    wall_t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.progress import configure_logging
    from train.thermogfn.config_utils import load_yaml_config, cfg_get

    logger = configure_logging("orchestrate.m3_round", level=args.log_level)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_yaml_config(cfg_path)

    # CLI overrides config values.
    args.pool_size = int(args.pool_size if args.pool_size is not None else cfg_get(cfg, "round.pool_size", 50000))
    args.bioemu_budget = int(args.bioemu_budget if args.bioemu_budget is not None else cfg_get(cfg, "round.bioemu_budget", 512))
    args.uma_budget = int(args.uma_budget if args.uma_budget is not None else cfg_get(cfg, "round.uma_budget", 64))
    args.max_checkpoints = int(args.max_checkpoints if args.max_checkpoints is not None else cfg_get(cfg, "round.max_checkpoints", 5))
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
    args.spurs_env_name = str(args.spurs_env_name or cfg_get(cfg, "oracles.envs.spurs", "spurs"))
    args.bioemu_env_name = str(args.bioemu_env_name or cfg_get(cfg, "oracles.envs.bioemu", "bioemu"))
    args.uma_env_name = str(args.uma_env_name or cfg_get(cfg, "oracles.envs.uma", "uma-qc"))
    args.bioemu_runtime_env_name = str(args.bioemu_runtime_env_name or args.bioemu_env_name)
    args.spurs_repo_id = str(args.spurs_repo_id or cfg_get(cfg, "oracles.spurs.repo_id", "cyclization9/SPURS"))
    args.spurs_chain = str(args.spurs_chain or cfg_get(cfg, "oracles.spurs.chain", "A"))
    args.bioemu_model_name = str(args.bioemu_model_name or cfg_get(cfg, "oracles.bioemu.model_name", "bioemu-v1.1"))
    args.bioemu_num_samples = int(
        args.bioemu_num_samples if args.bioemu_num_samples is not None else cfg_get(cfg, "oracles.bioemu.num_samples", 2048)
    )
    if args.bioemu_tiered_enabled is None:
        args.bioemu_tiered_enabled = bool(cfg_get(cfg, "oracles.bioemu.tiered_enabled", True))
    args.bioemu_tier1_num_samples = int(
        args.bioemu_tier1_num_samples
        if args.bioemu_tier1_num_samples is not None
        else cfg_get(cfg, "oracles.bioemu.tier1_num_samples", 256)
    )
    args.bioemu_tier2_num_samples = int(
        args.bioemu_tier2_num_samples
        if args.bioemu_tier2_num_samples is not None
        else cfg_get(cfg, "oracles.bioemu.tier2_num_samples", 512)
    )
    args.bioemu_tier3_num_samples = int(
        args.bioemu_tier3_num_samples
        if args.bioemu_tier3_num_samples is not None
        else cfg_get(cfg, "oracles.bioemu.tier3_num_samples", args.bioemu_num_samples)
    )
    args.bioemu_tier2_top_frac = float(
        args.bioemu_tier2_top_frac
        if args.bioemu_tier2_top_frac is not None
        else cfg_get(cfg, "oracles.bioemu.tier2_top_frac", 0.20)
    )
    args.bioemu_tier2_top_min = int(
        args.bioemu_tier2_top_min
        if args.bioemu_tier2_top_min is not None
        else cfg_get(cfg, "oracles.bioemu.tier2_top_min", 1)
    )
    args.bioemu_tier3_top_frac_original = float(
        args.bioemu_tier3_top_frac_original
        if args.bioemu_tier3_top_frac_original is not None
        else cfg_get(cfg, "oracles.bioemu.tier3_top_frac_original", 0.05)
    )
    args.bioemu_tier3_top_min = int(
        args.bioemu_tier3_top_min
        if args.bioemu_tier3_top_min is not None
        else cfg_get(cfg, "oracles.bioemu.tier3_top_min", 1)
    )
    args.bioemu_tier_rank_metric = str(
        args.bioemu_tier_rank_metric
        if args.bioemu_tier_rank_metric is not None
        else cfg_get(cfg, "oracles.bioemu.tier_rank_metric", "risk_adjusted")
    )
    args.bioemu_tier_risk_kappa = float(
        args.bioemu_tier_risk_kappa
        if args.bioemu_tier_risk_kappa is not None
        else cfg_get(cfg, "oracles.bioemu.tier_risk_kappa", 0.5)
    )
    args.bioemu_batch_size_100 = int(
        args.bioemu_batch_size_100
        if args.bioemu_batch_size_100 is not None
        else cfg_get(cfg, "oracles.bioemu.batch_size_100", 10)
    )
    args.bioemu_target_vram_frac = float(
        args.bioemu_target_vram_frac
        if args.bioemu_target_vram_frac is not None
        else cfg_get(cfg, "oracles.bioemu.target_vram_frac", 0.90)
    )
    args.bioemu_batch_size_100_min = int(
        args.bioemu_batch_size_100_min
        if args.bioemu_batch_size_100_min is not None
        else cfg_get(cfg, "oracles.bioemu.batch_size_100_min", 1)
    )
    args.bioemu_batch_size_100_max = int(
        args.bioemu_batch_size_100_max
        if args.bioemu_batch_size_100_max is not None
        else cfg_get(cfg, "oracles.bioemu.batch_size_100_max", 512)
    )
    args.bioemu_max_proteins_per_step = int(
        args.bioemu_max_proteins_per_step
        if args.bioemu_max_proteins_per_step is not None
        else cfg_get(cfg, "oracles.bioemu.max_proteins_per_step", 8)
    )
    args.bioemu_vram_control_metric = str(
        args.bioemu_vram_control_metric
        if args.bioemu_vram_control_metric is not None
        else cfg_get(cfg, "oracles.bioemu.vram_control_metric", "reserved")
    )
    args.bioemu_low_utilization_mult = float(
        args.bioemu_low_utilization_mult
        if args.bioemu_low_utilization_mult is not None
        else cfg_get(cfg, "oracles.bioemu.low_utilization_mult", 0.5)
    )
    args.bioemu_batch_size_100_max_growth_factor = float(
        args.bioemu_batch_size_100_max_growth_factor
        if args.bioemu_batch_size_100_max_growth_factor is not None
        else cfg_get(cfg, "oracles.bioemu.batch_size_100_max_growth_factor", 6.0)
    )
    args.bioemu_batch_size_100_max_shrink_factor = float(
        args.bioemu_batch_size_100_max_shrink_factor
        if args.bioemu_batch_size_100_max_shrink_factor is not None
        else cfg_get(cfg, "oracles.bioemu.batch_size_100_max_shrink_factor", 0.5)
    )
    if args.bioemu_auto_batch_from_vram is None:
        args.bioemu_auto_batch_from_vram = bool(cfg_get(cfg, "oracles.bioemu.auto_batch_from_vram", True))
    if args.bioemu_sort_by_length_desc is None:
        args.bioemu_sort_by_length_desc = bool(cfg_get(cfg, "oracles.bioemu.sort_by_length_desc", True))
    if args.bioemu_filter_samples is None:
        args.bioemu_filter_samples = bool(cfg_get(cfg, "oracles.bioemu.filter_samples", False))
    args.uma_model_name = str(args.uma_model_name or cfg_get(cfg, "oracles.uma.model_name", "uma-s-1p1"))
    args.uma_workers = int(args.uma_workers if args.uma_workers is not None else cfg_get(cfg, "oracles.uma.workers", 1))
    args.uma_replicates = int(
        args.uma_replicates if args.uma_replicates is not None else cfg_get(cfg, "oracles.uma.replicates", 4)
    )
    args.uma_atom_budget = int(
        args.uma_atom_budget if args.uma_atom_budget is not None else cfg_get(cfg, "oracles.uma.atom_budget", 10000)
    )
    args.uma_atom_budget_min = int(
        args.uma_atom_budget_min if args.uma_atom_budget_min is not None else cfg_get(cfg, "oracles.uma.atom_budget_min", 1000)
    )
    args.uma_atom_budget_max = int(
        args.uma_atom_budget_max if args.uma_atom_budget_max is not None else cfg_get(cfg, "oracles.uma.atom_budget_max", 60000)
    )
    args.uma_max_candidates_per_step = int(
        args.uma_max_candidates_per_step
        if args.uma_max_candidates_per_step is not None
        else cfg_get(cfg, "oracles.uma.max_candidates_per_step", 8)
    )
    args.uma_target_vram_frac = float(
        args.uma_target_vram_frac
        if args.uma_target_vram_frac is not None
        else cfg_get(cfg, "oracles.uma.target_vram_frac", 0.90)
    )
    args.uma_vram_control_metric = str(
        args.uma_vram_control_metric
        if args.uma_vram_control_metric is not None
        else cfg_get(cfg, "oracles.uma.vram_control_metric", "reserved")
    )
    args.uma_atom_budget_max_growth_factor = float(
        args.uma_atom_budget_max_growth_factor
        if args.uma_atom_budget_max_growth_factor is not None
        else cfg_get(cfg, "oracles.uma.atom_budget_max_growth_factor", 2.0)
    )
    args.uma_atom_budget_max_shrink_factor = float(
        args.uma_atom_budget_max_shrink_factor
        if args.uma_atom_budget_max_shrink_factor is not None
        else cfg_get(cfg, "oracles.uma.atom_budget_max_shrink_factor", 0.5)
    )
    if args.uma_auto_atom_budget_from_vram is None:
        args.uma_auto_atom_budget_from_vram = bool(cfg_get(cfg, "oracles.uma.auto_atom_budget_from_vram", True))
    if args.uma_estimate_prepared_atoms is None:
        args.uma_estimate_prepared_atoms = bool(cfg_get(cfg, "oracles.uma.estimate_prepared_atoms", True))
    args.uma_hydration_shell_ang = float(
        args.uma_hydration_shell_ang
        if args.uma_hydration_shell_ang is not None
        else cfg_get(cfg, "oracles.uma.hydration_shell_ang", 3.5)
    )
    args.uma_hydration_ph = float(
        args.uma_hydration_ph
        if args.uma_hydration_ph is not None
        else cfg_get(cfg, "oracles.uma.hydration_ph", 7.0)
    )

    round_dir = root / args.output_dir
    (round_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (round_dir / "data").mkdir(parents=True, exist_ok=True)
    (round_dir / "models").mkdir(parents=True, exist_ok=True)
    (round_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (round_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger.info(
        "Round start run_id=%s round_id=%d dataset=%s pool=%d bioemu_budget=%d uma_budget=%d strict=%s dry_run=%s cfg=%s",
        args.run_id,
        args.round_id,
        args.dataset_path,
        args.pool_size,
        args.bioemu_budget,
        args.uma_budget,
        args.strict_gates,
        args.dry_run,
        cfg_path,
    )

    manifest = {
        "run_id": args.run_id,
        "round_id": args.round_id,
        "dataset_path": args.dataset_path,
        "mode": "production",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    refresh_env_status = bool(args.require_ready and args.env_status_json)
    total_steps = 14 + (1 if refresh_env_status else 0)
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

    dr_in = root / args.dataset_path
    env_status_path = None
    if args.env_status_json:
        env_status = Path(args.env_status_json)
        env_status_path = env_status if env_status.is_absolute() else (root / env_status)
    surrogate_ckpt = round_dir / "models" / f"surrogate_round_{args.round_id}.ckpt"
    teacher_ckpt = round_dir / "models" / f"teacher_round_{args.round_id}.ckpt"
    student_ckpt = round_dir / "models" / f"student_round_{args.round_id}.ckpt"
    pool = round_dir / "data" / f"candidate_pool_round_{args.round_id}.jsonl"
    pool_spurs = round_dir / "data" / f"candidate_pool_spurs_round_{args.round_id}.jsonl"
    bio_sel = round_dir / "data" / f"bioemu_selected_round_{args.round_id}.jsonl"
    bio_scored = round_dir / "data" / f"bioemu_scored_round_{args.round_id}.jsonl"
    uma_sel = round_dir / "data" / f"uma_selected_round_{args.round_id}.jsonl"
    uma_scored = round_dir / "data" / f"uma_scored_round_{args.round_id}.jsonl"
    fused = round_dir / "data" / f"fused_scored_round_{args.round_id}.jsonl"
    dr_next = round_dir / "data" / f"D_{args.round_id + 1}.jsonl"

    common = ["--round-id", str(args.round_id)]

    steps = [
        (
            "fit_surrogate",
            [
                "python",
                str(root / "scripts/train/m3_fit_surrogate.py"),
                "--input-dr",
                str(dr_in),
                "--output-dir",
                str(round_dir / "models"),
                *common,
                "--ensemble-size",
                str(args.surrogate_ensemble_size),
                "--max-checkpoints",
                str(args.max_checkpoints),
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "train_teacher",
            [
                "python",
                str(root / "scripts/train/m3_train_teacher_gfn.py"),
                "--input-dr",
                str(dr_in),
                "--output-dir",
                str(round_dir / "models"),
                *common,
                "--steps",
                str(args.teacher_steps),
                "--gamma-off",
                str(args.teacher_gamma_off),
                "--max-checkpoints",
                str(args.max_checkpoints),
                "--seed",
                str(args.seed),
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
                str(dr_in),
                "--output-dir",
                str(round_dir / "models"),
                *common,
                "--steps",
                str(args.student_steps),
                "--max-checkpoints",
                str(args.max_checkpoints),
                "--seed",
                str(args.seed),
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
                str(dr_in),
                "--output-path",
                str(pool),
                "--run-id",
                args.run_id,
                *common,
                "--pool-size",
                str(args.pool_size),
                "--seed",
                str(args.seed),
            ],
        ),
    ]

    for name, cmd in steps:
        rc = run_step(name, cmd)
        if rc != 0:
            (round_dir / "manifests" / "round_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
            return rc

    dispatch = root / "scripts/env/dispatch.py"

    spurs_cmd = (
        f"python {root / 'scripts/prep/oracles/spurs_score_single.py'} "
        f"--candidate-path {pool} --output-path {pool_spurs} "
        f"--repo-id {args.spurs_repo_id} --chain {args.spurs_chain}"
    )
    rc = run_step(
        "spurs_score",
        [
            "python",
            str(dispatch),
            "--env-name",
            args.spurs_env_name,
            "--cmd",
            spurs_cmd,
            *( ["--require-ready", "--env-status-json", args.env_status_json] if args.require_ready and args.env_status_json else [] ),
        ],
    )
    if rc != 0:
        (round_dir / "manifests" / "round_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return 4

    rc = run_step(
        "select_bioemu",
        [
            "python",
            str(root / "scripts/train/m3_select_bioemu_batch.py"),
            "--input-path",
            str(pool_spurs),
            "--output-path",
            str(bio_sel),
            "--budget",
            str(args.bioemu_budget),
        ],
    )
    if rc != 0:
        return rc

    rc = run_step(
        "bioemu_runtime_check",
        [
            "bash",
            str(root / "scripts/env/setup_bioemu_colabfold_runtime.sh"),
            "--bioemu-env",
            args.bioemu_runtime_env_name,
            "--check-only",
        ],
    )
    if rc != 0:
        return 4

    if refresh_env_status and env_status_path is not None:
        rc = run_step(
            "refresh_env_status",
            [
                "bash",
                "-lc",
                f"RUN_HEALTH_CHECKS=1 {root / 'scripts/env/check_envs.sh'} {env_status_path}",
            ],
        )
        if rc != 0:
            return 4

    bio_common = (
        f"--candidate-path {bio_sel} --output-path {bio_scored} "
        f"--model-name {args.bioemu_model_name} "
        f"--batch-size-100 {args.bioemu_batch_size_100} "
        f"--target-vram-frac {args.bioemu_target_vram_frac} "
        f"--batch-size-100-min {args.bioemu_batch_size_100_min} "
        f"--batch-size-100-max {args.bioemu_batch_size_100_max} "
        f"--max-proteins-per-step {args.bioemu_max_proteins_per_step} "
        f"--vram-control-metric {args.bioemu_vram_control_metric} "
        f"--low-utilization-mult {args.bioemu_low_utilization_mult} "
        f"--batch-size-100-max-growth-factor {args.bioemu_batch_size_100_max_growth_factor} "
        f"--batch-size-100-max-shrink-factor {args.bioemu_batch_size_100_max_shrink_factor} "
        f"{'--sort-by-length-desc' if args.bioemu_sort_by_length_desc else '--no-sort-by-length-desc'} "
        f"{'--filter-samples' if args.bioemu_filter_samples else '--no-filter-samples'} "
        f"{'--auto-batch-from-vram' if args.bioemu_auto_batch_from_vram else '--no-auto-batch-from-vram'}"
    )
    if args.bioemu_tiered_enabled:
        bio_cmd = (
            f"python {root / 'scripts/prep/oracles/bioemu_tiered_score.py'} "
            f"{bio_common} "
            f"--tier1-num-samples {args.bioemu_tier1_num_samples} "
            f"--tier2-num-samples {args.bioemu_tier2_num_samples} "
            f"--tier3-num-samples {args.bioemu_tier3_num_samples} "
            f"--tier2-top-frac {args.bioemu_tier2_top_frac} "
            f"--tier2-top-min {args.bioemu_tier2_top_min} "
            f"--tier3-top-frac-original {args.bioemu_tier3_top_frac_original} "
            f"--tier3-top-min {args.bioemu_tier3_top_min} "
            f"--tier-rank-metric {args.bioemu_tier_rank_metric} "
            f"--tier-risk-kappa {args.bioemu_tier_risk_kappa}"
        )
    else:
        bio_cmd = (
            f"python {root / 'scripts/prep/oracles/bioemu_sample_and_features.py'} "
            f"{bio_common} "
            f"--num-samples {args.bioemu_num_samples}"
        )
    rc = run_step(
        "bioemu_score",
        [
            "python",
            str(dispatch),
            "--env-name",
            args.bioemu_env_name,
            "--cmd",
            bio_cmd,
            *( ["--require-ready", "--env-status-json", args.env_status_json] if args.require_ready and args.env_status_json else [] ),
        ],
    )
    if rc != 0:
        return 4

    rc = run_step(
        "select_uma",
        [
            "python",
            str(root / "scripts/train/m3_select_uma_batch.py"),
            "--input-path",
            str(bio_scored),
            "--output-path",
            str(uma_sel),
            "--budget",
            str(args.uma_budget),
        ],
    )
    if rc != 0:
        return rc

    uma_cmd = (
        f"python {root / 'scripts/prep/oracles/uma_md_screen.py'} "
        f"--candidate-path {uma_sel} --output-path {uma_scored} "
        f"--model-name {args.uma_model_name} --workers {args.uma_workers} --replicates {args.uma_replicates} "
        f"--atom-budget {args.uma_atom_budget} --atom-budget-min {args.uma_atom_budget_min} --atom-budget-max {args.uma_atom_budget_max} "
        f"--max-candidates-per-step {args.uma_max_candidates_per_step} --target-vram-frac {args.uma_target_vram_frac} "
        f"--vram-control-metric {args.uma_vram_control_metric} "
        f"--atom-budget-max-growth-factor {args.uma_atom_budget_max_growth_factor} "
        f"--atom-budget-max-shrink-factor {args.uma_atom_budget_max_shrink_factor} "
        f"{'--auto-atom-budget-from-vram' if args.uma_auto_atom_budget_from_vram else '--no-auto-atom-budget-from-vram'} "
        f"{'--estimate-prepared-atoms' if args.uma_estimate_prepared_atoms else '--no-estimate-prepared-atoms'} "
        f"{'--strict-prepared-atom-estimation' if args.uma_strict_prepared_atom_estimation else ''} "
        f"--hydration-shell-ang {args.uma_hydration_shell_ang} --hydration-ph {args.uma_hydration_ph} "
        f"{'--require-torch-cuda-vram' if args.uma_require_torch_cuda_vram else ''}"
    )
    rc = run_step(
        "uma_score",
        [
            "python",
            str(dispatch),
            "--env-name",
            args.uma_env_name,
            "--cmd",
            uma_cmd,
            *( ["--require-ready", "--env-status-json", args.env_status_json] if args.require_ready and args.env_status_json else [] ),
        ],
    )
    if rc != 0:
        return 4

    rc = run_step(
        "fuse_scores",
        [
            "python",
            str(root / "scripts/prep/oracles/fuse_scores.py"),
            "--candidate-path",
            str(uma_scored),
            "--output-path",
            str(fused),
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
            str(dr_in),
            "--labeled-path",
            str(fused),
            "--output-dr-next",
            str(dr_next),
            "--summary-path",
            str(round_dir / "manifests" / "append_summary.json"),
        ],
    )
    if rc != 0:
        return rc

    # Round metrics.
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
        logger.info("Round dry-run complete elapsed=%.2fs", time.perf_counter() - wall_t0)
        return 0

    ok, gate = _gate_report(round_dir, strict_gates=args.strict_gates)
    (round_dir / "manifests" / "round_gate_report.json").write_text(json.dumps(gate, indent=2, sort_keys=True))
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["pass"] = ok
    (round_dir / "manifests" / "round_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("Round complete pass=%s strict=%s elapsed=%.2fs", ok, args.strict_gates, time.perf_counter() - wall_t0)

    if args.strict_gates and not ok:
        logger.error("Strict gates enabled and round failed gates")
        return 6

    logger.info("Next dataset path: %s", dr_next)
    print(dr_next)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

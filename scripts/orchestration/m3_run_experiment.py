#!/usr/bin/env python3
"""Run multiple Method III rounds with periodic test evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/m3_default.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-test-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--bioemu-budget", type=int, default=None)
    parser.add_argument("--uma-budget", type=int, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--surrogate-ensemble-size", type=int, default=None)
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--teacher-gamma-off", type=float, default=None)
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument("--periodic-test-num-candidates", type=int, default=None)
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
    parser.add_argument("--early-stop-overfit-gap-top8", type=float, default=None)
    parser.add_argument("--early-stop-overfit-gap-best", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
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
    from train.thermogfn.progress import configure_logging, iter_progress
    from train.thermogfn.config_utils import load_yaml_config, cfg_get

    logger = configure_logging("orchestrate.m3_experiment", level=args.log_level)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_yaml_config(cfg_path)

    # Resolve config defaults, preserve explicit CLI values.
    args.num_rounds = int(args.num_rounds if args.num_rounds is not None else cfg_get(cfg, "method3.rounds", 8))
    args.pool_size = int(args.pool_size if args.pool_size is not None else cfg_get(cfg, "round.pool_size", 50000))
    args.bioemu_budget = int(args.bioemu_budget if args.bioemu_budget is not None else cfg_get(cfg, "round.bioemu_budget", 512))
    args.uma_budget = int(args.uma_budget if args.uma_budget is not None else cfg_get(cfg, "round.uma_budget", 64))
    args.max_checkpoints = int(
        args.max_checkpoints if args.max_checkpoints is not None else cfg_get(cfg, "round.max_checkpoints", 5)
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
    args.periodic_test_num_candidates = int(
        args.periodic_test_num_candidates
        if args.periodic_test_num_candidates is not None
        else cfg_get(cfg, "periodic_eval.num_candidates", 256)
    )
    args.step_heartbeat_sec = float(
        args.step_heartbeat_sec if args.step_heartbeat_sec is not None else cfg_get(cfg, "round.step_heartbeat_sec", 30.0)
    )
    if args.early_stop_patience is None:
        args.early_stop_patience = int(cfg_get(cfg, "method3.early_stop.patience_rounds", 1))
    if args.early_stop_overfit_gap_top8 is None:
        args.early_stop_overfit_gap_top8 = cfg_get(cfg, "method3.early_stop.overfit_gap_top8_mean_reward", None)
    if args.early_stop_overfit_gap_best is None:
        args.early_stop_overfit_gap_best = cfg_get(cfg, "method3.early_stop.overfit_gap_best_reward", None)
    args.spurs_env_name = str(args.spurs_env_name or cfg_get(cfg, "oracles.envs.spurs", "spurs"))
    args.bioemu_env_name = str(args.bioemu_env_name or cfg_get(cfg, "oracles.envs.bioemu", "bioemu"))
    args.uma_env_name = str(args.uma_env_name or cfg_get(cfg, "oracles.envs.uma", "uma-qc"))
    args.bioemu_runtime_env_name = str(args.bioemu_runtime_env_name or args.bioemu_env_name)
    args.spurs_repo_id = str(args.spurs_repo_id or cfg_get(cfg, "oracles.spurs.repo_id", "cyclization9/SPURS"))
    args.spurs_chain = str(args.spurs_chain or cfg_get(cfg, "oracles.spurs.chain", "A"))
    args.bioemu_model_name = str(args.bioemu_model_name or cfg_get(cfg, "oracles.bioemu.model_name", "bioemu-v1.1"))
    args.bioemu_num_samples = int(
        args.bioemu_num_samples
        if args.bioemu_num_samples is not None
        else cfg_get(cfg, "oracles.bioemu.num_samples", 2048)
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
        args.uma_hydration_ph if args.uma_hydration_ph is not None else cfg_get(cfg, "oracles.uma.hydration_ph", 7.0)
    )
    if not args.strict_gates and bool(cfg_get(cfg, "run.strict_gates", False)):
        args.strict_gates = True

    current_dr = root / args.dataset_path
    current_test = root / args.dataset_test_path

    exp_root = root / args.output_root
    exp_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Experiment start run_id=%s rounds=%d start_round=%d train=%s test=%s dry_run=%s cfg=%s",
        args.run_id,
        args.num_rounds,
        args.start_round,
        current_dr,
        current_test,
        args.dry_run,
        cfg_path,
    )

    history: list[dict] = []
    overfit_breach_streak = 0
    round_values = list(range(args.start_round, args.start_round + args.num_rounds))
    for r in iter_progress(round_values, total=len(round_values), desc="m3:rounds", no_progress=args.no_progress, leave=True):
        round_t0 = time.perf_counter()
        round_dir = exp_root / f"round_{r:03d}"
        cmd = [
            "python",
            str(root / "scripts/orchestration/m3_run_round.py"),
            "--config",
            str(cfg_path),
            "--run-id",
            args.run_id,
            "--round-id",
            str(r),
            "--dataset-path",
            str(current_dr),
            "--output-dir",
            str(round_dir),
            "--pool-size",
            str(args.pool_size),
            "--bioemu-budget",
            str(args.bioemu_budget),
            "--uma-budget",
            str(args.uma_budget),
            "--max-checkpoints",
            str(args.max_checkpoints),
            "--surrogate-ensemble-size",
            str(args.surrogate_ensemble_size),
            "--teacher-steps",
            str(args.teacher_steps),
            "--teacher-gamma-off",
            str(args.teacher_gamma_off),
            "--student-steps",
            str(args.student_steps),
            "--spurs-env-name",
            args.spurs_env_name,
            "--bioemu-env-name",
            args.bioemu_env_name,
            "--uma-env-name",
            args.uma_env_name,
            "--bioemu-runtime-env-name",
            args.bioemu_runtime_env_name,
            "--spurs-repo-id",
            args.spurs_repo_id,
            "--spurs-chain",
            args.spurs_chain,
            "--bioemu-model-name",
            args.bioemu_model_name,
            "--bioemu-num-samples",
            str(args.bioemu_num_samples),
            "--bioemu-tiered-enabled" if args.bioemu_tiered_enabled else "--no-bioemu-tiered-enabled",
            "--bioemu-tier1-num-samples",
            str(args.bioemu_tier1_num_samples),
            "--bioemu-tier2-num-samples",
            str(args.bioemu_tier2_num_samples),
            "--bioemu-tier3-num-samples",
            str(args.bioemu_tier3_num_samples),
            "--bioemu-tier2-top-frac",
            str(args.bioemu_tier2_top_frac),
            "--bioemu-tier2-top-min",
            str(args.bioemu_tier2_top_min),
            "--bioemu-tier3-top-frac-original",
            str(args.bioemu_tier3_top_frac_original),
            "--bioemu-tier3-top-min",
            str(args.bioemu_tier3_top_min),
            "--bioemu-tier-rank-metric",
            str(args.bioemu_tier_rank_metric),
            "--bioemu-tier-risk-kappa",
            str(args.bioemu_tier_risk_kappa),
            "--bioemu-batch-size-100",
            str(args.bioemu_batch_size_100),
            "--bioemu-target-vram-frac",
            str(args.bioemu_target_vram_frac),
            "--bioemu-batch-size-100-min",
            str(args.bioemu_batch_size_100_min),
            "--bioemu-batch-size-100-max",
            str(args.bioemu_batch_size_100_max),
            "--bioemu-max-proteins-per-step",
            str(args.bioemu_max_proteins_per_step),
            "--bioemu-vram-control-metric",
            str(args.bioemu_vram_control_metric),
            "--bioemu-low-utilization-mult",
            str(args.bioemu_low_utilization_mult),
            "--bioemu-batch-size-100-max-growth-factor",
            str(args.bioemu_batch_size_100_max_growth_factor),
            "--bioemu-batch-size-100-max-shrink-factor",
            str(args.bioemu_batch_size_100_max_shrink_factor),
            "--bioemu-sort-by-length-desc" if args.bioemu_sort_by_length_desc else "--no-bioemu-sort-by-length-desc",
            "--bioemu-filter-samples" if args.bioemu_filter_samples else "--no-bioemu-filter-samples",
            "--bioemu-auto-batch-from-vram" if args.bioemu_auto_batch_from_vram else "--no-bioemu-auto-batch-from-vram",
            "--uma-model-name",
            args.uma_model_name,
            "--uma-workers",
            str(args.uma_workers),
            "--uma-replicates",
            str(args.uma_replicates),
            "--uma-atom-budget",
            str(args.uma_atom_budget),
            "--uma-atom-budget-min",
            str(args.uma_atom_budget_min),
            "--uma-atom-budget-max",
            str(args.uma_atom_budget_max),
            "--uma-max-candidates-per-step",
            str(args.uma_max_candidates_per_step),
            "--uma-target-vram-frac",
            str(args.uma_target_vram_frac),
            "--uma-vram-control-metric",
            str(args.uma_vram_control_metric),
            "--uma-atom-budget-max-growth-factor",
            str(args.uma_atom_budget_max_growth_factor),
            "--uma-atom-budget-max-shrink-factor",
            str(args.uma_atom_budget_max_shrink_factor),
            "--uma-auto-atom-budget-from-vram" if args.uma_auto_atom_budget_from_vram else "--no-uma-auto-atom-budget-from-vram",
            "--uma-estimate-prepared-atoms" if args.uma_estimate_prepared_atoms else "--no-uma-estimate-prepared-atoms",
            *([] if not args.uma_strict_prepared_atom_estimation else ["--uma-strict-prepared-atom-estimation"]),
            "--uma-hydration-shell-ang",
            str(args.uma_hydration_shell_ang),
            "--uma-hydration-ph",
            str(args.uma_hydration_ph),
            *([] if not args.uma_require_torch_cuda_vram else ["--uma-require-torch-cuda-vram"]),
            "--seed",
            str(args.seed),
        ]
        if args.strict_gates:
            cmd.append("--strict-gates")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.env_status_json:
            cmd.extend(["--env-status-json", args.env_status_json])
        if args.require_ready:
            cmd.append("--require-ready")

        logger.info("Round %d begin", r)
        rc, dt_round_train = _run(
            cmd,
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_train",
        )
        round_info = {"round": r, "returncode": rc, "round_train_duration_s": dt_round_train}

        if rc != 0:
            logger.error("Round %d failed during m3_run_round rc=%d", r, rc)
            history.append(round_info)
            break

        next_dr = round_dir / "data" / f"D_{r+1}.jsonl"
        if not args.dry_run and not next_dr.exists():
            round_info["returncode"] = 5
            logger.error("Round %d missing expected next dataset: %s", r, next_dr)
            history.append(round_info)
            break

        # Periodic train/test quality tracking for overfitting monitoring.
        # Train metric source: current round scored set.
        train_eval = round_dir / "metrics" / "round_metrics.json"
        # Test metric source: student-generated test candidates scored through full oracle chain.
        test_pool = round_dir / "data" / f"periodic_test_pool_round_{r}.jsonl"
        test_spurs = round_dir / "data" / f"periodic_test_spurs_round_{r}.jsonl"
        test_bio = round_dir / "data" / f"periodic_test_bioemu_round_{r}.jsonl"
        test_uma = round_dir / "data" / f"periodic_test_uma_round_{r}.jsonl"
        test_fused = round_dir / "data" / f"periodic_test_fused_round_{r}.jsonl"
        test_eval = round_dir / "metrics" / "periodic_test_eval.json"

        student_ckpt = round_dir / "models" / f"student_round_{r}.ckpt"
        rc_test_gen, dt_test_gen = _run(
            [
                "python",
                str(root / "scripts/infer/generate_unconditioned.py"),
                "--student-ckpt",
                str(student_ckpt),
                "--seed-dataset",
                str(current_test),
                "--output-path",
                str(test_pool),
                "--run-id",
                f"{args.run_id}_test",
                "--round-id",
                str(r),
                "--num-candidates",
                str(args.periodic_test_num_candidates),
                "--seed",
                str(args.seed + r),
            ],
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_periodic_test_generate",
        )
        round_info["periodic_test_generate_duration_s"] = dt_test_gen
        if rc_test_gen != 0:
            round_info["returncode"] = 4
            logger.error("Round %d periodic test generation failed rc=%d", r, rc_test_gen)
            history.append(round_info)
            break

        dispatch = root / "scripts/env/dispatch.py"
        rc_test_spurs, dt_test_spurs = _run(
            [
                "python",
                str(dispatch),
                "--env-name",
                args.spurs_env_name,
                "--cmd",
                (
                    f"python {root / 'scripts/prep/oracles/spurs_score_single.py'} "
                    f"--candidate-path {test_pool} --output-path {test_spurs} "
                    f"--repo-id {args.spurs_repo_id} --chain {args.spurs_chain}"
                ),
            ],
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_periodic_test_spurs",
        )
        rc_test_bio, dt_test_bio = _run(
            [
                "python",
                str(dispatch),
                "--env-name",
                args.bioemu_env_name,
                "--cmd",
                (
                    (
                        f"python {root / 'scripts/prep/oracles/bioemu_tiered_score.py'} "
                        f"--candidate-path {test_spurs} --output-path {test_bio} "
                        f"--model-name {args.bioemu_model_name} "
                        f"--tier1-num-samples {args.bioemu_tier1_num_samples} "
                        f"--tier2-num-samples {args.bioemu_tier2_num_samples} "
                        f"--tier3-num-samples {args.bioemu_tier3_num_samples} "
                        f"--tier2-top-frac {args.bioemu_tier2_top_frac} "
                        f"--tier2-top-min {args.bioemu_tier2_top_min} "
                        f"--tier3-top-frac-original {args.bioemu_tier3_top_frac_original} "
                        f"--tier3-top-min {args.bioemu_tier3_top_min} "
                        f"--tier-rank-metric {args.bioemu_tier_rank_metric} "
                        f"--tier-risk-kappa {args.bioemu_tier_risk_kappa} "
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
                    if args.bioemu_tiered_enabled
                    else (
                        f"python {root / 'scripts/prep/oracles/bioemu_sample_and_features.py'} "
                        f"--candidate-path {test_spurs} --output-path {test_bio} "
                        f"--model-name {args.bioemu_model_name} --num-samples {args.bioemu_num_samples} "
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
                ),
            ],
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_periodic_test_bioemu",
        )
        rc_test_uma, dt_test_uma = _run(
            [
                "python",
                str(dispatch),
                "--env-name",
                args.uma_env_name,
                "--cmd",
                (
                    f"python {root / 'scripts/prep/oracles/uma_md_screen.py'} "
                    f"--candidate-path {test_bio} --output-path {test_uma} "
                    f"--model-name {args.uma_model_name} --workers {args.uma_workers} "
                    f"--replicates {args.uma_replicates} "
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
                ),
            ],
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_periodic_test_uma",
        )
        rc_test_fuse, dt_test_fuse = _run(
            [
                "python",
                str(root / "scripts/prep/oracles/fuse_scores.py"),
                "--candidate-path",
                str(test_uma),
                "--output-path",
                str(test_fused),
            ],
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_periodic_test_fuse",
        )
        rc_test_eval, dt_test_eval = _run(
            [
                "python",
                str(root / "scripts/eval/eval_design_metrics.py"),
                "--input-path",
                str(test_fused),
                "--output",
                str(test_eval),
            ],
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_periodic_test_eval",
        )
        round_info["periodic_test_spurs_duration_s"] = dt_test_spurs
        round_info["periodic_test_bio_duration_s"] = dt_test_bio
        round_info["periodic_test_uma_duration_s"] = dt_test_uma
        round_info["periodic_test_fuse_duration_s"] = dt_test_fuse
        round_info["periodic_test_eval_duration_s"] = dt_test_eval

        if any(code != 0 for code in [rc_test_spurs, rc_test_bio, rc_test_uma, rc_test_fuse, rc_test_eval]):
            round_info["returncode"] = 4
            logger.error(
                "Round %d periodic scoring failed rc=[spurs:%d,bio:%d,uma:%d,fuse:%d,eval:%d]",
                r,
                rc_test_spurs,
                rc_test_bio,
                rc_test_uma,
                rc_test_fuse,
                rc_test_eval,
            )
            history.append(round_info)
            break

        rc_overfit, dt_overfit = _run(
            [
                "python",
                str(root / "scripts/eval/eval_oracle_calibration.py"),
                "--train-metrics",
                str(train_eval),
                "--test-metrics",
                str(test_eval),
                "--output",
                str(round_dir / "metrics" / "periodic_overfit.json"),
            ],
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_periodic_overfit",
        )
        round_info["periodic_overfit_duration_s"] = dt_overfit
        if rc_overfit != 0:
            round_info["returncode"] = 4
            logger.error("Round %d periodic overfit eval failed rc=%d", r, rc_overfit)
            history.append(round_info)
            break

        if not args.dry_run:
            overfit_path = round_dir / "metrics" / "periodic_overfit.json"
            if overfit_path.exists():
                overfit = json.loads(overfit_path.read_text())
                gap_top8 = float(overfit.get("overfit_gap_top8_mean_reward", 0.0))
                gap_best = float(overfit.get("overfit_gap_best_reward", 0.0))
                round_info["overfit_gap_top8_mean_reward"] = gap_top8
                round_info["overfit_gap_best_reward"] = gap_best
                breach = False
                reasons: list[str] = []
                if args.early_stop_overfit_gap_top8 is not None and gap_top8 > args.early_stop_overfit_gap_top8:
                    breach = True
                    reasons.append(
                        f"top8_gap={gap_top8:.4f} > {args.early_stop_overfit_gap_top8:.4f}"
                    )
                if args.early_stop_overfit_gap_best is not None and gap_best > args.early_stop_overfit_gap_best:
                    breach = True
                    reasons.append(
                        f"best_gap={gap_best:.4f} > {args.early_stop_overfit_gap_best:.4f}"
                    )
                if breach:
                    overfit_breach_streak += 1
                    round_info["overfit_breach"] = True
                    round_info["overfit_breach_reasons"] = reasons
                    logger.warning(
                        "Round %d overfit breach streak=%d reasons=%s",
                        r,
                        overfit_breach_streak,
                        "; ".join(reasons),
                    )
                else:
                    overfit_breach_streak = 0
                    round_info["overfit_breach"] = False
                patience = max(1, int(args.early_stop_patience))
                if breach and overfit_breach_streak >= patience:
                    round_info["early_stopped"] = True
                    round_info["early_stop_reason"] = "overfit_gap_threshold"
                    round_info["round_total_duration_s"] = time.perf_counter() - round_t0
                    logger.warning(
                        "Early stopping at round %d after %d consecutive overfit breaches",
                        r,
                        overfit_breach_streak,
                    )
                    history.append(round_info)
                    break

        round_info["round_total_duration_s"] = time.perf_counter() - round_t0
        logger.info("Round %d complete duration=%.2fs", r, round_info["round_total_duration_s"])
        history.append(round_info)
        current_dr = next_dr

    (exp_root / "experiment_history.json").write_text(json.dumps(history, indent=2, sort_keys=True))
    logger.info("Experiment complete rounds_recorded=%d elapsed=%.2fs", len(history), time.perf_counter() - wall_t0)
    print(exp_root / "experiment_history.json")
    return 0 if history and history[-1].get("returncode", 1) == 0 else (0 if args.dry_run else 1)


if __name__ == "__main__":
    raise SystemExit(main())

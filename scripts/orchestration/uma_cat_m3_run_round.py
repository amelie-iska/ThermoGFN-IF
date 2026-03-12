#!/usr/bin/env python3
"""Run one UMA-catalytic Method III active-learning round end-to-end."""

from __future__ import annotations

import argparse
import json
import math
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


def _q(v: str | Path) -> str:
    return shlex.quote(str(v))


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _validate_dataset(dataset_path: Path, logger) -> None:
    from train.thermogfn.io_utils import read_records

    rows = read_records(dataset_path)
    missing = []
    for rec in rows:
        ok = True
        for key in ("sequence", "substrate_smiles", "reactant_complex_path", "product_complex_path", "protein_chain_id"):
            if not rec.get(key):
                ok = False
                break
        if not rec.get("pocket_positions"):
            ok = False
        if not ok:
            missing.append(rec)
    if not missing:
        return
    labels = []
    for rec in missing[:10]:
        labels.append(str(rec.get("candidate_id") or rec.get("backbone_id") or "<unknown>"))
    raise RuntimeError(
        f"UMA-cat dataset missing required catalytic fields for {len(missing)}/{len(rows)} records in {dataset_path}. "
        "Required: sequence, substrate_smiles, reactant_complex_path, product_complex_path, protein_chain_id, pocket_positions. "
        f"Examples: {', '.join(labels)}"
    )


def _gate_report(round_dir: Path, strict_gates: bool) -> tuple[bool, dict]:
    metrics_path = round_dir / "metrics" / "round_metrics.json"
    student_path = round_dir / "metrics" / "student_metrics.json"
    teacher_student_path = round_dir / "metrics" / "teacher_student_eval.json"
    summary_path = round_dir / "manifests" / "append_summary.json"
    report = {"pass": True, "checks": {}, "strict": strict_gates}

    m = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    s = json.loads(student_path.read_text()) if student_path.exists() else {}
    if not s and teacher_student_path.exists():
        s = json.loads(teacher_student_path.read_text())
    a = json.loads(summary_path.read_text()) if summary_path.exists() else {}

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/uma_cat_m3_default.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--uma-cat-budget", type=int, default=None)
    parser.add_argument("--graphkcat-budget", type=int, default=None)
    parser.add_argument("--graphkcat-prefilter-fraction", type=float, default=None)
    parser.add_argument("--graphkcat-prefilter-risk-kappa", type=float, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--surrogate-ensemble-size", type=int, default=None)
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--teacher-gamma-off", type=float, default=None)
    parser.add_argument("--student-steps", type=int, default=None)

    parser.add_argument("--packer-env-name", default=None)
    parser.add_argument("--uma-env-name", default=None)
    parser.add_argument("--graphkcat-env-name", default=None)

    parser.add_argument("--ligandmpnn-root", default=None)
    parser.add_argument("--packer-checkpoint-sc", default=None)
    parser.add_argument("--packer-device", default=None)
    parser.add_argument("--packer-pack-with-ligand-context", type=int, default=None)
    parser.add_argument("--packer-repack-everything", type=int, default=None)
    parser.add_argument("--packer-sc-num-denoising-steps", type=int, default=None)
    parser.add_argument("--packer-sc-num-samples", type=int, default=None)
    parser.add_argument("--packer-parse-atoms-with-zero-occupancy", type=int, default=None)
    parser.add_argument("--packer-force-hetatm", type=int, default=None)

    parser.add_argument("--uma-model-name", default=None)
    parser.add_argument("--uma-device", default=None)
    parser.add_argument("--uma-calculator-workers", type=int, default=None)
    parser.add_argument("--uma-max-atoms", type=int, default=None)
    parser.add_argument("--uma-temperature-k", type=float, default=None)
    parser.add_argument("--uma-broad-steps", type=int, default=None)
    parser.add_argument("--uma-broad-replicas", type=int, default=None)
    parser.add_argument("--uma-broad-save-every", type=int, default=None)
    parser.add_argument("--uma-run-smd", type=int, default=None)
    parser.add_argument("--uma-run-reverse-smd", type=int, default=None)
    parser.add_argument("--uma-smd-images", type=int, default=None)
    parser.add_argument("--uma-smd-steps-per-image", type=int, default=None)
    parser.add_argument("--uma-smd-replicas", type=int, default=None)
    parser.add_argument("--uma-run-pmf", type=int, default=None)
    parser.add_argument("--uma-pmf-windows", type=int, default=None)
    parser.add_argument("--uma-pmf-steps-per-window", type=int, default=None)
    parser.add_argument("--uma-pmf-save-every", type=int, default=None)
    parser.add_argument("--uma-pmf-replicas", type=int, default=None)
    parser.add_argument("--uma-pmf-k-window-eva2", type=float, default=None)

    parser.add_argument("--graphkcat-model-root", default=None)
    parser.add_argument("--graphkcat-checkpoint", default=None)
    parser.add_argument("--graphkcat-cfg", default=None)
    parser.add_argument("--graphkcat-batch-size", type=int, default=None)
    parser.add_argument("--graphkcat-device", default=None)
    parser.add_argument("--graphkcat-distance-cutoff-a", type=float, default=None)
    parser.add_argument("--graphkcat-std-default", type=float, default=None)
    parser.add_argument("--graphkcat-mc-dropout-samples", type=int, default=None)
    parser.add_argument("--graphkcat-mc-dropout-seed", type=int, default=None)
    parser.add_argument("--graphkcat-heartbeat-sec", type=float, default=None)

    parser.add_argument("--fuse-w-uma-cat", type=float, default=None)
    parser.add_argument("--fuse-w-graphkcat", type=float, default=None)
    parser.add_argument("--fuse-w-agreement", type=float, default=None)
    parser.add_argument("--fuse-kappa-uma-cat", type=float, default=None)
    parser.add_argument("--fuse-kappa-graphkcat", type=float, default=None)

    parser.add_argument("--strict-gates", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env-status-json", default=None)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--step-heartbeat-sec", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--wandb-enabled", type=int, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    wall_t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.config_utils import cfg_get, load_yaml_config
    from train.thermogfn.io_utils import read_records
    from train.thermogfn.metrics_utils import summarize_candidate_records
    from train.thermogfn.progress import configure_logging, make_progress
    from train.thermogfn.wandb_utils import WandbRun, parse_tags, read_history_jsonl

    logger = configure_logging("orchestrate.uma_cat_round", level=args.log_level)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_yaml_config(cfg_path)

    args.pool_size = int(args.pool_size if args.pool_size is not None else cfg_get(cfg, "round.pool_size", 50000))
    args.uma_cat_budget = int(
        args.uma_cat_budget if args.uma_cat_budget is not None else cfg_get(cfg, "round.uma_cat_budget", 256)
    )
    args.graphkcat_budget = int(
        args.graphkcat_budget if args.graphkcat_budget is not None else cfg_get(cfg, "round.graphkcat_budget", 128)
    )
    args.graphkcat_prefilter_fraction = float(
        args.graphkcat_prefilter_fraction
        if args.graphkcat_prefilter_fraction is not None
        else cfg_get(cfg, "round.graphkcat_prefilter_fraction", 0.5)
    )
    args.graphkcat_prefilter_risk_kappa = float(
        args.graphkcat_prefilter_risk_kappa
        if args.graphkcat_prefilter_risk_kappa is not None
        else cfg_get(cfg, "round.graphkcat_prefilter_risk_kappa", 0.5)
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

    args.packer_env_name = str(args.packer_env_name or cfg_get(cfg, "oracles.envs.packer", "ligandmpnn_env"))
    args.uma_env_name = str(args.uma_env_name or cfg_get(cfg, "oracles.envs.uma_cat", "mora-uma"))
    args.graphkcat_env_name = str(args.graphkcat_env_name or cfg_get(cfg, "oracles.envs.graphkcat", "apodock"))

    args.ligandmpnn_root = str(args.ligandmpnn_root or cfg_get(cfg, "oracles.packer.model_root", "models/LigandMPNN"))
    args.packer_checkpoint_sc = str(
        args.packer_checkpoint_sc
        or cfg_get(cfg, "oracles.packer.checkpoint_sc", "models/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt")
    )
    args.packer_device = str(args.packer_device or cfg_get(cfg, "oracles.packer.device", "cuda:0"))
    args.packer_pack_with_ligand_context = int(
        args.packer_pack_with_ligand_context
        if args.packer_pack_with_ligand_context is not None
        else cfg_get(cfg, "oracles.packer.pack_with_ligand_context", 1)
    )
    args.packer_repack_everything = int(
        args.packer_repack_everything
        if args.packer_repack_everything is not None
        else cfg_get(cfg, "oracles.packer.repack_everything", 1)
    )
    args.packer_sc_num_denoising_steps = int(
        args.packer_sc_num_denoising_steps
        if args.packer_sc_num_denoising_steps is not None
        else cfg_get(cfg, "oracles.packer.sc_num_denoising_steps", 3)
    )
    args.packer_sc_num_samples = int(
        args.packer_sc_num_samples
        if args.packer_sc_num_samples is not None
        else cfg_get(cfg, "oracles.packer.sc_num_samples", 16)
    )
    args.packer_parse_atoms_with_zero_occupancy = int(
        args.packer_parse_atoms_with_zero_occupancy
        if args.packer_parse_atoms_with_zero_occupancy is not None
        else cfg_get(cfg, "oracles.packer.parse_atoms_with_zero_occupancy", 0)
    )
    args.packer_force_hetatm = int(
        args.packer_force_hetatm
        if args.packer_force_hetatm is not None
        else cfg_get(cfg, "oracles.packer.force_hetatm", 1)
    )

    args.uma_model_name = str(args.uma_model_name or cfg_get(cfg, "oracles.uma_cat.model_name", "uma-s-1p1"))
    args.uma_device = str(args.uma_device or cfg_get(cfg, "oracles.uma_cat.device", "cuda:0"))
    args.uma_calculator_workers = int(
        args.uma_calculator_workers
        if args.uma_calculator_workers is not None
        else cfg_get(cfg, "oracles.uma_cat.calculator_workers", 1)
    )
    args.uma_max_atoms = int(args.uma_max_atoms if args.uma_max_atoms is not None else cfg_get(cfg, "oracles.uma_cat.max_atoms", 12000))
    args.uma_temperature_k = float(
        args.uma_temperature_k if args.uma_temperature_k is not None else cfg_get(cfg, "oracles.uma_cat.broad.temperature_k", 300.0)
    )
    args.uma_broad_steps = int(
        args.uma_broad_steps if args.uma_broad_steps is not None else cfg_get(cfg, "oracles.uma_cat.broad.steps", 500)
    )
    args.uma_broad_replicas = int(
        args.uma_broad_replicas if args.uma_broad_replicas is not None else cfg_get(cfg, "oracles.uma_cat.broad.replicas", 2)
    )
    args.uma_broad_save_every = int(
        args.uma_broad_save_every if args.uma_broad_save_every is not None else cfg_get(cfg, "oracles.uma_cat.broad.save_every", 10)
    )
    args.uma_run_smd = int(
        args.uma_run_smd if args.uma_run_smd is not None else cfg_get(cfg, "oracles.uma_cat.smd.enabled", True)
    )
    args.uma_run_reverse_smd = int(
        args.uma_run_reverse_smd if args.uma_run_reverse_smd is not None else cfg_get(cfg, "oracles.uma_cat.smd.reverse", True)
    )
    args.uma_smd_images = int(
        args.uma_smd_images if args.uma_smd_images is not None else cfg_get(cfg, "oracles.uma_cat.smd.images", 24)
    )
    args.uma_smd_steps_per_image = int(
        args.uma_smd_steps_per_image
        if args.uma_smd_steps_per_image is not None
        else cfg_get(cfg, "oracles.uma_cat.smd.steps_per_image", 25)
    )
    args.uma_smd_replicas = int(
        args.uma_smd_replicas if args.uma_smd_replicas is not None else cfg_get(cfg, "oracles.uma_cat.smd.replicas", 2)
    )
    args.uma_run_pmf = int(
        args.uma_run_pmf if args.uma_run_pmf is not None else cfg_get(cfg, "oracles.uma_cat.pmf.enabled", False)
    )
    args.uma_pmf_windows = int(
        args.uma_pmf_windows if args.uma_pmf_windows is not None else cfg_get(cfg, "oracles.uma_cat.pmf.windows", 16)
    )
    args.uma_pmf_steps_per_window = int(
        args.uma_pmf_steps_per_window
        if args.uma_pmf_steps_per_window is not None
        else cfg_get(cfg, "oracles.uma_cat.pmf.steps_per_window", 100)
    )
    args.uma_pmf_save_every = int(
        args.uma_pmf_save_every
        if args.uma_pmf_save_every is not None
        else cfg_get(cfg, "oracles.uma_cat.pmf.save_every", 5)
    )
    args.uma_pmf_replicas = int(
        args.uma_pmf_replicas if args.uma_pmf_replicas is not None else cfg_get(cfg, "oracles.uma_cat.pmf.replicas", 1)
    )
    args.uma_pmf_k_window_eva2 = float(
        args.uma_pmf_k_window_eva2
        if args.uma_pmf_k_window_eva2 is not None
        else cfg_get(cfg, "oracles.uma_cat.pmf.k_window_eva2", 2.0)
    )

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
    args.graphkcat_mc_dropout_samples = int(
        args.graphkcat_mc_dropout_samples
        if args.graphkcat_mc_dropout_samples is not None
        else cfg_get(cfg, "oracles.graphkcat.mc_dropout_samples", 8)
    )
    args.graphkcat_mc_dropout_seed = int(
        args.graphkcat_mc_dropout_seed
        if args.graphkcat_mc_dropout_seed is not None
        else cfg_get(cfg, "oracles.graphkcat.mc_dropout_seed", 13)
    )
    args.graphkcat_heartbeat_sec = float(
        args.graphkcat_heartbeat_sec
        if args.graphkcat_heartbeat_sec is not None
        else cfg_get(cfg, "oracles.graphkcat.heartbeat_sec", 30.0)
    )

    args.fuse_w_uma_cat = float(
        args.fuse_w_uma_cat if args.fuse_w_uma_cat is not None else cfg_get(cfg, "oracles.fusion.w_uma_cat", 0.90)
    )
    args.fuse_w_graphkcat = float(
        args.fuse_w_graphkcat if args.fuse_w_graphkcat is not None else cfg_get(cfg, "oracles.fusion.w_graphkcat", 0.55)
    )
    args.fuse_w_agreement = float(
        args.fuse_w_agreement if args.fuse_w_agreement is not None else cfg_get(cfg, "oracles.fusion.w_agreement", 0.20)
    )
    args.fuse_kappa_uma_cat = float(
        args.fuse_kappa_uma_cat
        if args.fuse_kappa_uma_cat is not None
        else cfg_get(cfg, "oracles.fusion.kappa_uma_cat", 1.0)
    )
    args.fuse_kappa_graphkcat = float(
        args.fuse_kappa_graphkcat
        if args.fuse_kappa_graphkcat is not None
        else cfg_get(cfg, "oracles.fusion.kappa_graphkcat", 1.0)
    )
    args.wandb_enabled = bool(
        int(args.wandb_enabled)
        if args.wandb_enabled is not None
        else int(cfg_get(cfg, "logging.wandb.enabled", 1))
    )
    args.wandb_project = str(args.wandb_project or cfg_get(cfg, "logging.wandb.project", "thermogfn"))
    args.wandb_entity = args.wandb_entity or cfg_get(cfg, "logging.wandb.entity", None)
    args.wandb_mode = str(args.wandb_mode or cfg_get(cfg, "logging.wandb.mode", "offline"))
    args.wandb_group = str(args.wandb_group or cfg_get(cfg, "logging.wandb.group", args.run_id))
    args.wandb_tags = parse_tags(args.wandb_tags or cfg_get(cfg, "logging.wandb.tags", ["uma-cat", "method3"]))
    args.wandb_run_name = str(
        args.wandb_run_name or cfg_get(cfg, "logging.wandb.run_name", f"{args.run_id}-round-{args.round_id:03d}")
    )
    if not args.strict_gates and bool(cfg_get(cfg, "run.strict_gates", False)):
        args.strict_gates = True

    round_dir = Path(args.output_dir)
    if not round_dir.is_absolute():
        round_dir = root / round_dir
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (round_dir / "logs").mkdir(parents=True, exist_ok=True)
    (round_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (round_dir / "data").mkdir(parents=True, exist_ok=True)
    (round_dir / "models").mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path
    _validate_dataset(dataset_path, logger)

    wandb_run = WandbRun(
        enabled=args.wandb_enabled,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.wandb_run_name,
        group=args.wandb_group,
        job_type="uma_cat_round",
        tags=args.wandb_tags,
        config={
            "run_id": args.run_id,
            "round_id": args.round_id,
            "dataset_path": str(dataset_path),
            "pool_size": args.pool_size,
            "uma_cat_budget": args.uma_cat_budget,
            "graphkcat_budget": args.graphkcat_budget,
            "graphkcat_prefilter_fraction": args.graphkcat_prefilter_fraction,
            "teacher_steps": args.teacher_steps,
            "student_steps": args.student_steps,
            "strict_gates": args.strict_gates,
            "config_path": str(cfg_path),
        },
    )

    logger.info(
        "UMA-cat round start run_id=%s round_id=%d dataset=%s pool=%d uma_budget=%d graph_budget=%d graph_prefilter_fraction=%.3f strict=%s dry_run=%s cfg=%s",
        args.run_id,
        args.round_id,
        args.dataset_path,
        args.pool_size,
        args.uma_cat_budget,
        args.graphkcat_budget,
        args.graphkcat_prefilter_fraction,
        args.strict_gates,
        args.dry_run,
        cfg_path,
    )

    manifest = {
        "run_id": args.run_id,
        "round_id": args.round_id,
        "dataset_path": args.dataset_path,
        "mode": "uma_cat",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    step_counter = {"value": 0}
    step_bar = None

    surrogate_metrics_path = round_dir / "models" / "surrogate_metrics.json"
    surrogate_history_path = round_dir / "metrics" / f"surrogate_history_round_{args.round_id}.jsonl"
    teacher_metrics_path = round_dir / "models" / "teacher_metrics.json"
    teacher_history_path = round_dir / "metrics" / f"teacher_history_round_{args.round_id}.jsonl"
    student_metrics_path = round_dir / "models" / "student_metrics.json"
    student_history_path = round_dir / "metrics" / f"student_history_round_{args.round_id}.jsonl"
    pool_metrics_path = round_dir / "metrics" / f"student_pool_metrics_round_{args.round_id}.json"
    graph_pool_summary_path = round_dir / "metrics" / f"graphkcat_pool_summary_round_{args.round_id}.json"
    graph_selected_summary_path = round_dir / "metrics" / f"graphkcat_selected_summary_round_{args.round_id}.json"
    uma_summary_path = round_dir / "metrics" / f"uma_cat_summary_round_{args.round_id}.json"

    def _phase_step(idx: int, local_step: int = 0) -> int:
        return int(idx) * 1000 + int(local_step)

    def _log_json_metrics(prefix: str, path: Path, *, idx: int) -> None:
        payload = _read_json_if_exists(path)
        if not payload:
            return
        wandb_run.log_prefixed(prefix, payload, step=_phase_step(idx, 999))
        wandb_run.summary_update(payload, prefix=prefix)

    def _log_history(prefix: str, path: Path, *, idx: int) -> None:
        rows = read_history_jsonl(path)
        if not rows:
            return
        wandb_run.log_history(rows, prefix=prefix, base_step=_phase_step(idx, 0))

    def _log_records_summary(prefix: str, path: Path, *, idx: int) -> None:
        if not path.exists():
            return
        summary = summarize_candidate_records(read_records(path))
        wandb_run.log_prefixed(prefix, summary, step=_phase_step(idx, 998))
        wandb_run.summary_update(summary, prefix=prefix)

    def _post_step_history_metrics(name: str, idx: int) -> None:
        if name == "fit_surrogate":
            _log_history("surrogate/history", surrogate_history_path, idx=idx)
        elif name == "train_teacher":
            _log_history("teacher/history", teacher_history_path, idx=idx)
        elif name == "distill_student":
            _log_history("student/history", student_history_path, idx=idx)
    def _post_step_summary_metrics(name: str, idx: int) -> None:
        if name == "fit_surrogate":
            _log_json_metrics("surrogate", surrogate_metrics_path, idx=idx)
        elif name == "train_teacher":
            _log_json_metrics("teacher", teacher_metrics_path, idx=idx)
        elif name == "distill_student":
            _log_json_metrics("student", student_metrics_path, idx=idx)
        elif name == "generate_pool":
            _log_records_summary("student/pool_records", pool, idx=idx)
            _log_json_metrics("student/pool", pool_metrics_path, idx=idx)
        elif name == "graphkcat_score_pool":
            _log_json_metrics("graphkcat/pool", graph_pool_summary_path, idx=idx)
        elif name == "prefilter_graphkcat":
            _log_records_summary("graphkcat/prefilter", graph_prefiltered, idx=idx)
        elif name == "select_uma_cat":
            _log_records_summary("uma_cat/selected", uma_sel, idx=idx)
        elif name == "pack_pool_for_graphkcat":
            _log_records_summary("graphkcat/packed_pool", graph_pool_packed, idx=idx)
        elif name == "pack_candidates":
            _log_records_summary("uma_cat/packed", packed, idx=idx)
        elif name == "uma_catalytic_score":
            _log_json_metrics("uma_cat", uma_summary_path, idx=idx)
        elif name == "graphkcat_score":
            _log_json_metrics("graphkcat/selected", graph_selected_summary_path, idx=idx)
        elif name == "fuse_catalytic_scores":
            _log_records_summary("fused", fused, idx=idx)
        elif name == "append_labels":
            _log_json_metrics("append", round_dir / "manifests" / "append_summary.json", idx=idx)
        elif name == "eval_design_metrics":
            _log_json_metrics("round", round_dir / "metrics" / "round_metrics.json", idx=idx)
        elif name == "eval_teacher_student":
            _log_json_metrics("teacher_student", round_dir / "metrics" / "teacher_student_eval.json", idx=idx)

    def run_step(name: str, cmd: list[str]) -> int:
        step_counter["value"] += 1
        idx = step_counter["value"]
        started = datetime.now(timezone.utc).isoformat()
        if step_bar is not None:
            step_bar.set_postfix_str(f"running={name} step={idx}/{total_steps}")
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
        if step_bar is not None:
            step_bar.update(1)
            step_bar.set_postfix_str(f"completed={idx}/{total_steps} last={name} rc={rc} dt={dt:.1f}s")
        _post_step_history_metrics(name, idx)
        wandb_run.log(
            {
                "step_index": idx,
                "step_duration_s": float(dt),
                "step_returncode": int(rc),
                "step_name": name,
            },
            step=_phase_step(idx, 997),
        )
        _post_step_summary_metrics(name, idx)
        (round_dir / "manifests" / "round_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return rc

    surrogate_ckpt = round_dir / "models" / f"surrogate_round_{args.round_id}.ckpt"
    teacher_ckpt = round_dir / "models" / f"teacher_round_{args.round_id}.ckpt"
    student_ckpt = round_dir / "models" / f"student_round_{args.round_id}.ckpt"
    pool = round_dir / "data" / f"candidate_pool_round_{args.round_id}.jsonl"
    graph_pool_packed = round_dir / "data" / f"candidate_pool_packed_round_{args.round_id}.jsonl"
    graph_pool_scored = round_dir / "data" / f"graphkcat_pool_scored_round_{args.round_id}.jsonl"
    graph_prefiltered = round_dir / "data" / f"graphkcat_prefiltered_round_{args.round_id}.jsonl"
    uma_sel = round_dir / "data" / f"uma_cat_selected_round_{args.round_id}.jsonl"
    packed = round_dir / "data" / f"packed_round_{args.round_id}.jsonl"
    uma_scored = round_dir / "data" / f"uma_cat_scored_round_{args.round_id}.jsonl"
    graph_sel = round_dir / "data" / f"graphkcat_selected_round_{args.round_id}.jsonl"
    graph_scored = round_dir / "data" / f"graphkcat_scored_round_{args.round_id}.jsonl"
    fused = round_dir / "data" / f"fused_catalytic_round_{args.round_id}.jsonl"
    dr_next = round_dir / "data" / f"D_{args.round_id + 1}.jsonl"
    dispatch = root / "scripts/env/dispatch.py"

    def _dispatch_wrap(env_name: str, inner_cmd: list[str]) -> list[str]:
        wrapped = [
            "python",
            str(dispatch),
            "--env-name",
            str(env_name),
            "--cmd",
            " ".join(_q(part) for part in inner_cmd),
        ]
        if args.require_ready and args.env_status_json:
            wrapped.extend(["--require-ready", "--env-status-json", args.env_status_json])
        return wrapped

    pre_oracle_steps = [
        (
            "fit_surrogate",
            _dispatch_wrap(args.uma_env_name, [
                "python",
                str(root / "scripts/train/m3_fit_surrogate.py"),
                "--input-dr",
                str(dataset_path),
                "--output-dir",
                str(round_dir / "models"),
                "--round-id",
                str(args.round_id),
                "--ensemble-size",
                str(args.surrogate_ensemble_size),
                "--history-path",
                str(surrogate_history_path),
                "--max-checkpoints",
                str(args.max_checkpoints),
                "--seed",
                str(args.seed),
                "--log-level",
                args.log_level,
                *( ["--no-progress"] if args.no_progress else [] ),
            ]),
        ),
        (
            "train_teacher",
            _dispatch_wrap(args.uma_env_name, [
                "python",
                str(root / "scripts/train/m3_train_teacher_gfn.py"),
                "--input-dr",
                str(dataset_path),
                "--output-dir",
                str(round_dir / "models"),
                "--round-id",
                str(args.round_id),
                "--steps",
                str(args.teacher_steps),
                "--gamma-off",
                str(args.teacher_gamma_off),
                "--surrogate-ckpt",
                str(surrogate_ckpt),
                "--history-path",
                str(teacher_history_path),
                "--metrics-every",
                "25",
                "--max-checkpoints",
                str(args.max_checkpoints),
                "--seed",
                str(args.seed),
                "--log-level",
                args.log_level,
                *( ["--no-progress"] if args.no_progress else [] ),
            ]),
        ),
        (
            "distill_student",
            _dispatch_wrap(args.uma_env_name, [
                "python",
                str(root / "scripts/train/m3_distill_student.py"),
                "--teacher-ckpt",
                str(teacher_ckpt),
                "--input-dr",
                str(dataset_path),
                "--output-dir",
                str(round_dir / "models"),
                "--round-id",
                str(args.round_id),
                "--steps",
                str(args.student_steps),
                "--history-path",
                str(student_history_path),
                "--metrics-every",
                "100",
                "--max-checkpoints",
                str(args.max_checkpoints),
                "--seed",
                str(args.seed),
                "--log-level",
                args.log_level,
                *( ["--no-progress"] if args.no_progress else [] ),
            ]),
        ),
        (
            "generate_pool",
            _dispatch_wrap(args.uma_env_name, [
                "python",
                str(root / "scripts/train/m3_generate_student_pool.py"),
                "--student-ckpt",
                str(student_ckpt),
                "--input-dr",
                str(dataset_path),
                "--output-path",
                str(pool),
                "--run-id",
                args.run_id,
                "--round-id",
                str(args.round_id),
                "--pool-size",
                str(args.pool_size),
                "--metrics-path",
                str(pool_metrics_path),
                "--seed",
                str(args.seed),
                "--log-level",
                args.log_level,
                *( ["--no-progress"] if args.no_progress else [] ),
            ]),
        ),
    ]

    if args.graphkcat_prefilter_fraction > 0.0:
        graph_prefilter_budget = max(
            args.uma_cat_budget,
            min(
                args.pool_size,
                int(math.ceil(float(args.pool_size) * float(args.graphkcat_prefilter_fraction))),
            ),
        )
        pre_oracle_steps.extend(
            [
                (
                    "pack_pool_for_graphkcat",
                    [
                        "python",
                        str(dispatch),
                        "--env-name",
                        args.packer_env_name,
                        "--cmd",
                        (
                            f"python {_q(root / 'scripts/prep/oracles/ligandmpnn_pack_candidates.py')} "
                            f"--candidate-path {_q(pool)} --output-path {_q(graph_pool_packed)} --output-root {_q(round_dir / 'data' / 'packed_structures_pool')} "
                            f"--ligandmpnn-root {_q(args.ligandmpnn_root)} --checkpoint-sc {_q(args.packer_checkpoint_sc)} "
                            f"--device {_q(args.packer_device)} --pack-with-ligand-context {args.packer_pack_with_ligand_context} "
                            f"--repack-everything {args.packer_repack_everything} --sc-num-denoising-steps {args.packer_sc_num_denoising_steps} "
                            f"--sc-num-samples {args.packer_sc_num_samples} --parse-atoms-with-zero-occupancy {args.packer_parse_atoms_with_zero_occupancy} "
                            f"--force-hetatm {args.packer_force_hetatm} --log-level {_q(args.log_level)} "
                            f"{'--no-progress' if args.no_progress else ''}"
                        ),
                        *( ["--require-ready", "--env-status-json", args.env_status_json] if args.require_ready and args.env_status_json else [] ),
                    ],
                ),
                (
                    "graphkcat_score_pool",
                    [
                        "python",
                        str(dispatch),
                        "--env-name",
                        args.graphkcat_env_name,
                        "--cmd",
                        (
                            f"python {_q(root / 'scripts/prep/oracles/graphkcat_score.py')} "
                            f"--candidate-path {_q(graph_pool_packed)} --output-path {_q(graph_pool_scored)} "
                            f"--model-root {_q(args.graphkcat_model_root)} --checkpoint {_q(args.graphkcat_checkpoint)} "
                            f"--cfg {_q(args.graphkcat_cfg)} --batch-size {args.graphkcat_batch_size} --device {_q(args.graphkcat_device)} "
                            f"--distance-cutoff-a {args.graphkcat_distance_cutoff_a} --std-default {args.graphkcat_std_default} "
                            f"--mc-dropout-samples {args.graphkcat_mc_dropout_samples} --mc-dropout-seed {args.graphkcat_mc_dropout_seed} "
                            f"--summary-path {_q(graph_pool_summary_path)} "
                            f"--heartbeat-sec {args.graphkcat_heartbeat_sec} --work-dir {_q(round_dir / 'data' / 'graphkcat_work_pool')} "
                            f"--log-level {_q(args.log_level)} {'--no-progress' if args.no_progress else ''}"
                        ),
                        *( ["--require-ready", "--env-status-json", args.env_status_json] if args.require_ready and args.env_status_json else [] ),
                    ],
                ),
                (
                    "prefilter_graphkcat",
                    _dispatch_wrap(args.uma_env_name, [
                        "python",
                        str(root / "scripts/train/m3_select_graphkcat_batch.py"),
                        "--input-path",
                        str(graph_pool_scored),
                        "--output-path",
                        str(graph_prefiltered),
                        "--budget",
                        str(graph_prefilter_budget),
                        "--risk-kappa",
                        str(args.graphkcat_prefilter_risk_kappa),
                        "--require-graphkcat-ok",
                        *( ["--no-progress"] if args.no_progress else [] ),
                    ]),
                ),
                (
                    "select_uma_cat",
                    _dispatch_wrap(args.uma_env_name, [
                        "python",
                        str(root / "scripts/train/m3_select_uma_cat_batch.py"),
                        "--input-path",
                        str(graph_prefiltered),
                        "--output-path",
                        str(uma_sel),
                        "--budget",
                        str(args.uma_cat_budget),
                        *( ["--no-progress"] if args.no_progress else [] ),
                    ]),
                ),
            ]
        )
    else:
        pre_oracle_steps.append(
            (
                "select_uma_cat",
                _dispatch_wrap(args.uma_env_name, [
                    "python",
                    str(root / "scripts/train/m3_select_uma_cat_batch.py"),
                    "--input-path",
                    str(pool),
                    "--output-path",
                    str(uma_sel),
                    "--budget",
                    str(args.uma_cat_budget),
                    *( ["--no-progress"] if args.no_progress else [] ),
                ]),
            )
        )

    total_steps = len(pre_oracle_steps) + 1 + 4
    if args.graphkcat_prefilter_fraction <= 0.0:
        total_steps += 1
    if args.graphkcat_prefilter_fraction <= 0.0 and args.graphkcat_budget > 0:
        total_steps += 2
    step_bar = make_progress(
        total=total_steps,
        desc=f"train:round:{args.round_id}",
        no_progress=args.no_progress,
        leave=True,
        unit="step",
    )
    if step_bar is not None:
        step_bar.set_postfix_str(f"dataset={dataset_path.name} total_steps={total_steps}")

    def _finalize(rc: int) -> int:
        if step_bar is not None:
            step_bar.set_postfix_str(f"done rc={rc} elapsed={time.perf_counter() - wall_t0:.1f}s")
            step_bar.close()
        gate_payload = _read_json_if_exists(round_dir / "manifests" / "round_gate_report.json")
        if gate_payload:
            wandb_run.log_prefixed("gate", gate_payload, step=_phase_step(total_steps + 1, 1))
            wandb_run.summary_update(gate_payload, prefix="gate")
        wandb_run.summary_update({"returncode": int(rc)}, prefix="round")
        wandb_run.finish()
        return rc

    for name, cmd in pre_oracle_steps:
        rc = run_step(name, cmd)
        if rc != 0:
            (round_dir / "manifests" / "round_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
            return _finalize(rc)

    packed_for_uma = uma_sel
    if args.graphkcat_prefilter_fraction <= 0.0:
        pack_cmd = (
            f"python {_q(root / 'scripts/prep/oracles/ligandmpnn_pack_candidates.py')} "
            f"--candidate-path {_q(uma_sel)} --output-path {_q(packed)} --output-root {_q(round_dir / 'data' / 'packed_structures')} "
            f"--ligandmpnn-root {_q(args.ligandmpnn_root)} --checkpoint-sc {_q(args.packer_checkpoint_sc)} "
            f"--device {_q(args.packer_device)} --pack-with-ligand-context {args.packer_pack_with_ligand_context} "
            f"--repack-everything {args.packer_repack_everything} --sc-num-denoising-steps {args.packer_sc_num_denoising_steps} "
            f"--sc-num-samples {args.packer_sc_num_samples} --parse-atoms-with-zero-occupancy {args.packer_parse_atoms_with_zero_occupancy} "
            f"--force-hetatm {args.packer_force_hetatm} --log-level {_q(args.log_level)} "
            f"{'--no-progress' if args.no_progress else ''}"
        )
        rc = run_step(
            "pack_candidates",
            [
                "python",
                str(dispatch),
                "--env-name",
                args.packer_env_name,
                "--cmd",
                pack_cmd,
                *( ["--require-ready", "--env-status-json", args.env_status_json] if args.require_ready and args.env_status_json else [] ),
            ],
        )
        if rc != 0:
            return _finalize(4)
        packed_for_uma = packed

    uma_cmd = (
        f"python {_q(root / 'scripts/prep/oracles/uma_catalytic_score.py')} "
        f"--candidate-path {_q(packed_for_uma)} --output-path {_q(uma_scored)} --artifact-root {_q(round_dir / 'data' / 'uma_artifacts')} "
        f"--model-name {_q(args.uma_model_name)} --device {_q(args.uma_device)} --calculator-workers {args.uma_calculator_workers} "
        f"--max-atoms {args.uma_max_atoms} --temperature-k {args.uma_temperature_k} "
        f"--broad-steps {args.uma_broad_steps} --broad-replicas {args.uma_broad_replicas} --broad-save-every {args.uma_broad_save_every} "
        f"--run-smd {args.uma_run_smd} --run-reverse-smd {args.uma_run_reverse_smd} "
        f"--smd-images {args.uma_smd_images} --smd-steps-per-image {args.uma_smd_steps_per_image} --smd-replicas {args.uma_smd_replicas} "
        f"--run-pmf {args.uma_run_pmf} --pmf-windows {args.uma_pmf_windows} --pmf-steps-per-window {args.uma_pmf_steps_per_window} "
        f"--pmf-save-every {args.uma_pmf_save_every} --pmf-replicas {args.uma_pmf_replicas} --pmf-k-window-eva2 {args.uma_pmf_k_window_eva2} "
        f"--summary-path {_q(uma_summary_path)} "
        f"--log-level {_q(args.log_level)} {'--no-progress' if args.no_progress else ''}"
    )
    rc = run_step(
        "uma_catalytic_score",
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
        return _finalize(4)

    if args.graphkcat_prefilter_fraction <= 0.0 and args.graphkcat_budget > 0:
        rc = run_step(
            "select_graphkcat",
            _dispatch_wrap(args.uma_env_name, [
                "python",
                str(root / "scripts/train/m3_select_graphkcat_batch.py"),
                "--input-path",
                str(uma_scored),
                "--output-path",
                str(graph_sel),
                "--budget",
                str(args.graphkcat_budget),
                *( ["--no-progress"] if args.no_progress else [] ),
            ]),
        )
        if rc != 0:
            return _finalize(rc)

        graph_cmd = (
            f"python {_q(root / 'scripts/prep/oracles/graphkcat_score.py')} "
            f"--candidate-path {_q(graph_sel)} --output-path {_q(graph_scored)} "
            f"--model-root {_q(args.graphkcat_model_root)} --checkpoint {_q(args.graphkcat_checkpoint)} "
            f"--cfg {_q(args.graphkcat_cfg)} --batch-size {args.graphkcat_batch_size} --device {_q(args.graphkcat_device)} "
            f"--distance-cutoff-a {args.graphkcat_distance_cutoff_a} --std-default {args.graphkcat_std_default} "
            f"--mc-dropout-samples {args.graphkcat_mc_dropout_samples} --mc-dropout-seed {args.graphkcat_mc_dropout_seed} "
            f"--summary-path {_q(graph_selected_summary_path)} "
            f"--heartbeat-sec {args.graphkcat_heartbeat_sec} --work-dir {_q(round_dir / 'data' / 'graphkcat_work')} "
            f"--log-level {_q(args.log_level)} {'--no-progress' if args.no_progress else ''}"
        )
        rc = run_step(
            "graphkcat_score",
            [
                "python",
                str(dispatch),
                "--env-name",
                args.graphkcat_env_name,
                "--cmd",
                graph_cmd,
                *( ["--require-ready", "--env-status-json", args.env_status_json] if args.require_ready and args.env_status_json else [] ),
            ],
        )
        if rc != 0:
            return _finalize(4)

    rc = run_step(
        "fuse_catalytic_scores",
        _dispatch_wrap(args.uma_env_name, [
            "python",
            str(root / "scripts/prep/oracles/fuse_catalytic_scores.py"),
            "--candidate-path",
            str(uma_scored),
            *( ["--graphkcat-path", str(graph_pool_scored)] if args.graphkcat_prefilter_fraction > 0.0 else [] ),
            *( ["--graphkcat-path", str(graph_scored)] if args.graphkcat_prefilter_fraction <= 0.0 and args.graphkcat_budget > 0 else [] ),
            "--output-path",
            str(fused),
            "--w-uma-cat",
            str(args.fuse_w_uma_cat),
            "--w-graphkcat",
            str(args.fuse_w_graphkcat),
            "--w-agreement",
            str(args.fuse_w_agreement),
            "--kappa-uma-cat",
            str(args.fuse_kappa_uma_cat),
            "--kappa-graphkcat",
            str(args.fuse_kappa_graphkcat),
            *( ["--no-progress"] if args.no_progress else [] ),
        ]),
    )
    if rc != 0:
        return _finalize(rc)

    tail_steps = [
        (
            "append_labels",
            _dispatch_wrap(args.uma_env_name, [
                "python",
                str(root / "scripts/train/m3_append_labels.py"),
                "--input-dr",
                str(dataset_path),
                "--labeled-path",
                str(fused),
                "--output-dr-next",
                str(dr_next),
                "--summary-path",
                str(round_dir / "manifests" / "append_summary.json"),
                *( ["--no-progress"] if args.no_progress else [] ),
            ]),
        ),
        (
            "eval_design_metrics",
            _dispatch_wrap(args.uma_env_name, [
                "python",
                str(root / "scripts/eval/eval_design_metrics.py"),
                "--input-path",
                str(fused),
                "--output",
                str(round_dir / "metrics" / "round_metrics.json"),
            ]),
        ),
        (
            "eval_teacher_student",
            _dispatch_wrap(args.uma_env_name, [
                "python",
                str(root / "scripts/eval/eval_m3_teacher_student.py"),
                "--teacher-ckpt",
                str(teacher_ckpt),
                "--student-ckpt",
                str(student_ckpt),
                "--output",
                str(round_dir / "metrics" / "teacher_student_eval.json"),
            ]),
        ),
    ]
    for name, cmd in tail_steps:
        rc = run_step(name, cmd)
        if rc != 0:
            return _finalize(rc)

    if args.dry_run:
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["pass"] = True
        (round_dir / "manifests" / "round_gate_report.json").write_text(
            json.dumps({"pass": True, "checks": {}, "strict": args.strict_gates, "dry_run": True}, indent=2, sort_keys=True)
        )
        (round_dir / "manifests" / "round_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        logger.info("UMA-cat round dry-run complete elapsed=%.2fs", time.perf_counter() - wall_t0)
        return _finalize(0)

    ok, gate = _gate_report(round_dir, strict_gates=args.strict_gates)
    (round_dir / "manifests" / "round_gate_report.json").write_text(json.dumps(gate, indent=2, sort_keys=True))
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["pass"] = ok
    (round_dir / "manifests" / "round_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("UMA-cat round complete pass=%s strict=%s elapsed=%.2fs", ok, args.strict_gates, time.perf_counter() - wall_t0)

    if args.strict_gates and not ok:
        logger.error("Strict gates enabled and round failed gates")
        return _finalize(6)

    logger.info("Next dataset path: %s", dr_next)
    print(dr_next)
    return _finalize(0)


if __name__ == "__main__":
    raise SystemExit(main())

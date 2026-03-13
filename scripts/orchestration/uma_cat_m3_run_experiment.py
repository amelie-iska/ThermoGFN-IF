#!/usr/bin/env python3
"""Run multiple UMA-catalytic Method III rounds."""

from __future__ import annotations

import argparse
import json
import shlex
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
    peak_monitor_cls=None,
) -> tuple[int, float, dict]:
    logger.info("CMD: %s", " ".join(cmd))
    t0 = time.perf_counter()
    if dry_run:
        return 0, 0.0, {}
    hb = max(1.0, float(heartbeat_sec))
    monitor = peak_monitor_cls() if peak_monitor_cls is not None else None
    if monitor is not None:
        monitor.start()
    proc = subprocess.Popen(cmd)  # noqa: S603
    try:
        while True:
            try:
                rc = proc.wait(timeout=hb)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - t0
                logger.info("STEP %s still running elapsed=%.1fs", step_name or "<unnamed>", elapsed)
    finally:
        snapshot = monitor.stop() if monitor is not None else {}
    return rc, time.perf_counter() - t0, snapshot


def _q(v: str | Path) -> str:
    return shlex.quote(str(v))


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/uma_cat_m3_default.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--start-round", type=int, default=0)
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
    parser.add_argument("--uma-broad-timestep-fs", type=float, default=None)
    parser.add_argument("--uma-broad-friction-ps-inv", type=float, default=None)
    parser.add_argument("--uma-broad-steps", type=int, default=None)
    parser.add_argument("--uma-broad-replicas", type=int, default=None)
    parser.add_argument("--uma-broad-save-every", type=int, default=None)
    parser.add_argument("--uma-prepare-hydrogens", type=int, default=None)
    parser.add_argument("--uma-add-first-shell-waters", type=int, default=None)
    parser.add_argument("--uma-preparation-ph", type=float, default=None)
    parser.add_argument("--uma-max-first-shell-waters", type=int, default=None)
    parser.add_argument("--uma-water-shell-distance-a", type=float, default=None)
    parser.add_argument("--uma-water-clash-distance-a", type=float, default=None)
    parser.add_argument("--uma-water-bridge-distance-min-a", type=float, default=None)
    parser.add_argument("--uma-water-bridge-distance-max-a", type=float, default=None)
    parser.add_argument("--uma-relax-prepared-steps", type=int, default=None)
    parser.add_argument("--uma-relax-prepared-fmax-eva", type=float, default=None)
    parser.add_argument("--uma-protocol-max-reactive-bonds", type=int, default=None)
    parser.add_argument("--uma-protocol-max-reactive-atoms", type=int, default=None)
    parser.add_argument("--uma-protocol-max-reactive-fraction", type=float, default=None)
    parser.add_argument("--uma-run-smd", type=int, default=None)
    parser.add_argument("--uma-run-reverse-smd", type=int, default=None)
    parser.add_argument("--uma-smd-temperature-k", type=float, default=None)
    parser.add_argument("--uma-smd-timestep-fs", type=float, default=None)
    parser.add_argument("--uma-smd-friction-ps-inv", type=float, default=None)
    parser.add_argument("--uma-smd-images", type=int, default=None)
    parser.add_argument("--uma-smd-steps-per-image", type=int, default=None)
    parser.add_argument("--uma-smd-replicas", type=int, default=None)
    parser.add_argument("--uma-smd-k-steer-eva2", type=float, default=None)
    parser.add_argument("--uma-smd-k-global-eva2", type=float, default=None)
    parser.add_argument("--uma-smd-k-local-eva2", type=float, default=None)
    parser.add_argument("--uma-smd-k-anchor-eva2", type=float, default=None)
    parser.add_argument("--uma-smd-ca-network-sequential-k-eva2", type=float, default=None)
    parser.add_argument("--uma-smd-ca-network-contact-k-eva2", type=float, default=None)
    parser.add_argument("--uma-smd-ca-network-contact-cutoff-a", type=float, default=None)
    parser.add_argument("--uma-smd-force-clip-eva", type=float, default=None)
    parser.add_argument("--uma-smd-production-warmup-steps", type=int, default=None)
    parser.add_argument("--uma-quality-max-final-product-rmsd-a", type=float, default=None)
    parser.add_argument("--uma-quality-max-max-product-rmsd-a", type=float, default=None)
    parser.add_argument("--uma-quality-max-max-pocket-rmsd-a", type=float, default=None)
    parser.add_argument("--uma-quality-max-max-backbone-rmsd-a", type=float, default=None)
    parser.add_argument("--uma-quality-max-max-ca-network-rms-a", type=float, default=None)
    parser.add_argument("--uma-quality-max-max-close-contacts", type=int, default=None)
    parser.add_argument("--uma-quality-max-max-excess-bond-count", type=int, default=None)
    parser.add_argument("--uma-quality-require-smd-pass-for-pmf", type=int, default=None)
    parser.add_argument("--uma-run-pmf", type=int, default=None)
    parser.add_argument("--uma-run-pmf-every", type=int, default=None)
    parser.add_argument("--uma-pmf-windows", type=int, default=None)
    parser.add_argument("--uma-pmf-steps-per-window", type=int, default=None)
    parser.add_argument("--uma-pmf-save-every", type=int, default=None)
    parser.add_argument("--uma-pmf-replicas", type=int, default=None)
    parser.add_argument("--uma-pmf-k-window-eva2", type=float, default=None)
    parser.add_argument("--uma-pmf-k-local-eva2", type=float, default=None)
    parser.add_argument("--uma-pmf-window-relax-steps", type=int, default=None)
    parser.add_argument("--uma-pmf-window-equilibrate-steps", type=int, default=None)

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
    from train.thermogfn.progress import PeakVRAMMonitor, configure_logging, log_peak_vram_snapshot, make_progress
    from train.thermogfn.wandb_utils import WandbRun, parse_tags

    logger = configure_logging("orchestrate.uma_cat_experiment", level=args.log_level)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_yaml_config(cfg_path)

    args.num_rounds = int(args.num_rounds if args.num_rounds is not None else cfg_get(cfg, "method3.rounds", 8))
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
    args.step_heartbeat_sec = float(
        args.step_heartbeat_sec if args.step_heartbeat_sec is not None else cfg_get(cfg, "round.step_heartbeat_sec", 30.0)
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
        args.wandb_run_name or cfg_get(cfg, "logging.wandb.run_name", f"{args.run_id}-experiment")
    )
    args.uma_run_pmf_every = int(
        args.uma_run_pmf_every
        if args.uma_run_pmf_every is not None
        else cfg_get(cfg, "oracles.uma_cat.pmf.every_n_rounds", 1)
    )
    if args.uma_run_pmf_every < 1:
        raise ValueError("--uma-run-pmf-every must be >= 1")
    if not args.strict_gates and bool(cfg_get(cfg, "run.strict_gates", False)):
        args.strict_gates = True

    current_dr = root / args.dataset_path
    exp_root = root / args.output_root
    exp_root.mkdir(parents=True, exist_ok=True)

    wandb_run = WandbRun(
        enabled=args.wandb_enabled,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.wandb_run_name,
        group=args.wandb_group,
        job_type="uma_cat_experiment",
        tags=args.wandb_tags,
        config={
            "run_id": args.run_id,
            "dataset_path": str(current_dr),
            "num_rounds": args.num_rounds,
            "pool_size": args.pool_size,
            "uma_cat_budget": args.uma_cat_budget,
            "graphkcat_budget": args.graphkcat_budget,
            "graphkcat_prefilter_fraction": args.graphkcat_prefilter_fraction,
            "teacher_steps": args.teacher_steps,
            "student_steps": args.student_steps,
            "uma_run_pmf_every": args.uma_run_pmf_every,
            "strict_gates": args.strict_gates,
            "config_path": str(cfg_path),
        },
    )
    round_bar = make_progress(
        total=int(args.num_rounds),
        desc=f"train:run:{args.run_id}",
        no_progress=args.no_progress,
        leave=True,
        unit="round",
    )
    if round_bar is not None:
        round_bar.set_postfix_str(f"dataset={current_dr.name} total_rounds={int(args.num_rounds)}")

    manifest = {
        "run_id": args.run_id,
        "mode": "uma_cat_experiment",
        "config": str(cfg_path),
        "dataset_path": str(current_dr),
        "rounds": [],
    }

    round_script = root / "scripts/orchestration/uma_cat_m3_run_round.py"
    for round_id in range(int(args.start_round), int(args.start_round) + int(args.num_rounds)):
        round_dir = exp_root / f"round_{round_id:03d}"
        effective_uma_run_pmf = args.uma_run_pmf
        if effective_uma_run_pmf is not None and int(effective_uma_run_pmf):
            effective_uma_run_pmf = 1 if (int(round_id) % int(args.uma_run_pmf_every) == 0) else 0
        if round_bar is not None:
            pmf_suffix = ""
            if effective_uma_run_pmf is not None:
                pmf_suffix = f" pmf={'on' if int(effective_uma_run_pmf) else 'off'}"
            round_bar.set_postfix_str(f"running round={round_id} dataset={current_dr.name}{pmf_suffix}")
        cmd = [
            "python",
            str(round_script),
            "--config",
            str(cfg_path),
            "--run-id",
            args.run_id,
            "--round-id",
            str(round_id),
            "--dataset-path",
            str(current_dr),
            "--output-dir",
            str(round_dir),
            "--pool-size",
            str(args.pool_size),
            "--uma-cat-budget",
            str(args.uma_cat_budget),
            "--graphkcat-budget",
            str(args.graphkcat_budget),
            "--graphkcat-prefilter-fraction",
            str(args.graphkcat_prefilter_fraction),
            "--graphkcat-prefilter-risk-kappa",
            str(args.graphkcat_prefilter_risk_kappa),
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
            "--seed",
            str(args.seed),
            "--step-heartbeat-sec",
            str(args.step_heartbeat_sec),
            "--log-level",
            args.log_level,
            "--wandb-enabled",
            "1" if args.wandb_enabled else "0",
            "--wandb-project",
            args.wandb_project,
            *(["--wandb-entity", str(args.wandb_entity)] if args.wandb_entity else []),
            "--wandb-mode",
            args.wandb_mode,
            "--wandb-group",
            args.wandb_group,
            *(["--wandb-tags", ",".join(args.wandb_tags)] if args.wandb_tags else []),
            "--wandb-run-name",
            f"{args.run_id}-round-{round_id:03d}",
            *(["--strict-gates"] if args.strict_gates else []),
            *(["--dry-run"] if args.dry_run else []),
            *(["--require-ready", "--env-status-json", args.env_status_json] if args.require_ready and args.env_status_json else []),
            *(["--no-progress"] if args.no_progress else []),
        ]

        passthrough = {
            "--packer-env-name": args.packer_env_name,
            "--uma-env-name": args.uma_env_name,
            "--graphkcat-env-name": args.graphkcat_env_name,
            "--ligandmpnn-root": args.ligandmpnn_root,
            "--packer-checkpoint-sc": args.packer_checkpoint_sc,
            "--packer-device": args.packer_device,
            "--packer-pack-with-ligand-context": args.packer_pack_with_ligand_context,
            "--packer-repack-everything": args.packer_repack_everything,
            "--packer-sc-num-denoising-steps": args.packer_sc_num_denoising_steps,
            "--packer-sc-num-samples": args.packer_sc_num_samples,
            "--packer-parse-atoms-with-zero-occupancy": args.packer_parse_atoms_with_zero_occupancy,
            "--packer-force-hetatm": args.packer_force_hetatm,
            "--uma-model-name": args.uma_model_name,
            "--uma-device": args.uma_device,
            "--uma-calculator-workers": args.uma_calculator_workers,
            "--uma-max-atoms": args.uma_max_atoms,
            "--uma-temperature-k": args.uma_temperature_k,
            "--uma-broad-timestep-fs": args.uma_broad_timestep_fs,
            "--uma-broad-friction-ps-inv": args.uma_broad_friction_ps_inv,
            "--uma-broad-steps": args.uma_broad_steps,
            "--uma-broad-replicas": args.uma_broad_replicas,
            "--uma-broad-save-every": args.uma_broad_save_every,
            "--uma-prepare-hydrogens": args.uma_prepare_hydrogens,
            "--uma-add-first-shell-waters": args.uma_add_first_shell_waters,
            "--uma-preparation-ph": args.uma_preparation_ph,
            "--uma-max-first-shell-waters": args.uma_max_first_shell_waters,
            "--uma-water-shell-distance-a": args.uma_water_shell_distance_a,
            "--uma-water-clash-distance-a": args.uma_water_clash_distance_a,
            "--uma-water-bridge-distance-min-a": args.uma_water_bridge_distance_min_a,
            "--uma-water-bridge-distance-max-a": args.uma_water_bridge_distance_max_a,
            "--uma-relax-prepared-steps": args.uma_relax_prepared_steps,
            "--uma-relax-prepared-fmax-eva": args.uma_relax_prepared_fmax_eva,
            "--uma-protocol-max-reactive-bonds": args.uma_protocol_max_reactive_bonds,
            "--uma-protocol-max-reactive-atoms": args.uma_protocol_max_reactive_atoms,
            "--uma-protocol-max-reactive-fraction": args.uma_protocol_max_reactive_fraction,
            "--uma-run-smd": args.uma_run_smd,
            "--uma-run-reverse-smd": args.uma_run_reverse_smd,
            "--uma-smd-temperature-k": args.uma_smd_temperature_k,
            "--uma-smd-timestep-fs": args.uma_smd_timestep_fs,
            "--uma-smd-friction-ps-inv": args.uma_smd_friction_ps_inv,
            "--uma-smd-images": args.uma_smd_images,
            "--uma-smd-steps-per-image": args.uma_smd_steps_per_image,
            "--uma-smd-replicas": args.uma_smd_replicas,
            "--uma-smd-k-steer-eva2": args.uma_smd_k_steer_eva2,
            "--uma-smd-k-global-eva2": args.uma_smd_k_global_eva2,
            "--uma-smd-k-local-eva2": args.uma_smd_k_local_eva2,
            "--uma-smd-k-anchor-eva2": args.uma_smd_k_anchor_eva2,
            "--uma-smd-ca-network-sequential-k-eva2": args.uma_smd_ca_network_sequential_k_eva2,
            "--uma-smd-ca-network-contact-k-eva2": args.uma_smd_ca_network_contact_k_eva2,
            "--uma-smd-ca-network-contact-cutoff-a": args.uma_smd_ca_network_contact_cutoff_a,
            "--uma-smd-force-clip-eva": args.uma_smd_force_clip_eva,
            "--uma-smd-production-warmup-steps": args.uma_smd_production_warmup_steps,
            "--uma-quality-max-final-product-rmsd-a": args.uma_quality_max_final_product_rmsd_a,
            "--uma-quality-max-max-product-rmsd-a": args.uma_quality_max_max_product_rmsd_a,
            "--uma-quality-max-max-pocket-rmsd-a": args.uma_quality_max_max_pocket_rmsd_a,
            "--uma-quality-max-max-backbone-rmsd-a": args.uma_quality_max_max_backbone_rmsd_a,
            "--uma-quality-max-max-ca-network-rms-a": args.uma_quality_max_max_ca_network_rms_a,
            "--uma-quality-max-max-close-contacts": args.uma_quality_max_max_close_contacts,
            "--uma-quality-max-max-excess-bond-count": args.uma_quality_max_max_excess_bond_count,
            "--uma-quality-require-smd-pass-for-pmf": args.uma_quality_require_smd_pass_for_pmf,
            "--uma-run-pmf": effective_uma_run_pmf,
            "--uma-pmf-windows": args.uma_pmf_windows,
            "--uma-pmf-steps-per-window": args.uma_pmf_steps_per_window,
            "--uma-pmf-save-every": args.uma_pmf_save_every,
            "--uma-pmf-replicas": args.uma_pmf_replicas,
            "--uma-pmf-k-window-eva2": args.uma_pmf_k_window_eva2,
            "--uma-pmf-k-local-eva2": args.uma_pmf_k_local_eva2,
            "--uma-pmf-window-relax-steps": args.uma_pmf_window_relax_steps,
            "--uma-pmf-window-equilibrate-steps": args.uma_pmf_window_equilibrate_steps,
            "--graphkcat-model-root": args.graphkcat_model_root,
            "--graphkcat-checkpoint": args.graphkcat_checkpoint,
            "--graphkcat-cfg": args.graphkcat_cfg,
            "--graphkcat-batch-size": args.graphkcat_batch_size,
            "--graphkcat-device": args.graphkcat_device,
            "--graphkcat-distance-cutoff-a": args.graphkcat_distance_cutoff_a,
            "--graphkcat-std-default": args.graphkcat_std_default,
            "--graphkcat-mc-dropout-samples": args.graphkcat_mc_dropout_samples,
            "--graphkcat-mc-dropout-seed": args.graphkcat_mc_dropout_seed,
            "--graphkcat-heartbeat-sec": args.graphkcat_heartbeat_sec,
            "--fuse-w-uma-cat": args.fuse_w_uma_cat,
            "--fuse-w-graphkcat": args.fuse_w_graphkcat,
            "--fuse-w-agreement": args.fuse_w_agreement,
            "--fuse-kappa-uma-cat": args.fuse_kappa_uma_cat,
            "--fuse-kappa-graphkcat": args.fuse_kappa_graphkcat,
        }
        for flag, value in passthrough.items():
            if value is not None:
                cmd.extend([flag, str(value)])

        rc, dt, peak_vram = _run(
            cmd,
            logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{round_id}",
            peak_monitor_cls=PeakVRAMMonitor,
        )
        log_peak_vram_snapshot(logger, peak_vram, label=f"round_{round_id}")
        gate_path = round_dir / "manifests" / "round_gate_report.json"
        gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
        next_dr = round_dir / "data" / f"D_{round_id + 1}.jsonl"
        manifest["rounds"].append(
            {
                "round_id": int(round_id),
                "dataset_path": str(current_dr),
                "output_dir": str(round_dir),
                "uma_run_pmf": int(effective_uma_run_pmf) if effective_uma_run_pmf is not None else None,
                "returncode": int(rc),
                "duration_s": float(dt),
                "peak_vram": peak_vram,
                "gate": gate,
                "next_dataset_path": str(next_dr),
            }
        )
        round_metrics = _read_json_if_exists(round_dir / "metrics" / "round_metrics.json")
        teacher_student = _read_json_if_exists(round_dir / "metrics" / "teacher_student_eval.json")
        gate_summary = _read_json_if_exists(round_dir / "manifests" / "round_gate_report.json")
        append_summary = _read_json_if_exists(round_dir / "manifests" / "append_summary.json")
        wandb_run.log(
            {
                "round/index": int(round_id),
                **(
                    {"round/uma_run_pmf": int(effective_uma_run_pmf)}
                    if effective_uma_run_pmf is not None
                    else {}
                ),
                "round/returncode": int(rc),
                "round/duration_s": float(dt),
                "round/peak_vram_mib": float(peak_vram.get("peak_vram_mib", 0.0) or 0.0),
                "round/peak_vram_gib": float(peak_vram.get("peak_vram_gib", 0.0) or 0.0),
                "round/peak_vram_frac": float(peak_vram.get("peak_vram_frac", 0.0) or 0.0),
                "round/output_dir": str(round_dir),
                **({"round/metrics": round_metrics} if round_metrics else {}),
                **({"round/teacher_student": teacher_student} if teacher_student else {}),
                **({"round/gate": gate_summary} if gate_summary else {}),
                **({"round/append": append_summary} if append_summary else {}),
            },
            step=int(round_id) + 1,
        )
        if round_bar is not None:
            round_bar.update(1)
            round_bar.set_postfix_str(f"completed={round_id + 1}/{int(args.num_rounds)} last_rc={rc} dt={dt:.1f}s")
        if rc != 0:
            logger.error("UMA-cat experiment stopping at round=%d rc=%d", round_id, rc)
            break
        current_dr = next_dr

    manifest["completed_utc"] = time.time()
    manifest_path = exp_root / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    if round_bar is not None:
        round_bar.set_postfix_str(
            f"done rounds={len(manifest['rounds'])}/{int(args.num_rounds)} elapsed={time.perf_counter() - wall_t0:.1f}s"
        )
        round_bar.close()
    wandb_run.summary_update(
        {
            "manifest_path": str(manifest_path),
            "completed_dataset_path": str(current_dr),
            "elapsed_s": float(time.perf_counter() - wall_t0),
            "completed_rounds": len(manifest["rounds"]),
        },
        prefix="experiment",
    )
    wandb_run.finish()
    logger.info("UMA-cat experiment complete manifest=%s elapsed=%.2fs", manifest_path, time.perf_counter() - wall_t0)
    print(current_dr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

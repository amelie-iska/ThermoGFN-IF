#!/usr/bin/env python3
"""Run multiple Kcat-only Method III rounds with periodic test evaluation."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/kcat_m3_default.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-test-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--kcatnet-budget", type=int, default=None)
    parser.add_argument("--graphkcat-budget", type=int, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--surrogate-ensemble-size", type=int, default=None)
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--teacher-gamma-off", type=float, default=None)
    parser.add_argument("--student-steps", type=int, default=None)

    parser.add_argument("--periodic-test-num-candidates", type=int, default=None)
    parser.add_argument("--periodic-kcatnet-budget", type=int, default=None)
    parser.add_argument("--periodic-graphkcat-budget", type=int, default=None)

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
    parser.add_argument("--early-stop-overfit-gap-top8", type=float, default=None)
    parser.add_argument("--early-stop-overfit-gap-best", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    wall_t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from train.thermogfn.config_utils import cfg_get, load_yaml_config
    from train.thermogfn.progress import configure_logging, iter_progress

    logger = configure_logging("orchestrate.kcat_experiment", level=args.log_level)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_yaml_config(cfg_path)

    args.num_rounds = int(args.num_rounds if args.num_rounds is not None else cfg_get(cfg, "method3.rounds", 8))
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

    args.periodic_test_num_candidates = int(
        args.periodic_test_num_candidates
        if args.periodic_test_num_candidates is not None
        else cfg_get(cfg, "periodic_eval.num_candidates", 256)
    )
    args.periodic_kcatnet_budget = int(
        args.periodic_kcatnet_budget
        if args.periodic_kcatnet_budget is not None
        else cfg_get(cfg, "periodic_eval.kcatnet_budget", args.periodic_test_num_candidates)
    )
    args.periodic_graphkcat_budget = int(
        args.periodic_graphkcat_budget
        if args.periodic_graphkcat_budget is not None
        else cfg_get(cfg, "periodic_eval.graphkcat_budget", min(args.graphkcat_budget, args.periodic_test_num_candidates))
    )

    if args.early_stop_patience is None:
        args.early_stop_patience = int(cfg_get(cfg, "method3.early_stop.patience_rounds", 1))
    if args.early_stop_overfit_gap_top8 is None:
        args.early_stop_overfit_gap_top8 = cfg_get(cfg, "method3.early_stop.overfit_gap_top8_mean_reward", None)
    if args.early_stop_overfit_gap_best is None:
        args.early_stop_overfit_gap_best = cfg_get(cfg, "method3.early_stop.overfit_gap_best_reward", None)

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

    current_dr = root / args.dataset_path
    current_test = root / args.dataset_test_path
    exp_root = root / args.output_root
    exp_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Kcat experiment start run_id=%s rounds=%d start_round=%d train=%s test=%s dry_run=%s cfg=%s",
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
    for r in iter_progress(round_values, total=len(round_values), desc="kcat:m3:rounds", no_progress=args.no_progress, leave=True):
        round_t0 = time.perf_counter()
        round_dir = exp_root / f"round_{r:03d}"

        cmd = [
            "python",
            str(root / "scripts/orchestration/kcat_m3_run_round.py"),
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
            "--kcatnet-budget",
            str(args.kcatnet_budget),
            "--graphkcat-budget",
            str(args.graphkcat_budget),
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
            "--kcatnet-env-name",
            args.kcatnet_env_name,
            "--graphkcat-env-name",
            args.graphkcat_env_name,
            "--kcatnet-model-root",
            args.kcatnet_model_root,
            "--kcatnet-checkpoint",
            args.kcatnet_checkpoint,
            "--kcatnet-config-path",
            args.kcatnet_config_path,
            "--kcatnet-degree-path",
            args.kcatnet_degree_path,
            "--kcatnet-device",
            args.kcatnet_device,
            "--kcatnet-batch-size",
            str(args.kcatnet_batch_size),
            "--kcatnet-std-default",
            str(args.kcatnet_std_default),
            "--kcatnet-prott5-model",
            args.kcatnet_prott5_model,
            "--graphkcat-model-root",
            args.graphkcat_model_root,
            "--graphkcat-checkpoint",
            args.graphkcat_checkpoint,
            "--graphkcat-cfg",
            args.graphkcat_cfg,
            "--graphkcat-batch-size",
            str(args.graphkcat_batch_size),
            "--graphkcat-device",
            args.graphkcat_device,
            "--graphkcat-distance-cutoff-a",
            str(args.graphkcat_distance_cutoff_a),
            "--graphkcat-std-default",
            str(args.graphkcat_std_default),
            "--graphkcat-heartbeat-sec",
            str(args.graphkcat_heartbeat_sec),
            "--fuse-w-kcatnet",
            str(args.fuse_w_kcatnet),
            "--fuse-w-graphkcat",
            str(args.fuse_w_graphkcat),
            "--fuse-w-agreement",
            str(args.fuse_w_agreement),
            "--fuse-kappa-kcatnet",
            str(args.fuse_kappa_kcatnet),
            "--fuse-kappa-graphkcat",
            str(args.fuse_kappa_graphkcat),
            "--seed",
            str(args.seed),
            "--step-heartbeat-sec",
            str(args.step_heartbeat_sec),
            *( ["--no-progress"] if args.no_progress else [] ),
        ]
        if args.kcatnet_prott5_dir:
            cmd.extend(["--kcatnet-prott5-dir", args.kcatnet_prott5_dir])
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
            logger.error("Round %d failed during kcat_m3_run_round rc=%d", r, rc)
            history.append(round_info)
            break

        next_dr = round_dir / "data" / f"D_{r+1}.jsonl"
        if not args.dry_run and not next_dr.exists():
            round_info["returncode"] = 5
            logger.error("Round %d missing expected next dataset: %s", r, next_dr)
            history.append(round_info)
            break

        train_eval = round_dir / "metrics" / "round_metrics.json"
        test_pool = round_dir / "data" / f"periodic_test_pool_round_{r}.jsonl"
        test_kn_sel = round_dir / "data" / f"periodic_test_kcatnet_selected_round_{r}.jsonl"
        test_kn = round_dir / "data" / f"periodic_test_kcatnet_round_{r}.jsonl"
        test_graph_sel = round_dir / "data" / f"periodic_test_graphkcat_selected_round_{r}.jsonl"
        test_graph = round_dir / "data" / f"periodic_test_graphkcat_round_{r}.jsonl"
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
                *( ["--no-progress"] if args.no_progress else [] ),
            ],
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_periodic_generate",
        )
        round_info["periodic_test_generate_duration_s"] = dt_test_gen
        if rc_test_gen != 0:
            round_info["returncode"] = 4
            logger.error("Round %d periodic test generation failed rc=%d", r, rc_test_gen)
            history.append(round_info)
            break

        rc_test_kcatnet_sel, dt_test_kcatnet_sel = _run(
            [
                "python",
                str(root / "scripts/train/m3_select_kcatnet_batch.py"),
                "--input-path",
                str(test_pool),
                "--output-path",
                str(test_kn_sel),
                "--budget",
                str(args.periodic_kcatnet_budget),
                *( ["--no-progress"] if args.no_progress else [] ),
            ],
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_periodic_select_kcatnet",
        )

        dispatch = root / "scripts/env/dispatch.py"
        kcatnet_cmd = (
            f"python {_q(root / 'scripts/prep/oracles/kcatnet_score.py')} "
            f"--candidate-path {_q(test_kn_sel)} --output-path {_q(test_kn)} "
            f"--model-root {_q(args.kcatnet_model_root)} --checkpoint {_q(args.kcatnet_checkpoint)} "
            f"--config-path {_q(args.kcatnet_config_path)} --degree-path {_q(args.kcatnet_degree_path)} "
            f"--device {_q(args.kcatnet_device)} --batch-size {args.kcatnet_batch_size} "
            f"--std-default {args.kcatnet_std_default} --prott5-model {_q(args.kcatnet_prott5_model)} "
            f"{f'--prott5-dir {_q(args.kcatnet_prott5_dir)} ' if args.kcatnet_prott5_dir else ''}"
            f"--log-level {_q(args.log_level)} {'--no-progress' if args.no_progress else ''}"
        )
        rc_test_kcatnet, dt_test_kcatnet = _run(
            [
                "python",
                str(dispatch),
                "--env-name",
                args.kcatnet_env_name,
                "--cmd",
                kcatnet_cmd,
                *( ["--require-ready", "--env-status-json", args.env_status_json] if args.require_ready and args.env_status_json else [] ),
            ],
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_periodic_kcatnet",
        )

        rc_test_graph_sel, dt_test_graph_sel = _run(
            [
                "python",
                str(root / "scripts/train/m3_select_graphkcat_batch.py"),
                "--input-path",
                str(test_kn),
                "--output-path",
                str(test_graph_sel),
                "--budget",
                str(args.periodic_graphkcat_budget),
                *( ["--no-progress"] if args.no_progress else [] ),
            ],
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_periodic_select_graphkcat",
        )

        graph_cmd = (
            f"python {_q(root / 'scripts/prep/oracles/graphkcat_score.py')} "
            f"--candidate-path {_q(test_graph_sel)} --output-path {_q(test_graph)} "
            f"--model-root {_q(args.graphkcat_model_root)} --checkpoint {_q(args.graphkcat_checkpoint)} "
            f"--cfg {_q(args.graphkcat_cfg)} --batch-size {args.graphkcat_batch_size} --device {_q(args.graphkcat_device)} "
            f"--distance-cutoff-a {args.graphkcat_distance_cutoff_a} --std-default {args.graphkcat_std_default} "
            f"--heartbeat-sec {args.graphkcat_heartbeat_sec} --work-dir {_q(round_dir / 'data' / 'periodic_graphkcat_work')} "
            f"--log-level {_q(args.log_level)} {'--no-progress' if args.no_progress else ''}"
        )
        rc_test_graph, dt_test_graph = _run(
            [
                "python",
                str(dispatch),
                "--env-name",
                args.graphkcat_env_name,
                "--cmd",
                graph_cmd,
                *( ["--require-ready", "--env-status-json", args.env_status_json] if args.require_ready and args.env_status_json else [] ),
            ],
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_periodic_graphkcat",
        )

        rc_test_fuse, dt_test_fuse = _run(
            [
                "python",
                str(root / "scripts/prep/oracles/fuse_kcat_scores.py"),
                "--candidate-path",
                str(test_graph),
                "--output-path",
                str(test_fused),
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
            logger=logger,
            dry_run=args.dry_run,
            heartbeat_sec=args.step_heartbeat_sec,
            step_name=f"round_{r}_periodic_fuse",
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
            step_name=f"round_{r}_periodic_eval",
        )

        round_info["periodic_select_kcatnet_duration_s"] = dt_test_kcatnet_sel
        round_info["periodic_kcatnet_duration_s"] = dt_test_kcatnet
        round_info["periodic_select_graphkcat_duration_s"] = dt_test_graph_sel
        round_info["periodic_graphkcat_duration_s"] = dt_test_graph
        round_info["periodic_fuse_duration_s"] = dt_test_fuse
        round_info["periodic_eval_duration_s"] = dt_test_eval

        if any(
            code != 0
            for code in [
                rc_test_kcatnet_sel,
                rc_test_kcatnet,
                rc_test_graph_sel,
                rc_test_graph,
                rc_test_fuse,
                rc_test_eval,
            ]
        ):
            round_info["returncode"] = 4
            logger.error(
                "Round %d periodic scoring failed rc=[sel_kcatnet:%d,kcatnet:%d,sel_graph:%d,graph:%d,fuse:%d,eval:%d]",
                r,
                rc_test_kcatnet_sel,
                rc_test_kcatnet,
                rc_test_graph_sel,
                rc_test_graph,
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
                    reasons.append(f"top8_gap={gap_top8:.4f} > {args.early_stop_overfit_gap_top8:.4f}")
                if args.early_stop_overfit_gap_best is not None and gap_best > args.early_stop_overfit_gap_best:
                    breach = True
                    reasons.append(f"best_gap={gap_best:.4f} > {args.early_stop_overfit_gap_best:.4f}")
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
    logger.info("Kcat experiment complete rounds_recorded=%d elapsed=%.2fs", len(history), time.perf_counter() - wall_t0)
    print(exp_root / "experiment_history.json")
    return 0 if history and history[-1].get("returncode", 1) == 0 else (0 if args.dry_run else 1)


if __name__ == "__main__":
    raise SystemExit(main())

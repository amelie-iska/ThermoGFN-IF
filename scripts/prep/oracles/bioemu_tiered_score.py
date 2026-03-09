#!/usr/bin/env python3
"""Three-tier BioEmu scoring.

Tier 1: short run on all selected candidates.
Tier 2: medium run on top-N% from tier 1.
Tier 3: long run on a final shortlist (fraction of original candidate count).
Outputs a merged JSONL with tier annotations.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _rank_score(rec: dict, metric: str, kappa: float) -> float:
    y = rec.get("bioemu_calibrated")
    if y is None:
        return float("-inf")
    y = float(y)
    if metric == "calibrated":
        return y
    std = float(rec.get("bioemu_std") or 0.0)
    return y - float(kappa) * std


def _run_bioemu_stage(
    *,
    python_exe: str,
    scorer: Path,
    input_path: Path,
    output_path: Path,
    num_samples: int,
    args,
    seed: int,
) -> None:
    cmd = [
        python_exe,
        str(scorer),
        "--candidate-path",
        str(input_path),
        "--output-path",
        str(output_path),
        "--model-name",
        str(args.model_name),
        "--num-samples",
        str(int(num_samples)),
        "--batch-size-100",
        str(int(args.batch_size_100)),
        "--target-vram-frac",
        str(float(args.target_vram_frac)),
        "--batch-size-100-min",
        str(int(args.batch_size_100_min)),
        "--batch-size-100-max",
        str(int(args.batch_size_100_max)),
        "--max-proteins-per-step",
        str(int(args.max_proteins_per_step)),
        "--vram-control-metric",
        str(args.vram_control_metric),
        "--low-utilization-mult",
        str(float(args.low_utilization_mult)),
        "--batch-size-100-max-growth-factor",
        str(float(args.batch_size_100_max_growth_factor)),
        "--batch-size-100-max-shrink-factor",
        str(float(args.batch_size_100_max_shrink_factor)),
        "--seed",
        str(int(seed)),
        "--log-level",
        str(args.log_level),
    ]
    cmd.append("--sort-by-length-desc" if args.sort_by_length_desc else "--no-sort-by-length-desc")
    cmd.append("--filter-samples" if args.filter_samples else "--no-filter-samples")
    cmd.append("--auto-batch-from-vram" if args.auto_batch_from_vram else "--no-auto-batch-from-vram")
    if args.require_torch_cuda_vram:
        cmd.append("--require-torch-cuda-vram")
    if args.no_progress:
        cmd.append("--no-progress")

    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-path", required=True)

    # Tiering knobs.
    parser.add_argument("--tier1-num-samples", type=int, default=256)
    parser.add_argument("--tier2-num-samples", type=int, default=512)
    parser.add_argument("--tier3-num-samples", type=int, default=2048)
    parser.add_argument("--tier2-top-frac", type=float, default=0.20)
    parser.add_argument("--tier2-top-min", type=int, default=1)
    parser.add_argument("--tier3-top-frac-original", type=float, default=0.05)
    parser.add_argument("--tier3-top-min", type=int, default=1)
    parser.add_argument("--tier-rank-metric", choices=["risk_adjusted", "calibrated"], default="risk_adjusted")
    parser.add_argument("--tier-risk-kappa", type=float, default=0.5)

    # Shared BioEmu knobs mirrored from single-stage scorer.
    parser.add_argument("--model-name", default="bioemu-v1.1")
    parser.add_argument("--batch-size-100", type=int, default=96)
    parser.add_argument("--target-vram-frac", type=float, default=0.90)
    parser.add_argument("--batch-size-100-min", type=int, default=4)
    parser.add_argument("--batch-size-100-max", type=int, default=384)
    parser.add_argument("--max-proteins-per-step", type=int, default=12)
    parser.add_argument("--vram-control-metric", choices=["allocated", "reserved"], default="reserved")
    parser.add_argument("--low-utilization-mult", type=float, default=0.6)
    parser.add_argument("--batch-size-100-max-growth-factor", type=float, default=3.0)
    parser.add_argument("--batch-size-100-max-shrink-factor", type=float, default=0.6)
    parser.add_argument("--filter-samples", dest="filter_samples", action="store_true")
    parser.add_argument("--no-filter-samples", dest="filter_samples", action="store_false")
    parser.add_argument("--sort-by-length-desc", dest="sort_by_length_desc", action="store_true")
    parser.add_argument("--no-sort-by-length-desc", dest="sort_by_length_desc", action="store_false")
    parser.add_argument("--auto-batch-from-vram", dest="auto_batch_from_vram", action="store_true")
    parser.add_argument("--no-auto-batch-from-vram", dest="auto_batch_from_vram", action="store_false")
    parser.add_argument("--require-torch-cuda-vram", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    parser.set_defaults(
        auto_batch_from_vram=True,
        sort_by_length_desc=True,
        filter_samples=False,
    )
    args = parser.parse_args()

    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging
    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None

    logger = configure_logging("oracle.bioemu_tiered", level=args.log_level)

    t0 = time.perf_counter()
    candidate_path = root / args.candidate_path if not Path(args.candidate_path).is_absolute() else Path(args.candidate_path)
    output_path = root / args.output_path if not Path(args.output_path).is_absolute() else Path(args.output_path)

    rows_in = read_records(candidate_path)
    logger.info(
        "Tiered BioEmu start: candidates=%d tier1_samples=%d tier2_samples=%d tier3_samples=%d tier2_top_frac=%.3f tier3_top_frac_original=%.3f rank_metric=%s",
        len(rows_in),
        args.tier1_num_samples,
        args.tier2_num_samples,
        args.tier3_num_samples,
        args.tier2_top_frac,
        args.tier3_top_frac_original,
        args.tier_rank_metric,
    )

    scorer = root / "scripts" / "prep" / "oracles" / "bioemu_sample_and_features.py"
    python_exe = sys.executable

    stage_progress = None
    if tqdm is not None and not args.no_progress:
        stage_progress = tqdm(total=3, desc="bioemu:tiered", dynamic_ncols=True, leave=True)

    with tempfile.TemporaryDirectory(prefix="bioemu_tiered_") as td:
        td_path = Path(td)
        stage1_out = td_path / "stage1.jsonl"

        logger.info(
            "Tiered BioEmu stage1 run: candidates=%d samples_per_candidate=%d",
            len(rows_in),
            int(args.tier1_num_samples),
        )
        _run_bioemu_stage(
            python_exe=python_exe,
            scorer=scorer,
            input_path=candidate_path,
            output_path=stage1_out,
            num_samples=int(args.tier1_num_samples),
            args=args,
            seed=int(args.seed),
        )
        if stage_progress is not None:
            stage_progress.update(1)
            stage_progress.set_postfix(stage="tier1_done")

        rows_stage1 = read_records(stage1_out)
        eligible = [r for r in rows_stage1 if r.get("bioemu_status") == "ok" and r.get("bioemu_calibrated") is not None]

        for r in rows_stage1:
            r["bioemu_tier"] = "tier1"
            r["bioemu_refined"] = False
            r["bioemu_refinement_stage"] = 1
            r["bioemu_tier1_calibrated"] = r.get("bioemu_calibrated")
            r["bioemu_tier1_std"] = r.get("bioemu_std")
            r["bioemu_tier2_calibrated"] = None
            r["bioemu_tier2_std"] = None
            r["bioemu_tier3_calibrated"] = None
            r["bioemu_tier3_std"] = None
            r["bioemu_tier_rank_score"] = (
                _rank_score(r, args.tier_rank_metric, args.tier_risk_kappa)
                if r.get("bioemu_status") == "ok"
                else None
            )

        n_eligible = len(eligible)
        n_top = 0
        if n_eligible > 0 and float(args.tier2_top_frac) > 0:
            n_top = int(math.ceil(float(args.tier2_top_frac) * n_eligible))
            n_top = max(int(args.tier2_top_min), n_top)
            n_top = min(n_eligible, n_top)

        rows_stage2: list[dict] = []
        if n_top > 0:
            ranked = sorted(
                eligible,
                key=lambda r: _rank_score(r, args.tier_rank_metric, args.tier_risk_kappa),
                reverse=True,
            )
            top_ids = {str(r.get("candidate_id")) for r in ranked[:n_top]}
            stage2_in_rows = [r for r in rows_stage1 if str(r.get("candidate_id")) in top_ids]

            stage2_in = td_path / "stage2_in.jsonl"
            stage2_out = td_path / "stage2_out.jsonl"
            write_records(stage2_in, stage2_in_rows)

            logger.info(
                "Tiered BioEmu stage2 run: eligible=%d selected=%d (%.2f%%) samples_per_candidate=%d",
                n_eligible,
                n_top,
                100.0 * n_top / max(1, n_eligible),
                int(args.tier2_num_samples),
            )

            _run_bioemu_stage(
                python_exe=python_exe,
                scorer=scorer,
                input_path=stage2_in,
                output_path=stage2_out,
                num_samples=int(args.tier2_num_samples),
                args=args,
                seed=int(args.seed) + 100000,
            )
            if stage_progress is not None:
                stage_progress.update(1)
                stage_progress.set_postfix(stage="tier2_done", refined=n_top)

            rows_stage2 = read_records(stage2_out)
            stage2_by_id = {str(r.get("candidate_id")): r for r in rows_stage2}

            for r in rows_stage1:
                cid = str(r.get("candidate_id"))
                r2 = stage2_by_id.get(cid)
                if r2 is None:
                    continue
                for k, v in r2.items():
                    if k.startswith("bioemu_"):
                        r[k] = v
                r["bioemu_tier"] = "tier2"
                r["bioemu_refined"] = True
                r["bioemu_refinement_stage"] = 2
                r["bioemu_tier2_calibrated"] = r.get("bioemu_calibrated")
                r["bioemu_tier2_std"] = r.get("bioemu_std")
        else:
            logger.info("Tiered BioEmu stage2 skipped: eligible=%d top=%d", n_eligible, n_top)
            if stage_progress is not None:
                stage_progress.update(1)
                stage_progress.set_postfix(stage="tier2_skipped", refined=0)

        # Stage 3: final shortlist based on original pool size.
        n_original = len(rows_stage1)
        stage2_eligible = [r for r in rows_stage2 if r.get("bioemu_status") == "ok" and r.get("bioemu_calibrated") is not None]
        n_stage2_eligible = len(stage2_eligible)
        n_tier3_target = int(math.ceil(float(args.tier3_top_frac_original) * float(max(1, n_original))))
        n_tier3 = 0
        if n_stage2_eligible > 0 and float(args.tier3_top_frac_original) > 0:
            n_tier3 = max(int(args.tier3_top_min), n_tier3_target)
            n_tier3 = min(n_stage2_eligible, n_tier3)

        if n_tier3 > 0:
            ranked2 = sorted(
                stage2_eligible,
                key=lambda r: _rank_score(r, args.tier_rank_metric, args.tier_risk_kappa),
                reverse=True,
            )
            top3_ids = {str(r.get("candidate_id")) for r in ranked2[:n_tier3]}
            stage3_in_rows = [r for r in rows_stage2 if str(r.get("candidate_id")) in top3_ids]

            stage3_in = td_path / "stage3_in.jsonl"
            stage3_out = td_path / "stage3_out.jsonl"
            write_records(stage3_in, stage3_in_rows)

            logger.info(
                "Tiered BioEmu stage3 run: original=%d stage2_eligible=%d selected=%d (target_frac_original=%.3f) samples_per_candidate=%d",
                n_original,
                n_stage2_eligible,
                n_tier3,
                float(args.tier3_top_frac_original),
                int(args.tier3_num_samples),
            )

            _run_bioemu_stage(
                python_exe=python_exe,
                scorer=scorer,
                input_path=stage3_in,
                output_path=stage3_out,
                num_samples=int(args.tier3_num_samples),
                args=args,
                seed=int(args.seed) + 200000,
            )
            if stage_progress is not None:
                stage_progress.update(1)
                stage_progress.set_postfix(stage="tier3_done", refined=n_tier3)

            rows_stage3 = read_records(stage3_out)
            stage3_by_id = {str(r.get("candidate_id")): r for r in rows_stage3}
            for r in rows_stage1:
                cid = str(r.get("candidate_id"))
                r3 = stage3_by_id.get(cid)
                if r3 is None:
                    continue
                for k, v in r3.items():
                    if k.startswith("bioemu_"):
                        r[k] = v
                r["bioemu_tier"] = "tier3"
                r["bioemu_refined"] = True
                r["bioemu_refinement_stage"] = 3
                r["bioemu_tier3_calibrated"] = r.get("bioemu_calibrated")
                r["bioemu_tier3_std"] = r.get("bioemu_std")
        else:
            logger.info(
                "Tiered BioEmu stage3 skipped: stage2_eligible=%d top3=%d target_frac_original=%.3f",
                n_stage2_eligible,
                n_tier3,
                float(args.tier3_top_frac_original),
            )
            if stage_progress is not None:
                stage_progress.update(1)
                stage_progress.set_postfix(stage="tier3_skipped", refined=0)

        write_records(output_path, rows_stage1)

    if stage_progress is not None:
        stage_progress.close()

    logger.info(
        "Tiered BioEmu complete: wrote=%d tier1_eligible=%d tier2_refined=%d tier3_refined=%d elapsed=%.2fs",
        len(rows_stage1),
        n_eligible,
        sum(1 for r in rows_stage1 if r.get("bioemu_refinement_stage") == 2),
        sum(1 for r in rows_stage1 if r.get("bioemu_refinement_stage") == 3),
        time.perf_counter() - t0,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compute baseline candidate records from training index using a generator backend."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path
import sys
import time


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _run_adflip_batch(
    *,
    root: Path,
    rows: list[dict],
    ckpt_path: str,
    env_name: str,
    device: str,
    steps: int,
    threshold: float,
    no_progress: bool,
) -> dict[int, str]:
    """Run one persistent ADFLIP worker process for all CIFs."""
    ckpt_abs = Path(ckpt_path).resolve()
    worker_script_abs = (root / "scripts/prep/oracles/adflip_batch_generate.py").resolve()
    with tempfile.TemporaryDirectory(prefix="adflip_batch_") as td:
        td_path = Path(td)
        inp = td_path / "inputs.jsonl"
        outp = td_path / "outputs.jsonl"
        with inp.open("w", encoding="utf-8") as fh:
            for idx, row in enumerate(rows):
                cif_path = (root / row["cif_path"]).resolve()
                fh.write(
                    json.dumps(
                        {
                            "idx": idx,
                            "cif_path": str(cif_path),
                            "stem": row["stem"],
                        },
                        sort_keys=True,
                    )
                )
                fh.write("\n")

        no_progress_arg = "--no-progress" if no_progress else ""
        inner = (
            "cd models/ADFLIP && "
            f"PYTHONPATH=. python {shlex.quote(str(worker_script_abs))} "
            f"--input-jsonl {shlex.quote(str(inp))} --output-jsonl {shlex.quote(str(outp))} "
            f"--ckpt {shlex.quote(str(ckpt_abs))} --device {shlex.quote(device)} --method adaptive "
            f"--steps {steps} --threshold {threshold} {no_progress_arg}"
        )
        cmd = ["conda", "run", "--no-capture-output", "-n", env_name, "bash", "-lc", inner]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ADFLIP batch worker failed rc={proc.returncode}")

        seq_by_idx: dict[int, str] = {}
        with outp.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                idx = int(rec["idx"])
                if not rec.get("ok"):
                    raise RuntimeError(
                        f"ADFLIP batch error at idx={idx} cif={rec.get('cif_path')}: {rec.get('error')}"
                    )
                seq = str(rec.get("sequence") or "")
                if not seq:
                    raise RuntimeError(f"ADFLIP returned empty sequence idx={idx} cif={rec.get('cif_path')}")
                seq_by_idx[idx] = seq

        if len(seq_by_idx) != len(rows):
            raise RuntimeError(f"ADFLIP batch output size mismatch: got={len(seq_by_idx)} expected={len(rows)}")
        return seq_by_idx


def _run_ligandmpnn_batch(
    *,
    root: Path,
    rows: list[dict],
    env_name: str,
    seed: int,
    model_type: str,
    checkpoint_ligand_mpnn: str,
    use_atom_context: int,
    parse_atoms_with_zero_occupancy: int,
    run_heartbeat_sec: float,
    batch_size: int,
    number_of_batches: int,
    temperature: float,
    no_progress: bool,
) -> dict[int, str]:
    """Run one persistent LigandMPNN worker process for all CIF/PDB rows."""
    ligand_root = (root / "models/LigandMPNN").resolve()
    if not ligand_root.exists():
        raise RuntimeError(f"LigandMPNN repository not found at {ligand_root}")
    if not (ligand_root / "run.py").exists():
        raise RuntimeError(f"LigandMPNN run.py missing at {ligand_root / 'run.py'}")

    worker_script_abs = (root / "scripts/prep/oracles/ligandmpnn_batch_generate.py").resolve()
    pycompat = (root / "scripts/vendor/pycompat").resolve()
    with tempfile.TemporaryDirectory(prefix="ligandmpnn_batch_") as td:
        td_path = Path(td)
        inp = td_path / "inputs.jsonl"
        outp = td_path / "outputs.jsonl"
        with inp.open("w", encoding="utf-8") as fh:
            for idx, row in enumerate(rows):
                cif_path = (root / row["cif_path"]).resolve()
                fh.write(
                    json.dumps(
                        {
                            "idx": idx,
                            "cif_path": str(cif_path),
                            "stem": row["stem"],
                        },
                        sort_keys=True,
                    )
                )
                fh.write("\n")

        no_progress_arg = "--no-progress" if no_progress else ""
        checkpoint_arg = ""
        if checkpoint_ligand_mpnn:
            ckpt_abs = (
                (root / checkpoint_ligand_mpnn).resolve()
                if not Path(checkpoint_ligand_mpnn).is_absolute()
                else Path(checkpoint_ligand_mpnn).resolve()
            )
            if not ckpt_abs.exists():
                raise RuntimeError(f"LigandMPNN checkpoint not found: {ckpt_abs}")
            checkpoint_arg = f"--checkpoint-ligand-mpnn {shlex.quote(str(ckpt_abs))}"

        inner = (
            f"export PYTHONPATH={shlex.quote(str(pycompat))}:${{PYTHONPATH:-}} && "
            f"python {shlex.quote(str(worker_script_abs))} "
            f"--input-jsonl {shlex.quote(str(inp))} --output-jsonl {shlex.quote(str(outp))} "
            f"--ligandmpnn-root {shlex.quote(str(ligand_root))} --model-type {shlex.quote(model_type)} "
            f"--seed {seed} --ligand-mpnn-use-atom-context {use_atom_context} "
            f"--parse-atoms-with-zero-occupancy {parse_atoms_with_zero_occupancy} "
            f"--batch-size {batch_size} --number-of-batches {number_of_batches} "
            f"--temperature {temperature} "
            f"--run-heartbeat-sec {run_heartbeat_sec} "
            f"{checkpoint_arg} {no_progress_arg}"
        )
        cmd = ["conda", "run", "--no-capture-output", "-n", env_name, "bash", "-lc", inner]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"LigandMPNN batch worker failed rc={proc.returncode}")

        seq_by_idx: dict[int, str] = {}
        with outp.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                idx = int(rec["idx"])
                if not rec.get("ok"):
                    raise RuntimeError(
                        f"LigandMPNN batch error at idx={idx} source={rec.get('source_path')}: {rec.get('error')}"
                    )
                seq = str(rec.get("sequence") or "")
                if not seq:
                    raise RuntimeError(f"LigandMPNN returned empty sequence idx={idx} source={rec.get('source_path')}")
                seq_by_idx[idx] = seq

        if len(seq_by_idx) != len(rows):
            raise RuntimeError(f"LigandMPNN batch output size mismatch: got={len(seq_by_idx)} expected={len(rows)}")
        return seq_by_idx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/m3_default.yaml")
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generator-backend", choices=("ligandmpnn", "adflip"), default=None)
    parser.add_argument("--ligandmpnn-env", default=None)
    parser.add_argument("--ligandmpnn-model-type", default=None)
    parser.add_argument("--ligandmpnn-checkpoint", default=None)
    parser.add_argument("--ligandmpnn-use-atom-context", type=int, default=None)
    parser.add_argument("--ligandmpnn-parse-atoms-with-zero-occupancy", type=int, default=None)
    parser.add_argument("--ligandmpnn-run-heartbeat-sec", type=float, default=None)
    parser.add_argument("--ligandmpnn-batch-size", type=int, default=None)
    parser.add_argument("--ligandmpnn-number-of-batches", type=int, default=None)
    parser.add_argument("--ligandmpnn-temperature", type=float, default=None)
    parser.add_argument("--adflip-env", default=None)
    parser.add_argument("--adflip-ckpt", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.features import deterministic_hash
    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging, iter_progress
    from train.thermogfn.schemas import validate_records, ensure_unique_ids
    from train.thermogfn.config_utils import load_yaml_config, cfg_get

    logger = configure_logging("prep.compute_baselines", level=args.log_level)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = load_yaml_config(cfg_path)
    args.generator_backend = str(args.generator_backend or cfg_get(cfg, "generator.backend", "ligandmpnn"))
    args.ligandmpnn_env = str(args.ligandmpnn_env or cfg_get(cfg, "generator.ligandmpnn.env_name", "ligandmpnn_env"))
    args.ligandmpnn_model_type = str(
        args.ligandmpnn_model_type or cfg_get(cfg, "generator.ligandmpnn.model_type", "ligand_mpnn")
    )
    args.ligandmpnn_checkpoint = str(args.ligandmpnn_checkpoint or cfg_get(cfg, "generator.ligandmpnn.checkpoint", ""))
    args.ligandmpnn_use_atom_context = int(
        args.ligandmpnn_use_atom_context
        if args.ligandmpnn_use_atom_context is not None
        else cfg_get(cfg, "generator.ligandmpnn.use_atom_context", 1)
    )
    args.ligandmpnn_parse_atoms_with_zero_occupancy = int(
        args.ligandmpnn_parse_atoms_with_zero_occupancy
        if args.ligandmpnn_parse_atoms_with_zero_occupancy is not None
        else cfg_get(cfg, "generator.ligandmpnn.parse_atoms_with_zero_occupancy", 0)
    )
    args.ligandmpnn_run_heartbeat_sec = float(
        args.ligandmpnn_run_heartbeat_sec
        if args.ligandmpnn_run_heartbeat_sec is not None
        else cfg_get(cfg, "generator.ligandmpnn.run_heartbeat_sec", 5.0)
    )
    args.adflip_env = str(args.adflip_env or cfg_get(cfg, "generator.adflip.env_name", "ADFLIP"))
    args.adflip_ckpt = str(
        args.adflip_ckpt or cfg_get(cfg, "generator.adflip.checkpoint", "models/ADFLIP/results/weights/ADFLIP_ICML_camera_ready.pt")
    )
    args.device = str(args.device or cfg_get(cfg, "generator.adflip.device", "cuda:0"))
    args.steps = int(args.steps if args.steps is not None else cfg_get(cfg, "generator.adflip.steps", 32))
    args.threshold = float(args.threshold if args.threshold is not None else cfg_get(cfg, "generator.adflip.threshold", 0.9))
    # Only fill from config when CLI does not pass explicit values.
    if args.ligandmpnn_batch_size is None:
        args.ligandmpnn_batch_size = int(cfg_get(cfg, "generator.ligandmpnn.batch_size", 1))
    if args.ligandmpnn_number_of_batches is None:
        args.ligandmpnn_number_of_batches = int(cfg_get(cfg, "generator.ligandmpnn.number_of_batches", 1))
    if args.ligandmpnn_temperature is None:
        args.ligandmpnn_temperature = float(cfg_get(cfg, "generator.ligandmpnn.temperature", 0.1))
    rows = read_records(root / args.index_path)
    logger.info("Loaded index rows=%d from %s", len(rows), root / args.index_path)
    out: list[dict] = []
    if args.generator_backend == "adflip":
        ckpt = (root / args.adflip_ckpt) if not Path(args.adflip_ckpt).is_absolute() else Path(args.adflip_ckpt)
        if not ckpt.exists():
            print(f"ADFLIP checkpoint not found: {ckpt}", file=sys.stderr)
            return 2
        logger.info(
            "Starting baseline generation backend=adflip env=%s device=%s steps=%d threshold=%.3f",
            args.adflip_env,
            args.device,
            args.steps,
            args.threshold,
        )
    else:
        logger.info(
            "Starting baseline generation backend=ligandmpnn env=%s model_type=%s use_atom_context=%d batch_size=%d n_batches=%d temp=%.3f",
            args.ligandmpnn_env,
            args.ligandmpnn_model_type,
            args.ligandmpnn_use_atom_context,
            args.ligandmpnn_batch_size,
            args.ligandmpnn_number_of_batches,
            args.ligandmpnn_temperature,
        )

    gen_t0 = time.perf_counter()
    try:
        if args.generator_backend == "adflip":
            seq_by_idx = _run_adflip_batch(
                root=root,
                rows=rows,
                ckpt_path=args.adflip_ckpt,
                env_name=args.adflip_env,
                device=args.device,
                steps=args.steps,
                threshold=args.threshold,
                no_progress=args.no_progress,
            )
        else:
            seq_by_idx = _run_ligandmpnn_batch(
                root=root,
                rows=rows,
                env_name=args.ligandmpnn_env,
                seed=args.seed,
                model_type=args.ligandmpnn_model_type,
                checkpoint_ligand_mpnn=args.ligandmpnn_checkpoint,
                use_atom_context=args.ligandmpnn_use_atom_context,
                parse_atoms_with_zero_occupancy=args.ligandmpnn_parse_atoms_with_zero_occupancy,
                run_heartbeat_sec=args.ligandmpnn_run_heartbeat_sec,
                batch_size=args.ligandmpnn_batch_size,
                number_of_batches=args.ligandmpnn_number_of_batches,
                temperature=args.ligandmpnn_temperature,
                no_progress=args.no_progress,
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Baseline generation failed backend=%s: %s", args.generator_backend, exc)
        return 4
    gen_elapsed = time.perf_counter() - gen_t0
    logger.info(
        "Completed baseline generation backend=%s for %d rows in %.2fs (avg=%.2fs/row)",
        args.generator_backend,
        len(rows),
        gen_elapsed,
        gen_elapsed / max(1, len(rows)),
    )

    for i, row in enumerate(iter_progress(rows, total=len(rows), desc="baselines:records", no_progress=args.no_progress)):
        seq = seq_by_idx.get(i)
        if not seq:
            cif_path = root / row["cif_path"]
            print(f"Baseline sequence generation failed for {cif_path}", file=sys.stderr)
            logger.error("baseline generation failed for cif_path=%s at index=%d", cif_path, i)
            return 4
        seed_seq = seq

        cid = deterministic_hash(f"{args.run_id}:baseline:{row['stem']}:{seed_seq}")
        atom_count = int(row.get("num_atoms") or 0)
        task_type = str(row.get("task_type") or "monomer")
        is_monomer = task_type == "monomer"
        decomposition = row.get("decomposition")
        if decomposition is None and task_type in {"ppi", "ligand"}:
            # Placeholder decomposition; should be replaced by curated component maps when available.
            decomposition = {
                "bound": "complex",
                "components": ["component_1", "component_2"],
                "stoichiometry": [1, 1],
            }
        rec = {
            "candidate_id": cid,
            "run_id": args.run_id,
            "round_id": 0,
            "task_type": task_type,
            "backbone_id": row["stem"],
            "seed_id": row["stem"],
            "sequence": seed_seq,
            "mutations": [],
            "K": 0,
            "prepared_atom_count": atom_count,
            "eligibility": {
                "bioemu": is_monomer,
                "uma_whole": atom_count <= 8000,
                "uma_local": atom_count > 8000,
            },
            "source": "baseline",
            "schema_version": "v1",
            "split": row["split"],
            "spec_path": row["spec_path"],
            "cif_path": row["cif_path"],
            "pack_unc": 0.0,
            "novelty": 0.0,
            "sequence_length": len(seed_seq),
        }
        for key in (
            "substrate_smiles",
            "product_smiles",
            "ligand_smiles",
            "smiles",
            "Smiles",
            "substrate",
            "product",
            "organism",
            "Organism",
            "ph",
            "pH",
            "temp",
            "Temp",
            "chain_id",
            "protein_path",
            "ligand_path",
            "complex_path",
        ):
            if key in row and row[key] is not None:
                rec[key] = row[key]
        if task_type in {"ppi", "ligand"}:
            rec["decomposition"] = decomposition
        out.append(rec)

    ensure_unique_ids(out, "candidate_id")
    summary = validate_records(out, "candidate")
    if summary.invalid:
        for err in summary.errors[:10]:
            print(err, file=sys.stderr)
        return 3

    write_records(root / args.output, out)
    logger.info("Wrote baseline records=%d to %s (elapsed=%.2fs)", len(out), root / args.output, time.perf_counter() - t0)
    print(root / args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

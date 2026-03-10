#!/usr/bin/env python3
"""Score candidates with GraphKcat for Kcat-mode active learning."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
import sys
import tempfile
import time


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "train").exists() and (parent / "scripts").exists():
            return parent
    raise RuntimeError("Could not locate repository root")


def _pick_substrate_smiles(rec: dict) -> str | None:
    for key in ("substrate_smiles", "Smiles", "smiles", "ligand_smiles"):
        val = rec.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            for item in val:
                if item is None:
                    continue
                s = str(item).strip()
                if s:
                    return s
            continue
        s = str(val).strip()
        if s:
            return s
    return None


def _to_float(value, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:  # noqa: BLE001
        return float(default)


def _cif_to_pdb(src: Path, dst: Path) -> None:
    from Bio.PDB import MMCIFParser, PDBIO

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(src.stem, str(src))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(dst))


def _materialize_protein_pdb(rec: dict, protein_out: Path, root: Path) -> None:
    candidate_paths: list[Path] = []
    for key in ("protein_path", "cif_path"):
        val = rec.get(key)
        if not val:
            continue
        p = Path(str(val))
        if not p.is_absolute():
            p = root / p
        candidate_paths.append(p)

    source = None
    for p in candidate_paths:
        if p.exists():
            source = p
            break
    if source is None:
        raise FileNotFoundError(
            f"no protein structure path found for candidate_id={rec.get('candidate_id')} "
            f"(checked keys: protein_path, cif_path)"
        )

    suffix = source.suffix.lower()
    if suffix == ".pdb":
        shutil.copy2(source, protein_out)
        return
    if suffix in {".cif", ".mmcif"}:
        _cif_to_pdb(source, protein_out)
        return
    raise ValueError(f"unsupported protein source extension: {source}")


def _materialize_ligand_sdf(rec: dict, ligand_out: Path, root: Path) -> str:
    ligand_path = rec.get("ligand_path")
    if ligand_path:
        p = Path(str(ligand_path))
        if not p.is_absolute():
            p = root / p
        if p.exists() and p.suffix.lower() == ".sdf":
            shutil.copy2(p, ligand_out)
            return str(ligand_out)

    smiles = _pick_substrate_smiles(rec)
    if not smiles:
        raise ValueError(
            f"missing ligand path and substrate smiles for candidate_id={rec.get('candidate_id')}"
        )

    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid substrate smiles for candidate_id={rec.get('candidate_id')}: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        raise RuntimeError(f"RDKit embedding failed for candidate_id={rec.get('candidate_id')}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:  # noqa: BLE001
        pass
    writer = Chem.SDWriter(str(ligand_out))
    writer.write(mol)
    writer.close()
    return str(ligand_out)


def _graphkcat_subprocess_env(base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    conda_prefix = env.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        llvm_path = Path(conda_prefix) / "lib" / "libLLVM-15.so"
        if llvm_path.exists():
            prior = env.get("LD_PRELOAD", "").strip()
            env["LD_PRELOAD"] = f"{llvm_path}:{prior}" if prior else str(llvm_path)
    return env


def _run_with_heartbeat(
    cmd: list[str],
    cwd: Path,
    logger,
    step_name: str,
    heartbeat_sec: float,
    env: dict[str, str] | None = None,
) -> int:
    logger.info("CMD: %s", " ".join(str(c) for c in cmd))
    t0 = time.perf_counter()
    hb = max(1.0, float(heartbeat_sec))
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env)  # noqa: S603
    while True:
        try:
            rc = proc.wait(timeout=hb)
            logger.info("STEP %s finished rc=%d elapsed=%.1fs", step_name, rc, time.perf_counter() - t0)
            return rc
        except subprocess.TimeoutExpired:
            logger.info("STEP %s still running elapsed=%.1fs", step_name, time.perf_counter() - t0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-root", default="models/GraphKcat")
    parser.add_argument("--checkpoint", default="models/GraphKcat/checkpoint/paper.pt")
    parser.add_argument("--cfg", default="TrainConfig_kcat_enz")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--distance-cutoff-a", type=float, default=8.0)
    parser.add_argument("--organism-default", default="unknown")
    parser.add_argument("--ph-default", type=float, default=7.0)
    parser.add_argument("--temp-default", type=float, default=30.0)
    parser.add_argument("--std-default", type=float, default=0.25)
    parser.add_argument("--heartbeat-sec", type=float, default=30.0)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_records
    from train.thermogfn.progress import configure_logging, iter_progress

    logger = configure_logging("oracle.graphkcat", level=args.log_level)
    if abs(float(args.distance_cutoff_a) - 8.0) > 1e-6:
        logger.warning(
            "graphkcat distance_cutoff_a=%.3f requested, but models/GraphKcat/predict.py uses fixed cutoff=8.0",
            args.distance_cutoff_a,
        )

    model_root = Path(args.model_root)
    if not model_root.is_absolute():
        model_root = root / model_root
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = root / ckpt

    if not model_root.exists():
        raise FileNotFoundError(f"GraphKcat model root not found: {model_root}")
    if not ckpt.exists():
        raise FileNotFoundError(f"GraphKcat checkpoint not found: {ckpt}")

    organism_set = model_root / "sub_utils" / "all_organism_set.npy"
    temp_set = model_root / "sub_utils" / "temp_set.npy"
    if not organism_set.exists() or not temp_set.exists():
        raise FileNotFoundError(
            "GraphKcat organism/temp set files missing under models/GraphKcat/sub_utils"
        )

    rows = read_records(root / args.candidate_path)
    logger.info(
        "GraphKcat scoring start: candidates=%d model_root=%s checkpoint=%s batch_size=%d device=%s",
        len(rows),
        model_root,
        ckpt,
        args.batch_size,
        args.device,
    )

    if args.work_dir:
        work_root = Path(args.work_dir)
        if not work_root.is_absolute():
            work_root = root / work_root
        work_root.mkdir(parents=True, exist_ok=True)
        td_ctx = tempfile.TemporaryDirectory(prefix="graphkcat_", dir=str(work_root))
    else:
        td_ctx = tempfile.TemporaryDirectory(prefix="graphkcat_")

    with td_ctx as td:
        td_path = Path(td)
        data_root = td_path / "data"
        data_root.mkdir(parents=True, exist_ok=True)

        csv_path = td_path / "input.csv"
        out_dir = td_path / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        fieldnames = ["id", "complex", "ligand", "protein", "Organism", "substrate", "Smiles", "pH", "Temp"]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

            for rec in iter_progress(rows, total=len(rows), desc="graphkcat:prepare", no_progress=args.no_progress):
                cid = str(rec.get("candidate_id") or "").strip()
                if not cid:
                    raise ValueError("candidate missing candidate_id")

                cdir = data_root / cid
                cdir.mkdir(parents=True, exist_ok=True)
                protein_path = cdir / f"{cid}_protein.pdb"
                ligand_path = cdir / f"{cid}_ligand.sdf"

                _materialize_protein_pdb(rec, protein_path, root)
                _materialize_ligand_sdf(rec, ligand_path, root)

                smiles = _pick_substrate_smiles(rec)
                if not smiles:
                    raise ValueError(f"missing substrate smiles for candidate_id={cid}")

                writer.writerow(
                    {
                        "id": cid,
                        "complex": "",
                        "ligand": str(ligand_path),
                        "protein": str(protein_path),
                        "Organism": rec.get("Organism") or rec.get("organism") or args.organism_default,
                        "substrate": rec.get("substrate") or "",
                        "Smiles": smiles,
                        "pH": _to_float(rec.get("pH", rec.get("ph")), args.ph_default),
                        "Temp": _to_float(rec.get("Temp", rec.get("temp")), args.temp_default),
                    }
                )

        cmd = [
            sys.executable,
            str(model_root / "predict.py"),
            "--csv_file",
            str(csv_path),
            "--output_dir",
            str(out_dir),
            "--batch_size",
            str(args.batch_size),
            "--cpkt_path",
            str(ckpt),
            "--device",
            str(args.device),
            "--cfg",
            str(args.cfg),
            "--organism_set_path",
            str(organism_set),
            "--temp_set_path",
            str(temp_set),
        ]

        predict_env = _graphkcat_subprocess_env(os.environ)
        if predict_env.get("LD_PRELOAD"):
            logger.info("GraphKcat using LD_PRELOAD=%s", predict_env["LD_PRELOAD"])
        rc = _run_with_heartbeat(
            cmd,
            cwd=model_root,
            logger=logger,
            step_name="graphkcat:predict",
            heartbeat_sec=args.heartbeat_sec,
            env=predict_env,
        )
        if rc != 0:
            raise RuntimeError(f"GraphKcat predict.py failed rc={rc}")

        out_csv = out_dir / "inference_results.csv"
        if not out_csv.exists():
            raise FileNotFoundError(f"GraphKcat output CSV missing: {out_csv}")

        by_id: dict[str, dict] = {}
        with out_csv.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rid = str(row.get("id") or "").strip()
                if rid:
                    by_id[rid] = row

        out_rows: list[dict] = []
        for rec in iter_progress(rows, total=len(rows), desc="graphkcat:merge", no_progress=args.no_progress):
            cid = str(rec.get("candidate_id") or "").strip()
            pred = by_id.get(cid)
            if pred is None:
                raise RuntimeError(f"GraphKcat missing prediction for candidate_id={cid}")
            try:
                rec["graphkcat_log_kcat"] = float(pred["pred_log_kcat_graphkcat"])
                rec["graphkcat_log_km"] = float(pred["pred_log_km_graphkcat"])
                rec["graphkcat_log_kcat_km"] = float(pred["pred_log_kcat_km_graphkcat"])
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"GraphKcat parse error candidate_id={cid}: {exc}") from exc
            rec["graphkcat_std"] = float(args.std_default)
            rec["graphkcat_status"] = "ok"
            out_rows.append(rec)

    write_records(root / args.output_path, out_rows)
    logger.info("GraphKcat scoring complete: wrote=%d elapsed=%.2fs", len(out_rows), time.perf_counter() - t0)
    print(root / args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

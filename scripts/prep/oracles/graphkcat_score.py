#!/usr/bin/env python3
"""Score candidates with GraphKcat for Kcat-mode active learning."""

from __future__ import annotations

import argparse
import csv
import math
import os
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


GRAPHKCAT_SUPPORTED_ATOMS = {"B", "Br", "C", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"}


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


def _write_protein_only_pdb(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    keep = ("ATOM  ", "TER", "END", "MODEL ", "ENDMDL")
    with src.open("r", encoding="utf-8") as in_fh, dst.open("w", encoding="utf-8") as out_fh:
        for line in in_fh:
            if line.startswith(keep):
                out_fh.write(line)


def _materialize_protein_pdb(rec: dict, protein_out: Path, root: Path) -> None:
    candidate_paths: list[Path] = []
    for key in (
        "reactant_protein_packed_path",
        "protein_path",
        "reactant_complex_packed_path",
        "complex_path",
        "cif_path",
        "reactant_complex_path",
    ):
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
        _write_protein_only_pdb(source, protein_out)
        return
    if suffix in {".cif", ".mmcif"}:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _cif_to_pdb(source, tmp_path)
            _write_protein_only_pdb(tmp_path, protein_out)
        finally:
            tmp_path.unlink(missing_ok=True)
        return
    raise ValueError(f"unsupported protein source extension: {source}")


def _validate_sdf_has_conformer(path: Path) -> None:
    from rdkit import Chem

    mol = None
    for sanitize in (True, False):
        mol = Chem.MolFromMolFile(str(path), sanitize=sanitize, removeHs=False)
        if mol is not None and mol.GetNumAtoms() > 0 and mol.GetNumConformers() > 0:
            return
    raise ValueError(f"ligand SDF is not parseable or has no conformer: {path}")


def _pick_graphkcat_substrate_mol(smiles: str, *, candidate_id: str | None = None):
    """Return the molecule GraphKcat should score from a possibly disconnected mixture."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid substrate smiles for candidate_id={candidate_id}: {smiles}")

    unsupported_atoms = sorted({atom.GetSymbol() for atom in mol.GetAtoms()} - GRAPHKCAT_SUPPORTED_ATOMS)
    if unsupported_atoms:
        raise ValueError(
            "GraphKcat unsupported ligand atom types for candidate_id="
            f"{candidate_id}: {','.join(unsupported_atoms)}"
        )

    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if not fragments:
        raise ValueError(f"GraphKcat found no ligand fragments for candidate_id={candidate_id}")

    bonded = [frag for frag in fragments if frag.GetNumBonds() > 0]
    if not bonded:
        raise ValueError(f"GraphKcat requires a bonded substrate molecule for candidate_id={candidate_id}")

    def sort_key(frag):
        symbols = [atom.GetSymbol() for atom in frag.GetAtoms()]
        has_carbon = any(sym == "C" for sym in symbols)
        heavy_atoms = sum(1 for atom in frag.GetAtoms() if atom.GetAtomicNum() > 1)
        return (1 if has_carbon else 0, heavy_atoms, frag.GetNumBonds(), frag.GetNumAtoms())

    selected = max(bonded, key=sort_key)
    selected_smiles = Chem.MolToSmiles(selected, canonical=True, isomericSmiles=True)
    policy = "full_smiles" if len(fragments) == 1 else "largest_bonded_carbon_fragment"
    return selected, selected_smiles, policy, len(fragments)


def _embed_molecule_robust(mol, *, candidate_id: str | None = None):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Work on a copy so callers can still inspect the original sanitized molecule.
    mol_h = Chem.AddHs(Chem.Mol(mol))
    attempts: list[tuple[int, bool, bool]] = []
    for seed in (13, 29, 73, 173, 997, 2027):
        attempts.append((seed, True, False))
    for seed in (13, 73, 997):
        attempts.append((seed, False, False))
    for seed in (13, 73, 997):
        attempts.append((seed, False, True))

    last_code = None
    for seed, enforce_chirality, random_coords in attempts:
        trial = Chem.Mol(mol_h)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.enforceChirality = bool(enforce_chirality)
        params.useRandomCoords = bool(random_coords)
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        try:
            params.maxAttempts = 2000
        except AttributeError:
            pass
        code = AllChem.EmbedMolecule(trial, params)
        last_code = code
        if code != 0:
            continue
        _optimize_embedded_molecule(trial)
        return trial, {
            "seed": int(seed),
            "enforce_chirality": bool(enforce_chirality),
            "use_random_coords": bool(random_coords),
        }

    # Final fallback: many conformers with random coordinates, then use the first valid conformer.
    trial = Chem.Mol(mol_h)
    params = AllChem.ETKDGv3()
    params.randomSeed = 31013
    params.enforceChirality = False
    params.useRandomCoords = True
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    ids = list(AllChem.EmbedMultipleConfs(trial, numConfs=24, params=params))
    if ids:
        keep = int(ids[0])
        for conf_id in ids[1:]:
            trial.RemoveConformer(int(conf_id))
        _optimize_embedded_molecule(trial, conf_id=keep)
        return trial, {
            "seed": 31013,
            "enforce_chirality": False,
            "use_random_coords": True,
            "multi_conf_fallback": True,
        }

    raise RuntimeError(
        f"RDKit embedding failed for candidate_id={candidate_id}; last_embed_code={last_code}"
    )


def _optimize_embedded_molecule(mol, *, conf_id: int = -1) -> None:
    from rdkit.Chem import AllChem

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", confId=conf_id, maxIters=500)
                return
    except Exception:  # noqa: BLE001
        pass
    try:
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
    except Exception:  # noqa: BLE001
        pass


def _materialize_ligand_sdf(rec: dict, ligand_out: Path, root: Path) -> dict[str, object]:
    ligand_path = rec.get("ligand_path")
    smiles = _pick_substrate_smiles(rec)
    if not smiles:
        raise ValueError(
            f"missing ligand path and substrate smiles for candidate_id={rec.get('candidate_id')}"
        )
    selected_mol, selected_smiles, policy, n_fragments = _pick_graphkcat_substrate_mol(
        smiles,
        candidate_id=str(rec.get("candidate_id") or ""),
    )

    if ligand_path and policy == "full_smiles":
        p = Path(str(ligand_path))
        if not p.is_absolute():
            p = root / p
        if p.exists() and p.suffix.lower() == ".sdf":
            shutil.copy2(p, ligand_out)
            _validate_sdf_has_conformer(ligand_out)
            return {
                "path": str(ligand_out),
                "smiles": selected_smiles,
                "source": "existing_sdf",
                "fragment_policy": policy,
                "n_fragments": int(n_fragments),
            }

    from rdkit import Chem

    mol, embed_info = _embed_molecule_robust(
        selected_mol,
        candidate_id=str(rec.get("candidate_id") or ""),
    )
    writer = Chem.SDWriter(str(ligand_out))
    writer.write(mol)
    writer.close()
    _validate_sdf_has_conformer(ligand_out)
    return {
        "path": str(ligand_out),
        "smiles": selected_smiles,
        "source": "rdkit_etkdg",
        "fragment_policy": policy,
        "n_fragments": int(n_fragments),
        "embed_info": embed_info,
    }


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
) -> tuple[int, dict]:
    logger.info("CMD: %s", " ".join(str(c) for c in cmd))
    t0 = time.perf_counter()
    hb = max(1.0, float(heartbeat_sec))
    from train.thermogfn.progress import PeakVRAMMonitor

    monitor = PeakVRAMMonitor()
    monitor.start()
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env)  # noqa: S603
    try:
        while True:
            try:
                rc = proc.wait(timeout=hb)
                logger.info("STEP %s finished rc=%d elapsed=%.1fs", step_name, rc, time.perf_counter() - t0)
                break
            except subprocess.TimeoutExpired:
                logger.info("STEP %s still running elapsed=%.1fs", step_name, time.perf_counter() - t0)
    finally:
        snapshot = monitor.stop()
    return rc, snapshot


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
    parser.add_argument("--mc-dropout-samples", type=int, default=8)
    parser.add_argument("--mc-dropout-seed", type=int, default=13)
    parser.add_argument("--heartbeat-sec", type=float, default=30.0)
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--summary-path", default="")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    root = _repo_root()
    sys.path.insert(0, str(root))

    from train.thermogfn.io_utils import read_records, write_json, write_records
    from train.thermogfn.metrics_utils import summarize_graphkcat_rows
    from train.thermogfn.progress import configure_logging, iter_progress, log_peak_vram_snapshot, make_progress

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
    overall_bar = make_progress(
        total=4,
        desc="graphkcat:overall",
        no_progress=args.no_progress,
        leave=True,
        unit="stage",
    )
    overall_bar.set_postfix_str(f"stage=prepare candidates={len(rows)}")
    try:
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
            prep_errors: dict[str, str] = {}
            graph_inputs: dict[str, dict[str, object]] = {}
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()

                for rec in iter_progress(rows, total=len(rows), desc="graphkcat:prepare", no_progress=args.no_progress):
                    cid = str(rec.get("candidate_id") or "").strip()
                    if not cid:
                        prep_errors[cid] = "candidate missing candidate_id"
                        continue

                    try:
                        cdir = data_root / cid
                        cdir.mkdir(parents=True, exist_ok=True)
                        protein_path = cdir / f"{cid}_protein.pdb"
                        ligand_path = cdir / f"{cid}_ligand.sdf"

                        _materialize_protein_pdb(rec, protein_path, root)
                        ligand_info = _materialize_ligand_sdf(rec, ligand_path, root)

                        smiles = str(ligand_info["smiles"])
                        if not smiles:
                            raise ValueError(f"missing substrate smiles for candidate_id={cid}")
                        graph_inputs[cid] = dict(ligand_info)

                        writer.writerow(
                            {
                                "id": cid,
                                "complex": "",
                                "ligand": str(ligand_info["path"]),
                                "protein": str(protein_path),
                                "Organism": rec.get("Organism") or rec.get("organism") or args.organism_default,
                                "substrate": rec.get("substrate") or "",
                                "Smiles": smiles,
                                "pH": _to_float(rec.get("pH", rec.get("ph")), args.ph_default),
                                "Temp": _to_float(rec.get("Temp", rec.get("temp")), args.temp_default),
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        prep_errors[cid] = str(exc)
                        logger.error("GraphKcat preparation failed candidate_id=%s error=%s", cid or "<unknown>", exc)

            prepared_ok = len(rows) - len(prep_errors)
            overall_bar.update(1)
            overall_bar.set_postfix(stage="prepare", prepared=prepared_ok, errors=len(prep_errors))

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
                "--mc_dropout_samples",
                str(args.mc_dropout_samples),
                "--mc_dropout_seed",
                str(args.mc_dropout_seed),
            ]

            predict_env = _graphkcat_subprocess_env(os.environ)
            if predict_env.get("LD_PRELOAD"):
                logger.info("GraphKcat using LD_PRELOAD=%s", predict_env["LD_PRELOAD"])
            overall_bar.set_postfix(stage="predict", prepared=prepared_ok, errors=len(prep_errors))
            rc, predict_peak_vram = _run_with_heartbeat(
                cmd,
                cwd=model_root,
                logger=logger,
                step_name="graphkcat:predict",
                heartbeat_sec=args.heartbeat_sec,
                env=predict_env,
            )
            log_peak_vram_snapshot(logger, predict_peak_vram, label="graphkcat:predict")
            overall_bar.update(1)
            overall_bar.set_postfix(stage="predict", prepared=prepared_ok, errors=len(prep_errors), rc=rc)
            if rc != 0 and len(prep_errors) != len(rows):
                raise RuntimeError(f"GraphKcat predict.py failed rc={rc}")
            if rc != 0 and len(prep_errors) == len(rows):
                logger.warning("GraphKcat predict skipped because all candidates failed preparation")

            out_csv = out_dir / "inference_results.csv"
            if not out_csv.exists() and len(prep_errors) != len(rows):
                raise FileNotFoundError(f"GraphKcat output CSV missing: {out_csv}")

            by_id: dict[str, dict] = {}
            if out_csv.exists():
                with out_csv.open("r", encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        rid = str(row.get("id") or "").strip()
                        if rid:
                            by_id[rid] = row

            out_rows: list[dict] = []
            for rec in iter_progress(rows, total=len(rows), desc="graphkcat:merge", no_progress=args.no_progress):
                row = dict(rec)
                cid = str(rec.get("candidate_id") or "").strip()
                if cid in graph_inputs:
                    row["graphkcat_input_smiles"] = graph_inputs[cid].get("smiles")
                    row["graphkcat_ligand_source"] = graph_inputs[cid].get("source")
                    row["graphkcat_fragment_policy"] = graph_inputs[cid].get("fragment_policy")
                    row["graphkcat_input_fragment_count"] = graph_inputs[cid].get("n_fragments")
                    if graph_inputs[cid].get("embed_info") is not None:
                        row["graphkcat_embed_info"] = graph_inputs[cid].get("embed_info")
                if cid in prep_errors:
                    row["graphkcat_status"] = "error"
                    row["graphkcat_error"] = prep_errors[cid]
                    row["graphkcat_std"] = float("nan")
                    out_rows.append(row)
                    continue
                pred = by_id.get(cid)
                if pred is None:
                    row["graphkcat_status"] = "error"
                    row["graphkcat_error"] = f"missing prediction for candidate_id={cid}"
                    row["graphkcat_std"] = float("nan")
                    out_rows.append(row)
                    continue
                try:
                    row["graphkcat_log_kcat"] = float(pred["pred_log_kcat_graphkcat"])
                    row["graphkcat_log_km"] = float(pred["pred_log_km_graphkcat"])
                    row["graphkcat_log_kcat_km"] = float(pred["pred_log_kcat_km_graphkcat"])
                    std_val = pred.get("pred_log_kcat_graphkcat_std")
                    if std_val not in {None, ""}:
                        row["graphkcat_std"] = float(std_val)
                        row["graphkcat_log_km_std"] = float(pred.get("pred_log_km_graphkcat_std", "nan"))
                        row["graphkcat_log_kcat_km_std"] = float(pred.get("pred_log_kcat_km_graphkcat_std", "nan"))
                        row["graphkcat_std_source"] = "mc_dropout"
                        row["graphkcat_uncertainty_samples"] = int(args.mc_dropout_samples)
                    else:
                        row["graphkcat_std"] = float(args.std_default)
                        row["graphkcat_std_source"] = "constant_default"
                        row["graphkcat_uncertainty_samples"] = 0
                except Exception as exc:  # noqa: BLE001
                    row["graphkcat_status"] = "error"
                    row["graphkcat_error"] = f"parse error candidate_id={cid}: {exc}"
                    row["graphkcat_std"] = float("nan")
                    out_rows.append(row)
                    continue
                row["graphkcat_status"] = "ok"
                out_rows.append(row)

        status_counts = summarize_graphkcat_rows(out_rows).get("status_counts", {})
        overall_bar.update(1)
        overall_bar.set_postfix(
            stage="merge",
            ok=int(status_counts.get("ok", 0)),
            errors=int(status_counts.get("error", 0)),
        )

        write_records(root / args.output_path, out_rows)
        summary = summarize_graphkcat_rows(out_rows)
        if 'predict_peak_vram' in locals():
            summary["predict_peak_vram"] = predict_peak_vram
        if args.summary_path:
            write_json(root / args.summary_path, summary)
        overall_bar.update(1)
        overall_bar.set_postfix(
            stage="write",
            written=len(out_rows),
            ok=int(summary.get("status_counts", {}).get("ok", 0)),
            errors=int(summary.get("status_counts", {}).get("error", 0)),
        )
        logger.info(
            "GraphKcat summary: ok_fraction=%.3f ok=%d/%d mean_log_kcat=%.4f",
            float(summary.get("ok_fraction", 0.0)),
            int(summary.get("status_counts", {}).get("ok", 0)),
            int(summary.get("n", 0)),
            float(summary.get("log_kcat", {}).get("mean", 0.0)),
        )
        logger.info("GraphKcat scoring complete: wrote=%d elapsed=%.2fs", len(out_rows), time.perf_counter() - t0)
        print(root / args.output_path)
        return 0
    finally:
        overall_bar.close()


if __name__ == "__main__":
    raise SystemExit(main())

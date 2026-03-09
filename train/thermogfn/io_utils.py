"""File I/O helpers with jsonl-first behavior and optional parquet support."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    rows: list[dict] = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError("Need fieldnames when writing empty CSV")
    fields = fieldnames or list(rows[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_records(path: str | Path) -> list[dict]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(p)
    if suffix == ".csv":
        return read_csv(p)
    if suffix == ".json":
        obj = read_json(p)
        if isinstance(obj, list):
            return obj
        raise ValueError(f"JSON file {p} does not contain a list")
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Parquet requires pandas+pyarrow") from exc
        df = pd.read_parquet(p)
        return df.to_dict(orient="records")
    raise ValueError(f"Unsupported record file extension: {p}")


def write_records(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        write_jsonl(p, rows)
        return
    if suffix == ".csv":
        write_csv(p, rows)
        return
    if suffix == ".json":
        write_json(p, rows)
        return
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Parquet requires pandas+pyarrow") from exc
        df = pd.DataFrame(rows)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        return
    raise ValueError(f"Unsupported record file extension: {p}")


def infer_record_path(base_dir: str | Path, name: str, prefer_parquet: bool = False) -> Path:
    ext = ".parquet" if prefer_parquet else ".jsonl"
    return Path(base_dir) / f"{name}{ext}"

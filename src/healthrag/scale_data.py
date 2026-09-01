from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from .core import PatientRecord, stable_canary, save_json
from .data import load_records

SCALE_LEVELS = [1000, 10000, 100000]
SCALE_EXTRA_SEEDS = [5103 + i for i in range(11)]


def _relabel(record: PatientRecord, alias: str) -> PatientRecord:
    canary = stable_canary(alias, record.patient_id)
    lines = []
    for line in record.text.splitlines():
        if line.startswith("PatientAlias:"):
            lines.append(f"PatientAlias: {alias}")
        elif line.startswith("ExperimentalSecurityCanary:"):
            lines.append(f"ExperimentalSecurityCanary: {canary}")
        else:
            lines.append(line)
    return PatientRecord(alias, record.patient_id, "\n".join(lines), canary, record.primary_condition, record.primary_medication)


def _load_extra(extra_dir: Path) -> list[PatientRecord]:
    files = sorted(extra_dir.glob("scale_shard_*.json"))
    if len(files) != 11:
        raise RuntimeError(f"Expected 11 extra normalized shards, found {len(files)}: {files}")
    out: list[PatientRecord] = []
    for p in files:
        out.extend(PatientRecord(**x) for x in json.loads(p.read_text()))
    return out


def _proportional_acl(base_acl: dict[str, list[str]], records: list[PatientRecord], n: int) -> tuple[dict[str, list[str]], list[str]]:
    if n not in SCALE_LEVELS:
        raise ValueError(n)
    mapping = {k: list(v) for k, v in base_acl.items()}
    for account in ["A", "B", "C", "D", "E"]:
        mapping.setdefault(account, [])
    protected: list[str] = []
    # Base 1,000 already has 200 patients per A-D plus 200 protected.
    base_authorized = set().union(*(set(mapping[a]) for a in ["A", "B", "C", "D"]))
    for r in records[:1000]:
        if r.alias not in base_authorized:
            protected.append(r.alias)
    # Every five new aliases are assigned A,B,C,D,protected. This keeps the
    # 80/20 design exact at 10K and 100K while assignments remain nested.
    buckets = ["A", "B", "C", "D", "P"]
    for j, r in enumerate(records[1000:n]):
        bucket = buckets[j % 5]
        if bucket == "P":
            protected.append(r.alias)
        else:
            mapping[bucket].append(r.alias)
    return mapping, protected


def merge_scale_corpus(base_records_path: Path, base_auth_path: Path, base_protected_path: Path, extra_dir: Path, outdir: Path) -> None:
    base = load_records(base_records_path)
    if len(base) != 1000:
        raise RuntimeError(f"Base corpus must contain exactly 1,000 records, got {len(base)}")
    base_auth = json.loads(base_auth_path.read_text())
    base_protected = json.loads(base_protected_path.read_text())
    if sum(len(base_auth.get(a, [])) for a in ["A", "B", "C", "D"]) != 800 or len(base_protected) != 200:
        raise RuntimeError("Base authorization corpus does not match the validated 800/200 split")

    extras = _load_extra(extra_dir)
    base_ids = {r.patient_id for r in base}
    unique: dict[str, PatientRecord] = {}
    for r in extras:
        if r.patient_id not in base_ids:
            unique.setdefault(r.patient_id, r)
    candidates = sorted(unique.values(), key=lambda r: hashlib.sha256(r.patient_id.encode()).hexdigest())
    if len(candidates) < 99000:
        raise RuntimeError(f"Need 99,000 unique extra patients after de-duplication, found {len(candidates)}")
    selected = candidates[:99000]
    relabeled = [_relabel(r, f"PAT-{i}") for i, r in enumerate(selected, start=1001)]
    records = base + relabeled
    if len(records) != 100000 or len({r.patient_id for r in records}) != 100000 or len({r.alias for r in records}) != 100000:
        raise RuntimeError("100K merged corpus uniqueness validation failed")

    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "records_100k.json", [r.__dict__ for r in records])
    save_json(outdir / "authorization_fixed.json", base_auth)
    save_json(outdir / "protected_fixed.json", base_protected)

    proportional_counts = {}
    for n in SCALE_LEVELS:
        mapping, protected = _proportional_acl(base_auth, records, n)
        save_json(outdir / f"authorization_proportional_{n}.json", mapping)
        save_json(outdir / f"protected_proportional_{n}.json", protected)
        proportional_counts[str(n)] = {
            "accounts": {a: len(mapping[a]) for a in ["A", "B", "C", "D", "E"]},
            "protected": len(protected),
        }

    text_lengths = [len(r.text) for r in records]
    manifest = {
        "master_patient_count": 100000,
        "nested_prefix_sizes": SCALE_LEVELS,
        "base_patient_count": 1000,
        "extra_patient_count": 99000,
        "extra_candidate_count_before_selection": len(candidates),
        "extra_seed_shards": SCALE_EXTRA_SEEDS,
        "base_seed": 4103,
        "synthea_version": "v4.0.0",
        "reference_date": "20260101",
        "state": "Massachusetts",
        "fixed_authorization_scope": {a: len(base_auth.get(a, [])) for a in ["A", "B", "C", "D", "E"]},
        "proportional_authorization": proportional_counts,
        "assignment_rule": "Base 1K preserved exactly; added patients are deterministically SHA-256 sorted and assigned A/B/C/D/protected in a repeating five-record cycle for proportional conditions.",
        "normalized_text_characters": {
            "min": min(text_lengths),
            "median": statistics.median(text_lengths),
            "mean": statistics.fmean(text_lengths),
            "max": max(text_lengths),
        },
    }
    save_json(outdir / "scale_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-records", type=Path, required=True)
    ap.add_argument("--base-auth", type=Path, required=True)
    ap.add_argument("--base-protected", type=Path, required=True)
    ap.add_argument("--extra-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("data/scale"))
    args = ap.parse_args()
    merge_scale_corpus(args.base_records, args.base_auth, args.base_protected, args.extra_dir, args.outdir)


if __name__ == "__main__":
    main()

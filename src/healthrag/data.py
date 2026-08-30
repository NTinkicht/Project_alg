from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

from .core import PatientRecord, stable_canary, save_json

SYNTHEA_URL = "https://raw.githubusercontent.com/synthetichealth/synthea-sample-data/main/downloads/latest/synthea_sample_data_fhir_latest.zip"
RESOURCE_TYPES = {"Patient", "Condition", "MedicationRequest", "Medication", "Observation", "Encounter", "AllergyIntolerance", "Procedure"}


def _code_display(resource: dict) -> str | None:
    code = resource.get("code") or resource.get("medicationCodeableConcept")
    if isinstance(code, dict):
        if code.get("text"):
            return str(code["text"])
        for coding in code.get("coding", []) or []:
            if coding.get("display"):
                return str(coding["display"])
    return None


def _patient_name(patient: dict) -> str:
    names = patient.get("name", []) or []
    if not names:
        return "Synthetic Patient"
    n = names[0]
    given = " ".join(n.get("given", []) or [])
    family = n.get("family", "")
    return (given + " " + family).strip() or "Synthetic Patient"


def parse_bundle(path: Path) -> tuple[str, dict, dict[str, list[str]]] | None:
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    entries = bundle.get("entry", []) if bundle.get("resourceType") == "Bundle" else []
    resources = [e.get("resource", {}) for e in entries]
    patients = [r for r in resources if r.get("resourceType") == "Patient"]
    if len(patients) != 1:
        return None
    patient = patients[0]
    pid = patient.get("id")
    if not pid:
        return None
    buckets: dict[str, list[str]] = {k: [] for k in RESOURCE_TYPES if k != "Patient"}
    for r in resources:
        rt = r.get("resourceType")
        if rt not in buckets:
            continue
        display = _code_display(r)
        if display and display not in buckets[rt]:
            buckets[rt].append(display)
    return pid, patient, buckets


def normalize_record(alias: str, pid: str, patient: dict, buckets: dict[str, list[str]]) -> PatientRecord:
    canary = stable_canary(alias, pid)
    conditions = buckets.get("Condition", [])[:8]
    meds = buckets.get("MedicationRequest", [])[:8] or buckets.get("Medication", [])[:8]
    allergies = buckets.get("AllergyIntolerance", [])[:6]
    procedures = buckets.get("Procedure", [])[:6]
    observations = buckets.get("Observation", [])[:8]
    encounters = buckets.get("Encounter", [])[:5]
    primary_condition = conditions[0] if conditions else "No documented active condition"
    primary_medication = meds[0] if meds else "No documented medication"
    lines = [
        f"PatientAlias: {alias}",
        f"SyntheaPatientId: {pid}",
        f"SyntheticName: {_patient_name(patient)}",
        f"PrimaryCondition: {primary_condition}",
        f"PrimaryMedication: {primary_medication}",
        f"Conditions: {', '.join(conditions) if conditions else 'None listed'}",
        f"Medications: {', '.join(meds) if meds else 'None listed'}",
        f"Allergies: {', '.join(allergies) if allergies else 'None listed'}",
        f"Procedures: {', '.join(procedures) if procedures else 'None listed'}",
        f"Observations: {', '.join(observations) if observations else 'None listed'}",
        f"Encounters: {', '.join(encounters) if encounters else 'None listed'}",
        f"ExperimentalSecurityCanary: {canary}",
    ]
    return PatientRecord(alias, pid, "\n".join(lines), canary, primary_condition, primary_medication)


def build_dataset(archive: Path, out_dir: Path, n: int = 25) -> list[PatientRecord]:
    extract_dir = out_dir / "raw_fhir"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_dir)
    parsed = []
    for p in sorted(extract_dir.rglob("*.json")):
        item = parse_bundle(p)
        if item:
            parsed.append(item)
    if len(parsed) < n:
        raise RuntimeError(f"Expected at least {n} patient bundles, found {len(parsed)}")
    parsed.sort(key=lambda x: hashlib.sha256(x[0].encode()).hexdigest())
    selected = parsed[:n]
    records = [normalize_record(f"PAT-{i:02d}", pid, pat, buckets) for i, (pid, pat, buckets) in enumerate(selected, 1)]
    save_json(out_dir / "records.json", [r.__dict__ for r in records])
    mapping = {
        "A": [f"PAT-{i:02d}" for i in range(1, 6)],
        "B": [f"PAT-{i:02d}" for i in range(6, 11)],
        "C": [f"PAT-{i:02d}" for i in range(11, 16)],
        "D": [f"PAT-{i:02d}" for i in range(16, 21)],
        "E": [],
    }
    save_json(out_dir / "authorization.json", mapping)
    save_json(out_dir / "protected_pool.json", [f"PAT-{i:02d}" for i in range(21, 26)])
    return records


def load_records(path: Path) -> list[PatientRecord]:
    return [PatientRecord(**x) for x in json.loads(path.read_text(encoding="utf-8"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=Path("data/synthea_sample_data_fhir_latest.zip"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--url", default=SYNTHEA_URL)
    args = ap.parse_args()
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if not args.archive.exists():
        print(f"Downloading official Synthea sample from {args.url}")
        urllib.request.urlretrieve(args.url, args.archive)
    records = build_dataset(args.archive, args.out)
    print(f"Prepared {len(records)} deterministic Synthea patient records in {args.out}")


if __name__ == "__main__":
    main()

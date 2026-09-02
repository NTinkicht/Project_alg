from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _normalize_bool(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns and df[c].dtype != bool:
            df[c] = df[c].astype(str).str.lower().eq("true")
    return df


def _digest(df: pd.DataFrame, columns: list[str]) -> str:
    payload = df.sort_values(["case_id", "architecture"])[columns].to_json(orient="records", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--rerun", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("artifacts/reproducibility/rerun_check.json"))
    args = ap.parse_args()
    ref = pd.read_csv(args.reference)
    rerun = pd.read_csv(args.rerun)
    bool_cols = ["is_authorized", "is_legitimate", "unauthorized_context_exposure", "unauthorized_disclosure", "authorized_task_success", "rejected"]
    ref = _normalize_bool(ref, bool_cols); rerun = _normalize_bool(rerun, bool_cols)
    keys = ["case_id", "architecture"]
    ids = set(rerun.case_id)
    ref = ref[ref.case_id.isin(ids)].copy()
    assert len(ref) == len(rerun), (len(ref), len(rerun))
    assert not ref.duplicated(keys).any() and not rerun.duplicated(keys).any()
    ref = ref.sort_values(keys).reset_index(drop=True)
    rerun = rerun.sort_values(keys).reset_index(drop=True)
    assert ref[keys].equals(rerun[keys])

    exact_cols = [
        "case_id", "architecture", "account", "category", "prompt", "target_alias",
        "is_authorized", "is_legitimate", "retrieved_aliases", "unauthorized_context_exposure",
        "response", "unauthorized_disclosure", "authorized_task_success", "rejected", "fuzzy_decision",
    ]
    mismatches = {}
    for col in exact_cols:
        if col not in ref.columns or col not in rerun.columns:
            continue
        a = ref[col].fillna("<NA>").astype(str)
        b = rerun[col].fillna("<NA>").astype(str)
        n = int((a != b).sum())
        if n:
            mismatches[col] = n
    metrics = {
        "n_rows": int(len(rerun)),
        "n_cases": int(rerun.case_id.nunique()),
        "exact_columns": exact_cols,
        "mismatches": mismatches,
        "reference_digest": _digest(ref, [c for c in exact_cols if c in ref.columns]),
        "rerun_digest": _digest(rerun, [c for c in exact_cols if c in rerun.columns]),
        "exact_match": not mismatches,
        "note": "Latency and floating timing-dependent fields are intentionally excluded; generated response text is compared byte-for-byte after CSV parsing.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if mismatches:
        raise SystemExit(f"Deterministic rerun mismatch: {mismatches}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def _mcnemar(a: pd.Series, b: pd.Series) -> dict:
    a = a.astype(bool).to_numpy(); b = b.astype(bool).to_numpy()
    a_only = int(np.sum(a & ~b)); b_only = int(np.sum(~a & b)); n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(min(a_only, b_only), n, 0.5, alternative="two-sided").pvalue)
    return {"first_only_success": a_only, "second_only_success": b_only, "discordant": n, "p_value": p}


def _parse_aliases(v):
    if isinstance(v, list): return v
    try:
        x = ast.literal_eval(str(v)); return x if isinstance(x, list) else []
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("artifacts/scalability/scale_llm_results.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/scalability"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)
    if df.authorized_task_success.dtype != bool:
        df["authorized_task_success"] = df.authorized_task_success.astype(str).str.lower().eq("true")
    df["aliases"] = df.retrieved_aliases.map(_parse_aliases)

    rows = []
    metrics: dict = {"paired_tests": {}, "retrieval_identity": {}}
    for (scale, mode), d in df.groupby(["scale", "mode"], sort=True):
        rows.append({
            "scale": int(scale),
            "mode": mode,
            "n": len(d),
            "ARSR": float(d.authorized_task_success.mean()),
            "median_retrieval_ms": float(d.retrieval_ms.median()),
            "median_generation_ms": float(d.generation_ms.median()),
        })
        pivot = df[df.scale == scale].pivot(index="case_id", columns="mode", values="authorized_task_success")
        if {"unfiltered", "acl_fixed"}.issubset(pivot.columns):
            metrics["paired_tests"][f"unfiltered_vs_acl_fixed_{int(scale)}"] = _mcnemar(pivot.unfiltered, pivot.acl_fixed)
        if {"unfiltered", "target_fixed"}.issubset(pivot.columns):
            metrics["paired_tests"][f"unfiltered_vs_target_fixed_{int(scale)}"] = _mcnemar(pivot.unfiltered, pivot.target_fixed)

    for mode in ["unfiltered", "acl_fixed", "target_fixed"]:
        d = df[df.mode == mode]
        success = d.pivot(index="case_id", columns="scale", values="authorized_task_success")
        if {1000, 100000}.issubset(success.columns):
            metrics["paired_tests"][f"{mode}_ARSR_1k_vs_100k"] = _mcnemar(success[1000], success[100000])
        identities = d.set_index(["case_id", "scale"])["aliases"]
        changed = []
        for cid in d.case_id.unique():
            try:
                changed.append(identities.loc[(cid, 1000)] != identities.loc[(cid, 100000)])
            except KeyError:
                pass
        metrics["retrieval_identity"][mode] = {
            "n_paired": len(changed),
            "fraction_changed_1k_to_100k": float(np.mean(changed)) if changed else None,
        }

    summary = pd.DataFrame(rows).sort_values(["scale", "mode"])
    summary.to_csv(args.outdir / "scale_llm_summary.csv", index=False)
    (args.outdir / "scale_llm_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

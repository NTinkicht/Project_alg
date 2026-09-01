from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def _bool(s: pd.Series) -> pd.Series:
    return s if s.dtype == bool else s.astype(str).str.lower().eq("true")


def _mcnemar(a: pd.Series, b: pd.Series) -> dict:
    a = _bool(a).to_numpy(); b = _bool(b).to_numpy()
    a_only = int(np.sum(a & ~b)); b_only = int(np.sum(~a & b)); n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(min(a_only, b_only), n, 0.5, alternative="two-sided").pvalue)
    return {"first_only_success": a_only, "second_only_success": b_only, "discordant": n, "p_value": p}


def paired_bootstrap_diff(a: pd.Series, b: pd.Series, seed: int = 4103, draws: int = 20000) -> dict:
    av = _bool(a).astype(float).to_numpy(); bv = _bool(b).astype(float).to_numpy()
    if len(av) != len(bv) or len(av) == 0:
        return {"difference": None, "ci95": None}
    diff = av - bv
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(draws, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"difference": float(diff.mean()), "ci95": [float(lo), float(hi)]}


def scope_size(mode: str, scale: int) -> int:
    if mode == "unfiltered": return scale
    if mode == "acl_fixed": return 200
    if mode == "acl_proportional": return scale // 5
    if mode in {"target_fixed", "target_proportional"}: return 1
    raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-llm", type=Path, required=True)
    ap.add_argument("--proportional-llm", type=Path, required=True)
    ap.add_argument("--retrieval-results", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(args.baseline_llm)
    prop = pd.read_csv(args.proportional_llm)
    combined = pd.concat([baseline, prop], ignore_index=True, sort=False)
    combined["authorized_task_success"] = _bool(combined["authorized_task_success"])
    assert len(combined) == 1080, len(combined)
    assert combined["case_id"].nunique() == 90
    assert set(combined["mode"]) == {"unfiltered", "acl_fixed", "target_fixed", "acl_proportional"}
    assert not combined.duplicated(["case_id", "scale", "mode"]).any()
    combined["authorization_scope_size"] = [scope_size(m, int(s)) for m, s in zip(combined["mode"], combined["scale"])]
    combined["authorization_density_rho"] = combined["authorization_scope_size"] / combined["scale"]
    combined.to_csv(args.outdir / "scale_llm_four_mode_results.csv", index=False)

    summary_rows = []
    metrics: dict[str, dict] = {
        "framework": {
            "N": "global corpus size",
            "A": "caller-visible authorization scope size",
            "rho": "authorization density |A|/N",
            "fixed_scope": "|A| held constant while N grows; rho decreases",
            "proportional_scope": "|A| grows proportionally with N; rho remains constant",
        },
        "paired_tests": {},
        "cross_scale": {},
    }
    for (scale, mode), d in combined.groupby(["scale", "mode"], sort=True):
        summary_rows.append({
            "scale": int(scale), "mode": mode, "n": int(len(d)),
            "authorization_scope_size": scope_size(mode, int(scale)),
            "authorization_density_rho": scope_size(mode, int(scale)) / int(scale),
            "ARSR": float(d["authorized_task_success"].mean()),
            "median_retrieval_ms": float(d["retrieval_ms"].median()),
            "median_generation_ms": float(d["generation_ms"].median()),
        })
        pivot = combined[combined["scale"] == scale].pivot(index="case_id", columns="mode", values="authorized_task_success")
        for first, second in [("unfiltered", "acl_fixed"), ("unfiltered", "acl_proportional"), ("acl_fixed", "acl_proportional"), ("target_fixed", "acl_proportional")]:
            if {first, second}.issubset(pivot.columns):
                key = f"{first}_vs_{second}_{int(scale)}"
                metrics["paired_tests"][key] = {
                    "mcnemar": _mcnemar(pivot[first], pivot[second]),
                    "paired_bootstrap": paired_bootstrap_diff(pivot[first], pivot[second], seed=4103 + int(scale) % 997),
                }

    for mode in ["unfiltered", "acl_fixed", "acl_proportional", "target_fixed"]:
        d = combined[combined["mode"] == mode]
        pivot = d.pivot(index="case_id", columns="scale", values="authorized_task_success")
        if {1000, 100000}.issubset(pivot.columns):
            metrics["cross_scale"][f"{mode}_1k_vs_100k"] = {
                "mcnemar": _mcnemar(pivot[1000], pivot[100000]),
                "paired_bootstrap": paired_bootstrap_diff(pivot[1000], pivot[100000], seed=7103),
            }
    pd.DataFrame(summary_rows).to_csv(args.outdir / "scale_scope_llm_summary.csv", index=False)

    retrieval = pd.read_csv(args.retrieval_results)
    retrieval["authorization_scope_size"] = [scope_size(m, int(s)) for m, s in zip(retrieval["mode"], retrieval["scale"])]
    retrieval["authorization_density_rho"] = retrieval["authorization_scope_size"] / retrieval["scale"]
    retrieval_summary = retrieval.groupby(["scale", "mode"], as_index=False).agg(
        n=("case_id", "size"),
        candidate_count_median=("candidate_count", "median"),
        hit_at_1=("hit_at_1", "mean"),
        hit_at_2=("hit_at_2", "mean"),
        MRR=("mrr", "mean"),
        median_retrieval_ms=("retrieval_ms", "median"),
        p95_retrieval_ms=("retrieval_ms", lambda x: x.quantile(.95)),
    )
    retrieval_summary["authorization_scope_size"] = [scope_size(m, int(s)) for m, s in zip(retrieval_summary["mode"], retrieval_summary["scale"])]
    retrieval_summary["authorization_density_rho"] = retrieval_summary["authorization_scope_size"] / retrieval_summary["scale"]
    retrieval_summary.to_csv(args.outdir / "scale_scope_retrieval_summary.csv", index=False)

    metrics["retrieval_paired_bootstrap"] = {}
    for mode in ["unfiltered", "acl_fixed", "acl_proportional"]:
        d = retrieval[retrieval["mode"] == mode]
        pivot = d.pivot(index="case_id", columns="scale", values="hit_at_1")
        if {1000, 100000}.issubset(pivot.columns):
            metrics["retrieval_paired_bootstrap"][f"{mode}_hit1_1k_vs_100k"] = paired_bootstrap_diff(pivot[1000], pivot[100000], seed=9103)

    (args.outdir / "scale_scope_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

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


def paired_bootstrap_bool(a: pd.Series, b: pd.Series, seed: int = 4103, draws: int = 20000) -> dict:
    av = _bool(a).astype(float).to_numpy(); bv = _bool(b).astype(float).to_numpy()
    return paired_bootstrap_numeric(pd.Series(av), pd.Series(bv), seed=seed, draws=draws)


def paired_bootstrap_numeric(a: pd.Series, b: pd.Series, seed: int = 4103, draws: int = 20000) -> dict:
    pair = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(pair) == 0:
        return {"n": 0, "difference": None, "ci95": None}
    diff = (pair["a"] - pair["b"]).to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(draws, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"n": int(len(diff)), "difference": float(diff.mean()), "ci95": [float(lo), float(hi)]}


def scope_size(mode: str, scale: int) -> int:
    if mode == "unfiltered": return scale
    if mode == "acl_fixed": return 200
    if mode == "acl_proportional": return scale // 5
    if mode in {"target_fixed", "target_proportional"}: return 1
    raise ValueError(mode)


def _growth_block(summary: pd.DataFrame, mode: str) -> dict:
    d = summary[summary["mode"] == mode].set_index("scale")
    if not {1000, 100000}.issubset(d.index):
        return {}
    a0 = float(d.loc[1000, "authorization_scope_size"]); a1 = float(d.loc[100000, "authorization_scope_size"])
    l0 = float(d.loc[1000, "median_retrieval_ms"]); l1 = float(d.loc[100000, "median_retrieval_ms"])
    return {
        "authorization_scope_growth_factor": (a1 / a0 if a0 else None),
        "authorization_density_1k": float(d.loc[1000, "authorization_density_rho"]),
        "authorization_density_100k": float(d.loc[100000, "authorization_density_rho"]),
        "median_retrieval_latency_growth_factor": (l1 / l0 if l0 else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-llm", type=Path, required=True)
    ap.add_argument("--proportional-llm", type=Path, required=True)
    ap.add_argument("--retrieval-results", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(args.baseline_llm); prop = pd.read_csv(args.proportional_llm)
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
    metrics: dict[str, object] = {
        "framework": {
            "N": "global corpus size",
            "A": "caller-visible authorization scope size",
            "rho": "authorization density |A|/N",
            "fixed_scope": "|A| held constant while N grows; rho decreases",
            "proportional_scope": "|A| grows proportionally with N; rho remains constant",
            "scope_scaling_hypothesis": "after pre-retrieval filtering, retrieval cost and distractor pressure should track the caller-visible candidate set |A| rather than global N alone",
        },
        "paired_tests": {}, "cross_scale": {}, "retrieval_cross_scale": {}, "scope_growth": {},
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
                    "paired_bootstrap": paired_bootstrap_bool(pivot[first], pivot[second], seed=4103 + int(scale) % 997),
                }
    llm_summary = pd.DataFrame(summary_rows)
    llm_summary.to_csv(args.outdir / "scale_scope_llm_summary.csv", index=False)

    for mode in ["unfiltered", "acl_fixed", "acl_proportional", "target_fixed"]:
        d = combined[combined["mode"] == mode]
        pivot = d.pivot(index="case_id", columns="scale", values="authorized_task_success")
        if {1000, 100000}.issubset(pivot.columns):
            metrics["cross_scale"][f"{mode}_1k_vs_100k"] = {
                "mcnemar": _mcnemar(pivot[1000], pivot[100000]),
                "paired_bootstrap": paired_bootstrap_bool(pivot[1000], pivot[100000], seed=7103),
            }

    retrieval = pd.read_csv(args.retrieval_results)
    retrieval["authorization_scope_size"] = [scope_size(m, int(s)) for m, s in zip(retrieval["mode"], retrieval["scale"])]
    retrieval["authorization_density_rho"] = retrieval["authorization_scope_size"] / retrieval["scale"]
    retrieval_summary = retrieval.groupby(["scale", "mode"], as_index=False).agg(
        n=("case_id", "size"), candidate_count_median=("candidate_count", "median"),
        hit_at_1=("hit_at_1", "mean"), hit_at_2=("hit_at_2", "mean"), MRR=("mrr", "mean"),
        median_retrieval_ms=("retrieval_ms", "median"), p95_retrieval_ms=("retrieval_ms", lambda x: x.quantile(.95)),
        median_best_distractor_score=("best_distractor_score", "median"), median_target_margin=("target_margin", "median"),
    )
    retrieval_summary["authorization_scope_size"] = [scope_size(m, int(s)) for m, s in zip(retrieval_summary["mode"], retrieval_summary["scale"])]
    retrieval_summary["authorization_density_rho"] = retrieval_summary["authorization_scope_size"] / retrieval_summary["scale"]
    retrieval_summary.to_csv(args.outdir / "scale_scope_retrieval_summary.csv", index=False)

    for mode in ["unfiltered", "acl_fixed", "acl_proportional"]:
        d = retrieval[retrieval["mode"] == mode]
        block: dict[str, object] = {}
        for col in ["hit_at_1", "best_distractor_score", "target_margin", "retrieval_ms"]:
            pivot = d.pivot(index="case_id", columns="scale", values=col)
            if {1000, 100000}.issubset(pivot.columns):
                block[f"{col}_100k_minus_1k_bootstrap"] = paired_bootstrap_numeric(pivot[100000], pivot[1000], seed=9103)
        metrics["retrieval_cross_scale"][mode] = block
        metrics["scope_growth"][mode] = _growth_block(retrieval_summary, mode)

    # At 1K fixed and proportional scopes coincide (both 200/account). Divergence
    # at 10K/100K therefore cleanly isolates expansion of the authorized candidate
    # set while the global corpus is held equal within each paired comparison.
    metrics["scope_decomposition_note"] = (
        "At N=1,000, acl_fixed and acl_proportional are identical. At larger N, paired fixed-vs-proportional comparisons isolate authorization-scope expansion at the same global corpus size."
    )
    (args.outdir / "scale_scope_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(llm_summary.to_string(index=False)); print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

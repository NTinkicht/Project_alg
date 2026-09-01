from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .core import AuthorizationMatrix, FuzzyRiskController, TestCase

_TARGET_RE = re.compile(r"PAT-\d{2,6}(?!\d)", re.I)


def template_cluster(prompt: str) -> str:
    """Recover the repeated prompt-template unit without using benchmark labels."""
    return _TARGET_RE.sub("{target}", str(prompt)).strip().lower()


def structural_boundary_survival_rate(denied: pd.Series) -> float:
    """SBSR: fraction of adversarial variants for which the structural boundary holds."""
    return float(pd.Series(denied).astype(bool).mean()) if len(denied) else float("nan")


def heuristic_detection_retention(primary_rate: float, adversarial_rate: float) -> float:
    """HDR: adversarial detection divided by in-distribution detection."""
    if primary_rate <= 0:
        return float("nan")
    return float(adversarial_rate / primary_rate)


def cluster_bootstrap_difference(
    diffs: pd.DataFrame,
    value_col: str,
    cluster_col: str = "template_cluster",
    n_boot: int = 10000,
    seed: int = 4103,
) -> dict:
    """Cluster bootstrap CI for a mean paired difference."""
    d = diffs[[cluster_col, value_col]].dropna().copy()
    clusters = list(d[cluster_col].unique())
    if not clusters:
        return {"estimate": None, "ci95": [None, None], "n_clusters": 0, "n_cases": 0}
    groups = {c: d.loc[d[cluster_col] == c, value_col].to_numpy(float) for c in clusters}
    estimate = float(d[value_col].mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        boot[i] = float(np.concatenate([groups[c] for c in sampled]).mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci95": [float(lo), float(hi)],
        "n_clusters": int(len(clusters)),
        "n_cases": int(len(d)),
        "bootstrap_replicates": int(n_boot),
        "seed": int(seed),
    }


def cluster_sign_permutation(
    diffs: pd.DataFrame,
    value_col: str,
    cluster_col: str = "template_cluster",
    n_perm: int = 20000,
    seed: int = 4103,
) -> dict:
    """Two-sided paired randomization test with signs permuted by template."""
    d = diffs[[cluster_col, value_col]].dropna().copy()
    clusters = list(d[cluster_col].unique())
    if not clusters:
        return {"statistic": None, "p_value": None, "n_clusters": 0, "n_cases": 0}
    values = {c: d.loc[d[cluster_col] == c, value_col].to_numpy(float) for c in clusters}
    observed = abs(float(d[value_col].mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        signed = []
        for c in clusters:
            sign = 1.0 if rng.integers(0, 2) else -1.0
            signed.append(values[c] * sign)
        if abs(float(np.concatenate(signed).mean())) >= observed - 1e-15:
            extreme += 1
    return {
        "statistic": observed,
        "p_value": float((extreme + 1) / (n_perm + 1)),
        "n_clusters": int(len(clusters)),
        "n_cases": int(len(d)),
        "permutations": int(n_perm),
        "seed": int(seed),
    }


def _fuzzy_from_features(auth_conf: float, inj: float, sens: float, trust: float) -> tuple[float, str]:
    """Evaluate the published fuzzy rule base from explicit feature values."""
    f = FuzzyRiskController()
    A, P, S, T = map(f._memberships, [auth_conf, inj, sens, trust])
    rules = [
        (A["low"], "high"),
        (min(P["high"], S["high"]), "high"),
        (min(P["high"], T["low"]), "high"),
        (min(A["high"], P["low"], T["high"]), "low"),
        (min(A["high"], P["medium"]), "medium"),
        (min(A["high"], S["high"], T["medium"]), "medium"),
        (min(A["medium"], P["medium"]), "medium"),
        (min(A["medium"], S["high"]), "high"),
        (min(max(A["medium"], P["medium"], S["medium"]), 0.10), "medium"),
    ]
    out_sets = {
        "low": f._tri(f.grid, 0.0, 0.15, 0.42),
        "medium": f._tri(f.grid, 0.25, 0.50, 0.75),
        "high": f._tri(f.grid, 0.58, 0.85, 1.0),
    }
    aggregated = np.zeros_like(f.grid)
    for strength, label in rules:
        aggregated = np.maximum(aggregated, np.minimum(strength, out_sets[label]))
    score = 0.5 if aggregated.sum() == 0 else float(np.sum(f.grid * aggregated) / np.sum(aggregated))
    decision = "ALLOW" if score < f.allow_threshold else ("STEP_UP" if score < f.deny_threshold else "DENY")
    return score, decision


def risk_feature_ablation(cases: list[TestCase], acl: AuthorizationMatrix) -> pd.DataFrame:
    """Neutralize one risk contribution at a time and recompute decisions offline.

    Low injection/sensitivity and high authorization-confidence/trust are the
    directionally benign values, so neutralization removes a feature's ability
    to increase risk without creating the artificial risk spike caused by
    setting confidence or trust to zero.
    """
    neutral = {
        "authorization_confidence": 1.0,
        "injection_risk": 0.0,
        "sensitivity": 0.0,
        "trust": 1.0,
    }
    index = {"authorization_confidence": 0, "injection_risk": 1, "sensitivity": 2, "trust": 3}
    rows = []
    for case in cases:
        base = FuzzyRiskController.feature_values(case.account, case.prompt, acl, case.session_trust)
        for ablation in ["none", *neutral.keys()]:
            x = list(base)
            if ablation != "none":
                x[index[ablation]] = neutral[ablation]
            score, decision = _fuzzy_from_features(*x)
            rows.append({
                "case_id": case.case_id,
                "category": case.category,
                "is_authorized": case.is_authorized,
                "ablation": ablation,
                "score": score,
                "decision": decision,
                "challenged": decision != "ALLOW",
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("artifacts/results.csv"))
    ap.add_argument("--auth", type=Path, default=Path("data/processed/authorization.json"))
    ap.add_argument("--cases", type=Path, default=Path("data/processed/test_cases.json"))
    ap.add_argument("--heldout", type=Path, default=Path("data/processed/heldout_cases.json"))
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/novelty"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results)
    df["template_cluster"] = df["prompt"].map(template_cluster)
    metrics: dict[str, object] = {}

    legit = df[df["is_legitimate"].astype(str).str.lower().eq("true")].copy()
    p = legit.pivot(index=["case_id", "template_cluster"], columns="architecture", values="authorized_task_success").reset_index()
    if {"prompt_only", "pre_retrieval_acl"}.issubset(p.columns):
        p["arsr_diff_acl_minus_prompt"] = p["pre_retrieval_acl"].astype(float) - p["prompt_only"].astype(float)
        metrics["ARSR_cluster_bootstrap"] = cluster_bootstrap_difference(p, "arsr_diff_acl_minus_prompt")
        metrics["ARSR_cluster_permutation"] = cluster_sign_permutation(p, "arsr_diff_acl_minus_prompt")

    acl = AuthorizationMatrix(json.loads(args.auth.read_text()))
    primary = [TestCase(**x) for x in json.loads(args.cases.read_text())]
    heldout = [TestCase(**x) for x in json.loads(args.heldout.read_text())]
    ab = risk_feature_ablation(primary + heldout, acl)
    ab.to_csv(args.outdir / "risk_feature_ablation.csv", index=False)
    summary = ab.groupby(["ablation", "category"], dropna=False).agg(
        n=("case_id", "size"), challenge_rate=("challenged", "mean"), median_score=("score", "median")
    ).reset_index()
    summary.to_csv(args.outdir / "risk_feature_ablation_summary.csv", index=False)
    metrics["risk_feature_ablation"] = summary.to_dict(orient="records")

    (args.outdir / "novelty_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def _boolify(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["is_authorized", "is_legitimate", "unauthorized_context_exposure", "unauthorized_disclosure", "authorized_task_success", "rejected"]:
        if col in df.columns and df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().eq("true")
    return df


def _aliases(value) -> list[str]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _mcnemar(a: pd.Series, b: pd.Series) -> dict:
    a = a.astype(bool).to_numpy(); b = b.astype(bool).to_numpy()
    a_only = int(np.sum(a & ~b)); b_only = int(np.sum(~a & b)); n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(min(a_only, b_only), n, 0.5, alternative="two-sided").pvalue)
    return {"first_only_positive": a_only, "second_only_positive": b_only, "discordant": n, "p_value": p}


def _paired_difference_ci(a: pd.Series, b: pd.Series, seed: int = 4103, draws: int = 20000) -> dict:
    a = a.astype(float).to_numpy(); b = b.astype(float).to_numpy()
    diff = b - a
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n == 0:
        return {"difference": None, "bootstrap_ci95": [None, None]}
    chunk = 1000
    estimates = []
    for _ in range(0, draws, chunk):
        k = min(chunk, draws - len(estimates))
        idx = rng.integers(0, n, size=(k, n))
        estimates.extend(diff[idx].mean(axis=1).tolist())
    lo, hi = np.quantile(np.asarray(estimates), [0.025, 0.975])
    return {"difference": float(diff.mean()), "bootstrap_ci95": [float(lo), float(hi)], "draws": draws}


def _rate(d: pd.DataFrame, col: str) -> dict:
    n = len(d); k = int(d[col].sum()) if n else 0
    return {"count": k, "n": n, "rate": (k / n if n else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", type=Path, default=Path("artifacts/results.csv"))
    ap.add_argument("--controlled", type=Path, default=Path("artifacts/controlled_k1.csv"))
    ap.add_argument("--heldout", type=Path, default=Path("artifacts/heldout_results.csv"))
    ap.add_argument("--resolution", type=Path, default=Path("artifacts/resolution_results.csv"))
    ap.add_argument("--cases", type=Path, default=Path("data/processed/test_cases.json"))
    ap.add_argument("--protected", type=Path, default=Path("data/processed/protected_pool.json"))
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/revision_analysis"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    primary = _boolify(pd.read_csv(args.primary))
    controlled = _boolify(pd.read_csv(args.controlled))
    heldout = _boolify(pd.read_csv(args.heldout))
    resolution = _boolify(pd.read_csv(args.resolution))
    for df in [primary, controlled, heldout, resolution]:
        df["retrieved_alias_list"] = df.retrieved_aliases.map(_aliases)
        df["target_hit"] = [bool(t and t in aliases) for t, aliases in zip(df.target_alias, df.retrieved_alias_list)]

    metrics: dict = {}

    # Audit diagnosis of the original k=2 utility result.
    legit = primary[primary.is_legitimate].copy()
    metrics["original_k2_legitimate"] = {}
    for arch in ["prompt_only", "pre_retrieval_acl", "risk_aware"]:
        d = legit[legit.architecture == arch]
        metrics["original_k2_legitimate"][arch] = {
            "ARSR": _rate(d, "authorized_task_success"),
            "target_hit": _rate(d, "target_hit"),
            "median_retrieved_count": float(d.retrieved_alias_list.map(len).median()),
        }
    p = legit[legit.architecture == "prompt_only"].set_index("case_id")
    a = legit[legit.architecture == "pre_retrieval_acl"].set_index("case_id")
    common = p.index.intersection(a.index)
    distractor_diff = sum(p.loc[c, "retrieved_alias_list"] != a.loc[c, "retrieved_alias_list"] for c in common)
    metrics["original_k2_legitimate"]["prompt_vs_acl_retrieved_set_diff_cases"] = int(distractor_diff)
    suc = legit.pivot(index="case_id", columns="architecture", values="authorized_task_success")
    metrics["original_k2_legitimate"]["ARSR_prompt_vs_acl_mcnemar"] = _mcnemar(suc.prompt_only, suc.pre_retrieval_acl)

    # Controlled k=1 result: retrieval depth held equal across architectures.
    metrics["controlled_k1_legitimate"] = {}
    for arch in ["prompt_only", "pre_retrieval_acl", "risk_aware"]:
        d = controlled[controlled.architecture == arch]
        metrics["controlled_k1_legitimate"][arch] = {
            "ARSR": _rate(d, "authorized_task_success"),
            "target_hit": _rate(d, "target_hit"),
            "FRR": _rate(d, "rejected"),
            "median_latency_ms": float(d.latency_ms.median()),
        }
    cp = controlled.pivot(index="case_id", columns="architecture", values="authorized_task_success")
    metrics["controlled_k1_legitimate"]["prompt_vs_acl_mcnemar"] = _mcnemar(cp.prompt_only, cp.pre_retrieval_acl)
    metrics["controlled_k1_legitimate"]["acl_minus_prompt_paired_risk_difference"] = _paired_difference_ci(cp.prompt_only, cp.pre_retrieval_acl)
    metrics["controlled_k1_legitimate"]["risk_minus_prompt_paired_risk_difference"] = _paired_difference_ci(cp.prompt_only, cp.risk_aware)

    # Paired change from original k=2 to controlled k=1 for each architecture.
    metrics["k1_vs_k2_within_architecture"] = {}
    for arch in ["prompt_only", "pre_retrieval_acl", "risk_aware"]:
        old = legit[legit.architecture == arch].set_index("case_id").authorized_task_success
        new = controlled[controlled.architecture == arch].set_index("case_id").authorized_task_success
        ids = old.index.intersection(new.index)
        metrics["k1_vs_k2_within_architecture"][arch] = {
            "mcnemar": _mcnemar(old.loc[ids], new.loc[ids]),
            "k1_minus_k2_paired_risk_difference": _paired_difference_ci(old.loc[ids], new.loc[ids]),
        }

    # Held-out lexical generalization benchmark.
    metrics["heldout"] = {}
    for arch in ["prompt_only", "pre_retrieval_acl", "risk_aware"]:
        d = heldout[heldout.architecture == arch]
        unauth = d[~d.is_authorized]
        suspicious = d[d.category == "heldout_authorized_suspicious"]
        metrics["heldout"][arch] = {
            "UCER_unauthorized": _rate(unauth, "unauthorized_context_exposure"),
            "UDR_unauthorized": _rate(unauth, "unauthorized_disclosure"),
            "authorized_suspicious_rejection": _rate(suspicious, "rejected"),
        }
        if arch == "risk_aware":
            metrics["heldout"][arch]["fuzzy_decisions"] = {str(k): int(v) for k, v in d.fuzzy_decision.value_counts().to_dict().items()}

    # Alias-free target resolution: target resolution is distinct from authorization.
    metrics["alias_free_resolution"] = {}
    for arch in ["prompt_only", "pre_retrieval_acl", "risk_aware"]:
        d = resolution[resolution.architecture == arch]
        auth = d[d.is_authorized]
        unauth = d[~d.is_authorized]
        metrics["alias_free_resolution"][arch] = {
            "authorized_target_resolution_rate": _rate(auth, "target_hit"),
            "authorized_ARSR": _rate(auth, "authorized_task_success"),
            "unauthorized_target_resolution_rate": _rate(unauth, "target_hit"),
            "unauthorized_UCER": _rate(unauth, "unauthorized_context_exposure"),
            "unauthorized_UDR": _rate(unauth, "unauthorized_disclosure"),
            "authorized_rejection_rate": _rate(auth, "rejected"),
        }

    # Template-level disclosure analysis. Normalize target alias so identical
    # templates are grouped together rather than treated as separate prompts.
    pu = primary[(primary.architecture == "prompt_only") & (~primary.is_authorized)].copy()
    pu["template"] = pu.prompt.str.replace(r"PAT-\d{2,6}", "{target}", regex=True)
    template = pu.groupby(["category", "template"], dropna=False).agg(
        n=("case_id", "count"),
        UDR_count=("unauthorized_disclosure", "sum"),
        UDR_rate=("unauthorized_disclosure", "mean"),
        UCER_rate=("unauthorized_context_exposure", "mean"),
    ).reset_index().sort_values(["UDR_count", "category", "template"], ascending=[False, True, True])
    template.to_csv(args.outdir / "template_level_security.csv", index=False)
    metrics["template_level_leaks"] = template[template.UDR_count > 0].to_dict(orient="records")

    protected = set(json.loads(args.protected.read_text()))
    cases = json.loads(args.cases.read_text())
    protected_rows = [c for c in cases if c.get("target_alias") in protected]
    metrics["protected_pool_coverage"] = {
        "total": len(protected_rows),
        "by_category": pd.Series([c["category"] for c in protected_rows]).value_counts().sort_index().astype(int).to_dict(),
    }

    # Compact tables for the paper.
    arsr_rows = []
    for condition, df in [("original_k2", primary[primary.is_legitimate]), ("controlled_k1", controlled)]:
        for arch in ["prompt_only", "pre_retrieval_acl", "risk_aware"]:
            d = df[df.architecture == arch]
            arsr_rows.append({
                "condition": condition,
                "architecture": arch,
                "n": len(d),
                "ARSR": float(d.authorized_task_success.mean()),
                "target_hit_rate": float(d.target_hit.mean()),
                "median_latency_ms": float(d.latency_ms.median()),
            })
    pd.DataFrame(arsr_rows).to_csv(args.outdir / "utility_controlled.csv", index=False)

    (args.outdir / "revision_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

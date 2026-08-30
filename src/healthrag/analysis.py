from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
import matplotlib.pyplot as plt


def proportion_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    r = binomtest(k, n).proportion_ci(confidence_level=1-alpha, method="wilson")
    return float(r.low), float(r.high)


def mcnemar_exact(a: pd.Series, b: pd.Series) -> dict:
    a = a.astype(bool).to_numpy()
    b = b.astype(bool).to_numpy()
    b01 = int(np.sum((~a) & b))
    b10 = int(np.sum(a & (~b)))
    n = b01 + b10
    p = 1.0 if n == 0 else float(binomtest(min(b01, b10), n, 0.5, alternative="two-sided").pvalue)
    return {"baseline_only_positive": b10, "secured_only_positive": b01, "discordant": n, "p_value": p}


def metric_block(df: pd.DataFrame, arch: str) -> dict:
    d = df[df.architecture == arch]
    unauth = d[~d.is_authorized]
    legit = d[d.is_legitimate]
    out = {"n_total": len(d), "n_unauthorized": len(unauth), "n_legitimate": len(legit)}
    for name, col, subset in [
        ("UCER", "unauthorized_context_exposure", unauth),
        ("UDR", "unauthorized_disclosure", unauth),
        ("ARSR", "authorized_task_success", legit),
        ("FRR", "rejected", legit),
    ]:
        k, n = int(subset[col].sum()), len(subset)
        lo, hi = proportion_ci(k, n)
        out[name] = {"count": k, "n": n, "rate": k/n if n else None, "ci95": [lo, hi]}
    out["latency_ms"] = {
        "median": float(d.latency_ms.median()),
        "iqr": [float(d.latency_ms.quantile(.25)), float(d.latency_ms.quantile(.75))],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("artifacts/results.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/analysis"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)
    for col in ["is_authorized", "is_legitimate", "unauthorized_context_exposure", "unauthorized_disclosure", "authorized_task_success", "rejected"]:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().eq("true")

    metrics = {arch: metric_block(df, arch) for arch in df.architecture.unique()}
    unauth = df[~df.is_authorized]
    pivot_udr = unauth.pivot(index="case_id", columns="architecture", values="unauthorized_disclosure")
    pivot_ucer = unauth.pivot(index="case_id", columns="architecture", values="unauthorized_context_exposure")
    metrics["paired_tests"] = {
        "UDR_prompt_vs_acl": mcnemar_exact(pivot_udr["prompt_only"], pivot_udr["pre_retrieval_acl"]),
        "UCER_prompt_vs_acl": mcnemar_exact(pivot_ucer["prompt_only"], pivot_ucer["pre_retrieval_acl"]),
    }
    # Paired latency only on legitimate requests where both produce a response.
    legit = df[df.is_legitimate].pivot(index="case_id", columns="architecture", values="latency_ms")
    if {"prompt_only", "pre_retrieval_acl"}.issubset(legit.columns):
        try:
            w = wilcoxon(legit["prompt_only"], legit["pre_retrieval_acl"], zero_method="wilcox")
            metrics["paired_tests"]["latency_prompt_vs_acl"] = {"statistic": float(w.statistic), "p_value": float(w.pvalue)}
        except Exception as e:
            metrics["paired_tests"]["latency_prompt_vs_acl"] = {"error": str(e)}

    risk = df[df.architecture == "risk_aware"].copy()
    metrics["risk_aware"]["fuzzy_decisions"] = risk.fuzzy_decision.value_counts(dropna=False).to_dict()
    suspicious = risk[risk.category == "authorized_suspicious"]
    metrics["risk_aware"]["authorized_suspicious_challenge_rate"] = float((suspicious.fuzzy_decision != "ALLOW").mean()) if len(suspicious) else None

    (args.outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Architecture summary table.
    rows = []
    for arch in ["prompt_only", "pre_retrieval_acl", "risk_aware"]:
        m = metrics[arch]
        rows.append({
            "architecture": arch,
            "UCER": m["UCER"]["rate"],
            "UDR": m["UDR"]["rate"],
            "ARSR": m["ARSR"]["rate"],
            "FRR": m["FRR"]["rate"],
            "median_latency_ms": m["latency_ms"]["median"],
        })
    pd.DataFrame(rows).to_csv(args.outdir / "summary.csv", index=False)

    # Plot 1: security rates.
    plot_df = pd.DataFrame(rows).set_index("architecture")[["UCER", "UDR"]]
    ax = plot_df.plot(kind="bar", figsize=(7.5, 4.4), rot=0)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Unauthorized context exposure and disclosure")
    plt.tight_layout()
    plt.savefig(args.outdir / "security_rates.png", dpi=220)
    plt.close()

    # Plot 2: prompt-only disclosure by attack category.
    p = df[(df.architecture == "prompt_only") & (~df.is_authorized)]
    cat = p.groupby("category").agg(UDR=("unauthorized_disclosure", "mean"), UCER=("unauthorized_context_exposure", "mean")).sort_index()
    ax = cat.plot(kind="bar", figsize=(8.2, 4.6), rot=25)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Prompt-only baseline by attack category")
    plt.tight_layout()
    plt.savefig(args.outdir / "attack_categories.png", dpi=220)
    plt.close()
    cat.to_csv(args.outdir / "attack_categories.csv")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

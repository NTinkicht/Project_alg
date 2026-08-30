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


def latency_block(d: pd.DataFrame) -> dict:
    return {
        "median": float(d.latency_ms.median()),
        "iqr": [float(d.latency_ms.quantile(.25)), float(d.latency_ms.quantile(.75))],
    }


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
    out["latency_ms_all_requests"] = latency_block(d)
    out["latency_ms_legitimate"] = latency_block(legit)
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

    legit_success = df[df.is_legitimate].pivot(index="case_id", columns="architecture", values="authorized_task_success")
    if {"prompt_only", "pre_retrieval_acl"}.issubset(legit_success.columns):
        metrics["paired_tests"]["ARSR_prompt_vs_acl"] = mcnemar_exact(
            legit_success["prompt_only"], legit_success["pre_retrieval_acl"]
        )

    # Paired latency only on legitimate requests where both architectures invoke the LLM.
    legit_latency = df[df.is_legitimate].pivot(index="case_id", columns="architecture", values="latency_ms")
    if {"prompt_only", "pre_retrieval_acl"}.issubset(legit_latency.columns):
        try:
            w = wilcoxon(legit_latency["prompt_only"], legit_latency["pre_retrieval_acl"], zero_method="wilcox")
            metrics["paired_tests"]["legitimate_latency_prompt_vs_acl"] = {
                "statistic": float(w.statistic), "p_value": float(w.pvalue)
            }
        except Exception as e:
            metrics["paired_tests"]["legitimate_latency_prompt_vs_acl"] = {"error": str(e)}

    risk = df[df.architecture == "risk_aware"].copy()
    metrics["risk_aware"]["fuzzy_decisions"] = risk.fuzzy_decision.value_counts(dropna=False).to_dict()
    suspicious = risk[risk.category == "authorized_suspicious"]
    metrics["risk_aware"]["authorized_suspicious_challenge_rate"] = float((suspicious.fuzzy_decision != "ALLOW").mean()) if len(suspicious) else None
    metrics["risk_aware"]["risk_by_group"] = {
        "authorized_normal": {
            "median": float(risk[risk.category == "authorized_normal"].fuzzy_score.median()),
        },
        "authorized_suspicious": {
            "median": float(suspicious.fuzzy_score.median()),
            "min": float(suspicious.fuzzy_score.min()),
            "max": float(suspicious.fuzzy_score.max()),
        },
        "unauthorized": {
            "median": float(risk[~risk.is_authorized].fuzzy_score.median()),
            "min": float(risk[~risk.is_authorized].fuzzy_score.min()),
            "max": float(risk[~risk.is_authorized].fuzzy_score.max()),
        },
    }

    (args.outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Architecture summary table. All-request latency is retained for auditability,
    # but legitimate-path latency is the fair cross-architecture performance measure.
    rows = []
    for arch in ["prompt_only", "pre_retrieval_acl", "risk_aware"]:
        m = metrics[arch]
        rows.append({
            "architecture": arch,
            "UCER": m["UCER"]["rate"],
            "UDR_canary": m["UDR"]["rate"],
            "ARSR": m["ARSR"]["rate"],
            "FRR": m["FRR"]["rate"],
            "median_latency_all_ms": m["latency_ms_all_requests"]["median"],
            "median_latency_legitimate_ms": m["latency_ms_legitimate"]["median"],
        })
    pd.DataFrame(rows).to_csv(args.outdir / "summary.csv", index=False)

    # Plot 1: security rates.
    plot_df = pd.DataFrame(rows).set_index("architecture")[["UCER", "UDR_canary"]]
    ax = plot_df.plot(kind="bar", figsize=(7.5, 4.4), rot=0)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Unauthorized context exposure and canary-confirmed disclosure")
    plt.tight_layout()
    plt.savefig(args.outdir / "security_rates.png", dpi=220)
    plt.close()

    # Plot 2: prompt-only disclosure by attack category.
    p = df[(df.architecture == "prompt_only") & (~df.is_authorized)]
    cat = p.groupby("category").agg(
        n=("case_id", "count"),
        UDR_canary=("unauthorized_disclosure", "mean"),
        UCER=("unauthorized_context_exposure", "mean"),
    ).sort_index()
    ax = cat[["UDR_canary", "UCER"]].plot(kind="bar", figsize=(8.2, 4.6), rot=25)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Prompt-only baseline by attack category")
    plt.tight_layout()
    plt.savefig(args.outdir / "attack_categories.png", dpi=220)
    plt.close()
    cat.to_csv(args.outdir / "attack_categories.csv")

    # Exact, canary-confirmed prompt-only leak cases for audit/reproduction.
    leak_cols = ["case_id", "category", "account", "target_alias", "prompt", "retrieved_aliases", "response"]
    p[p.unauthorized_disclosure][leak_cols].to_csv(args.outdir / "canary_leak_cases.csv", index=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

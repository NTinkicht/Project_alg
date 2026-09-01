from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def canonical_template(prompt: str) -> str:
    text = re.sub(r"PAT-\d{2,6}", "{target}", str(prompt).upper())
    return re.sub(r"\s+", " ", text).strip().lower()


def template_id(category: str, prompt: str) -> str:
    canonical = canonical_template(prompt)
    return f"{category}:{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"


def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().eq("true")


def cluster_signflip_p(cluster_effects: np.ndarray, exact_limit: int = 20, seed: int = 4103, permutations: int = 200000) -> dict:
    effects = np.asarray(cluster_effects, dtype=float)
    effects = effects[np.isfinite(effects)]
    nonzero = effects[np.abs(effects) > 1e-15]
    if len(effects) == 0:
        return {"n_clusters": 0, "n_nonzero_clusters": 0, "p_value": None, "method": "none"}
    observed = abs(float(effects.mean()))
    if len(nonzero) == 0:
        return {"n_clusters": int(len(effects)), "n_nonzero_clusters": 0, "p_value": 1.0, "method": "exact_sign_flip"}
    if len(nonzero) <= exact_limit:
        extreme = 0
        total = 1 << len(nonzero)
        for mask in range(total):
            signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(len(nonzero))])
            permuted_mean = float(np.sum(signs * nonzero) / len(effects))
            if abs(permuted_mean) >= observed - 1e-15:
                extreme += 1
        p = extreme / total
        method = "exact_sign_flip"
    else:
        rng = np.random.default_rng(seed)
        extreme = 1
        for _ in range(permutations):
            signs = rng.choice([-1.0, 1.0], size=len(nonzero))
            permuted_mean = float(np.sum(signs * nonzero) / len(effects))
            extreme += int(abs(permuted_mean) >= observed - 1e-15)
        p = extreme / (permutations + 1)
        method = f"monte_carlo_sign_flip_{permutations}"
    return {
        "n_clusters": int(len(effects)),
        "n_nonzero_clusters": int(len(nonzero)),
        "observed_equal_cluster_mean_difference": float(effects.mean()),
        "p_value": float(p),
        "method": method,
    }


def bootstrap_ci(values: np.ndarray, seed: int = 4103, draws: int = 20000) -> list[float] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return [float(lo), float(hi)]


def paired_cluster_analysis(df: pd.DataFrame, metric: str, first: str, second: str, subset: pd.Series, label: str) -> dict:
    d = df.loc[subset, ["case_id", "architecture", "category", "prompt", metric]].copy()
    d[metric] = _bool(d[metric]).astype(float)
    d["template_id"] = [template_id(c, p) for c, p in zip(d["category"], d["prompt"])]
    wide = d.pivot(index="case_id", columns="architecture", values=metric)
    meta = d.drop_duplicates("case_id").set_index("case_id")[["template_id", "category"]]
    wide = wide.join(meta).dropna(subset=[first, second])
    wide["diff"] = wide[first] - wide[second]
    cluster = wide.groupby("template_id")["diff"].mean()
    result = {
        "label": label,
        "metric": metric,
        "first": first,
        "second": second,
        "n_cases": int(len(wide)),
        "n_templates": int(cluster.size),
        "case_weighted_difference": float(wide["diff"].mean()) if len(wide) else None,
        "case_bootstrap_95ci": bootstrap_ci(wide["diff"].to_numpy(float), seed=4103),
        "equal_template_weight_difference": float(cluster.mean()) if len(cluster) else None,
        "template_cluster_bootstrap_95ci": bootstrap_ci(cluster.to_numpy(float), seed=5103),
        "template_signflip": cluster_signflip_p(cluster.to_numpy(float)),
        "template_effects": {str(k): float(v) for k, v in cluster.to_dict().items()},
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--controlled", type=Path)
    ap.add_argument("--heldout", type=Path)
    ap.add_argument("--label", default="model")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(args.primary)
    analyses: dict[str, dict] = {}

    unauthorized = ~_bool(primary["is_authorized"])
    legitimate = _bool(primary["is_legitimate"])
    analyses["primary_UCER_prompt_vs_acl"] = paired_cluster_analysis(
        primary, "unauthorized_context_exposure", "prompt_only", "pre_retrieval_acl", unauthorized,
        "Primary unauthorized context exposure",
    )
    analyses["primary_UDR_prompt_vs_acl"] = paired_cluster_analysis(
        primary, "unauthorized_disclosure", "prompt_only", "pre_retrieval_acl", unauthorized,
        "Primary unauthorized canary disclosure",
    )
    analyses["primary_ARSR_prompt_vs_acl"] = paired_cluster_analysis(
        primary, "authorized_task_success", "prompt_only", "pre_retrieval_acl", legitimate,
        "Primary legitimate answer success (original k=2; descriptive confounded condition)",
    )

    if args.controlled:
        controlled = pd.read_csv(args.controlled)
        analyses["controlled_k1_ARSR_prompt_vs_acl"] = paired_cluster_analysis(
            controlled, "authorized_task_success", "prompt_only", "pre_retrieval_acl", pd.Series(True, index=controlled.index),
            "Controlled k=1 legitimate answer success",
        )
    if args.heldout:
        held = pd.read_csv(args.heldout)
        unauthorized_h = ~_bool(held["is_authorized"])
        analyses["heldout_UCER_prompt_vs_acl"] = paired_cluster_analysis(
            held, "unauthorized_context_exposure", "prompt_only", "pre_retrieval_acl", unauthorized_h,
            "Held-out unauthorized context exposure",
        )
        analyses["heldout_UDR_prompt_vs_acl"] = paired_cluster_analysis(
            held, "unauthorized_disclosure", "prompt_only", "pre_retrieval_acl", unauthorized_h,
            "Held-out unauthorized canary disclosure",
        )

    payload = {
        "model_label": args.label,
        "interpretation": (
            "Template-cluster analyses treat normalized attack/question templates as the resampling/permutation unit. "
            "Case-level intervals are retained for transparency but are not interpreted as independent attack-strategy evidence."
        ),
        "analyses": analyses,
    }
    (args.outdir / "cluster_aware_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for key, value in analyses.items():
        sf = value["template_signflip"]
        rows.append({
            "analysis": key,
            "n_cases": value["n_cases"],
            "n_templates": value["n_templates"],
            "case_difference": value["case_weighted_difference"],
            "template_difference": value["equal_template_weight_difference"],
            "template_signflip_p": sf.get("p_value"),
            "nonzero_templates": sf.get("n_nonzero_clusters"),
        })
    pd.DataFrame(rows).to_csv(args.outdir / "cluster_aware_summary.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

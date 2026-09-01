from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _bool(s: pd.Series) -> pd.Series:
    return s if s.dtype == bool else s.astype(str).str.lower().eq("true")


def rate(df: pd.DataFrame, arch: str, field: str, subset: pd.Series) -> dict:
    d = df[(df["architecture"] == arch) & subset].copy()
    values = _bool(d[field])
    n = int(len(d)); k = int(values.sum())
    if n == 0:
        return {"n": 0, "count": 0, "rate": None}
    rng = np.random.default_rng(4103)
    arr = values.astype(float).to_numpy()
    idx = rng.integers(0, n, size=(20000, n))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"n": n, "count": k, "rate": k / n, "case_bootstrap_95ci": [float(lo), float(hi)]}


def summarize(primary: pd.DataFrame, controlled: pd.DataFrame, heldout: pd.DataFrame, label: str) -> dict:
    unauthorized = ~_bool(primary["is_authorized"])
    legitimate = _bool(primary["is_legitimate"])
    h_unauth = ~_bool(heldout["is_authorized"])
    out: dict = {"label": label, "primary": {}, "controlled_k1": {}, "heldout": {}}
    for arch in ["prompt_only", "pre_retrieval_acl", "risk_aware"]:
        out["primary"][arch] = {
            "UCER": rate(primary, arch, "unauthorized_context_exposure", unauthorized),
            "UDR": rate(primary, arch, "unauthorized_disclosure", unauthorized),
            "ARSR": rate(primary, arch, "authorized_task_success", legitimate),
        }
        out["controlled_k1"][arch] = {
            "ARSR": rate(controlled, arch, "authorized_task_success", pd.Series(True, index=controlled.index))
        }
        out["heldout"][arch] = {
            "UCER": rate(heldout, arch, "unauthorized_context_exposure", h_unauth),
            "UDR": rate(heldout, arch, "unauthorized_disclosure", h_unauth),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-primary", type=Path, required=True)
    ap.add_argument("--baseline-controlled", type=Path, required=True)
    ap.add_argument("--baseline-heldout", type=Path, required=True)
    ap.add_argument("--replication-primary", type=Path, required=True)
    ap.add_argument("--replication-controlled", type=Path, required=True)
    ap.add_argument("--replication-heldout", type=Path, required=True)
    ap.add_argument("--baseline-label", default="SmolLM2-360M-Instruct")
    ap.add_argument("--replication-label", default="Qwen2.5-0.5B-Instruct")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    bp = pd.read_csv(args.baseline_primary); bc = pd.read_csv(args.baseline_controlled); bh = pd.read_csv(args.baseline_heldout)
    rp = pd.read_csv(args.replication_primary); rc = pd.read_csv(args.replication_controlled); rh = pd.read_csv(args.replication_heldout)
    assert len(bp) == len(rp) == 9000
    assert len(bc) == len(rc) == 3000
    assert len(bh) == len(rh) == 900
    for a, b in [(bp, rp), (bc, rc), (bh, rh)]:
        assert set(a["case_id"]) == set(b["case_id"])

    payload = {
        "baseline": summarize(bp, bc, bh, args.baseline_label),
        "replication": summarize(rp, rc, rh, args.replication_label),
        "cross_model_claim_checks": {},
    }
    for label, primary, controlled, held in [(args.baseline_label, bp, bc, bh), (args.replication_label, rp, rc, rh)]:
        unauth = ~_bool(primary["is_authorized"])
        hunauth = ~_bool(held["is_authorized"])
        p_acl = primary[(primary["architecture"] == "pre_retrieval_acl") & unauth]
        r_acl = primary[(primary["architecture"] == "risk_aware") & unauth]
        c = controlled.pivot(index="case_id", columns="architecture", values="authorized_task_success")
        payload["cross_model_claim_checks"][label] = {
            "secured_primary_UCER_zero": bool((~_bool(p_acl["unauthorized_context_exposure"])).all() and (~_bool(r_acl["unauthorized_context_exposure"])).all()),
            "controlled_k1_architecture_success_vectors_identical": bool(
                (_bool(c["prompt_only"]) == _bool(c["pre_retrieval_acl"])).all() and
                (_bool(c["prompt_only"]) == _bool(c["risk_aware"])).all()
            ),
            "secured_heldout_UCER_zero": bool(
                not _bool(held[(held["architecture"] == "pre_retrieval_acl") & hunauth]["unauthorized_context_exposure"]).any() and
                not _bool(held[(held["architecture"] == "risk_aware") & hunauth]["unauthorized_context_exposure"]).any()
            ),
        }

    (args.outdir / "model_replication_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for model_key in ["baseline", "replication"]:
        model = payload[model_key]
        for section in ["primary", "controlled_k1", "heldout"]:
            for arch, metrics in model[section].items():
                for metric, val in metrics.items():
                    rows.append({"model": model["label"], "section": section, "architecture": arch, "metric": metric, **val})
    pd.DataFrame(rows).to_csv(args.outdir / "model_replication_summary.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

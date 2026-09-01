from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from .core import AuthorizationMatrix, DeterministicRiskController, FuzzyRiskController, TestCase


def _mcnemar(a: pd.Series, b: pd.Series) -> dict:
    a = a.astype(bool)
    b = b.astype(bool)
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(min(a_only, b_only), n, 0.5, alternative="two-sided").pvalue)
    return {"fuzzy_only": a_only, "rules_only": b_only, "discordant": n, "p_value": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth", type=Path, default=Path("data/processed/authorization.json"))
    ap.add_argument("--cases", type=Path, nargs="+", default=[Path("data/processed/test_cases.json"), Path("data/processed/heldout_cases.json")])
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/risk_eval"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    acl = AuthorizationMatrix(json.loads(args.auth.read_text()))
    cases: list[TestCase] = []
    for path in args.cases:
        cases.extend(TestCase(**x) for x in json.loads(path.read_text()))

    fuzzy = FuzzyRiskController()
    rules = DeterministicRiskController()
    rows = []
    for case in cases:
        for name, controller in [("fuzzy", fuzzy), ("deterministic_rules", rules)]:
            r = controller.assess(case.account, case.prompt, acl, case.session_trust)
            rows.append({
                "case_id": case.case_id,
                "source": "heldout" if case.case_id.startswith("H") else "primary",
                "category": case.category,
                "account": case.account,
                "is_authorized": case.is_authorized,
                "is_legitimate": case.is_legitimate,
                "expected_challenge": case.category in {"authorized_suspicious", "heldout_authorized_suspicious"},
                "controller": name,
                "score": r.score,
                "decision": r.decision,
                "challenged": r.decision != "ALLOW",
                "authorization_confidence": r.authorization_confidence,
                "injection_risk": r.injection_risk,
                "sensitivity": r.sensitivity,
                "session_trust": r.session_trust,
            })
    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "risk_decisions.csv", index=False)

    metrics: dict[str, dict] = {}
    for controller in ["fuzzy", "deterministic_rules"]:
        d = df[df.controller == controller]
        metrics[controller] = {}
        for source in ["primary", "heldout"]:
            s = d[d.source == source]
            normal = s[s.is_legitimate]
            suspicious = s[s.expected_challenge]
            unauthorized = s[~s.is_authorized]
            metrics[controller][source] = {
                "n": int(len(s)),
                "normal_allow_rate": float((normal.decision == "ALLOW").mean()) if len(normal) else None,
                "normal_deny_rate": float((normal.decision == "DENY").mean()) if len(normal) else None,
                "suspicious_challenge_rate": float(suspicious.challenged.mean()) if len(suspicious) else None,
                "suspicious_deny_rate": float((suspicious.decision == "DENY").mean()) if len(suspicious) else None,
                "unauthorized_deny_rate": float((unauthorized.decision == "DENY").mean()) if len(unauthorized) else None,
                "decision_counts": {str(k): int(v) for k, v in s.decision.value_counts().to_dict().items()},
                "median_score": float(s.score.median()) if len(s) else None,
            }

    auth_eval = df[df.is_authorized].pivot(index="case_id", columns="controller", values="challenged")
    if {"fuzzy", "deterministic_rules"}.issubset(auth_eval.columns):
        metrics["paired_challenge_mcnemar"] = _mcnemar(auth_eval["fuzzy"], auth_eval["deterministic_rules"])

    # Held-out lexical generalization is intentionally difficult: every held-out
    # prompt has injection_risk==0 under the detector frozen before case creation.
    heldout = df[df.source == "heldout"]
    metrics["heldout_injection_risk_zero_fraction"] = float((heldout.injection_risk == 0.0).mean()) if len(heldout) else None
    (args.outdir / "risk_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

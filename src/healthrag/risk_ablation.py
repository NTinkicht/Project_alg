from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import AuthorizationMatrix, FuzzyRiskController, TestCase

ABLATIONS = {
    "none": None,
    "no_authorization_confidence": "authorization_confidence",
    "no_injection_risk": "injection_risk",
    "no_sensitivity": "sensitivity",
    "no_session_trust": "session_trust",
}


def _ablate(features: tuple[float, float, float, float], feature: str | None) -> tuple[float, float, float, float]:
    auth, inj, sens, trust = features
    # Benign/neutral replacement values answer the practical question: what
    # happens when this signal contributes no suspicion to the decision?
    if feature == "authorization_confidence":
        auth = 0.95
    elif feature == "injection_risk":
        inj = 0.0
    elif feature == "sensitivity":
        sens = 0.35
    elif feature == "session_trust":
        trust = 0.95
    return auth, inj, sens, trust


def _fuzzy_decision(features: tuple[float, float, float, float]) -> tuple[float, str]:
    auth_conf, inj, sens, trust = features
    ctl = FuzzyRiskController()
    A, P, S, T = map(ctl._memberships, [auth_conf, inj, sens, trust])
    rules: list[tuple[float, str]] = [
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
        "low": ctl._tri(ctl.grid, 0.0, 0.15, 0.42),
        "medium": ctl._tri(ctl.grid, 0.25, 0.50, 0.75),
        "high": ctl._tri(ctl.grid, 0.58, 0.85, 1.0),
    }
    aggregated = np.zeros_like(ctl.grid)
    for strength, label in rules:
        aggregated = np.maximum(aggregated, np.minimum(strength, out_sets[label]))
    score = 0.5 if aggregated.sum() == 0 else float(np.sum(ctl.grid * aggregated) / np.sum(aggregated))
    decision = "ALLOW" if score < ctl.allow_threshold else ("STEP_UP" if score < ctl.deny_threshold else "DENY")
    return score, decision


def _rules_decision(features: tuple[float, float, float, float]) -> tuple[float, str]:
    auth_conf, inj, sens, trust = features
    if auth_conf < 0.50:
        return 0.90, "DENY"
    if inj >= 0.65 or (inj >= 0.40 and sens >= 0.70):
        return 0.78, "DENY"
    if inj >= 0.20 or sens >= 0.90 or trust < 0.55 or auth_conf < 0.90:
        return 0.50, "STEP_UP"
    return 0.20, "ALLOW"


def _source(case: TestCase) -> str:
    if case.case_id.startswith("H"):
        return "heldout"
    if case.case_id.startswith("A"):
        return "adaptive"
    return "primary"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--auth", type=Path, default=Path("data/processed/authorization.json"))
    ap.add_argument("--cases", type=Path, nargs="+", default=[
        Path("data/processed/test_cases.json"),
        Path("data/processed/heldout_cases.json"),
        Path("data/processed/adaptive_cases.json"),
    ])
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/risk_ablation"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    acl = AuthorizationMatrix(json.loads(args.auth.read_text()))
    cases: list[TestCase] = []
    for path in args.cases:
        cases.extend(TestCase(**x) for x in json.loads(path.read_text()))

    rows = []
    for case in cases:
        raw = FuzzyRiskController.feature_values(case.account, case.prompt, acl, case.session_trust)
        for ablation_name, feature in ABLATIONS.items():
            features = _ablate(raw, feature)
            for controller, fn in [("fuzzy", _fuzzy_decision), ("deterministic_rules", _rules_decision)]:
                score, decision = fn(features)
                rows.append({
                    "case_id": case.case_id,
                    "source": _source(case),
                    "category": case.category,
                    "is_authorized": case.is_authorized,
                    "is_legitimate": case.is_legitimate,
                    "controller": controller,
                    "ablation": ablation_name,
                    "score": score,
                    "decision": decision,
                    "challenged": decision != "ALLOW",
                    "authorization_confidence": features[0],
                    "injection_risk": features[1],
                    "sensitivity": features[2],
                    "session_trust": features[3],
                })
    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "risk_ablation_decisions.csv", index=False)

    metrics: dict = {"feature_neutral_values": {"authorization_confidence": 0.95, "injection_risk": 0.0, "sensitivity": 0.35, "session_trust": 0.95}, "results": {}}
    for (controller, ablation, source), d in df.groupby(["controller", "ablation", "source"]):
        suspicious = d[d.category.isin(["authorized_suspicious", "heldout_authorized_suspicious", "adaptive_authorized_suspicious"])]
        unauthorized = d[~d.is_authorized]
        normal = d[d.is_legitimate]
        metrics["results"][f"{controller}|{ablation}|{source}"] = {
            "n": int(len(d)),
            "suspicious_challenge_rate": float(suspicious.challenged.mean()) if len(suspicious) else None,
            "unauthorized_deny_rate": float((unauthorized.decision == "DENY").mean()) if len(unauthorized) else None,
            "normal_allow_rate": float((normal.decision == "ALLOW").mean()) if len(normal) else None,
            "median_injection_risk": float(d.injection_risk.median()),
            "median_sensitivity": float(d.sensitivity.median()),
            "median_session_trust": float(d.session_trust.median()),
        }
    (args.outdir / "risk_ablation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

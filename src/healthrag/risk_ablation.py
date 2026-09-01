from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


GRID = np.linspace(0.0, 1.0, 401)


def tri(x: float | np.ndarray, a: float, b: float, c: float):
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if a == b else np.clip((x - a) / (b - a), 0, 1)
    right = np.ones_like(x) if b == c else np.clip((c - x) / (c - b), 0, 1)
    return np.minimum(left, right)


def memberships(x: float) -> dict[str, float]:
    return {
        "low": float(tri(x, 0.0, 0.0, 0.5)),
        "medium": float(tri(x, 0.2, 0.5, 0.8)),
        "high": float(tri(x, 0.5, 1.0, 1.0)),
    }


def fuzzy_from_features(auth_conf: float, inj: float, sens: float, trust: float) -> tuple[float, str]:
    A, P, S, T = map(memberships, [auth_conf, inj, sens, trust])
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
        "low": tri(GRID, 0.0, 0.15, 0.42),
        "medium": tri(GRID, 0.25, 0.50, 0.75),
        "high": tri(GRID, 0.58, 0.85, 1.0),
    }
    aggregated = np.zeros_like(GRID)
    for strength, label in rules:
        aggregated = np.maximum(aggregated, np.minimum(strength, out_sets[label]))
    score = 0.5 if aggregated.sum() == 0 else float(np.sum(GRID * aggregated) / np.sum(aggregated))
    decision = "ALLOW" if score < 0.34 else ("STEP_UP" if score < 0.66 else "DENY")
    return score, decision


def rules_from_features(auth_conf: float, inj: float, sens: float, trust: float) -> tuple[float, str]:
    if auth_conf < 0.50:
        return 0.90, "DENY"
    if inj >= 0.65 or (inj >= 0.40 and sens >= 0.70):
        return 0.78, "DENY"
    if inj >= 0.20 or sens >= 0.90 or trust < 0.55 or auth_conf < 0.90:
        return 0.50, "STEP_UP"
    return 0.20, "ALLOW"


NEUTRAL = {
    "authorization_confidence": 0.95,
    "injection_risk": 0.0,
    "sensitivity": 0.35,
    "session_trust": 0.95,
}
FEATURES = list(NEUTRAL)


def classify(df: pd.DataFrame, controller: str, ablate: str | None) -> pd.Series:
    fn = fuzzy_from_features if controller == "fuzzy" else rules_from_features
    decisions = []
    for row in df.itertuples(index=False):
        vals = {
            "authorization_confidence": float(row.authorization_confidence),
            "injection_risk": float(row.injection_risk),
            "sensitivity": float(row.sensitivity),
            "session_trust": float(row.session_trust),
        }
        if ablate is not None:
            vals[ablate] = NEUTRAL[ablate]
        _, decision = fn(vals["authorization_confidence"], vals["injection_risk"], vals["sensitivity"], vals["session_trust"])
        decisions.append(decision)
    return pd.Series(decisions, index=df.index)


def _bool(s: pd.Series) -> pd.Series:
    return s if s.dtype == bool else s.astype(str).str.lower().eq("true")


def score_condition(df: pd.DataFrame, decision: pd.Series) -> dict:
    authorized = _bool(df["is_authorized"])
    legitimate = _bool(df["is_legitimate"])
    expected_challenge = _bool(df["expected_challenge"])
    suspicious = authorized & expected_challenge
    unauthorized = ~authorized
    normal = legitimate
    return {
        "n": int(len(df)),
        "normal_allow_rate": float((decision[normal] == "ALLOW").mean()) if normal.any() else None,
        "suspicious_challenge_rate": float((decision[suspicious] != "ALLOW").mean()) if suspicious.any() else None,
        "suspicious_deny_rate": float((decision[suspicious] == "DENY").mean()) if suspicious.any() else None,
        "unauthorized_deny_rate": float((decision[unauthorized] == "DENY").mean()) if unauthorized.any() else None,
        "decision_counts": {str(k): int(v) for k, v in decision.value_counts().to_dict().items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk-decisions", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.risk_decisions)
    base = raw[raw["controller"] == "fuzzy"].copy().reset_index(drop=True)
    results: dict[str, dict] = {}
    rows = []
    for controller in ["fuzzy", "deterministic_rules"]:
        results[controller] = {}
        for source in ["primary", "heldout"]:
            d = base[base["source"] == source].copy()
            conditions = [("full", None)] + [(f"without_{f}", f) for f in FEATURES]
            results[controller][source] = {}
            for label, feature in conditions:
                decision = classify(d, controller, feature)
                metrics = score_condition(d, decision)
                results[controller][source][label] = metrics
                rows.append({"controller": controller, "source": source, "ablation": label, **{k: v for k, v in metrics.items() if k != "decision_counts"}})

    held = base[(base["source"] == "heldout") & _bool(base["expected_challenge"])].copy()
    results["mechanistic_checks"] = {
        "heldout_suspicious_n": int(len(held)),
        "injection_risk_zero_fraction": float((held["injection_risk"] == 0.0).mean()) if len(held) else None,
        "sensitivity_default_035_fraction": float(np.isclose(held["sensitivity"], 0.35).mean()) if len(held) else None,
        "median_session_trust": float(held["session_trust"].median()) if len(held) else None,
        "interpretation": "Ablations neutralize one feature at a time to its benign reference value; they are diagnostic, not a learned causal model.",
    }
    pd.DataFrame(rows).to_csv(args.outdir / "risk_feature_ablation_summary.csv", index=False)
    (args.outdir / "risk_feature_ablation_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

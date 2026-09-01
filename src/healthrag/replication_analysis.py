from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _boolify(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["is_authorized", "is_legitimate", "unauthorized_context_exposure", "unauthorized_disclosure", "authorized_task_success", "rejected"]:
        if c in df.columns and df[c].dtype != bool:
            df[c] = df[c].astype(str).str.lower().eq("true")
    return df


def _rate(d: pd.DataFrame, col: str) -> dict:
    return {"count": int(d[col].sum()), "n": int(len(d)), "rate": float(d[col].mean()) if len(d) else None}


def architecture_block(df: pd.DataFrame) -> dict:
    out = {}
    for arch in ["prompt_only", "pre_retrieval_acl", "risk_aware"]:
        d = df[df.architecture == arch]
        unauth = d[~d.is_authorized]
        legit = d[d.is_legitimate]
        out[arch] = {
            "n": int(len(d)),
            "UCER_unauthorized": _rate(unauth, "unauthorized_context_exposure"),
            "UDR_unauthorized": _rate(unauth, "unauthorized_disclosure"),
            "ARSR_legitimate": _rate(legit, "authorized_task_success"),
            "FRR_legitimate": _rate(legit, "rejected"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", type=Path, required=True)
    ap.add_argument("--controlled", type=Path, required=True)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    primary = _boolify(pd.read_csv(args.primary))
    controlled = _boolify(pd.read_csv(args.controlled))
    heldout = _boolify(pd.read_csv(args.heldout))
    payload = {
        "model": args.model,
        "primary": architecture_block(primary),
        "controlled_k1": architecture_block(controlled),
        "heldout": architecture_block(heldout),
    }
    hs = heldout[(heldout.architecture == "risk_aware") & (heldout.category == "heldout_authorized_suspicious")]
    payload["heldout"]["risk_aware_authorized_suspicious"] = {
        "n": int(len(hs)),
        "challenge_or_deny_rate": float((hs.fuzzy_decision != "ALLOW").mean()) if len(hs) else None,
        "deny_rate": float((hs.fuzzy_decision == "DENY").mean()) if len(hs) else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

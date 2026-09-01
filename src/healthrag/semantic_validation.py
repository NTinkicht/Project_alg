from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .adaptive_attack import structural_intent_preserved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/processed/adaptive_pairs.json"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--required-fraction", type=float, default=0.95)
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/semantic_validation"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    pairs = json.loads(args.pairs.read_text())
    model = SentenceTransformer(args.model)
    originals = [x["original"] for x in pairs]
    adaptives = [x["adaptive"] for x in pairs]
    a = model.encode(originals, normalize_embeddings=True, show_progress_bar=False)
    b = model.encode(adaptives, normalize_embeddings=True, show_progress_bar=False)
    similarity = np.sum(a * b, axis=1)

    rows = []
    for pair, sim in zip(pairs, similarity):
        structural = structural_intent_preserved(pair["original"], pair["adaptive"], pair["target"])
        rows.append({
            "case_id": pair["case_id"],
            "source_template": pair["source_template"],
            "authorized": pair["authorized"],
            "cosine_similarity": float(sim),
            "structural_intent_preserved": bool(structural),
            "passes_similarity_threshold": bool(sim >= args.threshold),
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "adaptive_semantic_validation.csv", index=False)
    metrics = {
        "encoder": args.model,
        "threshold": args.threshold,
        "n": int(len(df)),
        "structural_intent_preserved_fraction": float(df.structural_intent_preserved.mean()),
        "similarity_pass_fraction": float(df.passes_similarity_threshold.mean()),
        "median_cosine_similarity": float(df.cosine_similarity.median()),
        "p05_cosine_similarity": float(df.cosine_similarity.quantile(0.05)),
        "minimum_cosine_similarity": float(df.cosine_similarity.min()),
        "required_fraction": args.required_fraction,
        "interpretation": "Automatic independent encoder check; not a substitute for human annotation.",
    }
    (args.outdir / "adaptive_semantic_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if metrics["structural_intent_preserved_fraction"] != 1.0:
        raise SystemExit("Adaptive mutation violated the explicit disclosure-intent predicate")
    if metrics["similarity_pass_fraction"] < args.required_fraction:
        raise SystemExit("Too many adaptive mutations failed the independent semantic-similarity threshold")


if __name__ == "__main__":
    main()

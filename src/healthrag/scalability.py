from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer

from .core import PatientRecord, TestCase
from .data import load_records

SCALES = [1000, 10000, 100000]


class ScaleIndex:
    """One master TF-IDF representation for nested-corpus experiments.

    The vectorizer is fitted once on the 100K master corpus. Smaller conditions
    restrict only candidate rows. This intentionally holds representation/IDF
    fixed so corpus candidate size, rather than a changing vocabulary, is the
    independent variable.
    """

    def __init__(self, records: list[PatientRecord]):
        self.records = records
        self.alias_to_index = {r.alias: i for i, r in enumerate(records)}
        started = time.perf_counter()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, dtype=np.float32)
        self.matrix = self.vectorizer.fit_transform([r.text for r in records])
        self.build_ms = (time.perf_counter() - started) * 1000

    @property
    def sparse_bytes(self) -> int:
        return int(self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes)

    def indices_for_aliases(self, aliases: set[str], n: int) -> np.ndarray:
        return np.array(sorted(i for a in aliases if (i := self.alias_to_index.get(a)) is not None and i < n), dtype=np.int32)

    def retrieve_metrics(self, query: str, target_alias: str, candidate_indices: np.ndarray, k: int = 2) -> dict:
        start = time.perf_counter()
        if len(candidate_indices) == 0:
            return {
                "retrieval_ms": (time.perf_counter() - start) * 1000,
                "candidate_count": 0,
                "retrieved_aliases": [],
                "target_rank": None,
                "hit_at_1": False,
                "hit_at_2": False,
                "mrr": 0.0,
                "target_score": None,
                "best_distractor_score": None,
                "target_margin": None,
            }
        q = self.vectorizer.transform([query])
        scores = self.matrix[candidate_indices].dot(q.T).toarray().ravel()
        take = min(k, len(scores))
        if take == len(scores):
            local_top = np.arange(len(scores))
        else:
            local_top = np.argpartition(-scores, take - 1)[:take]
        ranked = sorted(
            ((int(candidate_indices[j]), float(scores[j])) for j in local_top),
            key=lambda x: (-x[1], self.records[x[0]].alias),
        )
        top_aliases = [self.records[i].alias for i, _ in ranked]

        target_idx = self.alias_to_index.get(target_alias)
        target_rank = None; target_score = None; best_distractor = None; margin = None
        if target_idx is not None:
            matches = np.where(candidate_indices == target_idx)[0]
            if len(matches):
                pos = int(matches[0]); target_score = float(scores[pos])
                greater = int(np.sum(scores > target_score))
                equal_before = 0
                tied = np.where(scores == target_score)[0]
                for j in tied:
                    alias = self.records[int(candidate_indices[j])].alias
                    if alias < target_alias:
                        equal_before += 1
                target_rank = 1 + greater + equal_before
                if len(scores) > 1:
                    other = np.delete(scores, pos)
                    best_distractor = float(other.max()) if len(other) else None
                    margin = target_score - best_distractor if best_distractor is not None else None
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "retrieval_ms": elapsed,
            "candidate_count": int(len(candidate_indices)),
            "retrieved_aliases": top_aliases,
            "target_rank": target_rank,
            "hit_at_1": target_rank == 1,
            "hit_at_2": target_rank is not None and target_rank <= 2,
            "mrr": (1.0 / target_rank if target_rank else 0.0),
            "target_score": target_score,
            "best_distractor_score": best_distractor,
            "target_margin": margin,
        }

    def target_scoped_metrics(self, target_alias: str, allowed_aliases: set[str], n: int) -> dict:
        start = time.perf_counter()
        idx = self.alias_to_index.get(target_alias)
        ok = idx is not None and idx < n and target_alias in allowed_aliases
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "retrieval_ms": elapsed,
            "candidate_count": 1 if ok else 0,
            "retrieved_aliases": [target_alias] if ok else [],
            "target_rank": 1 if ok else None,
            "hit_at_1": bool(ok),
            "hit_at_2": bool(ok),
            "mrr": 1.0 if ok else 0.0,
            "target_score": None,
            "best_distractor_score": None,
            "target_margin": None,
        }


def _load_queries(path: Path, count: int) -> list[TestCase]:
    cases = [TestCase(**x) for x in json.loads(path.read_text())]
    legit = sorted((c for c in cases if c.is_legitimate), key=lambda c: c.case_id)
    if len(legit) < count:
        raise RuntimeError(f"Need {count} legitimate cases, found {len(legit)}")
    return legit[:count]


def _summarize(df: pd.DataFrame, index: ScaleIndex) -> tuple[pd.DataFrame, dict]:
    rows = []
    for (scale, mode), d in df.groupby(["scale", "mode"], sort=True):
        rows.append({
            "scale": int(scale),
            "mode": mode,
            "n_queries": len(d),
            "candidate_count_median": float(d.candidate_count.median()),
            "hit_at_1": float(d.hit_at_1.mean()),
            "hit_at_2": float(d.hit_at_2.mean()),
            "MRR": float(d.mrr.mean()),
            "median_retrieval_ms": float(d.retrieval_ms.median()),
            "p95_retrieval_ms": float(d.retrieval_ms.quantile(.95)),
            "median_best_distractor_score": float(d.best_distractor_score.dropna().median()) if d.best_distractor_score.notna().any() else None,
            "median_target_margin": float(d.target_margin.dropna().median()) if d.target_margin.notna().any() else None,
        })
    summary = pd.DataFrame(rows)
    stats: dict = {
        "index_build_ms": index.build_ms,
        "sparse_matrix_bytes": index.sparse_bytes,
        "matrix_shape": list(index.matrix.shape),
        "vocabulary_size": int(len(index.vectorizer.vocabulary_)),
        "design": "TF-IDF representation fitted once on 100K master; nested conditions change candidate rows only",
    }

    # Paired scale comparisons for the key unfiltered and fixed-ACL modes.
    for mode in ["unfiltered", "acl_fixed", "acl_proportional"]:
        d = df[df.mode == mode]
        pivot_latency = d.pivot(index="case_id", columns="scale", values="retrieval_ms")
        pivot_distractor = d.pivot(index="case_id", columns="scale", values="best_distractor_score")
        block = {}
        if {1000, 100000}.issubset(pivot_latency.columns):
            try:
                w = wilcoxon(pivot_latency[1000], pivot_latency[100000])
                block["latency_1k_vs_100k_wilcoxon"] = {"statistic": float(w.statistic), "p_value": float(w.pvalue)}
            except Exception as e:
                block["latency_1k_vs_100k_wilcoxon"] = {"error": str(e)}
        if {1000, 100000}.issubset(pivot_distractor.columns):
            pair = pivot_distractor[[1000, 100000]].dropna()
            if len(pair):
                try:
                    w = wilcoxon(pair[1000], pair[100000])
                    block["distractor_score_1k_vs_100k_wilcoxon"] = {"statistic": float(w.statistic), "p_value": float(w.pvalue), "n": len(pair)}
                except Exception as e:
                    block["distractor_score_1k_vs_100k_wilcoxon"] = {"error": str(e), "n": len(pair)}
        per_case_slope = []
        for _, g in d.groupby("case_id"):
            if len(g) == 3:
                rho, _ = spearmanr(g.scale.astype(float), g.retrieval_ms.astype(float))
                if np.isfinite(rho): per_case_slope.append(float(rho))
        block["median_per_case_spearman_scale_latency"] = float(np.median(per_case_slope)) if per_case_slope else None
        stats[mode] = block
    return summary, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("data/scale/records_100k.json"))
    ap.add_argument("--cases", type=Path, default=Path("data/scale/test_cases.json"))
    ap.add_argument("--fixed-auth", type=Path, default=Path("data/scale/authorization_fixed.json"))
    ap.add_argument("--scale-dir", type=Path, default=Path("data/scale"))
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/scalability"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.records)
    if len(records) != 100000:
        raise RuntimeError(f"Expected 100K master corpus, got {len(records)}")
    queries = _load_queries(args.cases, args.queries)
    fixed = json.loads(args.fixed_auth.read_text())
    index = ScaleIndex(records)

    rows = []
    for scale in SCALES:
        proportional = json.loads((args.scale_dir / f"authorization_proportional_{scale}.json").read_text())
        global_indices = np.arange(scale, dtype=np.int32)
        fixed_indices = {a: index.indices_for_aliases(set(fixed.get(a, [])), scale) for a in ["A", "B", "C", "D", "E"]}
        proportional_indices = {a: index.indices_for_aliases(set(proportional.get(a, [])), scale) for a in ["A", "B", "C", "D", "E"]}
        for case in queries:
            conditions = [
                ("unfiltered", global_indices, None),
                ("acl_fixed", fixed_indices[case.account], None),
                ("acl_proportional", proportional_indices[case.account], None),
            ]
            for mode, candidates, _ in conditions:
                m = index.retrieve_metrics(case.prompt, case.target_alias or "", candidates, k=2)
                rows.append({"case_id": case.case_id, "account": case.account, "target_alias": case.target_alias, "scale": scale, "mode": mode, **m})
            for mode, allowed in [
                ("target_fixed", set(fixed.get(case.account, []))),
                ("target_proportional", set(proportional.get(case.account, []))),
            ]:
                m = index.target_scoped_metrics(case.target_alias or "", allowed, scale)
                rows.append({"case_id": case.case_id, "account": case.account, "target_alias": case.target_alias, "scale": scale, "mode": mode, **m})
        print(f"Completed retrieval scalability scale={scale} for {len(queries)} paired queries", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "scalability_results.csv", index=False)
    summary, stats = _summarize(df, index)
    summary.to_csv(args.outdir / "scalability_summary.csv", index=False)
    (args.outdir / "scalability_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from .core import TestCase, authorized_task_success, build_context, system_prompt
from .data import load_records
from .experiment import TransformersLLM
from .scalability import SCALES, ScaleIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("data/scale/records_100k.json"))
    ap.add_argument("--cases", type=Path, default=Path("data/scale/test_cases.json"))
    ap.add_argument("--scale-dir", type=Path, default=Path("data/scale"))
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--queries", type=int, default=90)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard")

    records = load_records(args.records)
    cases = [TestCase(**x) for x in json.loads(args.cases.read_text())]
    legit = sorted((c for c in cases if c.is_legitimate), key=lambda c: c.case_id)[:args.queries]
    selected = legit[args.shard_index::args.num_shards]
    index = ScaleIndex(records)
    llm = TransformersLLM(args.model)

    rows = []
    for ci, case in enumerate(selected, 1):
        for scale in SCALES:
            proportional = json.loads((args.scale_dir / f"authorization_proportional_{scale}.json").read_text())
            allowed = set(proportional.get(case.account, []))
            candidates = index.indices_for_aliases(allowed, scale)
            r = index.retrieve_metrics(case.prompt, case.target_alias or "", candidates, k=2)
            retrieved = [records[index.alias_to_index[a]] for a in r["retrieved_aliases"]]
            context = build_context(retrieved)
            started = time.perf_counter()
            response = llm.generate(system_prompt(case.account, allowed, context), case.prompt)
            generation_ms = (time.perf_counter() - started) * 1000
            success = authorized_task_success(response, case.expected_fact, False)
            rows.append({
                "case_id": case.case_id,
                "account": case.account,
                "target_alias": case.target_alias,
                "scale": scale,
                "mode": "acl_proportional",
                "authorization_scope_size": len(allowed),
                "authorization_density_rho": len(allowed) / scale,
                "retrieved_aliases": r["retrieved_aliases"],
                "target_rank": r["target_rank"],
                "best_distractor_score": r["best_distractor_score"],
                "target_margin": r["target_margin"],
                "retrieval_ms": r["retrieval_ms"],
                "generation_ms": generation_ms,
                "authorized_task_success": bool(success),
                "response": response,
            })
        if ci % 5 == 0:
            print(f"Proportional scale-LLM shard {args.shard_index}: {ci}/{len(selected)} queries", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} acl_proportional scale-LLM rows from {len(selected)} base queries")


if __name__ == "__main__":
    main()

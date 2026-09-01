from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .core import AuthorizationMatrix, TestCase, authorized_task_success, build_context, system_prompt
from .data import load_records
from .experiment import TransformersLLM
from .scalability import SCALES, ScaleIndex


ALL_MODES = {"unfiltered", "acl_fixed", "acl_proportional", "target_fixed", "target_proportional"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("data/scale/records_100k.json"))
    ap.add_argument("--cases", type=Path, default=Path("data/scale/test_cases.json"))
    ap.add_argument("--fixed-auth", type=Path, default=Path("data/scale/authorization_fixed.json"))
    ap.add_argument("--scale-dir", type=Path, default=Path("data/scale"))
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--queries", type=int, default=90)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--modes", default="unfiltered,acl_fixed,target_fixed")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard")
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    if not modes or not set(modes).issubset(ALL_MODES):
        raise ValueError(f"modes must be a non-empty subset of {sorted(ALL_MODES)}")

    records = load_records(args.records)
    cases = [TestCase(**x) for x in json.loads(args.cases.read_text())]
    legit = sorted((c for c in cases if c.is_legitimate), key=lambda c: c.case_id)[:args.queries]
    selected = legit[args.shard_index::args.num_shards]
    fixed_map = json.loads(args.fixed_auth.read_text())
    fixed_acl = AuthorizationMatrix(fixed_map)
    index = ScaleIndex(records)
    llm = TransformersLLM(args.model)

    rows = []
    for ci, case in enumerate(selected, 1):
        for scale in SCALES:
            proportional_map = json.loads((args.scale_dir / f"authorization_proportional_{scale}.json").read_text())
            proportional_acl = AuthorizationMatrix(proportional_map)
            global_indices = np.arange(scale, dtype=np.int32)
            fixed_indices = index.indices_for_aliases(fixed_acl.allowed(case.account), scale)
            proportional_indices = index.indices_for_aliases(proportional_acl.allowed(case.account), scale)
            retrievals = {}
            if "unfiltered" in modes:
                retrievals["unfiltered"] = index.retrieve_metrics(case.prompt, case.target_alias or "", global_indices, k=2)
            if "acl_fixed" in modes:
                retrievals["acl_fixed"] = index.retrieve_metrics(case.prompt, case.target_alias or "", fixed_indices, k=2)
            if "acl_proportional" in modes:
                retrievals["acl_proportional"] = index.retrieve_metrics(case.prompt, case.target_alias or "", proportional_indices, k=2)
            if "target_fixed" in modes:
                retrievals["target_fixed"] = index.target_scoped_metrics(case.target_alias or "", fixed_acl.allowed(case.account), scale)
            if "target_proportional" in modes:
                retrievals["target_proportional"] = index.target_scoped_metrics(case.target_alias or "", proportional_acl.allowed(case.account), scale)

            for mode, r in retrievals.items():
                retrieved = [records[index.alias_to_index[a]] for a in r["retrieved_aliases"]]
                context = build_context(retrieved)
                started = time.perf_counter()
                response = llm.generate(system_prompt(case.account, fixed_acl.allowed(case.account), context), case.prompt)
                generation_ms = (time.perf_counter() - started) * 1000
                success = authorized_task_success(response, case.expected_fact, False)
                rows.append({
                    "case_id": case.case_id,
                    "account": case.account,
                    "target_alias": case.target_alias,
                    "scale": scale,
                    "mode": mode,
                    "retrieved_aliases": r["retrieved_aliases"],
                    "candidate_count": r.get("candidate_count"),
                    "target_rank": r["target_rank"],
                    "best_distractor_score": r["best_distractor_score"],
                    "target_margin": r["target_margin"],
                    "retrieval_ms": r["retrieval_ms"],
                    "generation_ms": generation_ms,
                    "authorized_task_success": bool(success),
                    "response": response,
                })
        if ci % 5 == 0:
            print(f"Scale-LLM shard {args.shard_index}: {ci}/{len(selected)} queries", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} scale-LLM rows from {len(selected)} base queries across modes={modes}")


if __name__ == "__main__":
    main()

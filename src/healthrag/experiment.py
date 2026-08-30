from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from .core import (
    AuthorizationMatrix,
    FuzzyRiskController,
    Retriever,
    RunResult,
    TestCase,
    authorized_task_success,
    build_context,
    system_prompt,
    unauthorized_context_exposure,
    unauthorized_disclosure,
)
from .data import load_records


class TransformersLLM:
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, sys_prompt: str, user_prompt: str, max_new_tokens: int = 64) -> str:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = self.torch.ones_like(input_ids)
        with self.torch.no_grad():
            out = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new = out[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip()


class MockLLM:
    """Used only for unit/integration tests, never for empirical paper results."""
    def generate(self, sys_prompt: str, user_prompt: str, max_new_tokens: int = 64) -> str:
        if "ExperimentalSecurityCanary:" in sys_prompt:
            # deterministic test helper: return first requested field-like line
            for line in sys_prompt.splitlines():
                if line.startswith("PrimaryCondition:"):
                    return line.split(":", 1)[1].strip()
        return "ACCESS_DENIED"


def run_case(case: TestCase, architecture: str, records, acl, retriever, risk, llm) -> RunResult:
    start = time.perf_counter()
    fuzzy = None
    rejected = False
    retrieved = []

    if architecture == "prompt_only":
        retrieved = retriever.retrieve(case.prompt, allowed_aliases=None, k=2)
    elif architecture == "pre_retrieval_acl":
        targets = [case.target_alias] if case.target_alias else []
        if targets and not acl.all_allowed(case.account, targets):
            rejected = True
        else:
            retrieved = retriever.retrieve(case.prompt, allowed_aliases=acl.allowed(case.account), k=2)
    elif architecture == "risk_aware":
        fuzzy = risk.assess(case.account, case.prompt, acl, case.session_trust)
        if fuzzy.decision == "DENY":
            rejected = True
        else:
            targets = [case.target_alias] if case.target_alias else []
            if targets and not acl.all_allowed(case.account, targets):
                rejected = True
            else:
                k = 1 if fuzzy.decision == "STEP_UP" else 2
                retrieved = retriever.retrieve(case.prompt, allowed_aliases=acl.allowed(case.account), k=k)
    else:
        raise ValueError(architecture)

    if rejected:
        response = "ACCESS_DENIED"
    else:
        context = build_context(retrieved)
        response = llm.generate(system_prompt(case.account, acl.allowed(case.account), context), case.prompt)

    latency = (time.perf_counter() - start) * 1000
    exposure = unauthorized_context_exposure(retrieved, case.account, acl)
    disclosure = unauthorized_disclosure(response, case.account, acl, records)
    task_success = case.is_legitimate and authorized_task_success(response, case.expected_fact, rejected)

    return RunResult(
        case_id=case.case_id,
        architecture=architecture,
        account=case.account,
        category=case.category,
        prompt=case.prompt,
        target_alias=case.target_alias,
        is_authorized=case.is_authorized,
        is_legitimate=case.is_legitimate,
        retrieved_aliases=[r.alias for r in retrieved],
        unauthorized_context_exposure=exposure,
        response=response,
        unauthorized_disclosure=disclosure,
        authorized_task_success=task_success,
        rejected=rejected,
        latency_ms=latency,
        fuzzy_score=fuzzy.score if fuzzy else None,
        fuzzy_decision=fuzzy.decision if fuzzy else None,
        authorization_confidence=fuzzy.authorization_confidence if fuzzy else None,
        injection_risk=fuzzy.injection_risk if fuzzy else None,
        sensitivity=fuzzy.sensitivity if fuzzy else None,
        session_trust=fuzzy.session_trust if fuzzy else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("data/processed/records.json"))
    ap.add_argument("--auth", type=Path, default=Path("data/processed/authorization.json"))
    ap.add_argument("--cases", type=Path, default=Path("data/processed/test_cases.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/results.csv"))
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    args = ap.parse_args()

    records = load_records(args.records)
    acl = AuthorizationMatrix(json.loads(args.auth.read_text()))
    cases = [TestCase(**x) for x in json.loads(args.cases.read_text())]
    retriever = Retriever(records)
    risk = FuzzyRiskController()
    llm = TransformersLLM(args.model)

    rows = []
    archs = ["prompt_only", "pre_retrieval_acl", "risk_aware"]
    for i, case in enumerate(cases, 1):
        for arch in archs:
            result = run_case(case, arch, records, acl, retriever, risk, llm)
            rows.append(result.to_dict())
        if i % 6 == 0:
            print(f"Completed {i}/{len(cases)} cases ({len(rows)} architecture executions)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} run records to {args.out}")


if __name__ == "__main__":
    main()

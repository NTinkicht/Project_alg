# Structural versus Heuristic Security in Healthcare RAG

This repository is the reproducible software artifact for **When Heuristics Fail but Boundaries Hold: Structural versus Heuristic Security in Healthcare RAG**. It evaluates authorization placement, wording-sensitive risk triage, context exposure, generated disclosure, and authorization-scope scaling using synthetic Synthea records.

## Current study at a glance

- Deterministic Synthea v4.0.0 corpus: **1,000 synthetic patients**, seed `4103`.
- Architectures: prompt-only policy, deterministic pre-retrieval ACL, and risk-aware triage followed by the same mandatory ACL.
- Primary benchmark: **3,000 cases / 9,000 architecture executions** with SmolLM2-360M-Instruct.
- Controlled utility benchmark: **1,000 cases / 3,000 executions** at `k=1`.
- Qwen2.5-0.5B-Instruct replication: **13,500 executions** across primary, controlled, held-out, and detector-targeted conditions.
- Additional strengthening evidence: 600 Smol detector-targeted executions, a 750-row exact Smol rerun, 1,080 scale generations, 35,000 risk-ablation decisions, cluster-aware inference, and property-based correspondence tests.

The authoritative reproduction guide, pinned revisions, workflow run, result manifest, checksums, and interpretation constraints are in [`paper/README_REPRODUCIBILITY.md`](paper/README_REPRODUCIBILITY.md). The preregistration is in [`paper/NOVELTY8_PREREGISTRATION.md`](paper/NOVELTY8_PREREGISTRATION.md), and frozen reported results are in [`paper/FINAL_RESULTS.md`](paper/FINAL_RESULTS.md).

The complete validated raw evidence snapshot is permanently archived at [`paper/evidence/healthrag_novelty8_final_actions_artifact.zip`](paper/evidence/healthrag_novelty8_final_actions_artifact.zip) with SHA-256 `a686d96614a57a5e96845abff191491c761fd3ad509b342e3c8b0a0640841b90`.

## Research design

The study compares three architectures using the same corpus, prompts, retriever, and model:

1. **Prompt-only policy baseline** - retrieval may include records outside the authenticated account's permissions; text instructions ask the model not to disclose them.
2. **Deterministic pre-retrieval ACL** - application code restricts the retrieval candidate set to patient records authorized for the authenticated identity.
3. **Risk-aware pre-retrieval authorization** - an interpretable risk layer returns `ALLOW`, `STEP_UP`, or `DENY`, followed by the same mandatory ACL. The risk layer may narrow access but never widen the authorized set.

The structural result is an implementation invariant under the stated threat model: if candidate construction correctly restricts retrieval to the authenticated identity's ACL, prompt wording cannot add an unauthorized record. Experiments and property tests confirm correspondence on tested inputs; they do not prove zero real-world leakage.

The heuristic experiment is deliberately narrow. Its baseline is a transparent fixed **22-entry lexical detector**. The preregistered detector-targeted search is **white-box**: it directly calls the detector's scoring functions using a frozen, detector-informed substitution vocabulary. Results therefore characterize this lexical detector family and must not be generalized to learned, embedding-based, LLM-based, or black-box detectors.

## Main metrics

- **UCER** - Unauthorized Context Exposure Rate: whether an unauthorized target enters model context.
- **Canary UDR** - Unauthorized Disclosure Rate: whether generated text emits an out-of-scope record's synthetic marker.
- **SBSR** - Structural Boundary Survival Rate on unauthorized adversarial requests.
- **HDR** - Heuristic Detection Retention after distribution shift.
- **ARSR** - Authorized Request Success Rate.
- **FRR** - False Rejection Rate for legitimate requests.

## Quick software tests

```bash
python -m pip install -e '.[test]'
pytest -q
```

## Reproduce the publication study

The publication corpus and full experiment should be run through the pinned workflows rather than the small developer defaults:

- [`.github/workflows/research.yml`](.github/workflows/research.yml) builds the 1,000-patient Synthea corpus and primary SmolLM2 evidence.
- [`.github/workflows/novelty8.yml`](.github/workflows/novelty8.yml) runs the preregistered strengthening, Qwen replication, semantic checks, ablations, exact rerun, and scale experiment.

For the exact commands, revisions, artifact identities, and validation rules, follow [`paper/README_REPRODUCIBILITY.md`](paper/README_REPRODUCIBILITY.md). Running `python -m healthrag.data` without publication parameters creates only the 25-patient developer fixture and does **not** reproduce the paper.

## Manuscript

Build the IEEEtran paper from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error healthrag_final.tex
pdflatex -interaction=nonstopmode -halt-on-error healthrag_final.tex
```

## Scope

This is a synthetic software-security experiment. It uses no real patients, clinicians, hospitals, clinical decisions, or human-subject research. Healthcare is the application domain, not evidence of clinical validation. The study does not provide cryptographic confidentiality, exhaustive formal verification, or universal claims about heuristic security.

# Risk-Aware Pre-Retrieval Authorization for Healthcare RAG

This repository is the reproducible software artifact for a focused conference-paper study of authorization placement in healthcare Retrieval-Augmented Generation (RAG).

## Research design

The primary experiment compares three architectures using the **same Synthea-derived patient corpus, prompts, retriever, and language model**:

1. **Prompt-only authorization (baseline)** - retrieval may include records outside the authenticated account's permissions; the LLM is instructed not to disclose them.
2. **Deterministic pre-retrieval ACL** - application code applies patient-level authorization before retrieval, so unauthorized records cannot enter the LLM context.
3. **Risk-aware pre-retrieval authorization** - an interpretable fuzzy risk layer produces `ALLOW`, `STEP_UP`, or `DENY`, followed by the same mandatory deterministic ACL. Fuzzy `ALLOW` never expands the authorized set.

The primary causal comparison is **(1) vs (2)**. Architecture (3) is a secondary adaptive-security extension, which avoids confounding the effect of authorization placement with the fuzzy layer.

## Dataset

The CI workflow downloads the official latest 100-patient Synthea FHIR R4 sample from the `synthetichealth/synthea-sample-data` repository. A deterministic SHA-256 ordering selects 25 patient bundles. Accounts A-D receive five patient aliases each, Account E receives none, and PAT-21..PAT-25 remain a protected pool. Each normalized patient document receives a unique experimental canary for objective disclosure measurement.

## Prompt suite

72 deterministic test cases:

- 18 legitimate authorized requests
- 12 authorized but suspicious requests (risk-layer evaluation)
- 42 unauthorized/adversarial requests across six categories

Each case is executed against all three architectures (216 architecture executions).

## Main metrics

- **UCER** - Unauthorized Context Exposure Rate: unauthorized tests where an out-of-scope patient record enters retrieved context.
- **UDR** - Unauthorized Disclosure Rate: unauthorized tests where the model emits an out-of-scope patient's unique canary.
- **ARSR** - Authorized Request Success Rate.
- **FRR** - False Rejection Rate for legitimate authorized requests.
- Latency and fuzzy decision distribution are secondary measures.

## Formal verification

`formal/secured.smv` models:

`LOGIN -> AUTHENTICATED -> RISK_CHECK -> ACL_CHECK -> RETRIEVE -> LLM_RESPONSE -> COMPLETE/DENIED`

Six CTL properties verify the authorization boundary and liveness. `formal/negative_control.smv` intentionally removes the ACL gate before retrieval, so critical properties should produce counterexamples.

## Local quick test

```bash
python -m pip install -e '.[test]'
pytest -q
```

## Full empirical reproduction

```bash
python -m pip install -e '.[experiment,test]'
python -m healthrag.data
python -m healthrag.prompts
python -m healthrag.experiment --model HuggingFaceTB/SmolLM2-360M-Instruct
python -m healthrag.analysis
```

The GitHub Actions workflow also downloads NuSMV 2.7.1, runs both formal models, and uploads all result artifacts.

## Scope

This is a synthetic software-security experiment. It uses no real patients, clinicians, hospitals, clinical decisions, or human-subject research. It does not evaluate medical correctness.

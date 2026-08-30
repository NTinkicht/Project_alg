# Authenticated Pre-Retrieval Authorization for Healthcare RAG

This repository is the reproducible software artifact for a focused conference-paper study of authentication, authorization placement, prompt injection, risk-aware handling, and formal verification in a synthetic healthcare Retrieval-Augmented Generation (RAG) system.

## Research design

The primary experiment compares three architectures using the **same Synthea-derived patient corpus, prompts, retriever, and language model**:

1. **Prompt-only authorization (baseline)** - retrieval may include records outside the authenticated account's permissions; the LLM is instructed not to disclose them.
2. **Deterministic pre-retrieval ACL** - application code applies patient-level authorization before retrieval, so unauthorized records cannot enter the LLM context.
3. **Risk-aware pre-retrieval authorization** - an interpretable fuzzy risk layer produces `ALLOW`, `STEP_UP`, or `DENY`, followed by the same mandatory deterministic ACL. Fuzzy `ALLOW` never expands the authorized set.

The primary causal comparison is **(1) vs (2)**. Architecture (3) is a secondary adaptive-security extension, which avoids confounding the effect of authorization placement with the fuzzy layer.

## Authentication boundary

Authentication and authorization are separate controls. The artifact contains a lightweight HMAC-signed bearer-token `Authenticator` that resolves a credential to a known software principal and rejects malformed, unknown, or tampered credentials. The adversarial RAG experiment begins after successful authentication so that authorization placement remains the independent variable; authentication failure is covered by unit tests and the NuSMV state model.

## Dataset

The CI workflow downloads a **pinned public Synthea FHIR R4 sample snapshot** from `synthetichealth/synthea-sample-data` commit `9959d9178ea28f4ec10f17ee238b6fabe6eb0de5`. A deterministic SHA-256 ordering selects 25 patient bundles. Accounts A-D receive five patient aliases each, Account E receives none, and PAT-21..PAT-25 remain a protected pool. Each normalized patient document receives a unique experimental canary for objective disclosure measurement.

## Prompt suite

72 deterministic test cases:

- 18 legitimate authorized requests
- 12 authorized but suspicious requests (risk-layer evaluation)
- 42 unauthorized/adversarial requests across six categories

Each case is executed against all three architectures (216 architecture-level run records).

## Main metrics

- **UCER** - Unauthorized Context Exposure Rate: unauthorized tests where an out-of-scope patient record enters retrieved context.
- **Canary UDR** - Unauthorized Disclosure Rate operationalized as exact emission of an out-of-scope patient's unique canary; this is intentionally a conservative lower bound on semantic leakage.
- **ARSR** - Authorized Request Success Rate.
- **FRR** - False Rejection Rate for legitimate authorized requests.
- Legitimate-path latency and fuzzy decision distribution are secondary measures.

## Formal verification

`formal/secured.smv` models:

`LOGIN -> AUTHENTICATED -> RISK_CHECK -> ACL_CHECK -> RETRIEVE -> LLM_RESPONSE -> COMPLETE/DENIED`

Seven CTL properties check the authentication gate, deterministic authorization boundary, fuzzy non-bypass property, and liveness. `formal/negative_control.smv` intentionally removes the authentication/ACL gates before retrieval; its corresponding security properties must be falsified and produce counterexample traces. CI fails if a secured property is false or if the negative control cannot falsify a boundary property.

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

The GitHub Actions workflow also runs NuSMV 2.7.1, records dependency and dataset versions, and uploads the empirical, statistical, and formal-verification artifacts.

## Scope

This is a synthetic software-security experiment. It uses no real patients, clinicians, hospitals, clinical decisions, or human-subject research. It does not evaluate medical correctness.

# Final verified 3,000-case results

Authoritative reproduction: GitHub Actions **Research Reproduction Run #11** (`33431353837`), research commit `c123d406b4346811ab703b2c7355441f1b8e6319`.

## Reproduction status

- Workflow conclusion: **success**
- Unit / architecture tests: **11 passed** (1 benign pytest collection warning)
- Synthea generator: **v4.0.0**, seed **4103**, reference date **2026-01-01**, Massachusetts
- Normalized synthetic patients: **1,000**
- Authorization corpus: **800** patients assigned across accounts A-D (200 each), **200** globally protected, account E has no patient permissions
- Publication benchmark: **3,000 cases**
- Historical regression suite: **72 cases / 42 unauthorized**, preserved separately and not pooled into the new sample
- Compared architectures: **3**
- Parallel shards: **12**
- LLM architecture executions: **9,000/9,000 completed**
- Unique `(case_id, architecture)` pairs: **9,000**; duplicates: **0**
- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Decoding: greedy (`do_sample=False`), `max_new_tokens=64`
- Formal verifier: **NuSMV 2.7.1**
- Secured NuSMV model: **7/7 CTL properties true**
- Deliberately insecure negative control: **4/4 corresponding safety properties false**, with counterexamples
- Authoritative Run #11 research artifact SHA-256: `6917088ea3654ef918344b9de1dd10621163c62279eec256e0e290a6fdc6f162`

## Benchmark composition

| Category | Cases | Unauthorized |
|---|---:|---:|
| Authorized normal | 1,000 | 0 |
| Direct unauthorized | 1,000 | 1,000 |
| Authorized suspicious | 250 | 0 |
| Direct injection | 150 | 150 |
| Privilege/debug | 150 | 150 |
| Context extraction | 150 | 150 |
| Role-play/obfuscated | 150 | 150 |
| Multi-step | 150 | 150 |
| **Total** | **3,000** | **1,750** |

## Main empirical outcomes

| Metric | Prompt-only | Pre-retrieval ACL | Risk-aware |
|---|---:|---:|---:|
| Unauthorized Context Exposure (UCER) | 1750/1750 (100%) | 0/1750 (0%) | 0/1750 (0%) |
| Canary Unauthorized Disclosure (UDR) | 5/1750 (0.286%) | 0/1750 (0%) | 0/1750 (0%) |
| Authorized Request Success (ARSR) | 790/1000 (79.0%) | 640/1000 (64.0%) | 640/1000 (64.0%) |
| False Rejection Rate | 0/1000 (0%) | 0/1000 (0%) | 0/1000 (0%) |

Wilson 95% confidence intervals:

- Prompt-only UCER: **99.781%-100%**
- Secured UCER: **0%-0.219%**
- Prompt-only UDR: **0.122%-0.667%**
- Secured UDR: **0%-0.219%**
- Prompt-only ARSR: **76.37%-81.41%**
- Secured ARSR: **60.98%-66.92%**
- FRR for each architecture: **0%-0.383%**

## Paired statistical tests

- UCER prompt-only vs. pre-retrieval ACL: 1,750 baseline-only positives, 0 ACL-only; exact probability below floating-point resolution; conservatively reported as **p < 10^-300**.
- Canary UDR prompt-only vs. pre-retrieval ACL: 5 baseline-only positives, 0 ACL-only; exact McNemar **p = 0.0625**. This secondary output-leakage difference is **not conventionally significant at alpha=0.05**.
- ARSR prompt-only vs. pre-retrieval ACL: 205 baseline-only successes, 55 ACL-only successes; exact McNemar **p = 1.629519646223584e-21**.
- Legitimate-request latency prompt-only vs. pre-retrieval ACL: Wilcoxon statistic **43,887**, **p = 5.560060802070198e-113**.

All five prompt-only canary disclosures occurred in the 150-case **multi-step** category (3.33% within that category). Every other unauthorized category had zero canary disclosures despite 100% prompt-only unauthorized context exposure.

Median legitimate-request end-to-end latency:

- Prompt-only: **11.961 s**
- Pre-retrieval ACL: **12.953 s**
- Risk-aware: **12.947 s**

All-request medians are **13.227 s**, **0.023 ms**, and **0.344 ms**, respectively. The near-zero secured all-request medians reflect early rejection of unauthorized requests before LLM inference and are not a fair cross-architecture model-serving latency comparison.

## Risk-aware controller behavior

Across all 3,000 cases:

- `ALLOW`: **1,000**
- `STEP_UP`: **219**
- `DENY`: **1,781**
- Authorized-suspicious challenge-or-denial rate: **100%** (219 STEP_UP, 31 DENY)
- Median fuzzy risk: ordinary authorized **0.246**, suspicious authorized **0.614**, unauthorized **0.757**

## Formal verification interpretation

The secured state machine satisfies all seven specified CTL properties, covering authentication gating, ACL-before-retrieval, retrieval implying successful authentication and ACL authorization, fuzzy-ALLOW non-bypass, risk-DENY transition, liveness for authenticated/authorized/non-denied requests, and authentication-failure denial. The deliberately insecure negative-control model violates all four corresponding safety properties, and NuSMV produces counterexamples.

## Scientific interpretation

The strongest result is structural: deterministic pre-retrieval authorization reduced unauthorized context exposure from **100% to 0%** across 1,750 unauthorized requests. The generated-output canary leakage rate was already rare in the prompt-only baseline, so the observed **5-to-0** reduction did not reach the conventional 0.05 significance threshold and is not overclaimed. The large run also revealed a real utility trade-off: authorized-task success fell from **79% to 64%** under the current secured retrieval implementation. That result is retained rather than tuned away post hoc. A preregistered follow-up should test target-scoped retrieval to reduce authorized distractors while preserving the same deterministic pre-retrieval authorization boundary.

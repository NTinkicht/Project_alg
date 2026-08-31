# Final verified results

Authoritative reproduction: GitHub Actions **Research Reproduction Run #9**, commit `09fa9ff99f1363d284504136e2fec3a2afc64cba` (completed successfully on 2026-08-30).

## Reproduction status

- Unit / architecture tests: **10 passed** (1 benign pytest collection warning)
- Pinned Synthea snapshot: `9959d9178ea28f4ec10f17ee238b6fabe6eb0de5`
- Synthetic patients: **25**
- Controlled requests: **72**
- Compared architectures: **3**
- LLM architecture executions: **216/216 completed**
- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Decoding: greedy (`do_sample=False`), `max_new_tokens=64`
- Formal verifier: **NuSMV 2.7.1**
- Secured NuSMV model: **7/7 CTL properties true**
- Deliberately insecure negative control: **4/4 boundary properties false**, each with a counterexample

## Primary empirical outcomes

| Metric | Prompt-only | Pre-retrieval ACL | Risk-aware |
|---|---:|---:|---:|
| Unauthorized Context Exposure (UCER) | 42/42 (100%) | 0/42 (0%) | 0/42 (0%) |
| Canary Unauthorized Disclosure (UDR) | 3/42 (7.14%) | 0/42 (0%) | 0/42 (0%) |
| Authorized Request Success (ARSR) | 13/18 (72.2%) | 15/18 (83.3%) | 15/18 (83.3%) |
| False Rejection Rate | 0/18 (0%) | 0/18 (0%) | 0/18 (0%) |

Exact paired tests:

- UCER prompt-only vs. pre-retrieval ACL: McNemar `p = 4.547473508864641e-13`
- UDR prompt-only vs. pre-retrieval ACL: McNemar `p = 0.25`
- ARSR prompt-only vs. pre-retrieval ACL: McNemar `p = 0.625`
- Legitimate-request latency prompt-only vs. pre-retrieval ACL: Wilcoxon `p = 3.814697265625e-05`

Median legitimate-request end-to-end latency:

- Prompt-only: **6.682 s** (IQR 6.280-7.415 s)
- Pre-retrieval ACL: **5.120 s** (IQR 4.998-6.124 s)
- Risk-aware: **5.115 s** (IQR 5.004-6.169 s)

## Risk-aware controller behavior

- `ALLOW`: 18
- `STEP_UP`: 10
- `DENY`: 44
- Authorized suspicious challenge rate: **100%**
- Median risk: ordinary authorized **0.246**, suspicious authorized **0.627**, unauthorized **0.757**

## Formal verification interpretation

The secured state-machine satisfies all seven specified CTL properties, including authentication non-bypass, ACL-before-retrieval, fuzzy non-override, safe denial transitions, and eventual completion for valid non-denied requests. The weakened negative-control model violates all four boundary properties, and NuSMV emits counterexamples that transition from `LOGIN` with failed authentication/ACL directly to `RETRIEVE`.

The authoritative raw artifact is attached to Run #9 as `healthrag-research-artifacts` (SHA-256 digest `28db43166f43e6c559576185976ed22176727504f707461cb73caa9270a4aaa8`).

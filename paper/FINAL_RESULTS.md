# HealthRAG structural-versus-heuristic strengthening: frozen final results

## Frozen execution

- Branch: `structural-heuristic-novelty8`
- Preregistered experiment commit: `7b9c0cbcb188da74c08ee4281061fbdb7a7b1f11`
- Workflow: `Structural vs Heuristic Security - Novelty 8 Strengthening`
- GitHub Actions run: `33565992867`
- Final workflow conclusion: **success**
- Final artifact: `healthrag-structural-heuristic-novelty8-artifacts`
- Final artifact ID: `9850437939`
- Final artifact SHA-256: `a686d96614a57a5e96845abff191491c761fd3ad509b342e3c8b0a0640841b90`

## Planned work completed

- Qwen2.5-0.5B-Instruct replication: **13,500/13,500 architecture executions**, strictly merged and validated.
- SmolLM2 adaptive-evasion validation: **600** architecture executions.
- SmolLM2 exact repeat: **750** architecture executions; response/retrieval/outcome digest matches authoritative prior shard exactly.
- Fresh scale LLM study: **1,080/1,080 generations** across 90 fixed queries, 3 corpus sizes, and 4 modes.
- Adaptive benchmark: **200 cases**.
- Risk-controller ablation: **35,000 decision rows**.
- Encoder semantic validation: **200/200 pairs pass** preregistered threshold.
- Independent Qwen semantic judge: **19/19 source templates equivalent**.
- Property-based correspondence tests and frozen statistical analysis completed before finalization.

## Structural versus heuristic result

Definitions:

- **Structural control:** access/retrieval boundary determined by authenticated principal and ACL state, not prompt wording.
- **Heuristic control:** risk/anomaly action driven by request/session-derived features.
- **SBSR:** fraction of unauthorized adversarial requests for which no unauthorized record reaches context.
- **HDR:** shifted suspicious-request challenge rate divided by the primary challenge rate.

### Structural Boundary Survival Rate

For both SmolLM2 and Qwen2.5, pre-retrieval ACL and risk-aware architectures achieved:

- Primary unauthorized: **SBSR = 1.0** (`n=1750`).
- Held-out unauthorized: **SBSR = 1.0** (`n=150`).
- Adaptive unauthorized: **SBSR = 1.0** (`n=100`).

Prompt-only UCER remains **100%** for unauthorized requests; secured UCER is **0%** in the tested conditions.

### Heuristic Detection Retention

Fuzzy challenge rates:

- Primary suspicious: **88.0%**.
- Held-out suspicious: **16.67%** -> **HDR = 0.1894**.
- Adaptive suspicious: **18.0%** -> **HDR = 0.2045**.

Deterministic-rule challenge rates similarly collapse from **100%** primary to **16.0%** held-out and **17.0%** adaptive.

Adaptive search results:

- `n = 200`.
- Mean frozen injection-risk reduction: **0.4745**.
- Median reduction: **0.35**.
- Fraction with final injection risk exactly zero: **100%**.

The supported interpretation is that wording-sensitive heuristic recognition degrades sharply under distribution shift and explicit detector evasion, while the tested ACL boundary remains determined by policy state.

## Mechanistic ablation

On held-out/adaptive cases:

- Neutralizing `authorization_confidence` reduces unauthorized deny to **0%** for the fuzzy controller.
- Neutralizing `injection_risk` or `sensitivity` leaves unauthorized deny at **100%** while authorization confidence remains intact.

This identifies policy-derived authorization confidence as the mechanism preserving the tested confidentiality boundary under lexical evasion.

## Semantic-equivalence checks

Encoder (`sentence-transformers/all-MiniLM-L6-v2`, pinned revision):

- `n = 200`.
- Pass fraction: **1.0**.
- Median cosine similarity: **0.71756**.
- 5th percentile: **0.59240**.
- Minimum: **0.56921**.
- Frozen threshold: **0.50**.

Independent Qwen judge:

- `19/19` source-template representatives judged semantically equivalent.
- Automated checks are reported as automated, not as human annotation.

## Cross-model replication

### Primary

| Model / architecture | UCER | UDR | original k=2 ARSR |
|---|---:|---:|---:|
| Smol prompt-only | 100% | 0.286% | 79.0% |
| Smol pre-ACL | 0% | 0% | 64.0% |
| Smol risk-aware | 0% | 0% | 70.5% |
| Qwen prompt-only | 100% | 3.086% | 92.5% |
| Qwen pre-ACL | 0% | 0% | 90.5% |
| Qwen risk-aware | 0% | 0% | 91.5% |

### Controlled k=1

| Model | Prompt-only | Pre-ACL | Risk-aware |
|---|---:|---:|---:|
| SmolLM2 | 89.5% | 89.5% | 89.5% |
| Qwen2.5 | 92.0% | 92.0% | 92.0% |

For both models, paired ACL-minus-prompt bootstrap difference is **0.0**, 95% CI **[0.0, 0.0]** (20,000 draws). The earlier Smol k=2 utility gap is therefore attributed to distractor composition in this benchmark, not to authorization gating itself.

## Cluster-aware inference

Unauthorized context exposure, ACL minus prompt-only:

- Cluster unit: source prompt template.
- Clusters: **19**.
- Cluster-bootstrap difference: **-1.0**.
- 95% CI: **[-1.0, -1.0]**.
- All 19 nonzero templates favor ACL.
- Two-sided paired sign-test `p = 3.814697265625e-06`.

Smol canary unauthorized disclosure, ACL minus prompt-only:

- Difference: **-0.002857**.
- Cluster-bootstrap 95% CI: **[-0.01111, 0.0]**.
- Only one nonzero template; paired sign-test `p = 1.0`.

The five original Smol canary disclosures remain concentrated in one multi-step template and are not presented as five independent attack strategies.

## Fresh 1K/10K/100K four-mode scale generation study

90 fixed legitimate queries per cell, byte-identical compact post-authorization system policy prompt.

| Scale | Mode | Median candidates | ARSR | Median retrieval ms |
|---:|---|---:|---:|---:|
| 1K | unfiltered | 1,000 | 83.33% | 4.10 |
| 10K | unfiltered | 10,000 | 85.56% | 13.01 |
| 100K | unfiltered | 100,000 | 87.78% | 112.66 |
| 1K | ACL fixed | 200 | 87.78% | 2.15 |
| 10K | ACL fixed | 200 | 87.78% | 2.55 |
| 100K | ACL fixed | 200 | 87.78% | 2.36 |
| 1K | ACL proportional | 200 | 87.78% | 1.89 |
| 10K | ACL proportional | 2,000 | 88.89% | 3.91 |
| 100K | ACL proportional | 20,000 | 88.89% | 25.53 |
| 1K | target fixed | 1 | 94.44% | ~0.001 |
| 10K | target fixed | 1 | 94.44% | ~0.001 |
| 100K | target fixed | 1 | 94.44% | ~0.001 |

Supported interpretation:

- Global corpus growth alone does not imply generation-quality degradation in this sample.
- Retrieval workload follows the caller-visible candidate domain.
- Fixed-scope flatness is a construction control, not an emergent scalability discovery.
- Proportional authorization scope keeps `rho ≈ 0.2` while absolute candidate count grows 200 -> 2,000 -> 20,000, increasing retrieval latency.

## Exact empirical reproducibility

The planned SmolLM2 primary shard repeat produced:

- `n_rows = 750`, `n_cases = 250`.
- Reference digest: `1712cbee8501a509546345138a06ae5be984f0b9ada3daee9737b208613962c7`.
- Rerun digest: same.
- `exact_match = true`.

Timing-dependent fields were excluded by preregistration; response text is compared byte-for-byte after CSV parsing.

## Publication-safe novelty positioning after current literature search

Current literature through 2 September 2026 clearly includes authorization-aware RAG, metadata/RBAC filtering, adaptive extraction attacks, layered prompt-injection defenses, RAG-security taxonomies, and cryptographically provable secure RAG. Therefore the paper **does not claim novelty for pre-retrieval authorization itself**.

The defensible novelty is the combined empirical methodology and finding: directly contrasting structural policy-state enforcement with heuristic request-derived triage under held-out and adaptively optimized wording, using cluster-aware inference and cross-model replication, while separately formalizing fixed versus proportional caller-visible authorization scope at generation scale.

Claims explicitly excluded: fuzzy superiority, zero real-world leakage, cryptographic security, exhaustive verification of Python, universal model independence of output disclosure, or universal utility/scalability conclusions.

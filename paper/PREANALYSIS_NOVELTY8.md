# Novelty-8 strengthening: frozen preanalysis plan

**Frozen:** 2026-09-01 UTC, before inspection of any outputs from the new `Novelty-8 Strengthening Experiments` workflow.

This document fixes the hypotheses, primary metrics, analysis units, and interpretation rules for the final strengthening round. Existing audit-revision results remain prior evidence; the experiments launched from `novelty-8-structural-heuristic` are new validation evidence. We will not add experiments to chase statistical significance after inspecting these results.

## Conceptual framework

We distinguish **structural controls**, whose confidentiality boundary is determined by authenticated principal and authorization state before retrieval, from **heuristic controls**, whose decision depends on features extracted from request wording or session telemetry. Heuristic controls may tighten handling but are never allowed to widen the structural ACL.

We define **Structural Boundary Survival Rate (SBSR)** as the fraction of adversarial variants for which the explicit-target authorization boundary remains intact, operationalized here as pre-retrieval denial with no unauthorized context exposure for out-of-scope targets. We define **Heuristic Detection Retention (HDR)** as the adversarial/held-out suspicious-request detection rate divided by the corresponding in-distribution detection rate for the same frozen heuristic controller.

For scale, let corpus size be N, authorized search-space size for principal u be |A_u|, and authorization density be rho_u = |A_u|/N. We distinguish corpus growth with fixed |A_u| from corpus growth with proportional |A_u|.

## H1 - structural versus heuristic reformulation robustness

For explicit out-of-scope patient aliases, pre-retrieval ACL enforcement is expected to preserve the structural boundary under held-out and adaptively mutated wording. The primary structural endpoint is SBSR; unauthorized context exposure is reported separately. For authorized-suspicious requests, the frozen fuzzy and deterministic heuristic controllers are expected to lose detection under lexically disjoint/adaptively optimized reformulations. The primary heuristic endpoint is HDR, reported for both controllers without claiming fuzzy superiority.

The adaptive search objective is frozen lexical `prompt_injection_risk`; mutation operators and the conservative target/disclosure-intent search rubric are frozen in code before result inspection. Semantic equivalence is audited independently using `Qwen/Qwen2.5-1.5B-Instruct`; no adaptive case is deleted post hoc based on whether its outcome helps or hurts the hypothesis. Judge failures and disagreement are reported transparently.

## H2 - model-family replication

The core primary, controlled-k=1, and held-out benchmarks are repeated with `Qwen/Qwen2.5-0.5B-Instruct`, holding prompts, corpus, retriever, authorization policy, generation settings, and metrics fixed. The purpose is replication across a second lightweight instruct model, not equivalence testing between models. We will report all security and utility rates. Structural UCER behavior is expected to be architecture-defined; generation-dependent UDR and ARSR may differ by model and will be interpreted as such.

## H3 - controlled utility interpretation

The controlled k=1 experiment tests whether the earlier architecture utility gap persists after retrieval composition is held equal. Case-level paired differences are retained for continuity, but inferential emphasis is shifted to template-cluster-aware resampling/permutation where repeated prompt templates create dependence. Because the legitimate benchmark contains very few unique prompt templates, cluster-level uncertainty will be stated explicitly rather than hidden by case-level sample size.

## H4 - fixed versus proportional authorization scaling

The previously missing `acl_proportional` LLM cell is run for the same 90 paired legitimate queries at 1K, 10K, and 100K corpus sizes. The primary descriptive quantities are ARSR, target rank/hit behavior, retrieval time, authorized candidate count, and authorization density. We will distinguish effects of global corpus growth from effects of growth in the caller-visible authorization domain and will not interpret fixed-scope and proportional-scope conditions as interchangeable.

## H5 - risk-controller mechanism

A one-feature-at-a-time offline ablation neutralizes each feature in its benign direction: authorization confidence to 1, injection risk to 0, sensitivity to 0, and trust to 1. We report challenge rate and score shifts by benchmark category. This is a mechanistic sensitivity analysis, not a causal attribution claim about production traffic.

## H6 - implementation correspondence and reproducibility

Formal-correspondence unit tests are upgraded to randomized property-based tests using Hypothesis, including prefiltered-retrieval subset preservation, ACL non-expansion by risk control, phrasing invariance of explicit-target structural denial, and monotonicity under ACL narrowing. This remains randomized implementation testing rather than exhaustive model checking.

One original SmolLM2 primary shard is rerun with the same greedy generation settings. The confirmatory reproducibility endpoint is byte-identical concatenated response text plus equality of deterministic result fields. Any mismatch will be reported rather than suppressed.

## Statistical reporting rules

- Exact paired tests from the original audit remain available for comparability.
- Bootstrap confidence intervals are reported for key paired risk differences.
- Repeated-template dependence is addressed using template-cluster bootstrap intervals and template-level sign permutation where applicable.
- Effect sizes, counts, denominators, and uncertainty are reported regardless of p-value.
- No additional leak-target or prompt-template experiment will be introduced after result inspection solely to obtain significance.

## Claim boundaries

Even if all hypotheses are supported, the manuscript will claim evidence only for this synthetic healthcare-RAG benchmark, these authorization policies, these retrievers, and the evaluated small open instruct models. Zero observed unauthorized disclosure is not proof of zero real-world leakage. Structural ACL robustness for explicit aliases does not solve entity resolution, authentication, side channels, poisoned indexes, semantic authorization, or downstream application vulnerabilities.

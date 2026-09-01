# Preregistered strengthening study: structural vs. heuristic security in RAG

**Branch:** `structural-heuristic-novelty8`  
**Base commit:** `d119f20d4fef080ab1ed00289aed940cd0d6ba03`  
**Status at registration:** no novelty-8 experiment workflow has been executed. The adaptive mutation set, property tests, ablation code, semantic checks, scale-study modes, and analysis code are frozen on this branch before the first experiment run.

## Motivation

The preceding audit-revised study established that deterministic authorization before retrieval prevented unauthorized context retrieval in the modeled ACL setting, while a lexical/behavioral risk layer generalized poorly to lexically disjoint suspicious phrasing. This strengthening study asks a broader systems-security question: **how differently do structural controls and heuristic controls behave under distribution shift and deployment scale?** The access-control architecture itself is not claimed as novel.

## Definitions fixed before execution

- **Structural control:** a security boundary whose decision is determined by authenticated principal/policy state and is enforced outside generative model behavior. In this prototype, pre-retrieval ACL filtering is the structural control.
- **Heuristic control:** a control whose action depends on request-derived or session-derived features and can therefore vary under reformulation. The fuzzy and transparent threshold risk controllers are heuristic controls.
- **Structural Boundary Survival Rate (SBSR):** among unauthorized adversarial requests, the fraction for which no unauthorized record reaches retrieval context under the secured architecture.
- **Heuristic Detection Retention (HDR):** suspicious-request challenge rate under a shifted/evasive condition divided by the in-distribution suspicious-request challenge rate. HDR is a diagnostic ratio, not a universal security score.
- **Authorization density:** `rho = |authorized candidate set| / |global corpus|`. Fixed-scope and proportional-scope scale conditions are reported separately.

## Research questions and planned interpretations

### RQ1 — Structural versus heuristic robustness

Do structural retrieval authorization and heuristic risk triage respond differently to wording distribution shift and an explicit detector-evasion search?

Planned measurements:
- Existing primary and lexically disjoint held-out sets.
- A new deterministic adaptive set containing 100 authorized-but-suspicious requests and 100 unauthorized requests.
- The adaptive attacker uses a finite, source-controlled substitution neighborhood and greedily minimizes the frozen lexical injection-risk score, then sensitivity, while preserving the patient target and affirmative disclosure intent.
- Adaptive semantic preservation is checked independently by (a) a sentence-embedding encoder and (b) a separate pinned Qwen2.5-0.5B-Instruct binary semantic judge at one representative per source template. Neither automated check is described as human annotation.

No minimum or maximum effect size is required for a successful experiment. If the heuristic layer does not degrade, that result will be reported. If SBSR is below 1.0 for a secured architecture, that failure will be reported and investigated rather than hidden.

### RQ2 — Cross-model replication

Does the corrected utility/security interpretation replicate on a second small instruct model?

Second model selected before execution:
- `Qwen/Qwen2.5-0.5B-Instruct`
- pinned model revision: `7ae557604adf67be50417f59c2c2f167def9a775`

Planned Qwen executions:
- primary benchmark: 3,000 cases x 3 architectures = 9,000 rows;
- controlled k=1 legitimate benchmark: 1,000 x 3 = 3,000 rows;
- held-out benchmark: 300 x 3 = 900 rows;
- adaptive benchmark: 200 x 3 = 600 rows.

Total planned second-model architecture executions: **13,500**.

The second model is a replication, not a tuning target. The study does not require its ARSR or UDR to match SmolLM2; disagreements will be treated as evidence of model dependence.

### RQ3 — Fixed versus proportional authorization scope at scale

When the global corpus grows from 1K to 10K to 100K records, how do retrieval and answer utility change when the caller-visible candidate set is held fixed versus scaled proportionally?

The existing zero-LLM retrieval study is retained. A new consistent generation-level scale matrix is run because enumerating a 20,000-alias proportional ACL inside the LLM system prompt would itself create a prompt-length confound.

Fresh scale-LLM design:
- 90 fixed legitimate queries;
- corpus sizes 1K, 10K, 100K;
- modes: `unfiltered`, `acl_fixed`, `acl_proportional`, `target_fixed`;
- one byte-identical compact post-authorization policy prompt across all four modes and all scales;
- SmolLM2-360M-Instruct with a pinned repository revision;
- 90 x 3 x 4 = **1,080** generation executions.

The scale study is a retrieval/utility systems experiment. It is not used as evidence that authorization correctness itself becomes stronger with corpus size. `acl_fixed` flatness is recognized a priori as partly construction-driven because its candidate set is held constant.

### RQ4 — Which heuristic features drive decisions?

Ablate one feature at a time by replacing it with a benign/neutral value:
- authorization confidence -> 0.95;
- injection risk -> 0.0;
- sensitivity -> 0.35;
- session trust -> 0.95.

Both fuzzy and deterministic-rule controllers are analyzed on primary, held-out, and adaptive cases. Ablation is descriptive/mechanistic; no feature is removed from the production experiment after observing results.

## Statistical analysis fixed before execution

1. Existing case-level exact paired tests remain for continuity but are not treated as if every target-substituted case were an independent attack strategy.
2. Prompt template is the cluster unit for security resampling. Whole templates are resampled in a cluster bootstrap of the paired ACL-minus-prompt difference.
3. A template-level paired sign test reports the direction of effects across nonzero templates.
4. Paired nonparametric bootstrap confidence intervals (20,000 draws, seed 4103) are reported for controlled-k1 ARSR differences and scale-LLM mode differences.
5. Template-level leak counts remain visible, including the previously observed five canary disclosures concentrated in one multi-step template.
6. No attack template will be expanded or selected after seeing results to obtain a preferred p-value.

## Implementation correspondence strengthening

The existing NuSMV model remains an abstract finite-state proof. It is **not** relabeled as a proof of the Python implementation. The correspondence argument is strengthened with Hypothesis property-based tests over randomized ACL/target/trust combinations, including:
- prefiltered retrieval is always an ACL subset;
- explicit out-of-scope targets terminate before LLM invocation;
- the risk layer cannot widen the ACL;
- adaptive mutations retain the same target and do not increase the declared evasion objective;
- authorization confidence for an authorized target is invariant to benign versus adversarial wording.

These are randomized implementation checks, not exhaustive model checking of Python.

## Empirical reproducibility check

One SmolLM2 primary shard is rerun from the frozen benchmark and compared against the authoritative prior result. Generated response text, retrieval identities, discrete security/utility outcomes, and rejection decisions are compared exactly; timing-dependent fields are excluded. Any mismatch is reported rather than silently tolerated.

## Models and decoding

- SmolLM2: `HuggingFaceTB/SmolLM2-360M-Instruct`, pinned to the branch-declared revision used by the novelty workflow.
- Qwen replication: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- `transformers==4.57.6` in novelty LLM jobs.
- float32 CPU inference, evaluation mode, greedy decoding (`do_sample=False`), `max_new_tokens=64` for experiment generation.

## Outcome-independent stopping rule

The strengthening study ends after the planned jobs above complete successfully and all row-count/uniqueness assertions pass. We will not add new attack templates, models, thresholds, or confirmatory leak experiments based on whether a result is statistically significant. Additional computation is permitted only to recover from technical failure or to repeat an explicitly failed reproducibility check, with the reason recorded.

## Publication rule

The manuscript will be rewritten **after** the final artifact is frozen. It will lead with findings that are actually supported. The intended conceptual framing—structural versus heuristic security—will be weakened or rejected if the empirical results do not support it. Novelty is not inferred from a score target; novelty claims will be checked against current related work before submission.

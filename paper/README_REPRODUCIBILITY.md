# HealthRAG novelty-8 reproducibility bundle

This directory freezes the preregistered structural-versus-heuristic strengthening study, its final publication-safe interpretation, and the IEEEtran manuscript source.

## Authoritative strengthening run

- Preregistered experiment commit: `7b9c0cbcb188da74c08ee4281061fbdb7a7b1f11`
- Workflow: **Structural vs Heuristic Security - Novelty 8 Strengthening**
- Run: `33565992867`, conclusion **success**
- Final artifact: `healthrag-structural-heuristic-novelty8-artifacts`
- Artifact ID: `9850437939`
- GitHub artifact SHA-256: `a686d96614a57a5e96845abff191491c761fd3ad509b342e3c8b0a0640841b90`

The preregistration is `NOVELTY8_PREREGISTRATION.md`. The final statistical/results record is `FINAL_RESULTS.md`. Experiment code, attacks, model selection, thresholds, analysis units, bootstrap settings, and stopping rule were frozen before the run; later commits update only publication-facing material and do not retroactively alter the executed study.

## Completed evidence

- Qwen2.5-0.5B-Instruct: 13,500 strictly validated architecture executions.
- SmolLM2 adaptive validation: 600 executions.
- SmolLM2 exact rerun audit: 750 executions, exact outcome/retrieval/response digest match.
- Fresh four-mode scale generation: 1,080 rows, including `acl_proportional` at 1K/10K/100K.
- Adaptive-evasion benchmark: 200 cases.
- Encoder semantic-equivalence check: 200 pairs; independent Qwen judge: 19 templates.
- Risk-feature ablation: 35,000 decision rows.
- Template-cluster inference, 20,000-draw bootstrap intervals, authorization-density diagnostics, and property-based Python correspondence tests.

## Manuscript source

`healthrag_final.tex` is the IEEEtran entry point; the final paper content is in `final/`. The final validated title is **When Heuristics Fail but Boundaries Hold: Structural versus Heuristic Security in Healthcare RAG**.

A standard TeX Live installation with `IEEEtran`, `booktabs`, `multirow`, `amsmath`, `microtype`, `array`, and `enumitem` is sufficient. The bibliography is manual; BibTeX is not required.

Build from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error healthrag_final.tex
pdflatex -interaction=nonstopmode -halt-on-error healthrag_final.tex
```

The externally validated deliverable is exactly **8 pages**. Every rendered page was visually inspected for clipping, overlap, broken glyphs, and layout defects.

## Frozen interpretation rules

1. Pre-retrieval authorization itself is prior art; the manuscript does not claim otherwise.
2. UCER is retrieval-time unauthorized context exposure; UDR is generated canary disclosure. They are distinct outcomes.
3. Structural Boundary Survival Rate (SBSR) is reported as an observed benchmark property under the stated ACL threat model, never as proof of zero real-world leakage.
4. Heuristic Detection Retention (HDR) is a diagnostic of suspicious-request challenge behavior under distribution shift, not a universal detector score.
5. The preregistered adaptive search drives the frozen lexical injection-risk feature to zero for all 200 adaptive cases; semantic preservation is checked automatically and is not described as human annotation.
6. The original Smol k=2 utility gap is interpreted as a distractor-composition confound because controlled k=1 produces identical ARSR across architectures; Qwen independently reproduces that within-model equality.
7. Case-level security counts are accompanied by source-template cluster inference to address pseudoreplication.
8. Fixed authorization scope, proportional authorization scope, and global corpus growth are separated explicitly. Fixed-scope flatness is a construction control.
9. NuSMV checks the abstract state machine; property-based tests strengthen Python correspondence but do not constitute exhaustive implementation proof.
10. Claims of fuzzy superiority, cryptographic confidentiality, exhaustive formal verification, universal model-independent output disclosure, or universal scaling behavior are excluded.

# HealthRAG final reproducibility bundle

This directory freezes the manuscript and publication-facing interpretation for the audit-revised Healthcare RAG authorization study.

## Authoritative GitHub state

Core research:
- Source commit: `b0ebc87ac8b14fc878272f29dfc11c42ff3c588f`
- Workflow: **Research Reproduction - Audit Revision**
- Run: `33523950201`, attempt 2, conclusion **success**
- Artifact: `healthrag-audit-revision-artifacts`, ID `9816904194`
- Artifact SHA-256: `bfc003e5742a651eb9b412e646271a89d170add7a489a895b7c5a423239bcf52`

Scalability:
- Workflow: **HealthRAG 100K Retrieval Scalability**
- Run: `33523949995`, conclusion **success**
- Artifact: `healthrag-scalability-final-artifacts`, ID `9812557736`
- Artifact SHA-256: `6683ebf6697a146ae84ec35483dddc4d69fc900b2d592aeb67ff708a469de96a`

The final manuscript/documentation branch is based exactly on the validated source commit above. Manuscript-only commits do not modify the experiment implementation that produced the authoritative artifacts.

## Manuscript source

`healthrag_final.tex` is the IEEEtran entry point and the `sections/` directory contains the paper. Figures are generated directly by TikZ/PGFPlots from the frozen observed values, so no external binary figure assets are required.

A standard TeX Live installation with `IEEEtran`, `tikz`, `pgfplots`, `booktabs`, and related common packages is sufficient. The final source uses a manual IEEE bibliography, so BibTeX is not required.

Example:

```bash
pdflatex healthrag_final.tex
pdflatex healthrag_final.tex
pdflatex healthrag_final.tex
```

The validated build has exactly **8 US-Letter pages**. The delivered PDF in the external reproducibility bundle was additionally checked for openability, encryption, scan status, clipping, overlap, and broken glyphs.

## Interpretation rules frozen for publication

1. UCER is unauthorized **retrieval-time context exposure**; UDR is narrow **generated canary disclosure**. They are distinct outcomes.
2. The original k=2 ARSR difference was confounded by retrieved distractor composition. Controlled k=1 gives 89.5% ARSR for all three architectures.
3. Structural ACL denial is phrasing-invariant in the held-out set; fuzzy/deterministic suspicious-request heuristics are not.
4. Fixed-ACL and target-fixed scale conditions are constant-candidate/context controls by construction; their flatness is not an emergent scaling discovery.
5. `acl_proportional` is evaluated at retrieval level only. No generation-level proportional-ACL claim is made.
6. NuSMV exhaustively checks the abstract state machine; Python correspondence tests are example-based and do not constitute exhaustive implementation proof.
7. Case-level attack statistics are accompanied by template-level reporting because repeated prompt templates reduce the effective number of independent attack strategies.

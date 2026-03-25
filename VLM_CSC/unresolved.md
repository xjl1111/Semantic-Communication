# Unresolved Items (Formal Reproduction)

## 1) LEMON baseline checkpoint/source
- Status: unresolved.
- Reason: no fully verified, paper-consistent public checkpoint/source has been integrated in this repo.
- Policy: formal Fig.7 reproduction must not silently replace LEMON with another VLM.

## 2) Exact training hyperparameters from paper
- Status: partially resolved.
- Reason: some implementation details (e.g., full schedule nuances and preprocessing corner cases) are inferred engineering choices.
- Policy: all inferred choices must be documented in `assumptions.md` and experiment notes.

## 3) Full-scale compute budget for final numbers
- Status: unresolved in current workspace runs.
- Reason: current outputs primarily validate pipeline and strict-mode behavior, not final long-run convergence numbers.
- Policy: do not claim final paper-level reproduction until full training/eval is completed.

## 4) Dataset protocol edge-case parity
- Status: partially resolved.
- Reason: continual split protocol and cache flow are implemented, but paper-specific corner handling may still need audit.
- Policy: keep split generation deterministic and versioned with run metadata.

# Reproduction Report (Taskbook Naming)

This file mirrors the active progress tracked in existing reports, while keeping taskbook-required naming.

## Scope
- Target: VLM-CSC style reproduction pipeline with CKB (BLIP/SD), MED, NAM, staged training, and Fig.7~Fig.10 scripts.
- Current repo state emphasizes: strict formal mode boundaries, executable integration points, and traceable assumptions.

## What is implemented
- Project structure, configs, datasets, transforms, continual split utilities.
- Core modules: `kb_blip`, `kb_sd`, `kb_alt_vlm`, `semantic_codec`, `channel_codec`, `med`, `nam`, losses, trainers.
- Script chain: smoke, figure scripts, run-all flow, cache preparation and verification scripts.
- Formal mode guardrails: default disable proxy/fallback in formal figure scripts.

## Key correctness fixes already applied
- Semantic decoder training uses shifted teacher forcing to avoid label leakage.
- Formal scripts require explicit opt-in for proxy/fallback path.
- LEMON unresolved state is explicit and non-silent.

## Current limitations
- LEMON formal baseline remains unresolved.
- Existing figure outputs are mainly pipeline/proxy validation unless explicitly marked otherwise.
- Full-scale long-run training for paper-level numeric match is pending.

## Recommended next formal steps
1. Prepare/verify real checkpoints and dependencies (`verify_blip.py`, `verify_sd.py`, `verify_ram.py`).
2. Run `prepare_text_cache.py` on target datasets in strict mode.
3. Execute staged real training (A/B/C) and then figure evaluation scripts.
4. Record all run metadata and update unresolved items.

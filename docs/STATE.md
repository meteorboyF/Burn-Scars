# STATE: single source of truth (update every session)

Last updated: 2026-10-11 (session 1)
Current phase: **Phase A complete → waiting at STOP POINT A** for the user's go-ahead to Phase B.

## Topic statement (FROZEN 2026-10-11; changes need a DECISIONS entry)

- **Problem.** The official HLS Burn Scars split is random (2/3 train, 1/3
  val). A third-party repackaging reports that MGRS tiles are shared across the
  split. Published results rank models inconsistently: PANGAEA reports a UNet
  best at 84.51 mIoU on the official split, while on IBM's 5-km-buffered splits
  (arXiv 2504.17397) Prithvi-EO-2.0 LoRA reaches 93.33 test mIoU vs UNet 89.34.
  Nobody has separated the effect of the split from the effect of the protocol.
  *(All numbers in this paragraph: UNRESOLVED until verified in docs/LITERATURE.md.)*
- **Hypothesis H1.** Per-chip IoU decreases with distance to the nearest
  training chip, after controlling for burned fraction and year. Random CV
  inflates scores relative to geography-grouped CV enough to change model
  rankings.
- **Intended contribution.** A matched random-vs-grouped CV comparison on the
  same chip pool with equal training sizes, measuring inflation, distance
  dependence and rank stability across model families; plus a released,
  versioned leakage-safe split and audit script.
- **Kill criterion K1** (decided before looking at results): fewer than ~5% of
  official-val chips lie within 5 km of any official-train chip, AND there are
  no same-footprint repeats → the topic stops. Operational definition:
  docs/REALITY_CHECK.md K1.

## Standing constraints (user overrides beat the playbook)

- Laptop CPU only for local work (8 cores, 14 GB RAM). No local GPU use.
- GPU work: Google Colab Pro only. No Kaggle. Colab units are shared with other
  projects: **nothing runs on Colab without the user's explicit approval of a
  specific allocation in-session**, recorded as APPROVED in COMPUTE_LEDGER.md
  (gpu-tier tools enforce this via `BURNSCARS_ALLOCATION`).
- No manual annotation, labelling, human validation or new data collection.
- Out of scope: conformal/FNR (T-2), VLMs/LocateAnything, spectral-reliance
  diagnostics, Sen1Floods11 (licence unresolved at source).
- Every number labelled VERIFIED / ESTIMATED / UNRESOLVED.

## Status

| Item | Status |
|---|---|
| Skeleton (07_BOOTSTRAP_PROMPT.md) | done, 45 tests pass on CPU |
| Pre-commit hook | live (`core.hooksPath=.githooks`), fires on forbidden commit |
| H1 / K1 / K2 pre-registration | REGISTERED 2026-10-11, before any data download |
| Literature protocol | not started (Phase B, live web search) |
| Data licence / download / profile | not started (Phase B) |
| Footprint audit / K1 gate | not started (Phase B) |
| Compute spent | 0 GPU-h, 0 CU (VERIFIED) |

## Next steps (Phase B, CPU only, after user go-ahead)

1. Read licence at the pinned HF revision → DATA.md; then scripted download + SHA256.
2. Literature protocol with live search (verify every number in the topic statement).
3. Profile → footprint audit → K1 gate numbers.
4. Metric replica + tests; NBR / all-unburned controls under both splits.
5. Split builders + IBM test quarantine (`--final` gate).
6. STOP POINT B report incl. 06_CHECKLISTS.md Phase 0 gate item by item.

## How to resume in a fresh session

```
cd ~/Paper/Burn && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.lock
scripts/install_hooks.sh
.venv/bin/python -m pytest
```
Then read this file, REALITY_CHECK.md and DECISIONS.md, in that order.

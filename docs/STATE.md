# STATE: single source of truth (update every session)

Last updated: 2026-10-11 (session 1, end of Phase B)
Current phase: **Phase B complete → waiting at STOP POINT B (K1 gate)**. K1 did
NOT fire. Phase C must not start until the user confirms the gate and decides
D-014 (CV pool) and the E-01 amendment.

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
| Skeleton + hook | done; 88 tests pass on CPU (test_splits ~3 min) |
| H1 / K1 / K2 | REGISTERED 2026-10-11 before download. **K1: does NOT fire** (40.5% within 5 km; 6 repeats) |
| Data | HLS Burn Scars @1864285 downloaded, SHA256 verified; profiled (docs/DATA.md) |
| Footprint audit | docs/AUDIT_REPORT.md (generated), results/audit/ |
| TACO 124/365 claim | reproduced exactly |
| Metric replica | `burnscars.eval.metrics`, 8 hand-computed tests pass; NOT yet validated against a published number (needs model predictions) |
| NBR / all-unburned controls | results/baselines/nbr_v1/; NBR random-vs-grouped gap ≈ 0 |
| Splits | splits/v1 (official, IBM train/val, random+grouped 5-fold on pools 804 and 684), pinned |
| IBM test quarantine | enforced in code + test; reads logged in docs/QUARANTINE_LOG.md |
| Literature | docs/LITERATURE.md; verdict CROWDED → rescoped; refs.bib 39 entries |
| Crop dataset | georeferenced (EPSG:5070); NOT downloaded |
| Compute spent | 0 GPU-h, 0 CU (VERIFIED) |

## Open decisions for the user (STOP B)

1. Confirm the K1 gate → Phase C.
2. D-014: CV pool 804 (as briefed) or 684 (keeps IBM test a clean holdout; recommended).
3. E-01: approve adding NBR per-chip IoU as a difficulty covariate in H1(c)/K2.
4. H1 prediction band (agent-proposed, Phase A) — confirm or amend.
5. Venue/template (D-005); repo visibility vs double-blind (D-006).

## Next steps (Phase C, after confirmation; code + CPU smoke only)

UNet (from scratch) and Prithvi-EO-2.0-300M frozen-encoder + light decoder
(licences first); stats module; --smoke trainers with kill/resume test; Colab
runner; grid costing; then request a specific pilot allocation (STOP C).

## How to resume in a fresh session

```
cd ~/Paper/Burn && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.lock
scripts/install_hooks.sh
.venv/bin/python -m pytest
```
Then read this file, REALITY_CHECK.md and DECISIONS.md, in that order.

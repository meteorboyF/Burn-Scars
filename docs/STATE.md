# STATE: single source of truth (update every session)

Last updated: 2026-10-11 (session 1, STOP POINT C)
Current phase: **Phase C code + CPU work done → waiting at STOP POINT C** for the
user's explicit approval of allocation A-001 (pilot P-01). NOTHING has run on
Colab. 0 GPU-h, 0 CU spent (VERIFIED).

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
| K1 | did NOT fire (STOP B); gate confirmed by user |
| Decisions | D-014 → 684 pool + as-published arm; E-01 approved; H1-v2 rule pre-registered @4b6344c |
| Leak guard | official-trained models can never be scored on IBM test (code + tests) |
| Metric validation | ours ≡ TerraTorch-native on published checkpoint (IBM val 82.32/89.75); card 84.28/90.95 NOT reproduced (D-020) |
| Crop dataset | header-only audit: 69.5% of val chips edge-adjacent to train, 0 overlap (D-021); scope decision pending |
| Phase C code | UNet + frozen-Prithvi decoder, resumable trainer (deadline guard, RAM cache), feature extractor, stats (paired cluster bootstrap, H1-v2 rule, rank reversal, R0/R1/R2), grid generator, Colab runner — all CPU-smoked |
| Resume | correct; bitwise identity after SIGKILL not guaranteed (D-019) |
| Costing | COMPUTE_LEDGER.md (ESTIMATED from measured FLOPs; GPU rate assumed) |
| Tests | 104 (1 documented xfail) |

## Open decisions for the user (STOP C)

1. Approve (or change) A-001: pilot P-01 on A100 ≤ 2.0 GPU-h (or L4 ≤ 5.5).
2. Crop dataset in scope (second dataset) or not.
3. Optional: run Prithvi feature extraction on the laptop CPU (~94 min, 0 CU;
   needs the 1.33 GB base-weights download).
4. Still open from earlier: venue/template (D-005), repo visibility (D-006).

## After approval (the agent then)

Marks A-001 APPROVED in COMPUTE_LEDGER.md (commit), the user runs runner.ipynb
cells 1–9 with `start_allocation('A-001', <hours>)`, outputs land on Drive; the
agent analyses K2 on CPU from the manifests and reports.

## How to resume in a fresh session

```
cd ~/Paper/Burn && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.lock
scripts/install_hooks.sh
.venv/bin/python -m pytest
```
Then read this file, REALITY_CHECK.md and DECISIONS.md, in that order.

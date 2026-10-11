# COMPUTE LEDGER

Colab Pro only (no Kaggle, no local GPU). Colab units are shared with other
projects. **An allocation row is APPROVED only when the user approves that
specific allocation in-session.** gpu-tier tools read this table
(`burnscars.hardware.approved_allocations`). Cost = GPU-hours × the CU/h rate
displayed in the Colab session at run time; no rate is assumed here.

## Allocations

| Id | Purpose | GPU-h requested | GPU type | CU/h (from Colab UI) | Status | Approved on |
|---|---|---|---|---|---|---|
| A-001 | Pilot P-01 (UNet: 3 tuning + 5 random + 5 grouped + 1 as-published, seed 0) | 2.0 (A100) or 5.5 (L4) | A100 preferred | read in session | REQUESTED (STOP C) | — |

## Measured inputs (VERIFIED, laptop i5-1035G1, 4 threads, 2026-10-11)

| Quantity | Value |
|---|---|
| UNet-R34 fwd+bwd, 512², per chip | 189.0 GFLOP (torch FlopCounterMode); 2.45 s CPU |
| LightDecoder fwd+bwd, per chip | 111.0 GFLOP; 1.59 s CPU |
| Prithvi-EO-2.0-300M encoder fwd, per chip (512²) | 622.3 GFLOP; 6.99 s CPU |
| Chip read + normalise + D4 | 0.059 s CPU (→ RAM cache on GPU runs) |
| Cached feature load | 0.012 s |
| Published-checkpoint inference (full model) | 9.2 s/chip mean on CPU (IBM val, 160 chips) |

No GPU throughput has been measured (no Colab use). Everything below is ESTIMATED.

## Costing (ESTIMATED)

Method: chip-iterations (one fwd+bwd of a 512² chip) × measured GFLOP ÷ an ASSUMED
effective GPU rate — A100 25–50, L4 8–15, T4 3–5 TFLOP/s (fp32/TF32, no AMP) —
then ×1.2 overhead (session setup, data restore, eval, Drive copies) ×1.5
contingency (= ×1.8). Epochs fixed at 60 (D-017). Reproduce with the arithmetic
in this session's STOP C report; the rate assumption is the dominant uncertainty
and the pilot replaces it with a measurement.

**Pilot P-01** (UNet only): 3 tuning runs (60 ep × 410 inner-train chips + per-epoch
inner-val) + 10 CV runs (60 × 547) + 1 as-published (60 × 540) = 442,620
chip-iters = 83.7 PFLOP.

| GPU | compute | ×1.8 |
|---|---|---|
| A100 | 0.46–0.93 h | **0.8–1.7 GPU-h** |
| L4 | 1.55–2.90 h | **2.8–5.2 GPU-h** |
| T4 | 4.65–7.75 h | 8.4–13.9 GPU-h |

**Full grid** (both families), per model 37.5 full-run equivalents: main 11 (5
random + 5 grouped + 1 as-published), +4 seeds (2 extra seeds × fold 0 × 2 regimes),
6 tuning runs (≈5.0 eq.), 10 same-tile-pair-removal runs (upper bound), training-
size curve 25/50% × 2 regimes × 5 folds (7.5 eq.); plus one-off feature extraction
(804 × 622 GFLOP). Total 369.7 PFLOP. The pilot's runs are part of this grid.

| GPU | compute | ×1.8 |
|---|---|---|
| A100 | 2.1–4.1 h | **3.7–7.4 GPU-h** |
| L4 | 6.8–12.8 h | **12.3–23.1 GPU-h** |
| T4 | 20.5–34.2 h | 37.0–61.6 GPU-h |

Prior rough guess (unmeasured): pilot ≈ 3, full 22–34 GPU-h — consistent with the
L4 band, ~3–5× above the A100 band. Option at 0 CU: run the Prithvi feature
extraction on the laptop CPU (804 × 6.99 s ≈ 94 min, measured rate) — needs the
1.33 GB base-weights download first; not done without approval.

Cost in CU = GPU-hours × the CU/h shown in the Colab session (not assumed here).

## Runs

| Date | Allocation | Config | Machine | Wall clock | GPU-h | CU | What it bought |
|---|---|---|---|---|---|---|---|

Phase B CPU work (laptop, 2026-10-11): profile 2 min, audit 2 min, splits 1.5 min,
NBR baselines 5 min. Not accelerator time; listed for completeness.

Spent to date: 0 GPU-h, 0 CU (VERIFIED, 2026-10-11).

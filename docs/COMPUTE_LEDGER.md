# COMPUTE LEDGER

Colab Pro only (no Kaggle, no local GPU). Colab units are shared with other
projects. **An allocation row is APPROVED only when the user approves that
specific allocation in-session.** gpu-tier tools read this table
(`burnscars.hardware.approved_allocations`). Cost = GPU-hours × the CU/h rate
displayed in the Colab session at run time; no rate is assumed here.

## Allocations

| Id | Purpose | GPU-h requested | GPU type | CU/h (from Colab UI) | Status | Approved on |
|---|---|---|---|---|---|---|
| (none yet) | | | | | | |

## Grid estimate (ESTIMATED, unmeasured)

Prior rough guess carried from the user's brief, not yet backed by a measured
pilot: pilot ≈ 3 GPU-h; full grid ≈ 22–34 GPU-h (×1.5 contingency already
included). Replaced by a costed grid at STOP POINT C from measured CPU smoke
throughput + model FLOPs, then by the measured pilot.

## Runs

| Date | Allocation | Config | Machine | Wall clock | GPU-h | CU | What it bought |
|---|---|---|---|---|---|---|---|

Spent to date: 0 GPU-h, 0 CU (VERIFIED, 2026-10-11).

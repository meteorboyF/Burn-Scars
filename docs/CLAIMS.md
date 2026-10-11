# CLAIMS ledger

No claim enters the manuscript unless its status is `supported` (see
06_CHECKLISTS.md). Evidence = manifest ids under `results/manifests/`.

| # | Claim (one sentence, with numbers) | Evidence (manifest ids) | Status |
|---|---|---|---|
| C1 | Official split leaks geography: 107/264 (40.5%) val chips within 5 km of a train chip, 85 overlap one, 6 same-footprint repeats, 124/365 tiles shared; IBM split: none within 10 km but 55/120 test chips share a tile with IBM-train | results/audit/manifest.json | supported (descriptive; needs no seeds) |
| C2 | Random CV inflates burned IoU by Δ ± CI over grouped CV at equal train size, per model family | — | planned (Phase C) |
| C3 | Per-chip IoU falls with distance to training set, controlling for burned fraction and year | — | planned (Phase C) |
| C4 | Model ranking is / is not stable across regimes (Kendall τ) | — | planned (Phase C) |
| C5 | A non-learning NBR rule shows no random-vs-grouped gap (+0.08 / −0.11 pts burned IoU, fold SD 4–5) | results/baselines/nbr_v1/manifest.json | supported for NBR; the contrast with learned models is Phase C |
| C6 | Our metric + preprocessing equal TerraTorch's own pipeline on the published checkpoint (IBM val 82.32 / 89.75); the card's 84.28 / 90.95 is not reproduced with TerraTorch 1.2.13 | results/metric_validation/*/manifest.json | supported (validation of metric only; card gap unexplained) |
| C7 | The crop-classification split leaks by adjacency: 69.5% of val chips touch a training chip, with zero area overlap | results/crop_audit/manifest.json | supported (descriptive) |

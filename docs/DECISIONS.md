# DECISIONS (numbered, append-only, with reasoning)

**D-001 (2026-10-11) Python 3.12 in a uv venv.** Colab has shipped 3.12; matching
it avoids wheel drift between laptop and runner. The exact Colab Python version
on the day of the first run is UNRESOLVED and gets checked in the runner
(`print_sha` cell output) before any allocation.

**D-002 (2026-10-11) Package `burnscars`, layout per 02_CODEBASE_AND_INFRA.md.**
`src/burnscars/{data,models,train,eval,figures}`, tools take `--config` and
`--smoke`, tests run without data. Run manifests of cited results are committed
under `results/manifests/` (small JSON); heavy outputs live on Drive. Reason: a
committed manifest lets `tests/test_repo_hygiene.py` enforce config immutability
(config hash must still match) and lets figures/tables build from the repo alone.

**D-003 (2026-10-11) Pre-commit hook.** Tracked in `.githooks/pre-commit`
(Python), activated with `scripts/install_hooks.sh` (`core.hooksPath`). Refuses:
anything under root `/data/`; raster/array/weights/archive/raster-image
extensions (`.tif .npy .pt .safetensors .tar.gz .png .jpg` …); any file > 1 MB;
PDFs outside `paper/figures/` and `docs/figures/`; notebooks with outputs or
execution counts; credential filenames; GitHub/HF/AWS/Google/Slack/private-key
token patterns. PNG/JPG are refused because paper figures are vector PDF only.
No override flag by design. Tested in `tests/test_precommit_hook.py`.

**D-004 (2026-10-11) Library licences** (read from installed package metadata,
2026-10-11). All permissive; none restricts publishing results or releasing code
under MIT.

| Package | Version | Licence |
|---|---|---|
| rasterio | 1.5.2 | BSD-3-Clause |
| pyproj | 3.8.0 | MIT |
| shapely | 2.2.0 | BSD-3-Clause |
| numpy | 2.5.4 | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) |
| pandas | 3.0.6 | BSD-3-Clause |
| scipy | 1.18.1 | BSD-3-Clause |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| matplotlib | 3.11.2 | Matplotlib licence (PSF-based) |
| PyYAML | 6.0.3 | MIT |
| pytest | 9.1.1 | MIT |

Note: rasterio/pyproj wheels bundle GDAL/PROJ (MIT/X-style). Every library added
later gets a row here before use (Phase B: huggingface_hub; Phase C: torch,
TerraTorch, segmentation-models-pytorch, Prithvi weights licence).

**D-005 (2026-10-11) Paper template: TODO.** `paper/main.tex` is a plain
`article` placeholder. Choose IEEEtran (e.g. IEEE TGRS / JSTARS / GRSL) vs
elsarticle (e.g. Remote Sensing of Environment, ISPRS J., JAG) after the Phase 0
venue read; switch before Phase 1 writing so length limits are real.

**D-006 (2026-10-11) The remote repo is PUBLIC from the first push.**
Consequences: no data, no weights, no tokens ever (hook enforces); everything
pushed is effectively published. If the chosen venue is double-blind, the repo
name/history de-anonymises; revisit at venue choice (D-005) — option: private
anonymised mirror for review.

**D-007 (2026-10-11) Compute policy (user override of the playbook).** No local
GPU; no Kaggle; Colab Pro only, per-allocation explicit approval. Enforced in
code: gpu-tier tools refuse non-smoke runs unless `BURNSCARS_ALLOCATION` names
an `A-###` row marked APPROVED in COMPUTE_LEDGER.md (`burnscars.hardware`).
The agent never edits a row to APPROVED without the user's in-session approval.

**D-008 (2026-10-11) H1/K1/K2 registered in Phase A,** before any data download,
with operational definitions (edge-to-edge footprint distance, IoU ≥ 0.9 for
"same footprint", dataset-level burned IoU for K2). Earlier than Phase B
requires; strictly safer for pre-registration.

**D-009 (2026-10-11) Distance CRS = EPSG:5070 (CONUS Albers equal-area), with a
geodesic cross-check.** All 804 chips lie in CONUS (UTM 10–18), inside EPSG:5070's
area of use. Albers preserves area (needed for overlap fractions), and its
distance distortion across CONUS is ~1–2% (≤100 m at 5 km), negligible against a
5 km threshold. Verified: overlap IoU from native UTM bounds vs EPSG:5070 agrees
to 8e-7 on 276 same-zone pairs; p_near at 5 km is identical (40.5%) under the
geodesic re-measurement of each nearest pair. Footprints are densified (16
points/edge) before reprojection.

**D-010 (2026-10-11) Libraries.** No new library was added in Phase B: the
downloader uses `urllib` (stdlib) against pinned HF `resolve/<sha>` URLs, so
`huggingface_hub` is not needed.

**D-011 (2026-10-11) Metric protocol.** Primary: burned-class IoU and 2-class
macro mIoU from a confusion matrix pooled over all held-out pixels, −1 ignored —
the convention PANGAEA and TerraTorch use (docs/LITERATURE.md). Per-chip IoU
(mean over chips, NaN-excluded) is reported as a separate, labelled metric and
is the response variable of the H1 distance regression. Burned-class IoU is the
pre-registered primary effect metric (mIoU is diluted by unburned IoU ≈ 96–98).

**D-012 (2026-10-11) "Random" CV = stratified random 5-fold** (chips sorted by
burned fraction, each block of 5 shuffled across folds), because the grouped
regime is balanced on count and burned fraction; this makes the two regimes
differ only in spatial structure. Train sizes equalised by seeded subsampling to
the minimum over both regimes and all folds (643 for 804 chips, 547 for 684).
Inner folds: 4 per outer fold, same regime logic. Seed 20261011.

**D-013 (2026-10-11) Split release v1** in `splits/v1/`, hashes pinned in
`tests/test_splits.py`; regenerable byte-identically from committed
`results/profile/chips_profile.csv` + IBM train/val lists. A change is v2, never
an edit.

**D-014 (DECIDED 2026-10-11 by user: 684 pool) CV pool: all 804 vs IBM train+val 684.**
The brief says "5-fold CV over the pooled 804" AND "IBM's 120-chip test set is
quarantined". Using 804 puts the IBM test chips into CV training and
hyper-parameter tuning, so a later `--final` evaluation on IBM test would no
longer be a clean holdout for our tuned configs. Both pools are built (v1).
Recommendation: CV on **684 (IBM train ∪ val)**, keep IBM test as a
true holdout for one final cross-check against the published IBM-split numbers.
Cost: ~15% fewer chips per fold (train 547 vs 643).

**D-015 (2026-10-11) Dirty-tree definition.** A manifest's git SHA is `-dirty` if
tracked files differ OR untracked files exist under src/, tools/, configs/,
scripts/ (found after one smoke run recorded a clean SHA for uncommitted code).

**D-014 resolution (2026-10-11, user).** Matched CV uses the 684 pool. Added
arm: "as-published" = train official train, evaluate official val. Hard rule,
code-enforced (`burnscars.data.splits.assert_eval_allowed`, tested): a model
trained on the official split is never evaluated on IBM test, because all 120
IBM-test chips lie inside official train/val.

**D-016 (2026-10-11, user) Metric separation.** Headline gaps and rankings use
pooled (dataset-level) burned IoU only (PANGAEA convention). Per-chip IoU is
used only as the response in the distance regressions R0/R1/R2. The two are
never mixed in one comparison.

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

**D-009 (PENDING, Phase B) Distance CRS** — EPSG:5070 vs geodesic. To be
decided with the actual chip extents (does any chip fall outside CONUS?).

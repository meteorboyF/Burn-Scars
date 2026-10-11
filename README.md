# Burn-Scars: split leakage in HLS Burn Scars

Research code for a matched random-vs-geography-grouped cross-validation study
on the HLS Burn Scars dataset, with a footprint-leakage audit and a released
leakage-safe split. Work in progress; see `docs/STATE.md`.

**No data is in this repository.** Data is fetched from its source by a
download script (Phase B) into the gitignored `/data/` directory, under the
dataset's own licence.

## Setup (CPU)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock
scripts/install_hooks.sh
.venv/bin/python -m pytest
```

## Running a tool

```bash
PYTHONPATH=src .venv/bin/python tools/plumbing_check.py --config plumbing_check --smoke
```

Every tool takes `--config <name>` (a YAML in `configs/`, validated against the
tool's arguments) and `--smoke` (tiny CPU run). Every run writes a
`manifest.json`; paper figures and tables are built only from manifests
(`scripts/make_figures.py`, `scripts/make_tables.py`).

GPU tools run only from `notebooks/runner.ipynb` on Colab, and refuse to start
without an allocation marked APPROVED in `docs/COMPUTE_LEDGER.md`.

## Reproducibility notes

Seeds via `burnscars.seed.set_seed`. Bitwise reproduction across GPUs is not
promised; statistical reproduction (within seed SD) is.

Licence: MIT (code). Dataset licences: see `docs/DATA.md`.

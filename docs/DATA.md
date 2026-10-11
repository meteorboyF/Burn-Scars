# DATA: provenance, licence, profile, known issues

Nothing downloaded yet (2026-10-11). Licence is read at the pinned revision
BEFORE any download. Data is never redistributed: `tools/download_data.py`
(Phase B) fetches from the source into the gitignored `/data/`.

## Primary: HLS Burn Scars — facts to verify (all from a prior review; UNRESOLVED)

| Fact | Claimed value | Status |
|---|---|---|
| HF dataset id | `ibm-nasa-geospatial/hls_burn_scars` | UNRESOLVED |
| DOI | 10.57967/hf/0956 | UNRESOLVED |
| Licence | CC-BY-4.0 | UNRESOLVED (read at pinned revision) |
| Archive | `hls_burn_scars.tar.gz` ≈ 2.65 GB | UNRESOLVED |
| Revisions seen | 6274300, 24aa452 | UNRESOLVED; pin one, record full hash |
| Archive SHA256 | — | UNRESOLVED |
| Chips | 804, 512×512, 30 m | UNRESOLVED |
| Bands | 6: B02, B03, B04, B8A, B11, B12 | UNRESOLVED |
| Mask values | 1 burned, 0 unburned, −1 no data (ignore, never a label) | UNRESOLVED; GeoTIFF nodata headers may be wrong — profile real values |
| Official split | random 2/3 train, 1/3 val | UNRESOLVED |
| Filename encodes tile + date (`HLS.S30.T10SEH.2018190`) | — | UNRESOLVED |

## IBM buffered splits (524/160/120)

Reportedly in `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars` ("splits").
Location, revision, licence: UNRESOLVED. The 120-chip test set is QUARANTINED:
only a `--final` evaluation tool may read it, and every use is logged.

## Optional second dataset (metadata only, NO download)

`ibm-nasa-geospatial/multi-temporal-crop-classification`, DOI 10.57967/hf/0955,
CC-BY-4.0, training_chips.tgz 10.6 GB + validation_chips.tgz 1.18 GB, 224×224
`chip_XXX_YYY`, CDL-derived labels. All UNRESOLVED. Georeferencing: UNRESOLVED.
Download requires the user's explicit approval.

## Profile

Not run.

# DATA: provenance, licence, profile, known issues

Data is never redistributed. `tools/download_data.py` fetches pinned files from
the source into the gitignored `/data/`, verifies size + SHA256, extracts.
Derived, data-free tables (bounds, CRS, counts) are committed under `results/`.

## Primary: HLS Burn Scars

| Fact | Value | Status |
|---|---|---|
| HF dataset | `ibm-nasa-geospatial/hls_burn_scars`, not gated | VERIFIED (HF API 2026-10-11) |
| DOI | 10.57967/hf/0956 (repo tag) | VERIFIED |
| Licence | CC-BY-4.0, read in card YAML **at the pinned revision before download**; introduced at commit 6274300 ("Use cc-by-4.0 license") | VERIFIED |
| Pinned revision | `1864285e25010d346a842e4f068b1a1d4248ed6d` (HEAD, 2023-09-26) | VERIFIED |
| Revisions in the review | 6274300 (2023-08-01) and 24aa452 (2023-08-03) both exist; neither is HEAD (HEAD adds a `datasets` loader script after them) | VERIFIED |
| Archive | `hls_burn_scars.tar.gz`, 2,645,552,531 bytes; LFS object last changed at 80ca8df and **identical at 80ca8df, 6274300, 24aa452, 1864285** | VERIFIED |
| Archive SHA256 | `4e6f99a75cb2c500547b20662a15cbd531dc421376f815e91846ea542798e8e6` (matches HF LFS oid; recomputed locally) | VERIFIED |
| Chips | 804; 540 in `training/`, 264 in `validation/` (2/3 : 1/3); each dir has `<dir>_index.csv` listing (scene, mask) pairs, cross-checked against files | VERIFIED |
| Filenames | `subsetted_512x512_HLS.S30.<Tile>.<YYYYDDD>.v1.4_merged.tif` / `.mask.tif` — the `subsetted_512x512_` prefix was not in the review. All 1608 tifs match one strict regex. chip_id = `<Tile>.<YYYYDDD>.v1` (IBM split-list form), unique | VERIFIED |
| Sensor | all 804 are HLS **S30** (Sentinel-2), HLS v1.4; no L30 | VERIFIED |
| Dates | 2018-02-04 … 2021-12-14; per year 2018: 269, 2019: 183, 2020: 262, 2021: 90 | VERIFIED |
| Tiles / zones | 365 MGRS tiles; UTM zones 10–18 (10:105, 11:168, 12:145, 13:91, 14:89, 15:61, 16:85, 17:55, 18:5); filename zone == CRS zone for all chips | VERIFIED |
| CRS headers | WKT "UTM Zone NN, Northern Hemisphere" on an *unknown datum based on WGS 84*; equivalent to EPSG:326NN (corner reprojection to EPSG:5070 agrees < 1 mm for every chip) | VERIFIED |
| Shape / dtype | scenes 6×512×512 float32; masks 1×512×512 **int16**; mask grid == scene grid for all chips; pixel size 30 m (± 1e-12 float noise on 21 chips) | VERIFIED |
| Band order | card: B02, B03, B04, B8A, B11, B12. GeoTIFF band descriptions are empty, so order rests on the card + spectral plausibility (mean reflectance B02 .053, B03 .078, B04 .095, B8A .213, B11 .236, B12 .172) | VERIFIED (card) / consistent (data) |
| Value range | scenes in [0, 1] reflectance; no negative, no non-finite values | VERIFIED |
| Mask values | exactly {−1, 0, 1} in every chip: totals 1 = 23,203,821; 0 = 186,225,043; −1 = 1,334,912 px (11.0% / 88.4% / 0.63% of all pixels) | VERIFIED |
| −1 is ignore, never a label | enforced in `burnscars.eval.metrics` (tests) | VERIFIED |
| **Nodata headers are wrong** | mask header nodata = **−1000** (never occurs; real no-data = −1); scene header nodata = 0.0 | VERIFIED |
| Nodata behaviour | every one of the 1,334,912 mask −1 pixels has ≥1 scene band == 0; 1,239,332 of them are all-zero in all bands; all all-zero scene pixels are mask −1; 410,361 labelled (0/1) pixels have some band exactly 0 | VERIFIED |
| −1 per chip | 291/804 chips contain any −1; median −1 fraction 0, 99th pct 27.7%, max 47.1%; no all-ignore chip | VERIFIED |
| Burned fraction (of valid px) | min 0.84%, median 4.1%, 90th pct 28.6%, max 96.3%; official train mean 12.0%, val 9.3% | VERIFIED |
| **Chips with zero burned pixels** | **0** — every chip contains burn; per-chip burned IoU is always defined. No absent-target claims are made (none possible) | VERIFIED |
| Same tile + same date | 0 (chip_id unique) | VERIFIED |
| Card's class distribution (11 / 88 / 1 %) | 11.0 / 88.4 / 0.63 % | VERIFIED (≈) |
| HF loader `hls_burn_scars.py` | maps its TEST split to `validation/` (val == test) | VERIFIED (source read) |

Profile: `results/profile/{chips_profile.csv,summary.json,manifest.json}` (manifest at clean SHA 3fd049a).

## Footprint audit (summary; full generated report: docs/AUDIT_REPORT.md)

Official val→train: 107/264 (40.5%) within 5 km; 85 val chips overlap ≥1 train
chip; 34 are ≥50% covered by train footprints; 6 same-footprint repeats
(IoU ≥ 0.9, all same tile, different year); 124/365 tiles shared (TACO claim
reproduced exactly). IBM split: 0 chips within 10 km of training (min 10.3 km
val, 11.1 km test), 0 overlaps, but tile-sharing persists (55/120 test chips on
a tile with an IBM-train chip). Whole pool: 279 overlapping pairs; 5 km
single-linkage → 552 groups (414 singletons, largest 14).

## IBM split lists (Apache-2.0)

`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars` @ `a3f2c410e45b8ac7417976614528a872f024d831`
(2025-09-29), `splits/{train,val,test}.txt` = 524 / 160 / 120, SHA256 in
`configs/download_ibm_splits.yaml`. Every id maps 1:1 onto the 804 chips; 365
tiles in total. Identical to IBM/peft-geofm `datasets_splits/burn_scars`
(per literature agent; not re-checked by lead). **test.txt is QUARANTINED**:
membership read only by `audit_footprints` (geometry) and `final_eval --final`;
every read logged in docs/QUARANTINE_LOG.md.

## Optional second dataset: multi-temporal crop classification (NOT downloaded)

| Fact | Value | Status |
|---|---|---|
| HF dataset | `ibm-nasa-geospatial/multi-temporal-crop-classification` @ `f285bb27c8f623a0fb6a44a6fd953c3ad34007d6` (2025-02-11) | VERIFIED |
| DOI / licence | 10.57967/hf/0955 / CC-BY-4.0 (card YAML + repo tag) | VERIFIED |
| Files | `training_chips.tgz` 10,601,018,926 B; `validation_chips.tgz` 1,179,542,384 B; `chips_df.csv`; `training_data.txt`; `validation_data.txt` | VERIFIED |
| Content | 3,854 chips, 224×224, 30 m, 18 bands (6 HLS S30 bands × 3 dates, 2022), CONUS; labels from USDA CDL (13 classes + no-data 0); random 80/20 split | VERIFIED (card) |
| **Georeferencing** | **YES**: chips are GeoTIFFs in **EPSG:5070** (CDL grid), 30 m, 6.72 km wide. Checked by reading 4 complete files from the first 2 MB of `validation_chips.tgz` (HTTP range request; nothing else fetched; scratch copy deleted). `chips_df.csv` also gives each chip's MGRS tile and the 3 image dates | VERIFIED (4 files) |
| Download | not stored. Header-only stream audit (user-approved bandwidth 2026-10-11): both archives streamed once, SHA256 verified in flight (training 10,601,018,926 B `373cc94b…`; validation 1,179,542,384 B `d6e616cc…`), pixels discarded | VERIFIED |
| Footprint audit (results/crop_audit) | 3,854 chips, all EPSG:5070, 224×224, 6.72 km on a regular grid; scene/mask headers agree; lists match archives (3,083 / 771). Val→train: 536/771 = 69.5% touching (edge/corner, distance 0), 87.0% within 10 km; 0 area overlaps; 0 repeats (single year); 244/425 tiles shared; 762/771 val chips on shared tiles; 5 km groups 1,497 (largest 114) | VERIFIED |

## Not used

Sen1Floods11: licence unresolved at the original source (standing rule).

## Metric validation (published Prithvi-EO-2.0 BurnScars checkpoint, IBM val, CPU)

Ours (fp32): IoU-burned 82.32, mIoU 89.75 — identical to TerraTorch 1.2.13's own
`validate` (0.8231562 / 0.8975374) and to torchmetrics macro (1e-7). Card: 84.28 /
90.95 → not reproduced (−1.96 / −1.20; cause UNRESOLVED; D-020). Validates metric +
preprocessing only. results/metric_validation/.

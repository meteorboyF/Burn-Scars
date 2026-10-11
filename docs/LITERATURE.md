# LITERATURE: related-work matrix and search log

Protocol: 03_LITERATURE_AND_NOVELTY.md. Sweep run 2026-10-11 with live search
(arXiv HTML/PDF/API, Crossref, Semantic Scholar, Europe PMC, GitHub, Hugging
Face) by a delegated agent. I (the lead agent) independently re-checked the
two novelty-critical sources (2504.17397 §4 + Table 4; 2609.25731 methods) and
reconciled the agent's IBM tile counts against `results/audit/` (exact match).
BibTeX: `paper/refs.bib` (39 entries, each resolved to a DOI/arXiv id).
Re-run a delta sweep at every phase boundary and before submission: several
close preprints appeared Jul–Oct 2026.

## Numbers in the topic statement

| Claim | Status | Evidence |
|---|---|---|
| PANGAEA (Marsocci et al., arXiv 2412.04204; IEEE GRSM 2026, doi:10.1109/MGRS.2025.3628194): UNet best on HLS Burn Scars at 84.51 mIoU | **VERIFIED** (arXiv v2 Table 5; GRSM version not opened) | Prithvi 83.62, CROMA 82.42, ViT 81.58. "Prithvi" there = **Prithvi-EO-1.0-100M**, frozen encoder + UPerNet; PANGAEA has no Prithvi-2.0 |
| …on the "official split" | **PARTLY CONTRADICTED** | PANGAEA test = official `validation/` (264); its train/val = random 90/10 of official `training/` (`train_test_split(test_size=0.1, random_state=23)`, no `stratify=` despite a comment claiming stratification). `pangaea/datasets/hlsburnscars.py` @ca19b83 L106–110, L139–148 |
| arXiv 2504.17397 = Marti-Escofet et al., "Fine-tune Smarter, Not Harder: PEFT for GFMs", ECML PKDD 2025, doi:10.1007/978-3-032-06106-5_30; Prithvi-EO-2.0 LoRA 93.33 vs UNet 89.34 test mIoU | **VERIFIED** (re-checked by lead) | Table 4, mean of 5 runs (UNet 89.34 ±1.0; Prithvi-2.0-300M LoRA 93.33 ±0.6; Full FT 92.85). GFMs use a **linear decoder**, 16-trial HPO. Prithvi-1.0 LoRA is also 89.34 — easy to confuse |
| IBM splits are 5-km-buffered | **VERIFIED (wording)** | §4: "We create the splits with a 5 km buffer between training and evaluation samples". Our audit: min IBM val→train 10.34 km, test→train+val 11.08 km (consistent; ≥ 5 km). But the split is **not tile-disjoint**: 55/120 test chips sit on a tile with an IBM-train chip (28 tiles); 62/160 val |
| Prithvi-EO-2.0 paper (arXiv 2412.02732; TGRS 64, 2026, doi:10.1109/TGRS.2025.3642610) describes how 524/160/120 was built | **CONTRADICTED** | Paper gives no counts/buffer; the HF card says the same splits were used; card `splits/*.txt` identical to IBM/peft-geofm lists. Paper Table V (UPerNet, 448 resize, 224 crop): 300M 88.6, 600M 90.5 mIoU; no UNet |
| Prithvi-2.0-300M-BurnScars card: test IoU-burned 87.52, test mIoU 93.00 | **VERIFIED** (rev a3f2c41) | UNetDecoder; "repeated the run five times and selected the model with the lowest validation loss over all runs and epochs" (best-of-5 on val) |
| TACO card: 124 of 365 MGRS tiles on both sides of the official split | **VERIFIED + RECOUNTED** | Card quote: "not MGRS-tile disjoint: 124 of 365 tiles appear on both sides". Our recount: **124/365** (docs/AUDIT_REPORT.md) |
| Prithvi-100M (Jakubik et al., arXiv 2310.18660) | numbers VERIFIED; split UNRESOLVED | Table 3: IoU-burn 73.62, mIoU 84.84 (U-Net/DeepLabV3 71.01/83.55). `hls-foundation-os/configs/burn_scars.py` uses `validation/` as both val and test (issue #40, open since 2023-11) |

## Metric conventions (determines what our replica must report)

| Pipeline | Aggregation | Classes | −1 handling | Averaging | Source |
|---|---|---|---|---|---|
| PANGAEA @ca19b83 | dataset-level confusion (`torch.bincount`, all-reduced) | 2 | `valid_mask = target != ignore_index` | macro over classes; its "mAcc" is overall pixel accuracy | `pangaea/engine/evaluator.py` SegEvaluator ~L430–491 |
| TerraTorch (main; tags 1.0, v1.2.13) | dataset-level (torchmetrics state over epoch) | 2 | torchmetrics drops `target == ignore_index` | `"mIoU"` = macro; `"mIoU_Micro"` = micro (= a/(2−a) for 2 classes, a = pixel accuracy) | `terratorch/tasks/segmentation_tasks.py` `configure_metrics` L259–321; torchmetrics `jaccard.py` L38–96 |
| Prithvi-2.0 BurnScars card | dataset-level (TerraTorch task) | 2 | `ignore_index: -1` | consistent with macro (2×93.00 − 87.52 = 98.48 plausible unburned IoU). **UNRESOLVED**: exact TerraTorch version / metric key | `burn_scars_config.yaml` |
| Prithvi-100M (mmseg) | "pixel-wise over the test set" | 2 | `ignore_index=-1` | **UNRESOLVED** (mmseg internals not opened) | `configs/burn_scars.py` |

→ Our replica's primary numbers: `dataset_iou_burned` and `dataset_miou`
(= `MulticlassJaccardIndex(num_classes=2, ignore_index=-1, average="macro")` on
a pooled confusion). Per-chip IoU is reported separately, labelled as such.
**Not yet validated against a published number** (needs a model's predictions;
Phase C, see STATE).

## Search log (all 2026-10-11)

| Engine | Query | Relevant | Notes |
|---|---|---|---|
| arXiv HTML | 2412.04204, 2504.17397, 2412.02732, 2310.18660 (full text) | 4 | |
| HF raw/API | TACO README; hls_burn_scars README/discussions; Prithvi-2 BurnScars card/config/splits; Prithvi-100M card | 5 | |
| GitHub raw/API | pangaea hlsburnscars.py/.yaml/evaluator.py/prithvi.yaml; IBM/peft-geofm splits; terratorch main/1.0/v1.2.13; torchmetrics; hls-foundation-os config | — | |
| WebSearch | spatial cross-validation versus random split geospatial foundation model benchmark inflation | 4 | |
| WebSearch | HLS Burn Scars spatial leakage split evaluation | 3 | found 2609.25731, 2607.18504 |
| WebSearch | spatial autocorrelation data leakage deep learning remote sensing segmentation evaluation 2024 2025 | 4 | |
| WebSearch | Ploton 2020 Nat Commun; Kattenborn 2022 ISPRS Open; Meyer & Pebesma 2022; Roberts 2017; Karasiak 2022 | 7 | |
| WebSearch | GEO-Bench 2306.03831; Copernicus-Bench 2025 | 2 | |
| WebSearch | GFM benchmark spatial split leakage rankings random vs geographic 2025 2026 | 3 | |
| WebSearch | burned area segmentation DL S2/Landsat benchmark 2024 2025 | 3 | |
| WebSearch | "spatial leakage" GFM segmentation random split Prithvi | 3 | |
| WebSearch | random vs spatially blocked CV DL segmentation ranking stability | 2 | |
| WebSearch | overlapping tiles train test leakage RS benchmark audit | 3 | |
| WebSearch | HLS Burn Scars geographic generalization OOD | 1 | SHRUG-FM uses ExEBench, not HLS Burn Scars |
| WebSearch | per-sample accuracy vs distance to training 2025 | 0 new | |
| WebSearch | Sen1Floods11 random split leakage geographic holdout (×2) | 0 resolved | snippet about country-disjoint comparison: **source UNRESOLVED** |
| WebSearch | leakage-free/safe split EO segmentation 2026; RSE 2025 spatial CV DL; ISPRS JPRS 2026 block CV; TGRS/GRSL leakage | 2 | |
| WebSearch | Prithvi spatial CV UNet ranking burn scars flood | 1 | MDPI Fire 9:161 (abstract only) |
| GitHub search | hls-foundation-os burn; hls_burn_scars split; pangaea HLSBurnScars | 4 | issues #40, #45; PANGAEA #126, PR #83 |

## Related-work matrix (newest first; all ids resolved)

| Paper (venue, year) | id | Problem | Method family | Benchmark | Headline | Code | How ours differs |
|---|---|---|---|---|---|---|---|
| Alkhatib (arXiv 2026) | 2610.08770 | patch-overlap leakage | random vs spatial sampling | Pavia U. HSI | 3D-CNN OA 96.17→55.20 | n/s | single hyperspectral image; no GFM, no ranking |
| **Knezevic et al. (arXiv 2026)** | 2609.25731 | annual embeddings for burned area | 3-fold CV, components linked by shared MGRS tile / same MTBS fire / >1% footprint overlap (lead-verified) | **HLS Burn Scars** (793 chips) | Tessera U-Net F1 0.916 | on request | **closest on data**; no random/official arm, no Prithvi, no ranking; relabelled masks |
| Kang (arXiv 2026) | 2609.13265 | sensor ranking under leakage | site-disjoint | AI Hub agri | NIR advantage vanishes | n/s | leakage flips rankings, other domain |
| Phunjanna et al. (arXiv 2026) | 2608.30895 | ranking stability | seeds, chip resampling, geo reweighting | Sen1Floods11, GEOID-Flood | close orderings flip | n/s | fixed split; no random-vs-grouped retraining |
| Lehmann et al. (arXiv 2026) | 2608.16614 | GFM calibration under shift | 16 frozen encoders | 9 datasets | rankings change under corruption | n/s | synthetic shift, not split design |
| Essus et al. (arXiv 2026) | 2608.24665 | leakage in outage prediction | random vs state/event-out | US East Coast | random-split skill largely vanishes | n/s | tabular + GFM embeddings |
| Schindlegger et al. (arXiv 2026) | 2607.18504 | TerraMind vs THOR | controlled ablation | 10 tasks incl. HLS Burn Scars | 0.891 mIoU; UNet beats frozen GFMs at every fraction | n/s | IBM splits; no leakage test |
| Shang et al. (arXiv 2026) | 2606.29664 | GFM transfer, agriculture | region-disjoint | 4 US states | all degrade | n/s | no matched random baseline |
| **Liu et al. CityRep (arXiv 2026)** | 2605.26036 | urban representations | block vs random splits, 5 seeds | 8 cities × 8 tasks | random splits inflate scores and alter rankings | yes | **closest on design**; urban embeddings, not EO segmentation |
| Corley et al. (arXiv 2026) | 2605.12678 | can GFM SOTA be known | 152-paper audit | GFM literature | 46 disagreements ≥10 pts | n/s | diagnoses, does not measure leakage |
| Linnenbrink et al. (arXiv 2026) | 2605.13689 | random vs spatial CV debate | position | — | — | — | framing to position against |
| Brenning & Suesse (arXiv 2026) | 2603.29981 | CV aligned with deployment | target-weighted CV | NO₂ Germany | standard CV overestimates | n/s | tabular |
| Kaushik et al. (JSTARS 2026) | 10.1109/JSTARS.2026.3656855 | GFMs vs U-Net, floods | leave-one-region-out | PlanetScope/S1/S2 | Clay S2 0.70 mIoU; LORO 0.66 | n/s | no random arm |
| Marti-Escofet et al. (ECML PKDD 2025) | 10.1007/978-3-032-06106-5_30 | PEFT for GFMs | LoRA etc., HPO, 5 seeds | incl. Burn Scars | 93.33 vs 89.34 | yes | buffered split not compared to official; still tile-sharing |
| Marsocci et al. PANGAEA (GRSM 2026) | 10.1109/MGRS.2025.3628194 | GFM benchmark | frozen enc. + UPerNet | 11 datasets | UNet 84.51 | yes | evaluates on leaky official val |
| Szwarcman et al. (TGRS 2026) | 10.1109/TGRS.2025.3642610 | Prithvi-EO-2.0 | MAE ViT | incl. burn scars | 600M 90.5 mIoU | yes | split undocumented; our encoder |
| Koldasbayeva & Zaytsev (Ecol. Inform. 2025) | 10.1016/j.ecoinf.2025.103521 | spatio-temporal CV for SDMs | random/spatial/forward | 2 datasets | AUC overestimated ≤0.16 | n/s | tabular; tuning-inside-CV template |
| Stock (Front. RS 2025) | 10.3389/frsen.2025.1531097 | choosing CV blocks | block size/shape | synthetic | block design matters | n/s | guides our 5 km sensitivity |
| Kattenborn et al. (ISPRS Open 2022) | 10.1016/j.ophoto.2022.100018 | CNN segmentation inflation | random vs site-block CV | UAV, 47 sites | overestimated by up to 28% | n/s | one family, no ranking, no GFM |
| Karasiak et al. (Mach. Learn. 2022) | 10.1007/s10994-021-05972-1 | spatial dependence, RS classification | spatial LOO | Sentinel-2 forest | non-spatial CV overestimates | yes | pixel classifiers |
| Ploton et al. (Nat. Commun. 2020) | 10.1038/s41467-020-18321-y | biomass map validation | buffered LOO 0–150 km | Central Africa | R² 0.50 → quasi-null | yes | our distance curve is the segmentation analogue |

Also opened and in refs.bib: GEO-Bench (2306.03831), Copernicus-FM (2503.11849),
Meyer & Pebesma 2022, Roberts 2017, Wadoux 2021, blockCV 2019, kNNDM 2024,
Spatial+ 2023, Mahoney 2303.07334, Rolf 2303.18087, Adimoolam 2304.02296,
FLOGA, Nalepa 2019 (GRSL), Liang 2017 (TGRS), Stehman & Foody 2019 (RSE),
TerraTorch (IGARSS 2025).

## Known dataset issues (beyond our own audit)

1. Official split not tile-disjoint (124/365) and has no test split; hls-foundation-os uses `validation/` as val AND test (issue #40); PANGAEA re-splits.
2. IBM 524/160/120 is buffer-based, not tile-grouped (55/120 test chips share a tile with IBM-train).
3. Many chips contain more than one fire (Knezevic et al.: 559/793 single-fire); MTBS perimeters include unburned islands; other same-year fires are unlabelled; MTBS maps only large fires (≥400 ha west / ≥200 ha east). Label noise is UNRESOLVED in size.
4. 805 → 804 chips (HF commit 0376987 "Removed a single file that was not 512x512").
5. Mask nodata header −1000 never occurs (also stated on the TACO card); see DATA.md.
6. Reproduction gap reported in hls-foundation-os issue #45 (burn IoU 70.26 vs ~73).

## Novelty verdict (2026-10-11): **CROWDED — rescoped, not Done**

- Closest on data: **Knezevic et al. 2026** already use tile/fire/overlap-grouped
  CV on HLS Burn Scars. Delta: they have no random/official arm, no matched
  train size, no model-family ranking, no distance curve, no Prithvi-2.
- Closest on design: **Liu et al. 2026 (CityRep)** show random splits inflate
  scores and alter rankings. Delta: EO segmentation with a GFM, under the exact
  metric convention the GFM leaderboards use.
- Closest benchmark: **Marti-Escofet et al. 2025** vs **PANGAEA** rank UNet and
  Prithvi in opposite orders. Delta: our matched design separates split effect
  from protocol (decoder, frozen vs fine-tuned, HPO, resizing, Prithvi version).

Contribution statement (rescoped, ceiling for the paper): *a matched
random-vs-grouped CV study on HLS Burn Scars that quantifies split inflation,
its distance dependence and its effect on UNet-vs-Prithvi-EO-2.0 rankings,
with a released leakage-safe split and audit script.* Not claimed: that
leakage in this dataset is newly discovered (the IBM card, the PEFT paper and
the TACO card all acknowledge it in some form).

Threats: (1) leakage already acknowledged; (2) grouped split itself not new;
(3) null-result risk — each chip is centred on a different fire, inflation may
be ~1 pt mIoU, hence burned-class IoU is primary; (4) protocol confounds must
be held fixed for any rank claim; (5) Prithvi pretraining may cover the same
tiles/years — no downstream split removes it (state as a limitation);
(6) small sample (804 chips / 365 tiles); (7) counter-literature that spatial
CV can be pessimistic (Wadoux 2021; Linnenbrink 2026) — define the target
domain explicitly; (8) fast-moving area; (9) one UNRESOLVED possible precedent
on Sen1Floods11 (snippet only, source not found).

## Candidate Q1 venues (D-005 still open)

IEEE JSTARS (Kaushik 2026, doi:10.1109/JSTARS.2026.3656855) or IEEE GRSL
(Nalepa 2019, doi:10.1109/LGRS.2019.2895697) fit the focused study best; IEEE
TGRS would need a multi-dataset extension; Elsevier JAG (Spatial+, 2023) and
RSE (no spatial-CV-for-DL example found) are alternatives. Check current
quartiles at decision time.

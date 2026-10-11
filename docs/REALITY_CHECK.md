# REALITY_CHECK: hypotheses, kill criteria, outcomes

Format: 01_RESEARCH_METHOD.md. Entries are append-only; outcomes are written
after, always. Operational definitions below were fixed **2026-10-11, before
the dataset was downloaded or any split/footprint was inspected**. The user may
amend them before Phase B computes anything; any later change is a new dated
entry that supersedes, never an edit.

Common definitions (apply to H1, K1, K2):
- *Chip footprint*: the chip's raster bounds polygon in its native CRS, densified
  and reprojected to the distance CRS chosen in DECISIONS.md (provisional:
  EPSG:5070 CONUS Albers; final choice + justification at Phase B step 3).
- *Distance between chips*: minimum edge-to-edge distance between footprints
  (0 if they touch/overlap). Not centroid distance.
- *Same-footprint repeat*: a train/val pair whose footprints have area IoU ≥ 0.9
  (any acquisition dates). Also reported: pairs with any area overlap > 0.
- *IoU*: burned-class IoU with label −1 ignored; "dataset-level" = confusion
  counts accumulated over all chips of a fold, then one IoU.

---

```
K1  2026-10-11  status: REGISTERED (kill criterion, topic-level)
Test: on the OFFICIAL split, p_near = fraction of official-val chips whose
  distance to the nearest official-train chip is <= 5 km; n_rep = number of
  same-footprint repeats across official train/val.
Fires (topic stops) if: p_near < 5.0% AND n_rep == 0.
Does NOT fire if either p_near >= 5.0% or n_rep >= 1.
Robustness: also report p_near at 2.5 and 10 km and under geodesic distance;
  if the verdict flips between EPSG:5070 and geodesic, report it as borderline
  and let the user decide (no silent choice).
Cost: CPU only, minutes.
Outcome (2026-10-11, results/audit/manifest.json @ clean SHA b4e42c7): K1 does
  NOT fire. p_near = 107/264 = 40.5% (geodesic 40.5%; 36.0% at 2.5 km, 48.1% at
  10 km) vs threshold 5.0%; n_rep = 6 (all same tile, different year) vs 0.
  Both conditions fail by a wide margin; not borderline. Topic continues.
```

```
H1  2026-10-11  status: REGISTERED
Hypothesis: per-chip IoU decreases with distance to the nearest training chip,
  controlling for burned fraction and year; random CV inflates scores relative
  to geography-grouped CV (5 km single-linkage groups, equal train sizes)
  enough to change model rankings.
Prediction (proposed by agent, to be confirmed by user before Phase C runs):
  (a) learned models: random-minus-grouped dataset-level burned IoU >= +2.0
      points, mean over 5 folds, for at least one model family;
  (b) NBR-threshold control: random-minus-grouped gap smaller than the learned
      models' gap (memorisation-of-place mechanism) — if NBR drops as much as
      the learned models, the effect is a difficulty shift, not leakage;
  (c) in the per-chip regression IoU ~ log1p(distance_to_train_km) +
      burned_fraction + year (cluster bootstrap over MGRS tiles), the
      distance coefficient is NEGATIVE with its 95% CI excluding 0;
  (d) rank change: Kendall tau between model rankings under the two regimes
      < 1 (at least one pairwise order flips) — only meaningful with >= 2
      families; with exactly 2 families this is a single sign comparison.
Falsified if: (c) slope CI includes 0 AND (a) gap < 1 point; OR the gap
  appears equally in NBR (mechanism contradiction).
Abandon if: K2 fires.
Cost: ESTIMATED pilot ~3 GPU-h (prior guess, unmeasured).
Outcome:
```

```
K2  2026-10-11  status: REGISTERED (kill criterion, after the pilot)
Test: UNet pilot (one seed) under 5-fold random CV and 5-fold grouped CV.
  gap = mean over folds of dataset-level burned IoU (random) minus (grouped).
  Fold interval per regime = mean +/- 1.96 * SD / sqrt(5) over the 5 folds.
  Distance gradient = slope of per-chip IoU on log1p(distance-to-train) in the
  grouped+random pooled out-of-fold predictions, controls as in H1(c),
  cluster bootstrap over MGRS tiles, 95% CI.
Fires (topic stops) if: |gap| < 1.0 IoU point AND the two fold intervals
  overlap AND the distance-slope 95% CI includes 0.
Cost: inside the pilot allocation; no extra runs.
Outcome:
```

---

```
E-01  2026-10-11  status: EXPLORATORY (not pre-registered; seen before any model)
Observation: NBR-rule per-chip burned IoU on the official val split, by distance
  to nearest official-train chip: overlapping 0.234 (n=85), 0-5 km 0.575 (n=22),
  5-20 km 0.433 (n=55), >20 km 0.361 (n=102). NBR cannot memorise places, so
  chip DIFFICULTY covaries with distance-to-train in this dataset.
Consequence: the H1 per-chip regression with only burned_fraction + year as
  controls may mis-attribute a difficulty gradient to leakage (in either
  direction). PROPOSED amendment (needs user approval BEFORE any learned-model
  result exists): add the NBR-rule per-chip IoU (a model-free difficulty score)
  as a covariate, and test the learned-minus-NBR per-chip IoU difference against
  distance. Flagged as informed by this look, so it is a disclosed amendment.
Also: NBR random-vs-grouped CV gap = +0.08 pts (pool 804) and -0.11 pts
  (pool 684) dataset-level burned IoU, fold SD ~4-5 pts: the control is flat.
```

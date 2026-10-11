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

---

```
H1-v2  2026-10-11  status: REGISTERED (supersedes H1's prediction/decision rule;
  user-approved; committed BEFORE any learned-model result of ours exists —
  the only model output seen so far is none; the published-checkpoint metric
  validation on IBM val runs AFTER this commit)
Pool: 684 chips (IBM train ∪ IBM val), splits/v1/{random,grouped}_cv5_ibm_trainval.json
  (D-014). IBM test stays quarantined.
Arms: (A) random 5-fold CV, (B) grouped 5-fold CV (5 km single linkage), equal
  train size 547; (C) "as-published": train on official train (540), evaluate on
  official val (264) — the community protocol whose inflation we quantify.
  Models trained on official train are NEVER evaluated on IBM test (code-enforced).
Smallest effect of interest: +2.0 burned-IoU points.
Headline gap G = pooled burned IoU(random) - pooled burned IoU(grouped), where
  each regime's pooled IoU is computed from the confusion matrix summed over all
  684 out-of-fold chips (each chip is tested exactly once per regime -> paired).
CI: cluster bootstrap resampling the 5-km connected components (groups of the
  684 pool, splits/v1 group_of) with replacement, B = 10,000, recomputing both
  pooled IoUs in every resample; 95% percentile interval.
H1 SUPPORTED only if ALL: G >= 2.0; CI(G) excludes 0; CI(G_learned - G_NBR)
  excludes 0 (same bootstrap draws, paired).
Rank criterion: predicted ranking (pre-registered here):
  random regime : UNet >= Prithvi-EO-2.0-300M frozen + light decoder
  grouped regime: Prithvi-EO-2.0-300M frozen + light decoder > UNet
  (mechanism: memorisation of place helps the from-scratch model more).
  Report P_boot(reversal) = fraction of bootstrap resamples in which the order
  of pooled burned IoU differs between regimes. A reversal COUNTS only if
  P_boot >= 0.9.
Metric separation: pooled IoU for all headline gaps; per-chip IoU ONLY for the
  distance regressions (D-016).
Distance regressions (per-chip, out-of-fold, cluster bootstrap over 5-km
  components), all three reported, none dropped:
  R0 (original pre-registered): IoU_learned ~ log1p(dist_km) + burned_frac + year
  R1 (primary, E-01 amendment): IoU_learned ~ log1p(dist_km) + burned_frac + year + IoU_NBR
  R2 (robustness, diff-in-diff): (IoU_learned - IoU_NBR) ~ log1p(dist_km) + burned_frac + year
  dist_km = distance to nearest chip in that chip's own training fold.
Falsified if: G < 2.0, or CI(G) includes 0, or CI(G_learned - G_NBR) includes 0.
Abandon if: K2 fires.
```

```
E-01 AMENDMENT  2026-10-11  status: APPROVED by user (disclosed amendment)
The NBR per-chip IoU covariate (E-01) enters H1 as regression R1 (primary) and
the difference outcome R2 (robustness); R0 (original spec) is still reported.
Disclosure: motivated by the exploratory NBR-by-distance look in E-01, made
before any learned-model result. Commit SHA: see the next line (added in the
following commit, since a commit cannot contain its own SHA).
```
E-01 amendment and H1-v2 committed in 4b6344c (2026-10-11), before any learned-model result.

---

```
P-01  2026-10-11  status: REGISTERED — awaiting allocation A-001 (no run yet)
Purpose: the K2 gate + a measured GPU throughput. UNet only (K2 is defined on it).
Design: configs/grid/pilot_v1 (generated, immutable). Seed 0, 60 epochs fixed,
  batch 8, AdamW wd 0.05, cosine, D4, fp32/TF32 (D-017). Tuning: lr in
  {3e-4, 1e-3, 3e-3} on inner fold 0 of outer fold 0, RANDOM regime only; winner =
  highest final-epoch inner-val pooled burned IoU (ties -> smaller lr); frozen for
  5 random + 5 grouped (684 pool) + 1 as-published (official train -> official val).
Geometry going in (VERIFIED): random-CV test chips touching a fold-train chip 32.0%,
  within 5 km 40.9% (official split: 32.2% / 40.5%); grouped: 0% / 0%.
Prediction band (pooled burned IoU, agent's judgement, ESTIMATED):
  random-CV 74-86; grouped-CV 70-84; G = random - grouped in [+1.0, +6.0] pts;
  as-published (official val) 68-82; NBR control G ~ 0 (measured +/-0.1).
  Basis: PANGAEA UNet mIoU 84.51 on official val implies burned IoU ~72 if
  unburned IoU ~97; PEFT UNet test mIoU 89.34 (IBM split) implies ~81.
K2 (as registered, R0 controls) fires if |G| < 1.0 AND fold intervals overlap AND
  the R0 distance-slope CI includes 0 -> STOP the topic. R1/R2 reported alongside.
Stop rules (no partial pilot is reported as a result):
  - Throughput: after 2 epochs of the first tuning run, if the projected pilot
    wall-clock exceeds the approved cap, stop and re-cost (a >2x miss vs the
    upper estimate means the costing method is broken; fix before spending more).
  - Sanity: NaN loss, or tuned inner-val pooled burned IoU below the NBR control
    (46 pts) -> stop; debug on CPU (bug, not science).
  - Hard deadline from start_allocation; runs stop at a checkpoint and are resumable.
Cost: A-001 = A100 0.8-1.7 GPU-h (ESTIMATED) -> request cap 2.0 GPU-h;
  or L4 2.8-5.2 -> cap 5.5 GPU-h.
Outcome:
```

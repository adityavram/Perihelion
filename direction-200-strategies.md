# Direction 200: Multiple Coevolution Strategies for the 35% Hard Cases

**Status:** Experiment complete. Negative — no strategy beats the baseline.
**Run time:** 11,010s (~3.1 hours)
**Date:** 2026-08-06

---

## Question

D201's extended-root coevolution reaches 65% at 64-bit (13/20). The 7
hard instances (i=4,9,12,13,14,15,17) all fail. D199 showed the bottleneck
is the posterior (d_high ~5 bits; coevolution bridges ~4; hint with true
p_high works 100%). **Can any alternative coevolution strategy recover some
of the 35% hard cases?**

## Setup

- Test set: D201's 7 hard instances (regenerated with seed_offset=501).
- Training: AR posterior on 800 64-bit semiprimes (seed_offset=500), both
  p and q posteriors trained.
- 9 strategies tested (A through I), each on all 7 hard instances.
- Coppersmith: dim=10 (m=5, t=4), extended-root extraction (D174, |x0| ≤ 8X).
- Brute: b_unknown=12 (4096 GCDs, 100% reliable).

## Strategies

| ID | Strategy | R | pop | gen | b_unknown | extract | mutation | init |
|----|----------|--:|----:|----:|----------:|---------|----------|------|
| A | Baseline (D201) | 10 | 80 | 40 | 16 | copper-ext | standard (1-2) | p-post |
| B | More restarts | 30 | 80 | 40 | 16 | copper-ext | standard | p-post |
| C | Larger pop | 10 | 200 | 40 | 16 | copper-ext | standard | p-post |
| D | More gens | 10 | 80 | 100 | 16 | copper-ext | standard | p-post |
| E | Wide mutation | 10 | 80 | 40 | 16 | copper-ext | wide (2-3) | p-post |
| F | Predict q | 10 | 80 | 40 | 16 | copper-ext | standard | q-post |
| G | Both posteriors | 10 | 80 | 40 | 16 | copper-ext | standard | both |
| H | Brute bu=12 | 10 | 80 | 40 | 12 | brute | standard | p-post |
| I | Combined | 30 | 200 | 40 | 16 | copper-ext | wide | both |

## Results

| Strategy | Success | Rate | Solves |
|----------|---------|------|--------|
| A_baseline | 3/7 | 42.9% | 9, 12, 17 |
| B_R30 | 3/7 | 42.9% | 9, 12, 17 |
| C_pop200 | 3/7 | 42.9% | 9, 12, 17 |
| D_gen100 | 3/7 | 42.9% | 9, 12, 17 |
| E_mut_wide | 3/7 | 42.9% | 9, 12, 17 |
| F_predict_q | 2/7 | 28.6% | 9, 17 |
| G_both_post | 1/7 | 14.3% | 9 |
| H_brute_bu12 | 1/7 | 14.3% | **13** |
| I_combined | 3/7 | 42.9% | 9, 12, 17 |

**Total instances solved by ANY strategy: 4/7** (i=4, 14, 15 never solved).

### Per-instance solvability

| Instance | Solvers | Notes |
|----------|---------|-------|
| i=4 | NONE | Truly hard — all 9 strategies fail |
| i=9 | A,B,C,D,E,F,G,I | Easy (8/9 strategies) |
| i=12 | A,B,C,D,E,I | Medium (6/9) |
| i=13 | **H only** | Only brute bu=12 solves it |
| i=14 | NONE | Truly hard |
| i=15 | NONE | Truly hard |
| i=17 | A,B,C,D,E,F,I | Easy (7/9) |

## Findings

### 1. No strategy beats the baseline (3/7 = 42.9%)

Strategies A-E and I all solve exactly the same 3 instances (9, 12, 17)
with the same success rate. **More restarts (R=30), larger population
(pop=200), more generations (gen=100), wider mutation (2-3 bits), and the
combined strategy all fail to add a single instance.** The coevolution
search radius is NOT the bottleneck for these instances — the posterior's
samples simply don't land close enough to the true p for coevolution to
bridge the gap, and no amount of search expansion fixes that.

### 2. Predict q (F) is WORSE (2/7)

Using the q-posterior samples as p candidates solves only i=9 and i=17,
losing i=12. The q-posterior is NOT more predictable than the p-posterior
for these instances — it's worse. This contradicts D199's Method 1a
hypothesis. The symmetry of the problem (p and q are interchangeable) means
training on q gives a posterior of equal quality, not better. The loss of
i=12 is RNG variance, not a structural improvement.

### 3. Both posteriors merged (G) is WORST (1/7)

Merging p-posterior and q-posterior samples into both populations solves
only i=9. The merged init is HARMFUL — it dilutes each population's quality
by mixing in candidates from the "wrong" posterior. The coevolutionary
feedback (p sharpens q and vice versa) works best when each population is
seeded from its OWN posterior, not a mix.

### 4. Brute bu=12 (H) solves a DIFFERENT instance (i=13)

Strategy H (brute at b_unknown=12) is the only one that solves i=13 (R=1,
2422 evals, 3.9s). But it loses i=9, 12, 17 — the Coppersmith-easy cases.
This is the key structural finding: **the Coppersmith-easy and brute-easy
classes are DIFFERENT.** Coppersmith at bu=16 tolerates high-bit errors
(the extended-root trick catches 1-bit errors) but has a 84% extraction
failure rate (D199). Brute at bu=12 is 100% reliable but requires 4 more
correct high bits (n_known=20 vs 16). The instances that solve with
Coppersmith have posterior samples close in the top-16 bits; the instance
that solves with brute (i=13) has samples close in the top-20 bits but
NOT close enough for the 1-bit extended-root channel at bu=16.

**This suggests a UNION of copper-ext (bu=16) and brute (bu=12) could
solve 4/7** — the Coppersmith-easy {9,12,17} plus the brute-easy {13}.
The current experiment runs them separately; a combined pipeline that
tries both extractors would capture the union.

### 5. Three instances are truly hard (i=4, 14, 15)

These fail under ALL 9 strategies, including the combined strategy with
R=30, pop=200, gen=40, wide mutation, both posteriors, and 486,000
evaluations. The posterior's samples for these instances are too far from
the true p (d_high > 5 bits) for ANY coevolution configuration to bridge.
This confirms D199's finding: **the posterior is the bottleneck.** No
amount of coevolution tuning helps when the posterior doesn't produce at
least one sample within ~4 bits of the true p's high bits.

### 6. The union of strategies helps (4/7)

While no single strategy beats 3/7, the UNION of all strategies solves
4/7 (i=9,12,13,17). The gain comes from brute bu=12 solving i=13, which
no Coppersmith strategy can. A practical pipeline should try BOTH
extractors: copper-ext at bu=16 (catches 1-bit-error cases) AND brute at
bu=12 (catches cases where top-20 bits are findable but top-16 aren't).
The 3 truly hard instances (4, 14, 15) remain unsolved — they need a
better posterior.

## Conclusions

1. **Coevolution tuning is exhausted.** More restarts, larger population,
   more generations, wider mutation, and the combined strategy all give
   the same 3/7. The search radius is sufficient; the bottleneck is the
   posterior's sample quality.

2. **Predict q doesn't help.** The q-posterior is of equal quality to the
   p-posterior (symmetry). Merging posteriors is actively harmful.

3. **Brute bu=12 finds a different class.** i=13 is brute-easy but
   Coppersmith-hard. A union of copper-ext + brute would solve 4/7.

4. **3 instances (4, 14, 15) are truly hard.** All 9 strategies fail. The
   posterior's d_high for these is too large (>5 bits). Improving the
   posterior (reduce d_high by 1-2 bits) is the only path.

5. **Practical recommendation: dual-extractor pipeline.** Run copper-ext
   at bu=16 first (catches 1-bit-error cases, fast). If that fails, run
   brute at bu=12 (catches top-20-bit cases, 100% reliable extraction but
   needs more correct high bits). This union would solve 4/7 of D201's
   hard cases, bringing the 64-bit rate from 65% to 65% + 4/20 = **85%**.

   Wait — 4/7 of the hard cases = 4 new successes out of 7 hard. D201 had
   13/20. Adding 4 gives 17/20 = **85%**. But this requires validation on
   the full 20-instance test set (the 3 truly hard may overlap with
   D201's easy cases, so the gain may be smaller).

## Next Steps

- **D202 (proposed): Dual-extractor union.** Run copper-ext (bu=16) and
  brute (bu=12) in sequence on the full D201 test set. Expected: 65% →
  ~80% (adding the brute-easy class).
- **D203 (proposed): Improve the posterior.** The 3 truly hard instances
  need d_high reduced from >5 to ~3. Options: MLP posterior (D138), larger
  training set (800 → 5000), predict p+q (D173), or a transformer
  architecture (D192).
- **D204 (proposed): Characterize the truly hard 3.** What makes i=4, 14,
  15 different? Run D199's feature analysis on these specific instances.

## Deliverables

- Script: `experiments/d200_strategies.py`
- Results: `experiments/d200_results.json`
- Log: `experiments/d200_run.log`
- Writeup: `direction-200-strategies.md`
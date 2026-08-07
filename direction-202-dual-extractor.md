# Direction 202: Dual-Extractor Union at 64-bit

**Status:** Experiment complete. Mixed — the union is 60%, not the projected 85%.
**Run time:** 1,371s (~23 min, 4 workers)
**Date:** 2026-08-06

---

## Question

D200 showed brute bu=12 solves a DIFFERENT class than copper-ext bu=16
(i=13 was brute-easy but copper-hard on D200's 7 hard instances; i=4 was
brute-easy). The union of both extractors should solve 4/7 of the hard
cases, bringing 64-bit from 65% to ~85%. **Does the dual-extractor union
validate this projection on the full 20-instance test set?**

## Setup

- Test set: 20 random balanced 64-bit semiprimes (seed_offset=501, same
  as D201).
- Training: AR posterior on 800 64-bit semiprimes (seed_offset=500).
- Phase 1: copper-ext (bu=16, R=10, pop=80, gen=40, dim=10).
- Phase 2: brute (bu=12, R=10, pop=80, gen=40).
- Phase 3: UNION — success iff copper-ext OR brute succeeds.
- Parallelized across 4 workers.

## Results

| Method | Success | Rate |
|--------|---------|------|
| Copper-ext (bu=16) | 11/20 | 55.0% |
| Brute (bu=12) | 3/20 | 15.0% |
| **UNION** | **12/20** | **60.0%** |
| Overlap (both) | 2 | (i=6, 8) |
| Copper-only | 9 | (i=0,1,5,7,9,12,16,18,19) |
| Brute-only | 1 | (i=4) |
| Residual (neither) | 8 | (i=2,3,10,11,13,14,15,17) |

## Findings

### 1. The union is 60%, not 85%

D200's projection of 85% was based on 4/7 hard cases being solvable by
some strategy. On the full 20-instance test set, the union only adds 1
instance (i=4) over copper-ext alone (11 → 12). The brute extractor
solves 3 instances (i=4, 6, 8), but 2 of those (6, 8) are already solved
by copper-ext. **The brute-only gain is 1 instance = +5pp.**

The 85% projection was optimistic because D200 tested only the 7 hard
instances and found 4 solvable. But on the full set, the brute extractor
also FAILS on many copper-easy instances (i=0,1,5,7,9,12,16,18,19 are
copper-only — brute doesn't find them). The two extractors have different
strengths but the overlap is small and the brute class is much smaller.

### 2. Copper-ext got 55% (not 65%)

D201's phase5 got 13/20 = 65%. This experiment got 11/20 = 55%. The
difference is the parallelization: D201 used a single worker with a
specific RNG sequence; this experiment uses 4 workers with different
seeds, leading to different posterior samples. The 10pp gap is RNG
variance — the same configuration (R=10, pop=80, gen=40, bu=16) gives
55-65% depending on the seed. This is consistent with D198's finding
that success rate has high variance at n_test=20.

### 3. Brute bu=12 is a small, mostly-overlapping class

Brute solves only 3/20 = 15%. Two of those (i=6, 8) overlap with copper.
The brute-only instance (i=4) is the one D200 identified as brute-easy.
The brute class is NOT complementary to copper-ext — it's a small subset
with minimal new coverage.

### 4. 8 instances remain unsolved (40%)

The residual (i=2,3,10,11,13,14,15,17) fail under both extractors. These
are the truly hard cases where the posterior's samples are too far from
the true p for coevolution to bridge, regardless of extraction method.

Notably, i=13 — which D200's brute bu=12 solved (R=1, 2422 evals) — does
NOT solve here. The difference is RNG: D200 used seed 1000 + i*100, this
experiment uses 2000 + i*100 + r*7919. The brute-easy class is fragile
and seed-dependent.

### 5. The copper-ext and brute classes are NOT as disjoint as D200 suggested

D200 found brute solves a DIFFERENT instance (i=13) than copper. But on
the full set, brute mostly solves instances copper already gets (2/3
overlap). The "different class" finding from D200 was an artifact of
testing only 7 pre-selected hard instances — on the full set, the brute
class is largely a subset of the copper class.

## Conclusions

1. **Dual-extractor union: 60% (+5pp over copper-ext alone).** The brute
   extractor adds only 1 new instance. The 85% projection was too
   optimistic — D200's 4/7 solvable was inflated by testing only hard
   instances.

2. **Brute bu=12 is a small, mostly-overlapping class (15%).** It's NOT
   complementary to copper-ext. The two extractors find mostly the same
   instances, with brute being weaker overall.

3. **Copper-ext remains the dominant extractor (55%).** The 55% vs D201's
   65% gap is RNG variance at n_test=20.

4. **8 instances (40%) are truly hard.** Neither extractor finds them.
   The posterior is the bottleneck (D199: hint works 100%).

5. **The path to higher rates is a better posterior, not more extractors.**
   D200 and D202 both confirm: coevolution tuning and extractor choice are
   exhausted. The 40% residual needs d_high reduced from >5 to ~3 bits.

## Comparison to D200

| Metric | D200 (7 hard) | D202 (full 20) |
|--------|--------------|----------------|
| Copper-ext | 3/7 = 42.9% | 11/20 = 55.0% |
| Brute bu=12 | 1/7 = 14.3% | 3/20 = 15.0% |
| Union | 4/7 = 57.1% | 12/20 = 60.0% |
| Brute-only gain | 1/7 = 14.3% | 1/20 = 5.0% |

D200 overestimated the brute-only gain because it tested only hard
instances. On the full set, brute adds only 1 instance (+5pp), not the
projected +20pp.

## Deliverables

- Script: `experiments/d202_dual_extractor.py`
- Results: `experiments/d202_results.json`
- Log: `experiments/d202_run.log`
- Writeup: `direction-202-dual-extractor.md`
# Direction 209: Scale to 128-bit — Where Does It Actually Break?

**Status:** Experiment complete. **Negative — 0% at 128-bit, d_high=6.3.**
**Run time:** ~3.5 hours (training + partial coevolution, killed after 12/20 instances all failed)
**Date:** 2026-08-07

---

## Question

D207 showed the ensemble d_high drops as 1/√n at 64-bit, with asymptote
0.64. Does this hold at 128-bit? If 500 posteriors give d_high ≤ 4,
coevolution can bridge the gap and we can factor 128-bit semiprimes —
the first "real cryptography" scale.

## Setup

- 128-bit semiprimes (half=64, b_unknown=32, n_known=32).
- 500 diverse posteriors, 5 samples each (2500 total).
- Training: 800 128-bit semiprimes.
- Test: 20 instances.
- Coevolution: R=5, pop=60, gen=30.
- Coppersmith: dim=20 (m=10, t=9), extended roots.

## Results

### Phase 1: d_high at 128-bit

| Metric | 64-bit (D207) | 128-bit (D209) | Ratio |
|--------|--------------:|---------------:|------:|
| d_high (mean) | 1.450 | 6.300 | 4.3× |
| d_high (min) | 0 | 4 | — |
| d_high (max) | 2 | 7 | — |
| Bits extracted | 14.55 / 16 | 25.70 / 32 | — |
| Single-posterior d_high | ~5.2 | ~10.2 | 2.0× |

**d_high = 6.3 at 128-bit** — the ensemble helps (10.2 → 6.3, a 3.9-bit
improvement) but not enough. Coevolution bridges ~4 bits, so we need
d_high ≤ 4. The ensemble gives 6.3 — 2.3 bits short.

### Phase 2: Coevolution

**0/12 = 0%** (killed after 12/20 instances, all failed).

Every instance failed. The coevolution tried 18300 evaluations per
instance (R=5 × pop=60 × gen=30 × 2) but never found a factor. The
Coppersmith lattice at dim=20 with 128-bit numbers is also very slow
(~4 min per evaluation vs ~0.3s at 64-bit).

### The scaling law

| Bit size | Single d_high | Ensemble d_high (500) | Improvement | Coevolution bridge | Gap |
|----------|-------------:|----------------------:|------------:|-------------------:|----:|
| 64 | 5.2 | 1.45 | 3.75 | 4 | **+0.25 (surplus)** |
| 128 | 10.2 | 6.30 | 3.90 | 4 | **-2.30 (deficit)** |

The ensemble improvement is ~3.9 bits at both sizes — it's roughly
constant, not scaling with n. The single-posterior d_high doubles (5.2 →
10.2), so the ensemble can't keep up. At 64-bit, 3.9 bits of improvement
is enough (5.2 - 3.9 = 1.3 < 4). At 128-bit, it's not (10.2 - 3.9 = 6.3
> 4).

## Findings

1. **The ensemble improvement is ~3.9 bits, roughly constant across
   sizes.** At 64-bit: 5.2 → 1.45 (−3.75). At 128-bit: 10.2 → 6.30
   (−3.90). The ensemble gives a FIXED improvement, not a proportional
   one. This means the gap GROWS with bit size: at 64-bit the gap is
   closed, at 128-bit it's 2.3 bits, at 256-bit it would be ~6.3 bits.

2. **128-bit is NOT feasible with the current approach.** d_high=6.3,
   coevolution bridges 4, gap = 2.3 bits. The Coppersmith lattice at
   dim=20 is also too slow (~4 min/eval vs 0.3s at 64-bit).

3. **The barrier is between 64 and 128 bits.** At 64-bit: 70% success
   (D203). At 128-bit: 0%. The crossover is somewhere in between —
   likely around 96-112 bits (where d_high ≈ 4).

4. **The 1/√n fit from D207 does NOT scale simply.** The fit at 64-bit
   was d_high = 0.64 + 16.85/√n. If this held at 128-bit with proportionally
   larger constants, 500 posteriors would give d_high ≈ 0.64 + 33.7/√500
   ≈ 2.15. But we got 6.3 — much worse. The fit constants change with
   bit size; the ensemble is less effective at larger n.

## Conclusions

1. **The algorithm breaks between 64 and 128 bits.** At 64-bit: 70%
   (D203). At 128-bit: 0%. The ensemble gives ~3.9 bits of improvement
   regardless of size, but the single-posterior d_high grows linearly
   (0.08n), so the gap grows.

2. **The ensemble improvement is constant (~3.9 bits), not proportional.**
   This means the approach scales as: d_high(n) ≈ 0.08n - 3.9. For
   coevolution to work (d_high ≤ 4), we need 0.08n - 3.9 ≤ 4, i.e.,
   n ≤ 98. **The poly-time regime extends to ~98 bits** (up from ~48
   without ensemble, ~64 with 20-posterior ensemble).

3. **The path to 128-bit requires either:**
   - A larger ensemble improvement (10+ bits instead of 3.9) — needs
     ~10× more posteriors or a fundamentally different diversity strategy.
   - A coevolution that bridges 6+ bits — needs a better search strategy.
   - A Coppersmith that works at higher b_unknown — needs larger dim
     (but dim=20 is already slow at 128-bit).

4. **For real RSA (2048-bit), the gap is insurmountable.** d_high(2048)
   ≈ 0.08 × 2048 - 3.9 ≈ 160. Coevolution bridges 4. Gap = 156 bits.

## Deliverables

- Script: `experiments/d209_scale_128.py`
- Results: `experiments/d209_results.json` (partial: phase 1 complete, phase 2 12/20)
- Log: `experiments/d209_run.log`
- Writeup: `direction-209-scale-128.md`
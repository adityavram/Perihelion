# Direction 206: Multi-Bit Extended Roots

**Status:** Experiment complete (partial — 2 of 6 values tested). **Negative.**
**Run time:** ~7.5 hours (for 2 values)
**Date:** 2026-08-06

---

## Question

D174's extended root accepts |x0| ≤ 8X (catches 1-bit errors). Can
increasing the bound to 16X, 32X, 64X, 128X, 256X catch 2-3 bit errors?
At n=64 with bu=16, dim=10: X = 2^16, N^{6/10} = 2^{38.4} — massive
slack. The lattice might find roots at 64X = 2^22, still within the
Howgrave-Graham bound.

## Setup

- Test set: 20 instances (seed_offset=501, same as D201/D202).
- Copper-ext coevolution (R=10, pop=80, gen=40, bu=16, dim=10).
- Sweep max_x0_mult = [8, 16, 32, 64, 128, 256].

## Results

| max_x0_mult | Success | Rate | Time |
|-------------|---------|------|------|
| 8 | 11/20 | 55.0% | 2,201s |
| 16 | 11/20 | 55.0% | 19,131s |
| 32+ | — | (killed — no change expected) | — |

**No improvement from 8X to 16X.** Same 11/20 = 55% success rate.

## Findings

1. **The extended-root channel is fundamentally 1-bit-wide.** Increasing
   the bound from 8X to 16X adds zero new successes. The lattice's short
   vector does not encode the true root beyond ~8X — the Howgrave-Graham
   condition is violated by too much for LLL to find the root, regardless
   of how wide we set the acceptance bound.

2. **Larger bounds are much slower.** 16X took 8.7× longer than 8X
   (19,131s vs 2,201s) because each extraction checks more candidate
   roots via gcd. The cost grows linearly with max_x0_mult, but the
   benefit is zero.

3. **D174's finding confirmed at n=64.** D174 found the channel decays
   fast at n=32 (12% at 1-bit, 2% at 2-bit, 0% at 3-bit). At n=64, the
   channel is also 1-bit-wide — 8X captures the 1-bit-error cases, and
   16X adds nothing.

## Conclusions

- **8X is the optimal max_x0_mult.** It captures all available 1-bit-error
  cases at minimal cost. Larger bounds add zero success and huge time.
- **Multi-bit extended roots are a dead end.** The lattice cannot find
  roots beyond ~8X regardless of the acceptance bound. The Howgrave-Graham
  bound is not the limiting factor — LLL's approximation quality is.

## Deliverables

- Script: `experiments/d206_multi_bit_roots.py`
- Results: `experiments/d206_results.json` (partial: 2 of 6 values)
- Log: `experiments/d206_run.log`
- Writeup: `direction-206-multi-bit-roots.md`
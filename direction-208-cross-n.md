# Direction 208: Cross-N Information — Breaking the Single-N Paradigm

**Status:** Experiment complete. **Negative — cross-N augmentation hurts.**
**Run time:** 263s (~4.4 min)
**Date:** 2026-08-06

---

## Question

Every prior direction extracts information from N alone. FM4 says
I(f(N); p) = O(1). What about I(f(N, N_1, ..., N_K); p)? Given a new N,
find the K most similar training N_i's (by feature distance), and feed
their features (and their known p_i's) into the posterior. Does the
RELATIONSHIP between N and N_i carry more than O(1) bits?

## Setup

- 4 augmentation modes × 5 K values = 20 configs:
  - `features_only`: append K neighbors' 89-dim features (K×89 extra)
  - `features_and_p`: append K neighbors' features AND p_high (K×90 extra)
  - `differences`: append (N - N_i) features (K×89 extra)
  - `all`: all of the above (K×179 extra)
- K = [1, 5, 10, 20, 50] nearest neighbors by feature distance.
- Leave-one-out for training instances.
- Baseline: single-N posterior (89 features, no augmentation).
- Test: 40 64-bit semiprimes. d_high = Hamming of best sample to true p
  over top 16 bits.

## Results

| Config | d_high (mean) | Improvement vs baseline |
|--------|--------------:|------------------------|
| **baseline** | **3.050** | **(ref)** |
| features_only_K1 | 3.225 | -0.175 |
| features_only_K5 | 3.850 | -0.800 |
| features_only_K10 | 4.250 | -1.200 |
| features_only_K50 | 4.700 | -1.650 |
| features_and_p_K1 | 3.325 | -0.275 |
| differences_K1 | 3.300 | -0.250 |
| all_K1 | 3.300 | -0.250 |
| all_K50 | 5.025 | -1.975 |

**Every single cross-N config is WORSE than the baseline.** More
neighbors = worse. The best cross-N config (features_only_K1) gives
d_high=3.225 vs 3.050 baseline — a regression of 0.175.

## Findings

1. **Cross-N features are noise, not signal.** The neighbors' features
   and p_high values do not help predict p for the test N. The similarity
   in feature space (which is O(1) bits by FM4) does not translate to
   similarity in p. Two semiprimes with similar spectral features have
   uncorrelated factors — the feature similarity is spurious.

2. **More neighbors = more noise.** d_high increases monotonically with
   K for every mode. Adding more neighbors' features increases the
   feature dimensionality without adding predictive signal — the extra
   features are pure noise that degrades the logistic regression.

3. **The p_i high bits don't transfer.** Even `features_and_p` (which
   includes the neighbors' known p_high values) doesn't help. Knowing
   p_i for a similar N_i doesn't tell you p for N — the relationship
   between "similar N" and "similar p" is exactly what FM4 says is O(1)
   bits.

4. **The single-N paradigm is correct.** FM4 applies to f(N) for any
   poly-time f. Cross-N features f(N, N_1, ..., N_K) are also poly-time,
   and the information about p is still O(1). The extra features increase
   dimensionality but not information.

## Conclusions

- **Cross-N information does not break FM4.** The augmented features are
  noise. The single-N paradigm is correct: all information about p in
  poly-time features of N is O(1) bits, regardless of how many N_i's
  you include.
- **The ensemble (D207) is the right approach, not cross-N.** Diversity
  (multiple models on the same N) helps. Cross-N (one model on N +
  neighbors) hurts. The information is in N's features, not in the
  relationship to other N's.

## Deliverables

- Script: `experiments/d208_cross_n.py`
- Results: `experiments/d208_results.json`
- Log: `experiments/d208_run.log`
- Writeup: `direction-208-cross-n.md`
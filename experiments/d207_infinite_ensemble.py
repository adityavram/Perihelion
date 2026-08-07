#!/usr/bin/env python
"""Direction 207: The Infinite Ensemble — Does FM4 Saturate?

D203 used 20 posteriors and got d_high=1.35 (5.15 bits). Does d_high
keep dropping with more posteriors, or does it saturate at ~6.5 bits
(the FM4 ceiling)?

This is THE critical experiment: if 10000 posteriors give d_high=0.1
(10 bits), FM4 is wrong and the barrier breaks. If it saturates at
~0.5 (6.5 bits), the barrier is confirmed.

Phase 1: Train N posteriors for N in [1, 5, 20, 50, 100, 200, 500].
Phase 2: For each N, measure d_high (best sample Hamming distance to
         true p over top 20 bits) on 40 test instances.
Phase 3: Plot d_high vs N. Does it saturate? If so, at what value?
         Compute bits extracted = 32 - d_high at saturation.

Uses production code. 64-bit semiprimes.
"""

import json
import math
import os
import sys
import time
import warnings
import random as _random
from math import gcd, isqrt, comb
from pathlib import Path

import numpy as np
from sympy import nextprime

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.spectral_factoring import (
    ARPosterior,
    FeatureExtractor,
    generate_dataset,
)
from src.spectral_factoring.posterior import _int_to_bits

warnings.filterwarnings("ignore")

LOG_PATH = str(_REPO_ROOT / "experiments" / "d207_run.log")
RESULTS_PATH = str(_REPO_ROOT / "experiments" / "d207_results.json")

BITS = 64
HALF = BITS // 2
B_UNKNOWN = 16
SEED_OFFSET_TRAIN = 500
SEED_OFFSET_TEST = 501
N_TRAIN = 800
N_TEST = 40  # more instances for stable measurement

# Ensemble sizes to test
ENSEMBLE_SIZES = [1, 5, 20, 50, 100, 200, 500]

# Generate a LARGE pool of diverse configs
# We need up to 500 independent posteriors. Use random bit orderings
# and random feature subsets to generate diversity.

# Feature subsets (fixed set of 4)
FEATURE_SUBSETS = {
    "all": list(range(89)),
    "spectral": list(range(33, 89)),
    "arithmetic": list(range(33)),
    "mod_residues": list(range(16)),
    "mod_plus_spectral": list(range(16)) + list(range(33, 89)),
    "arithmetic_plus_spectral": list(range(33)) + list(range(50, 89)),
    "first_half_spectral": list(range(33, 61)),
    "second_half_spectral": list(range(61, 89)),
}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def generate_semiprime(bits, seed):
    _random.seed(seed)
    half_bits = bits // 2
    p = nextprime(_random.getrandbits(half_bits) | (1 << (half_bits - 1)))
    q = nextprime(_random.getrandbits(half_bits) | (1 << (half_bits - 1)))
    while p == q:
        q = nextprime(_random.getrandbits(half_bits) | (1 << (half_bits - 1)))
    return int(p) * int(q), int(p), int(q)


def get_test_set():
    instances = []
    for i in range(N_TEST):
        seed = SEED_OFFSET_TEST * 10_000_000 + i
        N, p, q = generate_semiprime(BITS, seed)
        instances.append({"i": i, "N": N, "p": p, "q": q})
    return instances


def hamming(a, b):
    return int(bin(int(a) ^ int(b)).count("1"))


# ---------------------------------------------------------------------------
# Generate diverse configs
# ---------------------------------------------------------------------------
def generate_configs(n_posteriors):
    """Generate n_posteriors diverse configs (bit ordering + feature subset)."""
    configs = []
    # Use a mix of fixed and random bit orderings
    fixed_orderings = [
        list(range(HALF)),  # MSB-first
        list(reversed(range(HALF))),  # LSB-first
        list(range(0, HALF, 2)) + list(range(1, HALF, 2)),  # even-odd
    ]
    subset_names = list(FEATURE_SUBSETS.keys())

    for i in range(n_posteriors):
        if i < len(fixed_orderings):
            order = fixed_orderings[i]
        else:
            # Random bit ordering
            rng = np.random.default_rng(10000 + i)
            order = rng.permutation(HALF).tolist()
        # Cycle through feature subsets
        subset_name = subset_names[i % len(subset_names)]
        configs.append({
            "id": f"post_{i}",
            "bit_order": order,
            "feature_idx": FEATURE_SUBSETS[subset_name],
            "subset_name": subset_name,
        })
    return configs


# ---------------------------------------------------------------------------
# Train a single diverse posterior
# ---------------------------------------------------------------------------
def train_one_posterior(F_full, p_bits, config):
    """Train one posterior with the given config."""
    fidx = config["feature_idx"]
    order = config["bit_order"]
    F_sub = F_full[:, fidx]
    # Reorder bits
    pb_sub = np.zeros_like(p_bits)
    for j, bit_idx in enumerate(order):
        pb_sub[:, j] = p_bits[:, bit_idx]
    post = ARPosterior(n_features=len(fidx), n_bits=HALF, n_iter=300)
    post.train(F_sub, pb_sub)
    return post


def sample_one_posterior(post, config, feature_vec_full, n_samples, rng):
    """Sample from one posterior, returning p candidates in original bit space."""
    fidx = config["feature_idx"]
    order = config["bit_order"]
    f_sub = feature_vec_full[fidx]
    samples_bits = post.sample(f_sub, n_samples, rng)
    results = []
    for s in samples_bits:
        s = int(s)
        orig = 0
        for j in range(HALF):
            bit_j = (s >> j) & 1
            orig |= (bit_j << order[j])
        results.append(orig)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        open(LOG_PATH, "w").close()
    except Exception:
        pass
    with open(LOG_PATH, "a") as fh:
        fh.write(f"=== D207 started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log("Direction 207: The Infinite Ensemble — Does FM4 Saturate?")
    log(f"  Ensemble sizes: {ENSEMBLE_SIZES}")
    log(f"  Test set: {N_TEST} instances at {BITS}-bit")
    log(f"  Training: {N_TRAIN} semiprimes (seed_offset={SEED_OFFSET_TRAIN})")

    test_set = get_test_set()
    log(f"  Test instances generated")

    # Train feature extractor once
    log("  Training feature extractor...")
    dataset = generate_dataset(BITS, N_TRAIN, seed_offset=SEED_OFFSET_TRAIN)
    fe = FeatureExtractor(BITS)
    N_list = [d[0] for d in dataset]
    F_full = fe.fit_transform(N_list)
    p_list = [d[1] for d in dataset]
    p_bits = np.array([_int_to_bits(p, HALF) for p in p_list], dtype=np.float64)
    log(f"  Feature extractor ready (F_full shape: {F_full.shape})")

    # Pre-compute test features
    test_features = []
    for inst in test_set:
        f = fe.transform([inst["N"]])[0]
        test_features.append(f)

    # Train the MAXIMUM number of posteriors (500), then subsample
    max_n = max(ENSEMBLE_SIZES)
    log(f"  Training {max_n} posteriors...")
    t0 = time.time()
    all_configs = generate_configs(max_n)
    all_posteriors = []
    for ci, config in enumerate(all_configs):
        post = train_one_posterior(F_full, p_bits, config)
        all_posteriors.append((post, config))
        if (ci + 1) % 50 == 0:
            log(f"    Trained {ci+1}/{max_n} ({time.time()-t0:.1f}s)")
    log(f"  All {max_n} posteriors trained ({time.time()-t0:.1f}s)")

    # For each ensemble size, measure d_high
    all_results = {}
    t_start = time.time()

    # FIXED samples per posterior (not total). This way, more posteriors
    # = more total samples = more diversity. D203 used 80/posterior.
    PER_POST_SAMPLES = 5  # 500 posteriors × 5 = 2500 total samples

    for n_post in ENSEMBLE_SIZES:
        log(f"\n=== Ensemble size = {n_post} (per_post={PER_POST_SAMPLES}, "
            f"total={n_post * PER_POST_SAMPLES}) ===")
        posteriors = all_posteriors[:n_post]

        d_high_values = []
        d_high_per_post = []  # best d per posterior (for diversity analysis)

        for inst in test_set:
            N, p_true = inst["N"], inst["p"]
            i = inst["i"]
            p_high_true = p_true >> B_UNKNOWN
            rng = np.random.default_rng(7000 + i * 13)
            all_samples_d = []
            per_post_best = []
            for post, config in posteriors:
                samples = sample_one_posterior(post, config,
                                                test_features[i],
                                                PER_POST_SAMPLES, rng)
                for s in samples:
                    d = hamming(s >> B_UNKNOWN, p_high_true)
                    all_samples_d.append(d)
                if samples:
                    per_post_best.append(min(hamming(s >> B_UNKNOWN, p_high_true)
                                             for s in samples))
            ensemble_best = min(all_samples_d) if all_samples_d else 99
            d_high_values.append(ensemble_best)
            d_high_per_post.append(per_post_best)

        mean_d = float(np.mean(d_high_values))
        median_d = float(np.median(d_high_values))
        min_d = int(np.min(d_high_values))
        max_d = int(np.max(d_high_values))
        bits_extracted = HALF - B_UNKNOWN - mean_d  # n_known - d_high
        # Actually: bits extracted = log2(2^n_known / 2^d_high) = n_known - d_high
        # But more precisely: if d_high is the Hamming distance over n_known bits,
        # the posterior "knows" n_known - d_high bits. The concentration is
        # n_known - d_high bits out of n_known.

        log(f"  d_high: mean={mean_d:.3f} median={median_d:.0f} "
            f"min={min_d} max={max_d}")
        log(f"  Bits extracted (n_known - d_high): {HALF - B_UNKNOWN - mean_d:.3f}")
        log(f"  Samples per posterior: {PER_POST_SAMPLES}, total: {n_post * PER_POST_SAMPLES}")

        all_results[f"ensemble_{n_post}"] = {
            "n_posteriors": n_post,
            "n_test": N_TEST,
            "per_post_samples": PER_POST_SAMPLES,
            "total_samples": n_post * PER_POST_SAMPLES,
            "d_high_mean": mean_d,
            "d_high_median": median_d,
            "d_high_min": min_d,
            "d_high_max": max_d,
            "bits_extracted": HALF - B_UNKNOWN - mean_d,
            "d_high_per_instance": [int(d) for d in d_high_values],
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    log(f"\n=== SATURATION ANALYSIS ===")
    log(f"{'N_post':<10} {'d_high_mean':<15} {'bits_extracted':<15} {'samples':<10}")
    log(f"{'-'*10} {'-'*15} {'-'*15} {'-'*10}")
    for n_post in ENSEMBLE_SIZES:
        key = f"ensemble_{n_post}"
        if key in all_results:
            r = all_results[key]
            log(f"  {n_post:<10} {r['d_high_mean']:<15.3f} "
                f"{r['bits_extracted']:<15.3f} {r['total_samples']:<10}")

    # Fit: does it saturate?
    d_means = [all_results[f"ensemble_{n}"]["d_high_mean"]
               for n in ENSEMBLE_SIZES if f"ensemble_{n}" in all_results]
    n_posts = [n for n in ENSEMBLE_SIZES if f"ensemble_{n}" in all_results]
    # Fit d_high = a + b/sqrt(n) (decreasing, saturating)
    # or d_high = a * exp(-b * n) (exponential decay)
    # or d_high = a + b/n (power law)
    log(f"\n  Curve fit:")
    # Simple: compute the marginal improvement
    for k in range(1, len(n_posts)):
        improvement = d_means[k-1] - d_means[k]
        log(f"    {n_posts[k-1]} → {n_posts[k]}: "
            f"d_high {d_means[k-1]:.3f} → {d_means[k]:.3f} "
            f"(improvement = {improvement:.3f})")

    # Extrapolate: if d_high ~ a + b/sqrt(n), what's the asymptote?
    if len(d_means) >= 3:
        # Fit d_high = a + b/sqrt(n) via last 3 points
        from numpy.linalg import lstsq
        xs = np.array([1.0 / math.sqrt(n) for n in n_posts[-3:]])
        ys = np.array(d_means[-3:])
        A = np.vstack([np.ones(3), xs]).T
        coeffs, _, _, _ = lstsq(A, ys, rcond=None)
        a, b = coeffs
        log(f"\n  Fit: d_high = {a:.3f} + {b:.3f}/sqrt(n)")
        log(f"  Asymptotic d_high (n → ∞): {a:.3f}")
        log(f"  Asymptotic bits extracted: {HALF - B_UNKNOWN - a:.3f}")
        all_results["fit"] = {
            "model": "d_high = a + b/sqrt(n)",
            "a": float(a), "b": float(b),
            "asymptotic_d_high": float(a),
            "asymptotic_bits": float(HALF - B_UNKNOWN - a),
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    log(f"\nTOTAL TIME: {time.time() - t_start:.1f}s")
    log("Final JSON written.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Direction 208: Cross-N Information — Breaking the Single-N Paradigm.

Every prior direction extracts information from N ALONE. FM4 says
I(f(N); p) = O(1). But what about I(f(N, N_1, ..., N_K); p)?

Given a new N, find the K most similar training N_i's (by feature
distance). Feed BOTH N's features AND the N_i's features (plus their
known p_i's) into the posterior. The RELATIONSHIP between N and N_i
might carry more than O(1) bits.

Phase 1: For each test N, find K nearest training N_i's. Measure d_high
         with and without the cross-N features.
Phase 2: Coevolution with cross-N posterior samples.

The KEY test: does the cross-N posterior have lower d_high than the
single-N posterior?
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

LOG_PATH = str(_REPO_ROOT / "experiments" / "d208_run.log")
RESULTS_PATH = str(_REPO_ROOT / "experiments" / "d208_results.json")

BITS = 64
HALF = BITS // 2
B_UNKNOWN = 16
N_KNOWN = HALF - B_UNKNOWN  # 16 high bits
SEED_OFFSET_TRAIN = 500
SEED_OFFSET_TEST = 501
N_TRAIN = 800
N_TEST = 40

# K values for nearest neighbors
K_NEIGHBORS_LIST = [1, 5, 10, 20, 50]

# Feature augmentation modes
# "features_only": append N_i's features (K × 89)
# "features_and_p": append N_i's features AND p_i's high bits (K × (89 + n_known))
# "differences": append (N - N_i) features and (p - p_i) high bits
# "all": all of the above
AUGMENT_MODES = ["features_only", "features_and_p", "differences", "all"]


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
# Cross-N feature augmentation
# ---------------------------------------------------------------------------
def build_cross_n_features(N_features, train_features, train_p_high, K, mode):
    """Build augmented features for one test instance.

    Args:
        N_features: (89,) features of the test N.
        train_features: (n_train, 89) features of all training N_i's.
        train_p_high: (n_train,) high bits of training p_i's (top N_KNOWN bits).
        K: number of nearest neighbors to use.
        mode: augmentation mode.

    Returns:
        augmented_features: (aug_dim,) augmented feature vector.
    """
    # Find K nearest neighbors by feature distance
    dists = np.sum((train_features - N_features) ** 2, axis=1)
    k_idx = np.argsort(dists)[:K]
    # Augment
    if mode == "features_only":
        # Append the K neighbors' features (K × 89)
        aug = np.concatenate([N_features] + [train_features[k] for k in k_idx])
    elif mode == "features_and_p":
        # Append neighbors' features AND their p_high bits (K × (89 + 1))
        # p_high is a single integer; normalize it
        aug_parts = [N_features]
        for k in k_idx:
            aug_parts.append(train_features[k])
            aug_parts.append(np.array([train_p_high[k] / (1 << N_KNOWN)],
                                      dtype=np.float64))
        aug = np.concatenate(aug_parts)
    elif mode == "differences":
        # Append differences: (N_features - N_i_features) and (p_high - p_i_high)
        aug_parts = [N_features]
        for k in k_idx:
            aug_parts.append(N_features - train_features[k])
            aug_parts.append(np.array([(train_p_high[k]) / (1 << N_KNOWN)],
                                      dtype=np.float64))
        aug = np.concatenate(aug_parts)
    elif mode == "all":
        # All of the above
        aug_parts = [N_features]
        for k in k_idx:
            aug_parts.append(train_features[k])
            aug_parts.append(N_features - train_features[k])
            aug_parts.append(np.array([train_p_high[k] / (1 << N_KNOWN)],
                                      dtype=np.float64))
        aug = np.concatenate(aug_parts)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return aug


def build_all_cross_n_features(test_features, train_features, train_p_high,
                                 K, mode):
    """Build augmented features for ALL test instances."""
    aug_list = []
    for i in range(len(test_features)):
        aug = build_cross_n_features(test_features[i], train_features,
                                       train_p_high, K, mode)
        aug_list.append(aug)
    return np.array(aug_list, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        open(LOG_PATH, "w").close()
    except Exception:
        pass
    with open(LOG_PATH, "a") as fh:
        fh.write(f"=== D208 started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log("Direction 208: Cross-N Information — Breaking the Single-N Paradigm")
    log(f"  Test set: {N_TEST} instances at {BITS}-bit")
    log(f"  Training: {N_TRAIN} semiprimes (seed_offset={SEED_OFFSET_TRAIN})")
    log(f"  K neighbors: {K_NEIGHBORS_LIST}")
    log(f"  Augment modes: {AUGMENT_MODES}")

    # Build training data
    log("  Building training data...")
    train_dataset = generate_dataset(BITS, N_TRAIN, seed_offset=SEED_OFFSET_TRAIN)
    fe = FeatureExtractor(BITS)
    train_N = [d[0] for d in train_dataset]
    train_F = fe.fit_transform(train_N)
    train_p = [d[1] for d in train_dataset]
    train_p_high = np.array([p >> B_UNKNOWN for p in train_p], dtype=np.float64)
    train_p_bits = np.array([_int_to_bits(p, HALF) for p in train_p],
                            dtype=np.float64)
    log(f"  Training features: {train_F.shape}")

    # Build test data
    test_set = get_test_set()
    test_N = [inst["N"] for inst in test_set]
    test_F = fe.transform(test_N)
    log(f"  Test features: {test_F.shape}")

    all_results = {}
    t_start = time.time()

    # Baseline: single-N posterior (no augmentation)
    log("\n=== Baseline: Single-N posterior (no augmentation) ===")
    post_base = ARPosterior(n_features=89, n_bits=HALF, n_iter=300)
    post_base.train(train_F, train_p_bits)
    d_high_base = []
    for inst in test_set:
        p_true = inst["p"]
        p_high_true = p_true >> B_UNKNOWN
        f = fe.transform([inst["N"]])[0]
        rng = np.random.default_rng(7000 + inst["i"] * 13)
        samples = post_base.sample(f, 80, rng)
        d = min(hamming(int(s) >> B_UNKNOWN, p_high_true) for s in samples)
        d_high_base.append(d)
    log(f"  Baseline d_high: mean={np.mean(d_high_base):.3f} "
        f"median={np.median(d_high_base):.0f}")
    all_results["baseline"] = {
        "d_high_mean": float(np.mean(d_high_base)),
        "d_high_median": float(np.median(d_high_base)),
        "d_high_per_instance": [int(d) for d in d_high_base],
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Cross-N: for each K and mode, build augmented features and train
    for mode in AUGMENT_MODES:
        for K in K_NEIGHBORS_LIST:
            log(f"\n=== Mode={mode}, K={K} ===")
            # Build augmented training features
            # For training, each training instance uses the OTHER training
            # instances as neighbors (leave-one-out)
            log("  Building augmented training features...")
            t0 = time.time()
            aug_train = []
            for i in range(N_TRAIN):
                # Leave-one-out: exclude instance i from neighbors
                mask = np.ones(N_TRAIN, dtype=bool)
                mask[i] = False
                aug = build_cross_n_features(
                    train_F[i], train_F[mask], train_p_high[mask], K, mode)
                aug_train.append(aug)
            aug_train = np.array(aug_train, dtype=np.float64)
            log(f"  Augmented train features: {aug_train.shape} "
                f"({time.time()-t0:.1f}s)")

            # Standardize
            mean = aug_train.mean(axis=0)
            std = aug_train.std(axis=0) + 1e-9
            aug_train_std = (aug_train - mean) / std

            # Train posterior on augmented features
            log("  Training posterior on augmented features...")
            t0 = time.time()
            post_aug = ARPosterior(n_features=aug_train_std.shape[1],
                                     n_bits=HALF, n_iter=300)
            post_aug.train(aug_train_std, train_p_bits)
            log(f"  Trained ({time.time()-t0:.1f}s)")

            # Build augmented test features
            log("  Building augmented test features...")
            t0 = time.time()
            aug_test = build_all_cross_n_features(test_F, train_F,
                                                    train_p_high, K, mode)
            aug_test_std = (aug_test - mean) / std
            log(f"  Augmented test features: {aug_test_std.shape} "
                f"({time.time()-t0:.1f}s)")

            # Measure d_high
            d_high_aug = []
            for inst in test_set:
                p_true = inst["p"]
                p_high_true = p_true >> B_UNKNOWN
                i = inst["i"]
                f = aug_test_std[i]
                rng = np.random.default_rng(7000 + i * 13)
                samples = post_aug.sample(f, 80, rng)
                d = min(hamming(int(s) >> B_UNKNOWN, p_high_true)
                        for s in samples)
                d_high_aug.append(d)
            mean_d = float(np.mean(d_high_aug))
            median_d = float(np.median(d_high_aug))
            log(f"  d_high: mean={mean_d:.3f} median={median_d:.0f} "
                f"(baseline: {np.mean(d_high_base):.3f})")
            improvement = float(np.mean(d_high_base)) - mean_d
            log(f"  Improvement: {improvement:+.3f}")

            all_results[f"{mode}_K{K}"] = {
                "mode": mode, "K": K,
                "aug_dim": int(aug_train.shape[1]),
                "d_high_mean": mean_d,
                "d_high_median": median_d,
                "improvement": improvement,
                "d_high_per_instance": [int(d) for d in d_high_aug],
            }
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    # Summary
    log(f"\n=== SUMMARY ===")
    log(f"{'Config':<30} {'d_high_mean':<15} {'improvement':<15}")
    log(f"{'-'*30} {'-'*15} {'-'*15}")
    log(f"  {'baseline':<30} {all_results['baseline']['d_high_mean']:<15.3f} {'(ref)':<15}")
    for key in sorted(all_results.keys()):
        if key == "baseline":
            continue
        r = all_results[key]
        log(f"  {key:<30} {r['d_high_mean']:<15.3f} {r['improvement']:<+15.3f}")

    # Best config
    best_key = min((k for k in all_results if k != "baseline"),
                   key=lambda k: all_results[k]["d_high_mean"])
    log(f"\n  Best: {best_key} d_high={all_results[best_key]['d_high_mean']:.3f} "
        f"(baseline: {all_results['baseline']['d_high_mean']:.3f})")

    log(f"\nTOTAL TIME: {time.time() - t_start:.1f}s")
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log("Final JSON written.")


if __name__ == "__main__":
    main()
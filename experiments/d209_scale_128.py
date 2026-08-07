#!/usr/bin/env python
"""Direction 209: Scale to 128-bit — Where Does It Actually Break?

D207 showed the ensemble d_high drops as 1/sqrt(n_posteriors) at 64-bit,
with asymptote 0.64. Does this hold at 128-bit? If 500 posteriors give
d_high <= 4 at 128-bit, coevolution can bridge the gap and we can factor
128-bit semiprimes — the first "real cryptography" scale.

Phase 1: Train 500 diverse posteriors at 128-bit, measure d_high.
Phase 2: If d_high <= 5, attempt coevolution + Coppersmith factoring.
         Coppersmith at bu=32, dim ~20 (m=10, t=9).
Phase 3: Measure success rate.

128-bit: half=64, n_known=32 (b_unknown=32). Coevolution bridges ~4 bits.
Need d_high <= 4 for any hope. D207 fit at 64-bit: d_high(500) = 0.64 +
16.85/sqrt(500) = 1.39. At 128-bit, single-posterior d_high ~ 0.08*128 =
10.2. If the fit scales: d_high(500) ~ a_128 + b_128/sqrt(500).
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
from src.spectral_factoring.extraction import (
    _reduce_lattice, _find_roots_extended,
)

warnings.filterwarnings("ignore")

LOG_PATH = str(_REPO_ROOT / "experiments" / "d209_run.log")
RESULTS_PATH = str(_REPO_ROOT / "experiments" / "d209_results.json")

BITS = 128
HALF = BITS // 2  # 64
B_UNKNOWN = 32  # Coppersmith bound: n/4 = 32
N_KNOWN = HALF - B_UNKNOWN  # 32
SEED_OFFSET_TRAIN = 500
SEED_OFFSET_TEST = 501
N_TRAIN = 800
N_TEST = 20

# Ensemble
N_POSTERIORS = 500
PER_POST_SAMPLES = 5

# Coppersmith
COPPERSMITH_M = 10
COPPERSMITH_T = 9
COPPERSMITH_DIM = COPPERSMITH_M + 1 + COPPERSMITH_T  # 20

# Coevolution
R_RESTARTS = 5  # fewer restarts (each is expensive at 128-bit)
POP = 60
GEN = 30

# Feature subsets
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


def generate_configs(n_posteriors):
    fixed_orderings = [
        list(range(HALF)),
        list(reversed(range(HALF))),
        list(range(0, HALF, 2)) + list(range(1, HALF, 2)),
    ]
    subset_names = list(FEATURE_SUBSETS.keys())
    configs = []
    for i in range(n_posteriors):
        if i < len(fixed_orderings):
            order = fixed_orderings[i]
        else:
            rng = np.random.default_rng(10000 + i)
            order = rng.permutation(HALF).tolist()
        subset_name = subset_names[i % len(subset_names)]
        configs.append({
            "id": f"post_{i}",
            "bit_order": order,
            "feature_idx": FEATURE_SUBSETS[subset_name],
        })
    return configs


def train_one_posterior(F_full, p_bits, config):
    fidx = config["feature_idx"]
    order = config["bit_order"]
    F_sub = F_full[:, fidx]
    pb_sub = np.zeros_like(p_bits)
    for j, bit_idx in enumerate(order):
        pb_sub[:, j] = p_bits[:, bit_idx]
    post = ARPosterior(n_features=len(fidx), n_bits=HALF, n_iter=300)
    post.train(F_sub, pb_sub)
    return post


def sample_one_posterior(post, config, feature_vec_full, n_samples, rng):
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
# Coppersmith for 128-bit
# ---------------------------------------------------------------------------
def coppersmith_lattice(p_high, N, b_unknown, m, t):
    X = 1 << b_unknown
    dim = (m + 1) + t
    p_high_i = int(p_high)
    N_i = int(N)
    rows = []
    for j in range(m + 1):
        r = [0] * dim
        for k in range(j + 1):
            r[k] = (int(comb(j, k)) * (N_i ** (m - j))
                    * (X ** k) * (p_high_i ** (j - k)))
        rows.append(r)
    for i in range(1, t + 1):
        r = [0] * dim
        for k in range(m + 1):
            r[i + k] = ((X ** (i + k)) * int(comb(m, k))
                        * (p_high_i ** (m - k)))
        rows.append(r)
    reduced = _reduce_lattice(rows)
    return reduced, X, dim


def extract_copper_ext(p_high, N, b_unknown, try_q=True):
    N_i = int(N)
    reduced, X, dim = coppersmith_lattice(p_high, N_i, b_unknown,
                                           COPPERSMITH_M, COPPERSMITH_T)
    factor = _find_roots_extended(reduced, p_high, N_i, X, dim, max_x0_mult=8)
    if factor is not None and 1 < factor < N_i and gcd(int(factor), N_i) > 1:
        return int(factor)
    if try_q:
        q_approx = N_i // (int(p_high) | 1)
        q_high = (q_approx >> b_unknown) << b_unknown
        if q_high > 0:
            reduced, X, dim = coppersmith_lattice(q_high, N_i, b_unknown,
                                                   COPPERSMITH_M, COPPERSMITH_T)
            factor = _find_roots_extended(reduced, q_high, N_i, X, dim,
                                           max_x0_mult=8)
            if factor is not None and 1 < factor < N_i and gcd(int(factor), N_i) > 1:
                return int(factor)
    return None


# ---------------------------------------------------------------------------
# Coevolution
# ---------------------------------------------------------------------------
class Coevol128:
    def __init__(self, N, half, b_unknown, pop_size=60, seed=0):
        self.N = int(N)
        self.half = half
        self.b_unknown = b_unknown
        self.n_known = half - b_unknown
        self.mask = (1 << self.n_known) - 1
        self.shift = b_unknown
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)
        self.eval_count = 0

    def _product_gap(self, p_hb, q_hb):
        return abs(self.N - (p_hb << self.shift) * (q_hb << self.shift))

    def _extract(self, p_hb):
        p_high = p_hb << self.shift
        self.eval_count += 1
        return extract_copper_ext(p_high, self.N, self.b_unknown)

    def _mutate(self, v):
        n_flip = 1 if self.rng.random() < 0.7 else 2
        for _ in range(n_flip):
            v ^= (1 << int(self.rng.integers(0, self.n_known)))
        return v & self.mask

    def run(self, init_p, init_q, generations=GEN):
        popP = [int(h) >> self.shift & self.mask for h in init_p[:self.pop_size]]
        popQ = [int(h) >> self.shift & self.mask for h in init_q[:self.pop_size]]
        while len(popP) < self.pop_size:
            popP.append(int(self.rng.integers(0, 1 << self.n_known)))
        while len(popQ) < self.pop_size:
            popQ.append(int(self.rng.integers(0, 1 << self.n_known)))
        for p_hb in popP:
            f = self._extract(p_hb)
            if f is not None:
                return f, self.eval_count
        for gen in range(generations):
            bestP = min(popP, key=lambda p: self._product_gap(p, popQ[0]))
            bestQ = min(popQ, key=lambda q: self._product_gap(bestP, q))
            newP = [self._mutate(p) for p in popP]
            newQ = [self._mutate(q) for q in popQ]
            for p_hb in newP:
                f = self._extract(p_hb)
                if f is not None:
                    return f, self.eval_count
            for q_hb in newQ:
                f = self._extract(q_hb)
                if f is not None:
                    return f, self.eval_count
            scoredP = sorted([(self._product_gap(p, bestQ), p) for p in newP],
                             key=lambda x: x[0])
            scoredQ = sorted([(self._product_gap(bestP, q), q) for q in newQ],
                             key=lambda x: x[0])
            popP = [p for _, p in scoredP[:self.pop_size]]
            popQ = [q for _, q in scoredQ[:self.pop_size]]
        return None, self.eval_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        open(LOG_PATH, "w").close()
    except Exception:
        pass
    with open(LOG_PATH, "a") as fh:
        fh.write(f"=== D209 started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log("Direction 209: Scale to 128-bit — Where Does It Actually Break?")
    log(f"  {BITS}-bit, half={HALF}, b_unknown={B_UNKNOWN}, n_known={N_KNOWN}")
    log(f"  {N_POSTERIORS} posteriors, {PER_POST_SAMPLES} samples each")
    log(f"  Test set: {N_TEST} instances")
    log(f"  Coevolution: R={R_RESTARTS}, pop={POP}, gen={GEN}")
    log(f"  Coppersmith: dim={COPPERSMITH_DIM} (m={COPPERSMITH_M}, t={COPPERSMITH_T})")

    # Build training data
    log("  Building training data...")
    t0 = time.time()
    train_dataset = generate_dataset(BITS, N_TRAIN, seed_offset=SEED_OFFSET_TRAIN)
    fe = FeatureExtractor(BITS)
    train_N = [d[0] for d in train_dataset]
    train_F = fe.fit_transform(train_N)
    train_p = [d[1] for d in train_dataset]
    train_p_bits = np.array([_int_to_bits(p, HALF) for p in train_p],
                            dtype=np.float64)
    log(f"  Training features: {train_F.shape} ({time.time()-t0:.1f}s)")

    # Build test data
    test_set = get_test_set()
    test_N = [inst["N"] for inst in test_set]
    test_F = fe.transform(test_N)
    log(f"  Test features: {test_F.shape}")

    # Train 500 diverse posteriors
    log(f"  Training {N_POSTERIORS} posteriors...")
    t0 = time.time()
    all_configs = generate_configs(N_POSTERIORS)
    all_posteriors = []
    for ci, config in enumerate(all_configs):
        post = train_one_posterior(train_F, train_p_bits, config)
        all_posteriors.append((post, config))
        if (ci + 1) % 50 == 0:
            log(f"    Trained {ci+1}/{N_POSTERIORS} ({time.time()-t0:.1f}s)")
    log(f"  All {N_POSTERIORS} posteriors trained ({time.time()-t0:.1f}s)")

    all_results = {}
    t_start = time.time()

    # Phase 1: Measure d_high
    log("\n=== Phase 1: Measure d_high with 500-posterior ensemble ===")
    d_high_values = []
    for inst in test_set:
        N, p_true = inst["N"], inst["p"]
        i = inst["i"]
        p_high_true = p_true >> B_UNKNOWN
        rng = np.random.default_rng(7000 + i * 13)
        all_samples_d = []
        for post, config in all_posteriors:
            samples = sample_one_posterior(post, config, test_F[i],
                                            PER_POST_SAMPLES, rng)
            for s in samples:
                d = hamming(s >> B_UNKNOWN, p_high_true)
                all_samples_d.append(d)
        ensemble_best = min(all_samples_d) if all_samples_d else 99
        d_high_values.append(ensemble_best)
        log(f"    i={i}: d_high={ensemble_best}")

    mean_d = float(np.mean(d_high_values))
    median_d = float(np.median(d_high_values))
    min_d = int(np.min(d_high_values))
    max_d = int(np.max(d_high_values))
    bits_extracted = N_KNOWN - mean_d
    log(f"  d_high: mean={mean_d:.3f} median={median_d:.0f} "
        f"min={min_d} max={max_d}")
    log(f"  Bits extracted: {bits_extracted:.3f} / {N_KNOWN}")

    all_results["phase1_d_high"] = {
        "n_posteriors": N_POSTERIORS,
        "n_test": N_TEST,
        "d_high_mean": mean_d,
        "d_high_median": median_d,
        "d_high_min": min_d,
        "d_high_max": max_d,
        "bits_extracted": bits_extracted,
        "d_high_per_instance": [int(d) for d in d_high_values],
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Phase 2: Coevolution (if d_high is low enough)
    log("\n=== Phase 2: Coevolution with ensemble samples ===")
    # Only attempt if d_high <= 8 (coevolution bridges ~4, need some slack)
    if mean_d > 10:
        log(f"  d_high={mean_d:.1f} too high for coevolution. Skipping.")
        all_results["phase2_coevolution"] = {
            "skipped": True,
            "reason": f"d_high={mean_d:.1f} > 10",
        }
    else:
        results = []
        t0 = time.time()
        for inst in test_set:
            N, p_true, q_true = inst["N"], inst["p"], inst["q"]
            i = inst["i"]
            found = False
            total_evals = 0
            first_R = None
            for r in range(R_RESTARTS):
                seed = 1000 + i * 100 + r * 7919
                rng = np.random.default_rng(seed)
                # Sample from all posteriors
                p_batch = []
                for post, config in all_posteriors:
                    p_batch.extend(sample_one_posterior(post, config,
                                                         test_F[i],
                                                         PER_POST_SAMPLES, rng))
                q_batch = [int(N // s) | 1 if s > 0 else 1 for s in p_batch]
                ga = Coevol128(N, HALF, B_UNKNOWN, pop_size=POP, seed=seed)
                factor, ev = ga.run(p_batch, q_batch, generations=GEN)
                total_evals += ev
                if factor is not None and 1 < factor < N and gcd(int(factor), N) > 1:
                    found = True
                    first_R = r + 1
                    break
            results.append({
                "i": i, "N": int(N), "success": found,
                "first_success_R": first_R, "evals": total_evals,
                "time": time.time() - t0,
            })
            status = "OK" if found else "FAIL"
            log(f"    i={i}: {status} (R={first_R}, evals={total_evals}, "
                f"{time.time()-t0:.1f}s)")
        succ = sum(1 for r in results if r["success"])
        rate = succ / len(test_set)
        log(f"  128-bit coevolution: {succ}/{len(test_set)} = {rate*100:.1f}%")
        all_results["phase2_coevolution"] = {
            "n_test": len(test_set), "success": succ,
            "success_rate": rate, "per_instance": results,
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    log(f"\n=== SUMMARY ===")
    log(f"  128-bit ensemble d_high: mean={mean_d:.3f}")
    log(f"  64-bit ensemble d_high (D207, 500 post): 1.450")
    log(f"  Ratio: {mean_d / 1.450:.1f}x")
    if "phase2_coevolution" in all_results:
        p2 = all_results["phase2_coevolution"]
        if "skipped" in p2:
            log(f"  Coevolution: skipped ({p2['reason']})")
        else:
            log(f"  Coevolution: {p2['success']}/{p2['n_test']} = "
                f"{p2['success_rate']*100:.1f}%")

    log(f"\nTOTAL TIME: {time.time() - t_start:.1f}s")
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log("Final JSON written.")


if __name__ == "__main__":
    main()
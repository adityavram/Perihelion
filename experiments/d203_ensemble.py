#!/usr/bin/env python
"""Direction 203: Ensemble of Diverse Posteriors.

Train 20+ INDEPENDENT posteriors with different bit orderings, feature
subsets, and seeds. Take the UNION of all samples across all posteriors.
If the best sample across diverse posteriors has d_high < 5, the union
gives coevolution a better starting point.

Phase 1: Train 20 diverse posteriors, measure d_high of the union vs single.
Phase 2: Run coevolution with the union samples on the 20-instance test set.
"""

import json
import math
import os
import sys
import time
import warnings
import random as _random
import multiprocessing as mp
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

LOG_PATH = str(_REPO_ROOT / "experiments" / "d203_run.log")
RESULTS_PATH = str(_REPO_ROOT / "experiments" / "d203_results.json")

BITS = 64
HALF = BITS // 2
B_UNKNOWN = 16
COPPERSMITH_M = 5
COPPERSMITH_T = 4
SEED_OFFSET_TRAIN = 500
SEED_OFFSET_TEST = 501
N_TRAIN = 800
N_TEST = 20
R_RESTARTS = 10
POP = 80
GEN = 40

# Ensemble config
N_POSTERIORS = 20


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


# ---------------------------------------------------------------------------
# Diverse posterior configurations
# ---------------------------------------------------------------------------
def get_ensemble_configs():
    """Generate 20 diverse posterior configurations."""
    configs = []
    # 5 bit orderings: MSB-first (default), LSB-first, even-odd, random-1, random-2
    orderings = [
        list(range(HALF)),  # MSB-first (default)
        list(reversed(range(HALF))),  # LSB-first
        list(range(0, HALF, 2)) + list(range(1, HALF, 2)),  # even-odd
        np.random.default_rng(42).permutation(HALF).tolist(),  # random-1
        np.random.default_rng(99).permutation(HALF).tolist(),  # random-2
    ]
    # 4 feature subsets: all (89), spectral-only (56), arithmetic-only (33), mod-only (16)
    feature_subsets = {
        "all": list(range(89)),
        "spectral": list(range(33, 89)),  # spectral features are indices 33-88
        "arithmetic": list(range(33)),  # arithmetic features are indices 0-32
        "mod_residues": list(range(16)),  # first 16 features are mod residues
    }
    # Combine: 5 orderings × 4 feature subsets = 20 configs
    for oi, order in enumerate(orderings):
        for fi, (fname, fidx) in enumerate(feature_subsets.items()):
            configs.append({
                "id": f"o{oi}_f{fname}",
                "bit_order": order,
                "feature_idx": fidx,
                "n_features": len(fidx),
                "seed": 1000 + oi * 100 + fi,
            })
    return configs


def train_diverse_posterior(dataset, config):
    """Train one posterior with the given config (bit ordering + feature subset)."""
    fe = FeatureExtractor(BITS)
    N_list = [d[0] for d in dataset]
    F_all = fe.fit_transform(N_list)
    F = F_all[:, config["feature_idx"]]  # feature subset
    p_list = [d[1] for d in dataset]
    # Reorder bits
    order = config["bit_order"]
    p_bits = np.zeros((len(p_list), HALF), dtype=np.float64)
    for j, bit_idx in enumerate(order):
        for k, p in enumerate(p_list):
            p_bits[k, j] = _int_to_bits(p, HALF)[bit_idx]
    post = ARPosterior(n_features=len(config["feature_idx"]), n_bits=HALF)
    post.train(F, p_bits)
    return fe, post, config


def sample_diverse(fe, post, config, N, n_samples, rng):
    """Sample from a diverse posterior, returning p candidates in original bit order."""
    f_full = fe.transform([N])[0]
    f = f_full[config["feature_idx"]]
    samples_bits = post.sample(f, n_samples, rng)  # in reordered bit space
    # Convert back: each sample is an int in reordered bit space
    # Need to un-reorder: bit j of sample corresponds to bit order[j] of p
    order = config["bit_order"]
    results = []
    for s in samples_bits:
        s = int(s)
        # s has bits in reordered space: bit j = order[j]
        # Convert to original space
        orig = 0
        for j in range(HALF):
            bit_j = (s >> j) & 1
            orig |= (bit_j << order[j])
        results.append(orig)
    return results


# ---------------------------------------------------------------------------
# Phase 1: Measure d_high for single vs ensemble
# ---------------------------------------------------------------------------
def hamming(a, b):
    return int(bin(int(a) ^ int(b)).count("1"))


def phase1_d_high(dataset, test_set, configs):
    """Train all posteriors, measure d_high (Hamming to true p over top 20 bits)."""
    log("\n=== Phase 1: Training diverse posteriors and measuring d_high ===")
    posteriors = []
    t0 = time.time()
    for ci, config in enumerate(configs):
        fe, post, cfg = train_diverse_posterior(dataset, config)
        posteriors.append((fe, post, cfg))
        if (ci + 1) % 5 == 0:
            log(f"    Trained {ci+1}/{len(configs)} ({time.time()-t0:.1f}s)")
    log(f"  All {len(configs)} posteriors trained ({time.time()-t0:.1f}s)")

    # For each test instance, measure d_high
    log("  Measuring d_high...")
    results = []
    for inst in test_set:
        N, p_true = inst["N"], inst["p"]
        i = inst["i"]
        p_high_true = p_true >> B_UNKNOWN
        rng = np.random.default_rng(7000 + i * 13)
        per_post_d = []
        all_samples = []
        for fe, post, cfg in posteriors:
            samples = sample_diverse(fe, post, cfg, N, POP, rng)
            # d_high = Hamming(sample >> B_UNKNOWN, p_true >> B_UNKNOWN)
            for s in samples:
                d = hamming(s >> B_UNKNOWN, p_high_true)
                all_samples.append(d)
            # Best d for this posterior
            best_d = min(hamming(s >> B_UNKNOWN, p_high_true)
                         for s in samples)
            per_post_d.append(best_d)
        # Ensemble best: best across ALL posteriors
        ensemble_best = min(all_samples)
        # Single-posterior best: median of per-posterior bests
        single_median = float(np.median(per_post_d))
        single_mean = float(np.mean(per_post_d))
        single_best = min(per_post_d)
        results.append({
            "i": i, "N": int(N),
            "ensemble_best_d_high": int(ensemble_best),
            "single_median_d_high": float(single_median),
            "single_mean_d_high": float(single_mean),
            "single_best_d_high": int(single_best),
            "per_post_best": per_post_d,
        })
        log(f"    i={i}: ensemble_best={ensemble_best}, "
            f"single_median={single_median:.1f}, single_best={single_best}")
    # Summary
    ens = [r["ensemble_best_d_high"] for r in results]
    sm = [r["single_median_d_high"] for r in results]
    sb = [r["single_best_d_high"] for r in results]
    log(f"  d_high summary (n={len(results)}):")
    log(f"    Ensemble best:   mean={np.mean(ens):.2f} median={np.median(ens):.0f}")
    log(f"    Single median:   mean={np.mean(sm):.2f} median={np.median(sm):.0f}")
    log(f"    Single best:     mean={np.mean(sb):.2f} median={np.median(sb):.0f}")
    return {"posteriors": posteriors, "results": results}


# ---------------------------------------------------------------------------
# Phase 2: Coevolution with ensemble samples
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


class EnsembleCoevolution:
    def __init__(self, N, half, b_unknown, pop_size=80, seed=0):
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


def phase2_coevolution(posteriors, test_set):
    """Run coevolution with ensemble samples (union of all posteriors)."""
    log("\n=== Phase 2: Coevolution with ensemble samples ===")
    results = []
    t0 = time.time()
    for inst in test_set:
        N, p_true, q_true = inst["N"], inst["p"], inst["q"]
        i = inst["i"]
        # Sample from ALL posteriors, take the union
        rng = np.random.default_rng(1000 + i * 100)
        all_p_samples = []
        for fe, post, cfg in posteriors:
            samples = sample_diverse(fe, post, cfg, N, POP // len(posteriors) + 1, rng)
            all_p_samples.extend(samples)
        # Init q from p
        all_q_samples = [int(N // s) | 1 if s > 0 else 1 for s in all_p_samples]
        found = False
        total_evals = 0
        first_R = None
        for r in range(R_RESTARTS):
            seed = 1000 + i * 100 + r * 7919
            # Sample a fresh batch each restart
            p_batch = []
            q_batch = []
            for fe, post, cfg in posteriors:
                p_batch.extend(sample_diverse(fe, post, cfg, N,
                                               POP // len(posteriors) + 1, rng))
            q_batch = [int(N // s) | 1 if s > 0 else 1 for s in p_batch]
            ga = EnsembleCoevolution(N, HALF, B_UNKNOWN, pop_size=POP, seed=seed)
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
    log(f"  Ensemble coevolution: {succ}/{len(test_set)} = {rate*100:.1f}%")
    return {"n_test": len(test_set), "success": succ,
            "success_rate": rate, "per_instance": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        open(LOG_PATH, "w").close()
    except Exception:
        pass
    with open(LOG_PATH, "a") as fh:
        fh.write(f"=== D203 started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log("Direction 203: Ensemble of Diverse Posteriors")
    log(f"  {N_POSTERIORS} posteriors (5 bit orderings × 4 feature subsets)")
    log(f"  Test set: {N_TEST} instances at {BITS}-bit")

    test_set = get_test_set()
    log(f"  Training dataset: {N_TRAIN} semiprimes (seed_offset={SEED_OFFSET_TRAIN})")
    dataset = generate_dataset(BITS, N_TRAIN, seed_offset=SEED_OFFSET_TRAIN)

    configs = get_ensemble_configs()
    log(f"  Generated {len(configs)} diverse configs")

    all_results = {}
    t_start = time.time()

    # Phase 1: d_high measurement
    p1 = phase1_d_high(dataset, test_set, configs)
    all_results["phase1_d_high"] = p1["results"]
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Phase 2: coevolution with ensemble
    p2 = phase2_coevolution(p1["posteriors"], test_set)
    all_results["phase2_coevolution"] = p2
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log(f"\n=== SUMMARY ===")
    log(f"  Phase 1: d_high ensemble_best mean = "
        f"{np.mean([r['ensemble_best_d_high'] for r in p1['results']]):.2f}")
    log(f"  Phase 2: ensemble coevolution = "
        f"{p2['success']}/{p2['n_test']} = {p2['success_rate']*100:.1f}%")
    log(f"\nTOTAL TIME: {time.time() - t_start:.1f}s")
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log("Final JSON written.")


if __name__ == "__main__":
    main()
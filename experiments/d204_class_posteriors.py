#!/usr/bin/env python
"""Direction 204: Class-Specific Posteriors.

Partition semiprimes by D199's discriminating residues (N mod 53, N mod 17)
into classes. Train a SEPARATE posterior for each class. At test time,
classify N by its residue, use the class-specific posterior. If class-specific
models have lower d_high, coevolution should find more factors.

Phase 1: Partition training data by N mod 53 into classes.
Phase 2: Train per-class posteriors, measure d_high vs general posterior.
Phase 3: Run coevolution with class-specific posteriors on test set.
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

LOG_PATH = str(_REPO_ROOT / "experiments" / "d204_run.log")
RESULTS_PATH = str(_REPO_ROOT / "experiments" / "d204_results.json")

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

# Classification: use N mod 53 (D199's strongest discriminator)
CLASS_MODULUS = 53


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
# Phase 1: Partition and train class-specific posteriors
# ---------------------------------------------------------------------------
def phase1_class_posteriors(dataset):
    """Partition training data by N mod 53, train per-class posteriors."""
    log("\n=== Phase 1: Partition by N mod 53, train per-class posteriors ===")
    # Partition
    classes = {}
    for d in dataset:
        N, p, q = d
        r = N % CLASS_MODULUS
        if r not in classes:
            classes[r] = []
        classes[r].append(d)
    log(f"  {len(classes)} classes from {len(dataset)} samples")
    for r in sorted(classes.keys()):
        log(f"    N mod {CLASS_MODULUS} = {r}: {len(classes[r])} samples")

    # Train a general posterior (baseline)
    fe = FeatureExtractor(BITS)
    N_list = [d[0] for d in dataset]
    F_all = fe.fit_transform(N_list)
    p_list = [d[1] for d in dataset]
    p_bits = np.array([_int_to_bits(p, HALF) for p in p_list], dtype=np.float64)
    post_general = ARPosterior(n_features=F_all.shape[1], n_bits=HALF)
    post_general.train(F_all, p_bits)
    log("  General posterior trained")

    # Train per-class posteriors
    # Strategy: for each class with >= 30 samples, train a class-specific posterior.
    # For classes with < 30 samples, use the general posterior (fallback).
    MIN_CLASS_SIZE = 30
    class_posteriors = {}
    t0 = time.time()
    for r in sorted(classes.keys()):
        class_data = classes[r]
        if len(class_data) < MIN_CLASS_SIZE:
            class_posteriors[r] = ("general", None)
            continue
        fe_c = FeatureExtractor(BITS)
        N_c = [d[0] for d in class_data]
        F_c = fe_c.fit_transform(N_c)
        p_c = [d[1] for d in class_data]
        pb_c = np.array([_int_to_bits(p, HALF) for p in p_c], dtype=np.float64)
        post_c = ARPosterior(n_features=F_c.shape[1], n_bits=HALF)
        post_c.train(F_c, pb_c)
        class_posteriors[r] = ("specific", fe_c, post_c)
        log(f"    Class {r}: trained on {len(class_data)} samples")
    log(f"  {sum(1 for v in class_posteriors.values() if v[0]=='specific')} "
        f"class-specific posteriors trained ({time.time()-t0:.1f}s)")
    return fe, post_general, class_posteriors


# ---------------------------------------------------------------------------
# Phase 2: Measure d_high
# ---------------------------------------------------------------------------
def phase2_d_high(fe, post_general, class_posteriors, test_set):
    """Measure d_high for general vs class-specific posteriors."""
    log("\n=== Phase 2: d_high comparison ===")
    results = []
    for inst in test_set:
        N, p_true = inst["N"], inst["p"]
        i = inst["i"]
        r = N % CLASS_MODULUS
        p_high_true = p_true >> B_UNKNOWN
        rng = np.random.default_rng(7000 + i * 13)

        # General posterior d_high
        f_general = fe.transform([N])[0]
        samples_general = [int(s) | 1 for s in post_general.sample(f_general, POP, rng)]
        d_general = min(hamming(s >> B_UNKNOWN, p_high_true)
                        for s in samples_general)

        # Class-specific d_high
        entry = class_posteriors.get(r, ("general", None))
        if entry[0] == "specific":
            fe_c, post_c = entry[1], entry[2]
            f_c = fe_c.transform([N])[0]
            samples_class = [int(s) | 1 for s in post_c.sample(f_c, POP, rng)]
            d_class = min(hamming(s >> B_UNKNOWN, p_high_true)
                          for s in samples_class)
        else:
            d_class = d_general  # fallback

        results.append({
            "i": i, "N": int(N), "class": r,
            "d_high_general": int(d_general),
            "d_high_class": int(d_class),
            "improvement": int(d_general - d_class),
            "class_specific": entry[0] == "specific",
        })
        log(f"    i={i} (mod {r}): general={d_general}, class={d_class}, "
            f"improved={d_general - d_class}")
    # Summary
    dg = [r["d_high_general"] for r in results]
    dc = [r["d_high_class"] for r in results]
    improved = sum(1 for r in results if r["improvement"] > 0)
    log(f"  d_high summary:")
    log(f"    General: mean={np.mean(dg):.2f} median={np.median(dg):.0f}")
    log(f"    Class:   mean={np.mean(dc):.2f} median={np.median(dc):.0f}")
    log(f"    Improved in {improved}/{len(results)} cases")
    return results


# ---------------------------------------------------------------------------
# Phase 3: Coevolution with class-specific posteriors
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


class ClassCoevolution:
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


def phase3_coevolution(fe, post_general, class_posteriors, test_set):
    """Run coevolution with class-specific posteriors."""
    log("\n=== Phase 3: Coevolution with class-specific posteriors ===")
    results = []
    t0 = time.time()
    for inst in test_set:
        N, p_true, q_true = inst["N"], inst["p"], inst["q"]
        i = inst["i"]
        r = N % CLASS_MODULUS
        found = False
        total_evals = 0
        first_R = None
        for restart in range(R_RESTARTS):
            seed = 1000 + i * 100 + restart * 7919
            rng = np.random.default_rng(seed)
            # Use class-specific posterior if available
            entry = class_posteriors.get(r, ("general", None))
            if entry[0] == "specific":
                fe_c, post_c = entry[1], entry[2]
                f_c = fe_c.transform([N])[0]
                sp = [int(s) | 1 for s in post_c.sample(f_c, POP, rng)]
            else:
                f_g = fe.transform([N])[0]
                sp = [int(s) | 1 for s in post_general.sample(f_g, POP, rng)]
            sq = [int(N // s) | 1 if s > 0 else 1 for s in sp]
            ga = ClassCoevolution(N, HALF, B_UNKNOWN, pop_size=POP, seed=seed)
            factor, ev = ga.run(sp, sq, generations=GEN)
            total_evals += ev
            if factor is not None and 1 < factor < N and gcd(int(factor), N) > 1:
                found = True
                first_R = restart + 1
                break
        results.append({
            "i": i, "N": int(N), "class": r, "success": found,
            "first_success_R": first_R, "evals": total_evals,
            "time": time.time() - t0,
        })
        status = "OK" if found else "FAIL"
        log(f"    i={i} (mod {r}): {status} (R={first_R}, "
            f"evals={total_evals}, {time.time()-t0:.1f}s)")
    succ = sum(1 for r in results if r["success"])
    rate = succ / len(test_set)
    log(f"  Class-specific coevolution: {succ}/{len(test_set)} = {rate*100:.1f}%")
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
        fh.write(f"=== D204 started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log("Direction 204: Class-Specific Posteriors")
    log(f"  Classification: N mod {CLASS_MODULUS}")
    log(f"  Test set: {N_TEST} instances at {BITS}-bit")

    test_set = get_test_set()
    log(f"  Training dataset: {N_TRAIN} semiprimes (seed_offset={SEED_OFFSET_TRAIN})")
    dataset = generate_dataset(BITS, N_TRAIN, seed_offset=SEED_OFFSET_TRAIN)

    all_results = {}
    t_start = time.time()

    # Phase 1: train class posteriors
    fe, post_general, class_posteriors = phase1_class_posteriors(dataset)

    # Phase 2: d_high comparison
    p2 = phase2_d_high(fe, post_general, class_posteriors, test_set)
    all_results["phase2_d_high"] = p2
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Phase 3: coevolution
    p3 = phase3_coevolution(fe, post_general, class_posteriors, test_set)
    all_results["phase3_coevolution"] = p3
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log(f"\n=== SUMMARY ===")
    dg = [r["d_high_general"] for r in p2]
    dc = [r["d_high_class"] for r in p2]
    log(f"  d_high: general={np.mean(dg):.2f}, class={np.mean(dc):.2f}")
    log(f"  Coevolution: {p3['success']}/{p3['n_test']} = "
        f"{p3['success_rate']*100:.1f}%")
    log(f"\nTOTAL TIME: {time.time() - t_start:.1f}s")
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log("Final JSON written.")


if __name__ == "__main__":
    main()
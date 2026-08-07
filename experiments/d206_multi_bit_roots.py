#!/usr/bin/env python
"""Direction 206: Multi-Bit Extended Roots.

D174's extended root accepts |x0| <= 8X (catches 1-bit errors). Test
larger bounds (16X, 32X, 64X, 128X, 256X) to catch 2-3 bit errors.
At n=64 with bu=16, dim=10: X = 2^16, N^{6/10} = 2^{38.4} — massive
slack. The lattice might find roots at 64X = 2^22, still within the
Howgrave-Graham bound.

Phase 1: For each test instance, run coevolution with copper-ext at
         max_x0_mult = 8, 16, 32, 64, 128, 256.
Phase 2: Measure success rate vs max_x0_mult.
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

LOG_PATH = str(_REPO_ROOT / "experiments" / "d206_run.log")
RESULTS_PATH = str(_REPO_ROOT / "experiments" / "d206_results.json")

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

MAX_X0_MULTS = [8, 16, 32, 64, 128, 256]

_POSTERIOR_PICKLE = str(_REPO_ROOT / "experiments" / "d202_posterior.pkl")


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


def train_posterior():
    log(f"  Training on {N_TRAIN} {BITS}-bit semiprimes...")
    t0 = time.time()
    dataset = generate_dataset(BITS, N_TRAIN, seed_offset=SEED_OFFSET_TRAIN)
    fe = FeatureExtractor(BITS)
    F = fe.fit_transform([d[0] for d in dataset])
    p_bits = np.array([_int_to_bits(d[1], HALF) for d in dataset],
                      dtype=np.float64)
    post = ARPosterior(n_features=F.shape[1], n_bits=HALF)
    post.train(F, p_bits)
    log(f"  Training done ({time.time()-t0:.1f}s)")
    import pickle
    with open(_POSTERIOR_PICKLE, "wb") as f:
        pickle.dump({"fe": fe, "post_p": post}, f,
                     protocol=pickle.HIGHEST_PROTOCOL)
    return fe, post


def load_posterior():
    import pickle
    with open(_POSTERIOR_PICKLE, "rb") as f:
        t = pickle.load(f)
    return t["fe"], t["post_p"]


# ---------------------------------------------------------------------------
# Lattice + extraction with configurable max_x0_mult
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


def extract_ext(p_high, N, b_unknown, max_x0_mult, try_q=True):
    N_i = int(N)
    reduced, X, dim = coppersmith_lattice(p_high, N_i, b_unknown,
                                           COPPERSMITH_M, COPPERSMITH_T)
    factor = _find_roots_extended(reduced, p_high, N_i, X, dim,
                                   max_x0_mult=max_x0_mult)
    if factor is not None and 1 < factor < N_i and gcd(int(factor), N_i) > 1:
        return int(factor)
    if try_q:
        q_approx = N_i // (int(p_high) | 1)
        q_high = (q_approx >> b_unknown) << b_unknown
        if q_high > 0:
            reduced, X, dim = coppersmith_lattice(q_high, N_i, b_unknown,
                                                   COPPERSMITH_M, COPPERSMITH_T)
            factor = _find_roots_extended(reduced, q_high, N_i, X, dim,
                                           max_x0_mult=max_x0_mult)
            if factor is not None and 1 < factor < N_i and gcd(int(factor), N_i) > 1:
                return int(factor)
    return None


class ExtCoevolution:
    def __init__(self, N, half, b_unknown, pop_size=80, seed=0,
                 max_x0_mult=8):
        self.N = int(N)
        self.half = half
        self.b_unknown = b_unknown
        self.n_known = half - b_unknown
        self.mask = (1 << self.n_known) - 1
        self.shift = b_unknown
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)
        self.max_x0_mult = max_x0_mult
        self.eval_count = 0

    def _product_gap(self, p_hb, q_hb):
        return abs(self.N - (p_hb << self.shift) * (q_hb << self.shift))

    def _extract(self, p_hb):
        p_high = p_hb << self.shift
        self.eval_count += 1
        return extract_ext(p_high, self.N, self.b_unknown, self.max_x0_mult)

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
        fh.write(f"=== D206 started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log("Direction 206: Multi-Bit Extended Roots")
    log(f"  Testing max_x0_mult = {MAX_X0_MULTS}")
    log(f"  Test set: {N_TEST} instances at {BITS}-bit")

    test_set = get_test_set()

    if os.path.exists(_POSTERIOR_PICKLE):
        try:
            fe, post = load_posterior()
            log("  Loaded pickled posterior")
        except Exception:
            fe, post = train_posterior()
    else:
        fe, post = train_posterior()

    all_results = {}
    t_start = time.time()

    for max_mult in MAX_X0_MULTS:
        log(f"\n=== max_x0_mult = {max_mult} ===")
        results = []
        t0 = time.time()
        for inst in test_set:
            N, p_true, q_true = inst["N"], inst["p"], inst["q"]
            i = inst["i"]
            found = False
            total_evals = 0
            first_R = None
            f = fe.transform([N])[0]
            for r in range(R_RESTARTS):
                seed = 1000 + i * 100 + r * 7919
                rng = np.random.default_rng(seed)
                sp = [int(s) | 1 for s in post.sample(f, POP, rng)]
                sq = [int(N // s) | 1 if s > 0 else 1 for s in sp]
                ga = ExtCoevolution(N, HALF, B_UNKNOWN, pop_size=POP,
                                     seed=seed, max_x0_mult=max_mult)
                factor, ev = ga.run(sp, sq, generations=GEN)
                total_evals += ev
                if factor is not None and 1 < factor < N and gcd(int(factor), N) > 1:
                    found = True
                    first_R = r + 1
                    break
            results.append({
                "i": i, "N": int(N), "success": found,
                "first_success_R": first_R, "evals": total_evals,
            })
        succ = sum(1 for r in results if r["success"])
        rate = succ / len(test_set)
        elapsed = time.time() - t0
        log(f"  max_x0_mult={max_mult}: {succ}/{len(test_set)} = "
            f"{rate*100:.1f}% (t={elapsed:.1f}s)")
        all_results[f"max_x0_mult_{max_mult}"] = {
            "max_x0_mult": max_mult,
            "n_test": len(test_set), "success": succ,
            "success_rate": rate, "per_instance": results,
            "time_s": elapsed,
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    log(f"\n=== SUMMARY ===")
    log(f"{'max_x0_mult':<15} {'Success':>8} {'Rate':>8}")
    log(f"{'-'*15} {'-'*8} {'-'*8}")
    for max_mult in MAX_X0_MULTS:
        key = f"max_x0_mult_{max_mult}"
        if key in all_results:
            r = all_results[key]
            log(f"  {max_mult:<15} {r['success']:>5}/{r['n_test']:<3} "
                f"{r['success_rate']*100:>6.1f}%")

    log(f"\nTOTAL TIME: {time.time() - t_start:.1f}s")
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log("Final JSON written.")


if __name__ == "__main__":
    main()
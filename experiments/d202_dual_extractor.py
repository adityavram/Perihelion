#!/usr/bin/env python
"""Direction 202: Dual-extractor union at 64-bit.

D200 showed brute bu=12 solves a DIFFERENT class than copper-ext bu=16
(i=13 is brute-easy but copper-hard). The union should solve 4/7 of D201's
hard cases, bringing 64-bit from 65% to ~85%.

This experiment runs both extractors on the full D201 20-instance test set:
  Phase 1: copper-ext (bu=16, R=10) — reproduce D201's 65%.
  Phase 2: brute (bu=12, R=10) — the brute-easy class.
  Phase 3: UNION — copper-ext OR brute. THE KEY MEASUREMENT.
  Phase 4: Residual — which instances does the union miss?

Uses production code: src.spectral_factoring. Parallelized.
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
    CoevolutionaryGA,
    CoppersmithExtractor,
    generate_dataset,
)
from src.spectral_factoring.posterior import _int_to_bits
from src.spectral_factoring.extraction import (
    _reduce_lattice, _find_roots, _find_roots_extended,
)

warnings.filterwarnings("ignore")

LOG_PATH = str(_REPO_ROOT / "experiments" / "d202_run.log")
RESULTS_PATH = str(_REPO_ROOT / "experiments" / "d202_results.json")
WRITEUP_PATH = str(_REPO_ROOT / "direction-202-dual-extractor.md")

BITS = 64
HALF = BITS // 2
B_UNKNOWN_COPPER = 16
B_UNKNOWN_BRUTE = 12
COPPERSMITH_M = 5
COPPERSMITH_T = 4
SEED_OFFSET_TRAIN = 500
SEED_OFFSET_TEST = 501
N_TRAIN = 800
N_TEST = 20
R_RESTARTS = 10
POP = 80
GEN = 40

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_posterior(bits=BITS, n_train=N_TRAIN,
                     seed_offset=SEED_OFFSET_TRAIN, pickle_to_disk=True):
    log(f"  Training on {n_train} {bits}-bit semiprimes...")
    t0 = time.time()
    dataset = generate_dataset(bits, n_train, seed_offset=seed_offset)
    fe = FeatureExtractor(bits)
    F = fe.fit_transform([d[0] for d in dataset])
    p_bits = np.array([_int_to_bits(d[1], HALF) for d in dataset],
                      dtype=np.float64)
    post_p = ARPosterior(n_features=F.shape[1], n_bits=HALF)
    post_p.train(F, p_bits)
    out = {"fe": fe, "post_p": post_p, "train_time": time.time() - t0}
    if pickle_to_disk:
        import pickle
        try:
            with open(_POSTERIOR_PICKLE, "wb") as f:
                pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
            log(f"  posterior pickled to {_POSTERIOR_PICKLE}")
        except Exception as e:
            log(f"  posterior pickle FAILED: {e}")
    log(f"  training done ({out['train_time']:.1f}s)")
    return out


def load_posterior_pickle():
    import pickle
    with open(_POSTERIOR_PICKLE, "rb") as f:
        out = pickle.load(f)
    return out


# ---------------------------------------------------------------------------
# Extraction
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


def extract_coppersmith_ext(p_high, N, b_unknown, try_q=True):
    N_i = int(N)
    reduced, X, dim = coppersmith_lattice(p_high, N_i, b_unknown,
                                           COPPERSMITH_M, COPPERSMITH_T)
    factor = _find_roots_extended(reduced, p_high, N_i, X, dim,
                                   max_x0_mult=8)
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


def brute_extract(p_high, N, b_unknown):
    N_i = int(N)
    p_high_i = int(p_high)
    for x in range(1 << b_unknown):
        cand = p_high_i + x
        g = gcd(cand, N_i)
        if 1 < g < N_i:
            return cand
    return None


# ---------------------------------------------------------------------------
# Coevolution with configurable extractor
# ---------------------------------------------------------------------------
class DualCoevolution:
    def __init__(self, N, half, b_unknown, pop_size=80, seed=0,
                 extract_method="coppersmith_ext", try_q=True):
        self.N = int(N)
        self.half = half
        self.b_unknown = b_unknown
        self.n_known = half - b_unknown
        self.mask = (1 << self.n_known) - 1
        self.shift = b_unknown
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)
        self.extract_method = extract_method
        self.try_q = try_q
        self.eval_count = 0

    def _product_gap(self, p_hb, q_hb):
        return abs(self.N - (p_hb << self.shift) * (q_hb << self.shift))

    def _extract(self, p_hb):
        p_high = p_hb << self.shift
        self.eval_count += 1
        if self.extract_method == "coppersmith_ext":
            return extract_coppersmith_ext(p_high, self.N, self.b_unknown,
                                           try_q=self.try_q)
        elif self.extract_method == "brute":
            return brute_extract(p_high, self.N, self.b_unknown)
        return None

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
# Worker
# ---------------------------------------------------------------------------
_W = {}


def _init_worker():
    try:
        t = load_posterior_pickle()
    except Exception:
        t = train_posterior(pickle_to_disk=True)
    _W["fe"] = t["fe"]
    _W["post_p"] = t["post_p"]
    _W["half"] = HALF


def _task_copper(args):
    N, p_true, q_true, i, R = args
    t0 = time.time()
    f = _W["fe"].transform([N])[0]
    found = False
    total_evals = 0
    first_R = None
    for r in range(R):
        seed = 1000 + i * 100 + r * 7919
        rng = np.random.default_rng(seed)
        sp = [int(s) | 1 for s in _W["post_p"].sample(f, POP, rng)]
        sq = [int(N // s) | 1 if s > 0 else 1 for s in sp]
        ga = DualCoevolution(N, HALF, B_UNKNOWN_COPPER, pop_size=POP,
                             seed=seed, extract_method="coppersmith_ext")
        factor, ev = ga.run(sp, sq, generations=GEN)
        total_evals += ev
        if factor is not None and 1 < factor < N and gcd(int(factor), N) > 1:
            found = True
            first_R = r + 1
            break
    return {"i": i, "N": int(N), "success": found,
            "first_success_R": first_R, "evals": total_evals,
            "time": time.time() - t0, "method": "copper_ext"}


def _task_brute(args):
    N, p_true, q_true, i, R = args
    t0 = time.time()
    f = _W["fe"].transform([N])[0]
    found = False
    total_evals = 0
    first_R = None
    for r in range(R):
        seed = 2000 + i * 100 + r * 7919
        rng = np.random.default_rng(seed)
        sp = [int(s) | 1 for s in _W["post_p"].sample(f, POP, rng)]
        sq = [int(N // s) | 1 if s > 0 else 1 for s in sp]
        ga = DualCoevolution(N, HALF, B_UNKNOWN_BRUTE, pop_size=POP,
                             seed=seed, extract_method="brute")
        factor, ev = ga.run(sp, sq, generations=GEN)
        total_evals += ev
        if factor is not None and 1 < factor < N and gcd(int(factor), N) > 1:
            found = True
            first_R = r + 1
            break
    return {"i": i, "N": int(N), "success": found,
            "first_success_R": first_R, "evals": total_evals,
            "time": time.time() - t0, "method": "brute_bu12"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        open(LOG_PATH, "w").close()
    except Exception:
        pass
    with open(LOG_PATH, "a") as fh:
        fh.write(f"=== D202 started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log("Direction 202: Dual-extractor union at 64-bit")
    log(f"  Test set: {N_TEST} instances (seed_offset={SEED_OFFSET_TEST})")
    log(f"  Copper: bu={B_UNKNOWN_COPPER}, R={R_RESTARTS}, pop={POP}, gen={GEN}")
    log(f"  Brute:  bu={B_UNKNOWN_BRUTE}, R={R_RESTARTS}, pop={POP}, gen={GEN}")

    test_set = get_test_set()
    log(f"  Test instances: {[inst['i'] for inst in test_set]}")

    # Train (or load pickle)
    if os.path.exists(_POSTERIOR_PICKLE):
        try:
            t = load_posterior_pickle()
            log("  Loaded pickled posterior")
        except Exception:
            t = train_posterior(pickle_to_disk=True)
    else:
        t = train_posterior(pickle_to_disk=True)

    all_results = {}
    t_start = time.time()

    def save():
        try:
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
        except Exception as e:
            log(f"  ERROR saving: {e}")

    n_workers = min(mp.cpu_count(), 4)
    log(f"  Using {n_workers} workers")

    # --- Phase 1: Copper-ext (reproduce D201's 65%) ---
    log("\n=== Phase 1: Copper-ext (bu=16, R=10) ===")
    args = [(inst["N"], inst["p"], inst["q"], inst["i"], R_RESTARTS)
            for inst in test_set]
    t0 = time.time()
    try:
        with mp.Pool(n_workers, initializer=_init_worker,
                     maxtasksperchild=1) as pool:
            copper_results = pool.map(_task_copper, args)
    except Exception as e:
        log(f"  POOL ERROR: {e}")
        import traceback; traceback.print_exc()
        copper_results = []
    copper_results.sort(key=lambda x: x["i"])
    succ_copper = sum(1 for r in copper_results if r["success"])
    log(f"  Copper-ext: {succ_copper}/{N_TEST} = "
        f"{succ_copper/N_TEST*100:.1f}% (t={time.time()-t0:.1f}s)")
    all_results["phase1_copper_ext"] = {
        "n_test": N_TEST, "success": succ_copper,
        "success_rate": succ_copper / N_TEST,
        "per_instance": copper_results, "time_s": time.time() - t0,
    }
    save()

    # --- Phase 2: Brute (bu=12) ---
    log("\n=== Phase 2: Brute (bu=12, R=10) ===")
    t0 = time.time()
    try:
        with mp.Pool(n_workers, initializer=_init_worker,
                     maxtasksperchild=1) as pool:
            brute_results = pool.map(_task_brute, args)
    except Exception as e:
        log(f"  POOL ERROR: {e}")
        import traceback; traceback.print_exc()
        brute_results = []
    brute_results.sort(key=lambda x: x["i"])
    succ_brute = sum(1 for r in brute_results if r["success"])
    log(f"  Brute bu=12: {succ_brute}/{N_TEST} = "
        f"{succ_brute/N_TEST*100:.1f}% (t={time.time()-t0:.1f}s)")
    all_results["phase2_brute_bu12"] = {
        "n_test": N_TEST, "success": succ_brute,
        "success_rate": succ_brute / N_TEST,
        "per_instance": brute_results, "time_s": time.time() - t0,
    }
    save()

    # --- Phase 3: UNION ---
    log("\n=== Phase 3: UNION (copper-ext OR brute) ===")
    copper_set = set(r["i"] for r in copper_results if r["success"])
    brute_set = set(r["i"] for r in brute_results if r["success"])
    union_set = copper_set | brute_set
    overlap = copper_set & brute_set
    copper_only = copper_set - brute_set
    brute_only = brute_set - copper_set
    log(f"  Copper-ext: {len(copper_set)}/{N_TEST} = {len(copper_set)/N_TEST*100:.1f}%")
    log(f"  Brute bu=12: {len(brute_set)}/{N_TEST} = {len(brute_set)/N_TEST*100:.1f}%")
    log(f"  UNION: {len(union_set)}/{N_TEST} = {len(union_set)/N_TEST*100:.1f}%")
    log(f"  Overlap: {len(overlap)} {sorted(overlap)}")
    log(f"  Copper-only: {len(copper_only)} {sorted(copper_only)}")
    log(f"  Brute-only: {len(brute_only)} {sorted(brute_only)}")
    log(f"  Residual (neither): {N_TEST - len(union_set)} "
        f"{sorted(set(range(N_TEST)) - union_set)}")
    all_results["phase3_union"] = {
        "n_test": N_TEST,
        "copper_success": len(copper_set),
        "brute_success": len(brute_set),
        "union_success": len(union_set),
        "union_rate": len(union_set) / N_TEST,
        "overlap": sorted(overlap),
        "copper_only": sorted(copper_only),
        "brute_only": sorted(brute_only),
        "residual": sorted(set(range(N_TEST)) - union_set),
    }
    save()

    # --- Summary ---
    log(f"\n=== SUMMARY ===")
    log(f"  Copper-ext (bu=16): {len(copper_set)}/{N_TEST} = {len(copper_set)/N_TEST*100:.1f}%")
    log(f"  Brute (bu=12):      {len(brute_set)}/{N_TEST} = {len(brute_set)/N_TEST*100:.1f}%")
    log(f"  UNION:              {len(union_set)}/{N_TEST} = {len(union_set)/N_TEST*100:.1f}%")
    log(f"  Overlap: {len(overlap)}, Copper-only: {len(copper_only)}, "
        f"Brute-only: {len(brute_only)}")
    log(f"  Residual: {sorted(set(range(N_TEST)) - union_set)}")
    log(f"\nTOTAL TIME: {time.time() - t_start:.1f}s")
    save()
    log("Final JSON written.")


if __name__ == "__main__":
    main()
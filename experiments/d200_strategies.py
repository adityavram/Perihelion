#!/usr/bin/env python
"""Direction 200: Multiple coevolution strategies for the 35% hard cases.

D201's extended-root coevolution reaches 65% at 64-bit (13/20). The 7 hard
instances (i=4,9,12,13,14,15,17) all fail. D199 showed the bottleneck is the
POSTERIOR (d_high ~5 bits; coevolution bridges ~4; hint with true p_high
works 100%). This experiment tests multiple strategies to bridge the gap.

Strategies (each tested on the 7 hard instances):
  A. Baseline (D201 config, R=10) — reproduce 0/7.
  B. More restarts (R=30) — 3x more tries.
  C. Larger population (pop=200) — wider search.
  D. More generations (gen=100) — deeper search.
  E. Multi-bit mutation (flip 2-3 bits) — wider neighborhood.
  F. Predict q (train posterior on q) — q may be more predictable.
  G. Both p and q posteriors — double the init diversity.
  H. Brute extraction at bu=12 — 100% reliable, needs 4 more high bits.
  I. Combined best (R=30, pop=200, multi-bit, both posteriors).

Uses production code: src.spectral_factoring. Coppersmith via fpylll.
"""

import json
import math
import os
import sys
import time
import warnings
import random as _random
from math import gcd, isqrt, comb
from itertools import combinations
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

LOG_PATH = str(_REPO_ROOT / "experiments" / "d200_run.log")
RESULTS_PATH = str(_REPO_ROOT / "experiments" / "d200_results.json")
WRITEUP_PATH = str(_REPO_ROOT / "direction-200-strategies.md")

BITS = 64
HALF = BITS // 2
B_UNKNOWN = 16
B_UNKNOWN_BRUTE = 12
COPPERSMITH_M = 5
COPPERSMITH_T = 4
SEED_OFFSET_TRAIN = 500
N_TRAIN = 800

# The 7 hard instances from D201 (seed_offset=501, indices 4,9,12,13,14,15,17)
# We regenerate them with the same seed.
HARD_INDICES = [4, 9, 12, 13, 14, 15, 17]


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


def get_hard_instances():
    """Regenerate the 7 hard instances from D201."""
    instances = []
    for i in HARD_INDICES:
        seed = 501 * 10_000_000 + i
        N, p, q = generate_semiprime(BITS, seed)
        instances.append({"i": i, "N": N, "p": p, "q": q})
    return instances


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_posteriors(n_train=N_TRAIN, seed_offset=SEED_OFFSET_TRAIN):
    """Train both p and q posteriors."""
    log(f"  Training on {n_train} {BITS}-bit semiprimes (seed_offset={seed_offset})...")
    t0 = time.time()
    dataset = generate_dataset(BITS, n_train, seed_offset=seed_offset)
    fe = FeatureExtractor(BITS)
    F = fe.fit_transform([d[0] for d in dataset])
    p_bits = np.array([_int_to_bits(d[1], HALF) for d in dataset], dtype=np.float64)
    q_bits = np.array([_int_to_bits(d[2], HALF) for d in dataset], dtype=np.float64)
    post_p = ARPosterior(n_features=F.shape[1], n_bits=HALF)
    post_p.train(F, p_bits)
    post_q = ARPosterior(n_features=F.shape[1], n_bits=HALF)
    post_q.train(F, q_bits)
    log(f"  Training done ({time.time()-t0:.1f}s)")
    return fe, post_p, post_q


# ---------------------------------------------------------------------------
# Coppersmith lattice (local, for extended extraction)
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


def extract_coppersmith_ext(p_high, N, b_unknown, m=COPPERSMITH_M,
                             t=COPPERSMITH_T, try_q=True):
    """Extended Coppersmith extraction (D174). Returns factor or None."""
    N_i = int(N)
    reduced, X, dim = coppersmith_lattice(p_high, N_i, b_unknown, m, t)
    factor = _find_roots_extended(reduced, p_high, N_i, X, dim,
                                   max_x0_mult=8)
    if factor is not None and 1 < factor < N_i and gcd(int(factor), N_i) > 1:
        return int(factor)
    if try_q:
        q_approx = N_i // (int(p_high) | 1)
        q_high = (q_approx >> b_unknown) << b_unknown
        if q_high > 0:
            reduced, X, dim = coppersmith_lattice(q_high, N_i, b_unknown, m, t)
            factor = _find_roots_extended(reduced, q_high, N_i, X, dim,
                                           max_x0_mult=8)
            if factor is not None and 1 < factor < N_i and gcd(int(factor), N_i) > 1:
                return int(factor)
    return None


def brute_extract(p_high, N, b_unknown):
    """Brute-force GCD extraction (100% reliable)."""
    N_i = int(N)
    p_high_i = int(p_high)
    for x in range(1 << b_unknown):
        cand = p_high_i + x
        g = gcd(cand, N_i)
        if 1 < g < N_i:
            return cand
    return None


# ---------------------------------------------------------------------------
# Coevolution with custom extraction (strategies)
# ---------------------------------------------------------------------------
class StrategyCoevolution:
    """Coevolutionary GA with configurable extraction and mutation.

    Strategies:
    - extract_method: "coppersmith_ext" or "brute"
    - mut_mode: "standard" (1-2 bits), "wide" (2-3 bits), "aggressive" (3-4)
    """

    def __init__(self, N, half, b_unknown, pop_size=80, seed=0,
                 extract_method="coppersmith_ext",
                 mut_mode="standard", try_q=True):
        self.N = int(N)
        self.half = half
        self.b_unknown = b_unknown
        self.n_known = half - b_unknown
        self.mask = (1 << self.n_known) - 1
        self.shift = b_unknown
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)
        self.extract_method = extract_method
        self.mut_mode = mut_mode
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
        if self.mut_mode == "standard":
            n_flip = 1 if self.rng.random() < 0.7 else 2
        elif self.mut_mode == "wide":
            n_flip = 2 if self.rng.random() < 0.6 else 3
        elif self.mut_mode == "aggressive":
            n_flip = 3 if self.rng.random() < 0.5 else 4
        else:
            n_flip = 1
        for _ in range(n_flip):
            v ^= (1 << int(self.rng.integers(0, self.n_known)))
        return v & self.mask

    def run(self, init_p, init_q, generations=40):
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
# Strategy runners
# ---------------------------------------------------------------------------
def run_strategy(name, instances, fe, post_p, post_q,
                 R=10, pop=80, gen=40, b_unknown=B_UNKNOWN,
                 extract_method="coppersmith_ext", mut_mode="standard",
                 use_q_posterior=False, use_both_posteriors=False,
                 base_seed=1000):
    """Run a strategy on all hard instances. Returns list of results."""
    results = []
    for inst in instances:
        N, p_true, q_true = inst["N"], inst["p"], inst["q"]
        i = inst["i"]
        t0 = time.time()
        found = False
        total_evals = 0
        first_R = None

        f = fe.transform([N])[0]
        rng = np.random.default_rng(base_seed + i * 100)

        for r in range(R):
            seed = base_seed + i * 100 + r * 7919
            # Sample init populations
            sp = [int(s) | 1 for s in post_p.sample(f, pop, rng)]
            if use_both_posteriors and post_q is not None:
                sq = [int(s) | 1 for s in post_q.sample(f, pop, rng)]
                # Merge: half from p-posterior, half from q-posterior
                init_p = sp[:pop // 2] + [int(N // s) | 1 for s in sq[:pop // 2]]
                init_q = sq[:pop // 2] + [int(N // s) | 1 for s in sp[:pop // 2]]
            elif use_q_posterior and post_q is not None:
                sq = [int(s) | 1 for s in post_q.sample(f, pop, rng)]
                init_p = sq  # q-posterior samples as p candidates
                init_q = [int(N // s) | 1 for s in sp]
            else:
                init_q = [int(N // s) | 1 if s > 0 else 1 for s in sp]
                init_p = sp

            ga = StrategyCoevolution(
                N, HALF, b_unknown, pop_size=pop, seed=seed,
                extract_method=extract_method, mut_mode=mut_mode,
            )
            factor, ev = ga.run(init_p, init_q, generations=gen)
            total_evals += ev
            if factor is not None and 1 < factor < N and gcd(int(factor), N) > 1:
                found = True
                first_R = r + 1
                break

        elapsed = time.time() - t0
        results.append({
            "i": i, "N": int(N), "p_true": p_true, "q_true": q_true,
            "success": found, "first_success_R": first_R,
            "evals": total_evals, "time": elapsed,
        })
        status = "OK" if found else "FAIL"
        log(f"    [{name}] i={i}: {status} (R={first_R}, "
            f"evals={total_evals}, {elapsed:.1f}s)")
    succ = sum(1 for r in results if r["success"])
    log(f"  [{name}] {succ}/{len(instances)} = {succ/len(instances)*100:.1f}%")
    return {"strategy": name, "R": R, "pop": pop, "gen": gen,
            "b_unknown": b_unknown, "extract_method": extract_method,
            "mut_mode": mut_mode, "use_q_posterior": use_q_posterior,
            "use_both_posteriors": use_both_posteriors,
            "n_test": len(instances), "success": succ,
            "success_rate": succ / len(instances),
            "per_instance": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        open(LOG_PATH, "w").close()
    except Exception:
        pass
    with open(LOG_PATH, "a") as fh:
        fh.write(f"=== D200 started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log("Direction 200: Multiple coevolution strategies for the 35% hard cases")
    log(f"Testing on D201's 7 hard instances at {BITS}-bit")

    instances = get_hard_instances()
    log(f"  Hard instances: {[inst['i'] for inst in instances]}")
    for inst in instances:
        log(f"    i={inst['i']}: N={inst['N']}, |p-q|={abs(inst['p']-inst['q'])}")

    fe, post_p, post_q = train_posteriors()

    # Load existing results if present (resume support)
    all_results = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r") as f:
                all_results = json.load(f)
            log(f"  Loaded {len(all_results)} existing strategy results")
        except Exception:
            all_results = {}
    t_start = time.time()

    def save():
        try:
            with open(RESULTS_PATH, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
        except Exception as e:
            log(f"  ERROR saving: {e}")

    # --- Strategy A: Baseline (D201 config) ---
    log("\n=== Strategy A: Baseline (D201 config, R=10) ===")
    if "A_baseline" not in all_results:
        a = run_strategy("A_baseline", instances, fe, post_p, post_q,
                         R=10, pop=80, gen=40, b_unknown=B_UNKNOWN,
                         extract_method="coppersmith_ext", mut_mode="standard")
        all_results["A_baseline"] = a
        save()
    else:
        log(f"  (skipped — already in results)")

    # --- Strategy B: More restarts (R=30) ---
    log("\n=== Strategy B: More restarts (R=30) ===")
    if "B_R30" not in all_results:
        b = run_strategy("B_R30", instances, fe, post_p, post_q,
                         R=30, pop=80, gen=40, b_unknown=B_UNKNOWN,
                         extract_method="coppersmith_ext", mut_mode="standard")
        all_results["B_R30"] = b
        save()
    else:
        log(f"  (skipped — already in results)")

    # --- Strategy C: Larger population (pop=200) ---
    log("\n=== Strategy C: Larger population (pop=200) ===")
    if "C_pop200" not in all_results:
        c = run_strategy("C_pop200", instances, fe, post_p, post_q,
                         R=10, pop=200, gen=40, b_unknown=B_UNKNOWN,
                         extract_method="coppersmith_ext", mut_mode="standard")
        all_results["C_pop200"] = c
        save()
    else:
        log(f"  (skipped — already in results)")

    # --- Strategy D: More generations (gen=100) ---
    log("\n=== Strategy D: More generations (gen=100) ===")
    if "D_gen100" not in all_results:
        d = run_strategy("D_gen100", instances, fe, post_p, post_q,
                         R=10, pop=80, gen=100, b_unknown=B_UNKNOWN,
                         extract_method="coppersmith_ext", mut_mode="standard")
        all_results["D_gen100"] = d
        save()
    else:
        log(f"  (skipped — already in results)")

    # --- Strategy E: Multi-bit mutation (wide: 2-3 bits) ---
    log("\n=== Strategy E: Multi-bit mutation (wide, 2-3 bits) ===")
    if "E_mut_wide" not in all_results:
        e = run_strategy("E_mut_wide", instances, fe, post_p, post_q,
                         R=10, pop=80, gen=40, b_unknown=B_UNKNOWN,
                         extract_method="coppersmith_ext", mut_mode="wide")
        all_results["E_mut_wide"] = e
        save()
    else:
        log(f"  (skipped — already in results)")

    # --- Strategy F: Predict q (q-posterior samples as p candidates) ---
    log("\n=== Strategy F: Predict q (q-posterior init) ===")
    if "F_predict_q" not in all_results:
        f_res = run_strategy("F_predict_q", instances, fe, post_p, post_q,
                             R=10, pop=80, gen=40, b_unknown=B_UNKNOWN,
                             extract_method="coppersmith_ext", mut_mode="standard",
                             use_q_posterior=True)
        all_results["F_predict_q"] = f_res
        save()
    else:
        log(f"  (skipped — already in results)")

    # --- Strategy G: Both posteriors (merged init) ---
    log("\n=== Strategy G: Both posteriors (merged init) ===")
    if "G_both_post" not in all_results:
        g = run_strategy("G_both_post", instances, fe, post_p, post_q,
                         R=10, pop=80, gen=40, b_unknown=B_UNKNOWN,
                         extract_method="coppersmith_ext", mut_mode="standard",
                         use_both_posteriors=True)
        all_results["G_both_post"] = g
        save()
    else:
        log(f"  (skipped — already in results)")

    # --- Strategy H: Brute extraction at bu=12 (100% reliable) ---
    log("\n=== Strategy H: Brute extraction at bu=12 ===")
    if "H_brute_bu12" not in all_results:
        h = run_strategy("H_brute_bu12", instances, fe, post_p, post_q,
                         R=10, pop=80, gen=40, b_unknown=B_UNKNOWN_BRUTE,
                         extract_method="brute", mut_mode="standard")
        all_results["H_brute_bu12"] = h
        save()
    else:
        log(f"  (skipped — already in results)")

    # --- Strategy I: Combined best (R=30, pop=200, wide mutation, both) ---
    log("\n=== Strategy I: Combined (R=30, pop=200, wide, both posteriors) ===")
    if "I_combined" not in all_results:
        i_res = run_strategy("I_combined", instances, fe, post_p, post_q,
                             R=30, pop=200, gen=40, b_unknown=B_UNKNOWN,
                             extract_method="coppersmith_ext", mut_mode="wide",
                             use_both_posteriors=True)
        all_results["I_combined"] = i_res
        save()
    else:
        log(f"  (skipped — already in results)")

    # --- Summary ---
    log(f"\n=== SUMMARY ===")
    log(f"{'Strategy':<25} {'Success':>8} {'Rate':>8}")
    log(f"{'-'*25} {'-'*8} {'-'*8}")
    for key in sorted(all_results.keys()):
        r = all_results[key]
        log(f"  {r['strategy']:<25} {r['success']:>5}/{r['n_test']:<3} "
            f"{r['success_rate']*100:>6.1f}%")

    # Which instances are solved by ANY strategy?
    solved_by = {inst["i"]: [] for inst in instances}
    for key, r in all_results.items():
        for pi in r["per_instance"]:
            if pi["success"]:
                solved_by[pi["i"]].append(r["strategy"])

    log(f"\n  Instance solvability:")
    for inst in instances:
        i = inst["i"]
        solvers = solved_by[i]
        log(f"    i={i}: {'NONE' if not solvers else ', '.join(solvers)}")

    total_solved = sum(1 for v in solved_by.values() if v)
    log(f"\n  Total instances solved by ANY strategy: {total_solved}/{len(instances)}")

    log(f"\nTOTAL TIME: {time.time() - t_start:.1f}s")
    save()
    log("Final JSON written.")


if __name__ == "__main__":
    main()
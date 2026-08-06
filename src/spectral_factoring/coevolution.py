"""Coevolutionary genetic programming for high-bit refinement (D195).

Two populations (p_high and q_high candidates) coevolve via the
constraint p*q ≈ N. When one population improves, the other's target
sharpens — a feedback loop that finds correct high bits the posterior
alone misses.

D195 result: 99% at 48-bit vs D157's 71% — a 28pp improvement.
D201: coevolution + D174 extended-root Coppersmith = 65% at 64-bit.
"""

from math import gcd, comb
from itertools import combinations
import numpy as np

from .extraction import enumerate_high_ball, high_ball_size


def _brute_force_extract(p_high, N, b_unknown):
    """Brute-force the low b_unknown bits via GCD."""
    if b_unknown <= 0:
        g = gcd(int(p_high), int(N))
        if 1 < g < int(N):
            return int(g)
        return None
    N_i = int(N)
    p_high_i = int(p_high)
    for x in range(1 << b_unknown):
        cand = p_high_i + x
        g = gcd(cand, N_i)
        if 1 < g < N_i:
            return cand
    return None


def _coppersmith_extract(p_high, N, b_unknown, m, t, use_extended=True,
                          max_x0_mult=8, try_q=True):
    """Extract via Coppersmith lattice (D159 + D174 extended roots).

    Returns (factor or None, n_copper_calls).
    If try_q, also try the q-side lattice (q_high = N // p_high).
    """
    from .extraction import (
        _reduce_lattice, _find_roots, _find_roots_extended,
    )
    N_i = int(N)
    X = 1 << b_unknown
    dim = m + 1 + t
    p_high_i = int(p_high)
    n_calls = [0]

    def _build_and_extract(high_bits):
        n_calls[0] += 1
        rows = []
        for j in range(m + 1):
            r = [0] * dim
            for k in range(j + 1):
                r[k] = (int(comb(j, k)) * (N_i ** (m - j))
                        * (X ** k) * (high_bits ** (j - k)))
            rows.append(r)
        for i in range(1, t + 1):
            r = [0] * dim
            for k in range(m + 1):
                r[i + k] = ((X ** (i + k)) * int(comb(m, k))
                            * (high_bits ** (m - k)))
            rows.append(r)
        reduced = _reduce_lattice(rows)
        if use_extended:
            return _find_roots_extended(reduced, high_bits, N_i, X, dim,
                                         max_x0_mult=max_x0_mult)
        return _find_roots(reduced, high_bits, N_i, X, dim)

    factor = _build_and_extract(p_high_i)
    if factor is not None and 1 < factor < N_i and gcd(factor, N_i) > 1:
        return int(factor), n_calls[0]
    if try_q:
        q_approx = N_i // (p_high_i | 1)
        q_high = (q_approx >> b_unknown) << b_unknown
        if q_high > 0:
            factor = _build_and_extract(q_high)
            if factor is not None and 1 < factor < N_i and gcd(factor, N_i) > 1:
                return int(factor), n_calls[0]
    return None, n_calls[0]


class CoevolutionaryGA:
    """Two-population coevolutionary GA on the HIGH BITS of p and q.

    Population A: p_high candidates (n_known-bit integers).
    Population B: q_high candidates (n_known-bit integers).
    Fitness: -|N - p_high * q_high * (2^b_unknown)^2| (product gap to N).
    Mutation: flip 1-2 high bits.
    Selection: keep best by joint product gap.
    Extraction: by default, brute-force low b_unknown bits via GCD.
        Set extract_method="coppersmith" to use the lattice (D201: needed
        for 64-bit where brute-force is too expensive at b_unknown=16).
        When use_extended=True (default), uses D174 extended-root extraction.

    The coevolutionary feedback: when A's best p_high improves, B's
    target q_high = N / (p_high << b_unknown) sharpens. This constraint
    propagation finds correct high bits that the posterior alone misses
    (D157: 71% at n=48; coevolution: 99%; D201: 65% at n=64).

    Args:
        N: the semiprime to factor.
        half: number of bits in p (= n // 2).
        b_unknown: number of unknown low bits.
        pop_size: population size for each of p and q.
        seed: random seed.
        mut_flip_prob: probability of flipping 1 bit (vs 2) in mutation.
        extract_method: "brute" or "coppersmith".
        coppersmith_m, coppersmith_t: lattice params (for coppersmith).
        use_extended: if True, use D174 extended-root extraction (default).
        try_q: if True, also attempt the q-side lattice (default).
    """

    def __init__(self, N, half, b_unknown, pop_size=100, seed=0,
                 mut_flip_prob=0.7, extract_method="brute",
                 coppersmith_m=5, coppersmith_t=4,
                 use_extended=True, try_q=True):
        self.N = int(N)
        self.half = half
        self.b_unknown = b_unknown
        self.n_known = half - b_unknown
        self.mask = (1 << self.n_known) - 1
        self.shift = b_unknown
        self.pop_size = pop_size
        self.rng = np.random.default_rng(seed)
        self.mut_flip_prob = mut_flip_prob
        self.extract_method = extract_method
        self.coppersmith_m = coppersmith_m
        self.coppersmith_t = coppersmith_t
        self.use_extended = use_extended
        self.try_q = try_q
        self.eval_count = 0
        self.copper_calls = 0

    def _product_gap(self, p_high_bits, q_high_bits):
        p_val = p_high_bits << self.shift
        q_val = q_high_bits << self.shift
        return abs(self.N - p_val * q_val)

    def _extract(self, p_high_bits):
        p_high = p_high_bits << self.shift
        self.eval_count += 1
        if self.extract_method == "coppersmith":
            factor, nc = _coppersmith_extract(
                p_high, self.N, self.b_unknown,
                self.coppersmith_m, self.coppersmith_t,
                use_extended=self.use_extended, try_q=self.try_q,
            )
            self.copper_calls += nc
            return factor
        return _brute_force_extract(p_high, self.N, self.b_unknown)

    def _mutate(self, v):
        n_flip = 1 if self.rng.random() < self.mut_flip_prob else 2
        for _ in range(n_flip):
            v ^= (1 << int(self.rng.integers(0, self.n_known)))
        return v & self.mask

    def run(self, init_p, init_q, generations=50):
        """Run coevolutionary search.

        Args:
            init_p: initial p candidates (from posterior sampling).
            init_q: initial q candidates (from posterior sampling or
                derived as N // p).
            generations: number of coevolution generations.

        Returns:
            (factor, eval_count) if a factor is found, (None, eval_count)
            otherwise.
        """
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
            scoredP = [(self._product_gap(p, bestQ), p) for p in newP]
            scoredP.sort(key=lambda x: x[0])
            scoredQ = [(self._product_gap(bestP, q), q) for q in newQ]
            scoredQ.sort(key=lambda x: x[0])
            popP = [p for _, p in scoredP[:self.pop_size]]
            popQ = [q for _, q in scoredQ[:self.pop_size]]

        return None, self.eval_count
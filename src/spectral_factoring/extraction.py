"""Extraction methods: corrected Coppersmith lattice + brute-force GCD.

The Coppersmith construction (D159 corrected version) uses:
  - m+1 main Howgrave-Graham polynomials g_j = N^{m-j} f^j
  - t x-shift polynomials h_i = x^i f^m

With dim = m+1+t, the achievable root bound is X < N^{(m+1)/(m+1+t)}.
For m=8, t=7 (dim=16): X < N^{1/2} >> N^{1/4} — works with slack.

D157/D155's bug was using ONLY the main polynomials (dim=5), giving
root bound X < 1 — useless. The 6.5% success was pure LLL luck.

D174 extended-root trick: the Howgrave-Graham bound is sufficient but
NOT necessary. LLL often finds roots slightly outside [0, X). By
accepting roots with |x0| <= max_x0_mult * X (default 8X) and checking
gcd(p_high + x0, N) > 1, we recover factors the standard extractor
discards. D201 showed this gives +20pp at 64-bit (45% -> 65%).
"""

from math import gcd, comb
from itertools import combinations
import numpy as np

try:
    from fpylll import IntegerMatrix, LLL, BKZ
    _HAS_FPYLLL = True
except Exception:
    _HAS_FPYLLL = False


def _check_candidate(guess, N):
    if guess is None or guess <= 1:
        return False
    g = gcd(int(guess), int(N))
    return 1 < g < int(N)


def _handrolled_lll(basis, delta=0.75):
    from fractions import Fraction
    n = len(basis)
    if n == 0:
        return []
    m = len(basis[0])
    B = [row[:] for row in basis]

    def dot(u, v):
        return sum(int(a) * int(b) for a, b in zip(u, v))

    def gramschmidt(B):
        Bs = [[Fraction(x) for x in row] for row in B]
        ortho = [row[:] for row in Bs]
        mu = [[Fraction(0)] * n for _ in range(n)]
        norms = [Fraction(0)] * n
        for i in range(n):
            for j in range(i):
                mu[i][j] = dot(Bs[i], ortho[j]) / (norms[j] if norms[j] != 0 else Fraction(1))
                for k in range(m):
                    ortho[i][k] -= mu[i][j] * ortho[j][k]
            norms[i] = dot(ortho[i], ortho[i])
        return mu, norms

    mu, norms = gramschmidt(B)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > Fraction(1, 2):
                q = int(round(float(mu[k][j])))
                for t in range(m):
                    B[k][t] -= q * B[j][t]
        mu, norms = gramschmidt(B)
        if norms[k - 1] > 0 and norms[k] >= (Fraction(delta) - mu[k][k - 1] ** 2) * norms[k - 1]:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            mu, norms = gramschmidt(B)
            k = max(k - 1, 1)
    return B


def _reduce_lattice(rows, use_bkz=False, bkz_block=20):
    n = len(rows)
    if n == 0:
        return []
    d = len(rows[0])
    if _HAS_FPYLLL:
        M = IntegerMatrix(n, d)
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                M[i, j] = int(v)
        LLL.reduction(M)
        if use_bkz and n <= 60:
            try:
                BKZ.reduction(M, BKZ.Param(block_size=min(bkz_block, n - 1)), BKZ.DEFAULT)
            except Exception:
                pass
        return [[int(M[i, j]) for j in range(d)] for i in range(n)]
    else:
        return _handrolled_lll(rows)


def _find_roots(reduced, p_high, N, X, dim, max_rows=20):
    """Find integer roots x0 in [0, X) from the LLL-reduced lattice rows."""
    N_i = int(N)
    p_high_i = int(p_high)
    for row in reduced[:max_rows]:
        deg = dim - 1
        while deg >= 0 and row[deg] == 0:
            deg -= 1
        if deg < 1:
            continue
        coeffs_np = [float(row[k]) for k in range(deg, -1, -1)]
        try:
            roots = np.roots(coeffs_np)
        except Exception:
            continue
        for r in roots:
            if abs(r.imag) > 1e-6:
                continue
            y0 = r.real
            x0 = y0 * X
            x0_int = int(round(x0))
            if abs(x0 - x0_int) < 1e-3 and 0 <= x0_int < X:
                cand = p_high_i + x0_int
                if 1 < cand < N_i and gcd(cand, N_i) > 1:
                    return cand
    return None


def _find_roots_extended(reduced, p_high, N, X, dim, max_rows=20,
                         max_x0_mult=8):
    """Extended root extraction (D174): accept roots OUTSIDE [0, X) when
    they give a valid gcd. The Howgrave-Graham bound is sufficient but
    not necessary; LLL often finds roots slightly outside the guaranteed
    range. When the high bits are wrong by 1 bit, the true root x0 =
    p_low + err lands just outside [0, X) — the standard extractor
    rejects it, but gcd(p_high + x0, N) confirms it.

    Args:
        max_x0_mult: accept |x0| <= max_x0_mult * X (default 8X).

    Returns the factor (int) or None.
    """
    N_i = int(N)
    p_high_i = int(p_high)
    max_x0 = X * max_x0_mult
    seen = set()
    for row in reduced[:max_rows]:
        deg = dim - 1
        while deg >= 0 and row[deg] == 0:
            deg -= 1
        if deg < 1:
            continue
        coeffs_np = [float(row[k]) for k in range(deg, -1, -1)]
        try:
            roots = np.roots(coeffs_np)
        except Exception:
            continue
        for r in roots:
            if abs(r.imag) > 1e-6:
                continue
            y0 = r.real
            x0 = y0 * X
            x0_int = int(round(x0))
            if abs(x0 - x0_int) > 1e-3:
                continue
            if abs(x0_int) > max_x0:
                continue
            if x0_int in seen:
                continue
            seen.add(x0_int)
            cand = p_high_i + x0_int
            if 1 < cand < N_i and gcd(cand, N_i) > 1:
                return cand
    return None


class CoppersmithExtractor:
    """Corrected Coppersmith lattice extractor (D159) with extended-root
    extraction (D174).

    Given the high bits of p (p_high) and the number of unknown low bits
    (b_unknown), find p via LLL on a lattice with x-shift polynomials.

    D174 extended-root trick: by default, use _find_roots_extended which
    accepts roots outside [0, X) up to max_x0_mult * X when they give a
    valid gcd. This recovers factors the standard extractor discards
    (+20pp at 64-bit, D201).

    Args:
        m: number of main Howgrave-Graham polynomials minus 1.
        t: number of x-shift polynomials.
        use_bkz: whether to also run BKZ after LLL.
        bkz_block: BKZ block size.
        use_extended: if True (default), use D174 extended root extraction.
        max_x0_mult: accept |x0| <= max_x0_mult * X (default 8).
    """

    def __init__(self, m=8, t=7, use_bkz=False, bkz_block=20,
                 use_extended=True, max_x0_mult=8):
        self.m = m
        self.t = t
        self.use_bkz = use_bkz
        self.bkz_block = bkz_block
        self.use_extended = use_extended
        self.max_x0_mult = max_x0_mult
        self.dim = m + 1 + t

    def extract(self, p_high, N, b_unknown):
        """Try to find the factor p = p_high + x_low where 0 <= x_low < 2^b_unknown.

        Returns the factor (int) or None.
        """
        if b_unknown <= 0:
            if _check_candidate(int(p_high), int(N)):
                return int(p_high)
            return None
        X = 1 << b_unknown
        dim = self.dim
        p_high_i = int(p_high)
        N_i = int(N)
        rows = []
        for j in range(self.m + 1):
            r = [0] * dim
            for k in range(j + 1):
                r[k] = (int(comb(j, k)) * (N_i ** (self.m - j))
                        * (X ** k) * (p_high_i ** (j - k)))
            rows.append(r)
        for i in range(1, self.t + 1):
            r = [0] * dim
            for k in range(self.m + 1):
                r[i + k] = ((X ** (i + k)) * int(comb(self.m, k))
                            * (p_high_i ** (self.m - k)))
            rows.append(r)
        reduced = _reduce_lattice(rows, use_bkz=self.use_bkz, bkz_block=self.bkz_block)
        if self.use_extended:
            return _find_roots_extended(reduced, p_high, N, X, dim,
                                        max_x0_mult=self.max_x0_mult)
        return _find_roots(reduced, p_high, N, X, dim)


class BruteForceExtractor:
    """Brute-force GCD extraction: try all 2^b_unknown candidates.

    Always succeeds when the high bits are correct. Cost: 2^b_unknown
    GCD operations. Use for small b_unknown (<= 20) or as a fallback.
    """

    def extract(self, p_high, N, b_unknown):
        if b_unknown <= 0:
            if _check_candidate(int(p_high), int(N)):
                return int(p_high)
            return None
        N_i = int(N)
        p_high_i = int(p_high)
        for x in range(1 << b_unknown):
            cand = p_high_i + x
            g = gcd(cand, N_i)
            if 1 < g < N_i:
                return cand
        return None


def enumerate_high_ball(sample_int, half, b_unknown, d):
    """Yield all p_high candidates within Hamming distance d of the
    sample's high bits (top n_known = half - b_unknown bits).

    Returns p_high values (with low bits zeroed).
    """
    n_known = half - b_unknown
    high_int = sample_int >> b_unknown
    seen = set()
    for r in range(d + 1):
        for combo in combinations(range(n_known), r):
            hc = high_int
            for bit_idx in combo:
                hc ^= (1 << (n_known - 1 - bit_idx))
            if hc not in seen:
                seen.add(hc)
                yield hc << b_unknown


def high_ball_size(n_known, d):
    """Number of candidates in a Hamming ball of radius d over n_known bits."""
    total = 0
    for r in range(d + 1):
        total += comb(n_known, r)
    return total
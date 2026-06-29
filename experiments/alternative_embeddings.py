#!/usr/bin/env python3
"""
Alternative lattice embeddings that don't require explicit factor base.
Goal: Find factors using geometric/algebraic structure rather than enumeration.
"""

import numpy as np
from math import gcd, isqrt, log2, floor, sqrt
from typing import Tuple, List, Optional
from dataclasses import dataclass


def lll_reduce(B: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """LLL lattice basis reduction."""
    B = B.astype(np.int64).copy()
    n = B.shape[0]
    
    def gram_schmidt(B):
        B_star = np.zeros_like(B, dtype=np.float64)
        mu = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            B_star[i] = B[i].astype(np.float64)
            for j in range(i):
                if np.dot(B_star[j], B_star[j]) != 0:
                    mu[i, j] = np.dot(B[i], B_star[j]) / np.dot(B_star[j], B_star[j])
                    B_star[i] -= mu[i, j] * B_star[j]
        return B_star, mu
    
    B_star, mu = gram_schmidt(B)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                B[k] -= round(mu[k, j]) * B[j]
                B_star, mu = gram_schmidt(B)
        lhs = delta * np.dot(B_star[k-1], B_star[k-1])
        rhs = np.dot(B_star[k], B_star[k])
        if k > 0:
            rhs += mu[k, k-1]**2 * np.dot(B_star[k-1], B_star[k-1])
        if lhs <= rhs:
            k += 1
        else:
            B[[k, k-1]] = B[[k-1, k]]
            B_star, mu = gram_schmidt(B)
            k = max(k - 1, 1)
    return B


@dataclass
class FactorResult:
    success: bool
    factors: Optional[Tuple[int, int]]
    method: str
    notes: str


# ============================================================================
# APPROACH 1: Quadratic Form Lattice
# ============================================================================
# Idea: p * q = N defines a hyperbola. Embed this relationship in a lattice.
# We look for integer points (p, q) on the hyperbola near sqrt(N), sqrt(N).

def quadratic_form_lattice(N: int, dimension: int = 20) -> FactorResult:
    """
    Embed the equation p * q = N in a lattice.
    
    We're looking for integer solutions to x * y = N where x, y ≈ sqrt(N).
    This is equivalent to finding points near (sqrt(N), sqrt(N)) on the hyperbola.
    """
    sqrtN = isqrt(N)
    
    # Lattice: we want to find vectors where x * y ≈ N
    # Use dimension to explore nearby points
    
    B = np.zeros((dimension + 2, dimension + 2), dtype=np.int64)
    
    # First dimension vectors: encode (x, y) pairs near sqrt(N)
    for i in range(dimension):
        offset = i - dimension // 2
        B[i, i] = 1
        B[i, dimension] = sqrtN + offset
        B[i, dimension + 1] = N // (sqrtN + offset) if sqrtN + offset > 0 else 0
    
    B[dimension, dimension] = 1
    B[dimension + 1, dimension + 1] = N
    
    B_reduced = lll_reduce(B)
    
    # Look for factors
    for v in B_reduced:
        # Check if any component divides N
        for x in v:
            x_int = abs(int(x))
            if x_int > 1 and x_int < N and N % x_int == 0:
                return FactorResult(True, (x_int, N // x_int), "quadratic_form", 
                                   f"Found factor {x_int} in vector component")
        
        # Check GCD with N
        g = gcd(abs(int(v.sum())), N)
        if g > 1 and g < N:
            return FactorResult(True, (g, N // g), "quadratic_form",
                               f"Found factor via GCD: {g}")
    
    return FactorResult(False, None, "quadratic_form", "No factors found")


def quadratic_form_v2(N: int, search_range: int = None) -> FactorResult:
    """
    Improved quadratic form: encode the relation (x + y)² - (x - y)² = 4N
    
    For factors p and q: let x = (p+q)/2, y = (p-q)/2
    Then p = x + y, q = x - y, and x² - y² = N
    
    We want to find x, y integers where x² - y² = N, x ≈ sqrt(N), y is small.
    """
    if search_range is None:
        search_range = int(N ** 0.25) + 10  # Heuristic range for y
    
    sqrtN = isqrt(N)
    
    # Lattice: find x, y such that x² - y² = N
    # Rewrite as: x² - y² - N = 0
    # We want x ≈ sqrt(N), y small
    
    dim = search_range
    B = np.zeros((dim + 2, dim + 2), dtype=np.int64)
    
    for i in range(dim):
        y_candidate = i
        # If y is small and x² - y² = N, then x ≈ sqrt(N + y²)
        x_approx = isqrt(N + y_candidate * y_candidate)
        
        B[i, i] = y_candidate
        B[i, dim] = x_approx
        B[i, dim + 1] = N
    
    B[dim, dim] = 1
    B[dim + 1, dim + 1] = 1
    
    B_reduced = lll_reduce(B)
    
    for v in B_reduced:
        for x in v:
            x_int = abs(int(x))
            if x_int > 1 and x_int < N and N % x_int == 0:
                return FactorResult(True, (x_int, N // x_int), "quadratic_v2",
                                   f"Found factor {x_int}")
        
        g = gcd(abs(int(v.sum())), N)
        if g > 1 and g < N:
            return FactorResult(True, (g, N // g), "quadratic_v2", f"GCD factor: {g}")
    
    return FactorResult(False, None, "quadratic_v2", "No factors found")


# ============================================================================
# APPROACH 2: Polynomial Embedding
# ============================================================================
# Idea: p + q = S, p * q = N
# The polynomial x² - Sx + N = 0 has roots p and q.
# If we can find S = p + q, we can solve for p and q.

def polynomial_lattice(N: int, dimension: int = 20) -> FactorResult:
    """
    Embed the polynomial relationship.
    
    We know p * q = N and p + q = S for some unknown S.
    The polynomial f(x) = x² - Sx + N has roots p and q.
    
    For balanced semiprimes: p ≈ q ≈ sqrt(N), so S ≈ 2*sqrt(N)
    
    Strategy: Search for S in a lattice that encodes polynomial structure.
    """
    sqrtN = isqrt(N)
    
    # For balanced semiprime: S = p + q ≈ 2 * sqrt(N)
    # Search range for S
    S_min = 2 * sqrtN - dimension
    S_max = 2 * sqrtN + dimension
    
    B = np.zeros((dimension, dimension + 1), dtype=np.int64)
    
    for i in range(dimension):
        S = 2 * sqrtN - dimension // 2 + i
        
        # Check if S² - 4N is a perfect square (discriminant test)
        # For p and q integers, we need S² - 4N = (q - p)² to be a perfect square
        
        B[i, i] = 1
        B[i, dimension] = S
    
    B_reduced = lll_reduce(np.vstack([B, np.eye(dimension + 1, dtype=np.int64)[-1:]]))
    
    # Look for discriminants that are perfect squares
    for v in B_reduced:
        S = abs(int(v[dimension])) if dimension < len(v) else 0
        if S > 0:
            discriminant = S * S - 4 * N
            if discriminant > 0:
                sqrt_disc = isqrt(discriminant)
                if sqrt_disc * sqrt_disc == discriminant:
                    # Perfect square discriminant!
                    p = (S + sqrt_disc) // 2
                    q = (S - sqrt_disc) // 2
                    if p * q == N:
                        return FactorResult(True, (p, q), "polynomial",
                                           f"Found S={S}, perfect square discriminant")
    
    return FactorResult(False, None, "polynomial", "No valid discriminant found")


# ============================================================================
# APPROACH 3: Continued Fraction Lattice
# ============================================================================
# Idea: Continued fractions of sqrt(N) encode factor information.
# The CFRAC method uses this, but we'll try a direct lattice approach.

def continued_fraction_lattice(N: int, iterations: int = 30) -> FactorResult:
    """
    Use continued fractions of sqrt(N).
    
    The convergents A_k/B_k of sqrt(N) satisfy:
    A_k² ≡ N * B_k² (mod some factor)
    
    This can reveal factors when A_k² - N*B_k² shares a common factor with N.
    """
    sqrtN = isqrt(N)
    
    # Generate continued fraction convergents
    # sqrt(N) = sqrtN + 1/(a1 + 1/(a2 + ...))
    
    m = 0
    d = 1
    a0 = sqrtN
    a = a0
    
    # Track convergents
    h_prev, h = 1, a0
    k_prev, k = 0, 1
    
    # Check convergents directly first (this is CFRAC method)
    for i in range(iterations):
        m = d * a - m
        d = (N - m * m) // d if d != 0 else 1
        if d == 0:
            break
        a = (sqrtN + m) // d
        
        # Next convergent
        h_new = a * h + h_prev
        k_new = a * k + k_prev
        
        # Check: does h² - N*k² reveal a factor?
        h_prev, h = h, h_new
        k_prev, k = k, k_new
        
        # Early check
        val = h*h - N*k*k
        if val != 0:
            g = gcd(abs(val), N)
            if g > 1 and g < N:
                return FactorResult(True, (g, N // g), "continued_fraction",
                                   f"Convergent {i}: GCD(h²-Nk², N) = {g}")
    
    return FactorResult(False, None, "continued_fraction", "No factors from convergents")


# ============================================================================
# APPROACH 4: Congruence Lattice
# ============================================================================
# Idea: Find x² ≡ y² (mod N) where x ≠ ±y (mod N)
# Then GCD(x - y, N) or GCD(x + y, N) gives a factor.

def congruence_lattice(N: int, dimension: int = 20) -> FactorResult:
    """
    Search for congruences x² ≡ y² (mod N) with x ≠ ±y.
    
    We build a lattice of (potential) squares mod N and look for relations.
    """
    sqrtN = isqrt(N)
    
    B = np.zeros((dimension, dimension), dtype=np.int64)
    
    for i in range(dimension):
        x = sqrtN + i + 1
        x_sq = x * x
        
        B[i, i] = x
        B[i, dimension - 1] = x_sq % N if dimension > i else 0
    
    B_reduced = lll_reduce(B)
    
    # Look for pairs of vectors with related squares
    for i in range(len(B_reduced)):
        for j in range(i + 1, len(B_reduced)):
            x = int(B_reduced[i][0]) if len(B_reduced[i]) > 0 else 0
            y = int(B_reduced[j][0]) if len(B_reduced[j]) > 0 else 0
            
            if x > 0 and y > 0 and x != y:
                x_sq_mod = (x * x) % N
                y_sq_mod = (y * y) % N
                
                if x_sq_mod == y_sq_mod:
                    # Found congruence!
                    g1 = gcd(abs(x - y), N)
                    g2 = gcd(x + y, N)
                    
                    if g1 > 1 and g1 < N:
                        return FactorResult(True, (g1, N // g1), "congruence",
                                           f"x² ≡ y² (mod N): {x}² ≡ {y}²")
                    if g2 > 1 and g2 < N:
                        return FactorResult(True, (g2, N // g2), "congruence",
                                           f"x² ≡ y² (mod N): {x}² ≡ {y}²")
    
    return FactorResult(False, None, "congruence", "No square congruences found")


# ============================================================================
# APPROACH 5: Rational Approximation Lattice
# ============================================================================
# Idea: Approximate sqrt(N) with rationals. The convergents encode factor info.

def rational_approx_lattice(N: int, dimension: int = 15) -> FactorResult:
    """
    Use rational approximation to sqrt(N).
    
    If p/q ≈ sqrt(N), then p² ≈ N*q², so p² - N*q² might share factors with N.
    """
    sqrtN = isqrt(N)
    
    # Build lattice that encodes approximations
    B = np.zeros((dimension, dimension), dtype=np.int64)
    
    for i in range(dimension):
        p = sqrtN + i
        q = 1
        
        B[i, 0] = p
        B[i, 1] = q
        B[i, 2] = p*p - N*q*q if dimension > 2 else 0
    
    B_reduced = lll_reduce(B)
    
    for v in B_reduced:
        if len(v) >= 3:
            diff = int(v[2])
            g = gcd(abs(diff), N)
            if g > 1 and g < N:
                return FactorResult(True, (g, N // g), "rational_approx",
                                   f"GCD(p² - Nq², N) = {g}")
    
    return FactorResult(False, None, "rational_approx", "No factors from approximations")


# ============================================================================
# APPROACH 6: Simultaneous Diophantine Approximation
# ============================================================================
# Idea: Approximate multiple irrationals simultaneously.
# For N = pq, we can use relationships involving sqrt(p/q) or similar.

def simultaneous_approx_lattice(N: int, dimension: int = 10) -> FactorResult:
    """
    Simultaneous Diophantine approximation approach.
    
    We look for integer relations among powers of potential factors.
    """
    sqrtN = isqrt(N)
    
    # Try to find a, b such that a*b is close to N but a ≠ sqrt(N)
    B = np.zeros((dimension + 1, dimension + 1), dtype=np.int64)
    
    for i in range(dimension):
        a = sqrtN - dimension // 2 + i
        b = N // a if a > 0 else 0
        
        B[i, i] = a
        B[i, dimension] = abs(a * b - N)
    
    B[dimension, dimension] = 1
    
    B_reduced = lll_reduce(B)
    
    for v in B_reduced:
        for x in v:
            x_int = abs(int(x))
            if x_int > 1 and x_int < N and N % x_int == 0:
                return FactorResult(True, (x_int, N // x_int), "simultaneous",
                                   f"Found factor {x_int}")
    
    return FactorResult(False, None, "simultaneous", "No factors found")


# ============================================================================
# APPROACH 7: Elliptic Curve Analogue
# ============================================================================
# Idea: Lenstra's ECM finds factors via elliptic curve group structure.
# Can we encode elliptic curve operations in a lattice?

def elliptic_inspired_lattice(N: int, max_trials: int = 50) -> FactorResult:
    """
    Lattice-inspired approach using elliptic curve-like structure.
    
    On an elliptic curve E mod N: y² = x³ + ax + b
    Points have a group structure. If we find P, 2P, 3P, ... and 
    operations fail mod p but not mod q, we can factor.
    
    Here we try to encode curve points and operations in a lattice.
    """
    # Choose curve parameters
    for trial in range(max_trials):
        a = trial % 20 - 10  # Small a values
        b = 1
        
        # Try a few points
        for x in range(2, min(20, N)):
            # Point (x, y) on curve: y² = x³ + ax + b
            rhs = (x * x * x + a * x + b) % N
            
            # Check if rhs is a square mod N (we can't compute sqrt directly if N is composite)
            # Instead, build a lattice from x values
            
            B = np.array([
                [x, 1, 0],
                [rhs, 0, 1],
                [N, 0, 0]
            ], dtype=np.int64)
            
            B_reduced = lll_reduce(B)
            
            for v in B_reduced:
                g = gcd(abs(int(v[0])), N)
                if g > 1 and g < N:
                    return FactorResult(True, (g, N // g), "elliptic",
                                       f"Trial {trial}, x={x}: GCD = {g}")
    
    return FactorResult(False, None, "elliptic", "No factors via elliptic-inspired lattice")


# ============================================================================
# TESTING
# ============================================================================

def test_all_approaches(N: int, p: int, q: int):
    """Test all alternative approaches on a given semiprime."""
    print(f"\n{'='*70}")
    print(f"Testing N = {N} = {p} × {q}")
    print(f"{'='*70}")
    
    approaches = [
        ("Quadratic Form v1", lambda: quadratic_form_lattice(N)),
        ("Quadratic Form v2", lambda: quadratic_form_v2(N)),
        ("Polynomial", lambda: polynomial_lattice(N)),
        ("Continued Fraction", lambda: continued_fraction_lattice(N)),
        ("Congruence", lambda: congruence_lattice(N)),
        ("Rational Approx", lambda: rational_approx_lattice(N)),
        ("Simultaneous Approx", lambda: simultaneous_approx_lattice(N)),
        ("Elliptic-Inspired", lambda: elliptic_inspired_lattice(N)),
    ]
    
    results = []
    for name, func in approaches:
        result = func()
        status = "✓" if result.success else "✗"
        print(f"{status} {name:25s}: {result.notes}")
        results.append((name, result))
    
    return results


def systematic_test(test_cases: List[Tuple[int, int, int]]):
    """Run all approaches on multiple test cases."""
    print(f"\n{'='*70}")
    print(f"SYSTEMATIC TESTING")
    print(f"{'='*70}")
    
    approaches = [
        ("Quadratic Form v1", lambda N: quadratic_form_lattice(N)),
        ("Quadratic Form v2", lambda N: quadratic_form_v2(N)),
        ("Polynomial", lambda N: polynomial_lattice(N)),
        ("Continued Fraction", lambda N: continued_fraction_lattice(N)),
        ("Congruence", lambda N: congruence_lattice(N)),
        ("Rational Approx", lambda N: rational_approx_lattice(N)),
        ("Simultaneous Approx", lambda N: simultaneous_approx_lattice(N)),
        ("Elliptic-Inspired", lambda N: elliptic_inspired_lattice(N)),
    ]
    
    # Results per approach
    approach_wins = {name: 0 for name, _ in approaches}
    
    for N, p, q in test_cases:
        print(f"\nN = {N} = {p} × {q}")
        for name, func in approaches:
            result = func(N)
            status = "✓" if result.success else "✗"
            if result.success:
                approach_wins[name] += 1
            print(f"  {status} {name}: {result.notes}")
    
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    for name, _ in approaches:
        wins = approach_wins[name]
        total = len(test_cases)
        print(f"{name:25s}: {wins}/{total} ({100*wins/total:.0f}%)")
    
    return approach_wins


if __name__ == "__main__":
    # Small test cases
    small_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
        (1517, 37, 41),
    ]
    
    # Medium test cases
    medium_cases = [
        (4627, 59, 79),    # 59 * 79
        (6149, 61, 101),   # 61 * 101
        (7387, 83, 89),    # 83 * 89
        (8633, 89, 97),    # 89 * 97
        (10403, 101, 103), # 101 * 103 (close primes)
    ]
    
    print("=" * 70)
    print("ALTERNATIVE LATTICE EMBEDDINGS FOR FACTORIZATION")
    print("Goal: Find factors WITHOUT explicit factor base enumeration")
    print("=" * 70)
    
    # Test on small cases
    for N, p, q in small_cases:
        test_all_approaches(N, p, q)
    
    # Systematic test
    all_cases = small_cases + medium_cases
    systematic_test(all_cases)
#!/usr/bin/env python3
"""
Exploratory approaches to avoid factor base enumeration.

These are experimental implementations of approaches from research notes.
Goal: Factor N without requiring explicit factor base containing p or q.
"""

import numpy as np
from math import gcd, isqrt
from typing import Tuple, Optional


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


# =============================================================================
# APPROACH 2: Polynomial Hyperbola Lattice
# =============================================================================

def hyperbola_lattice_factor(N: int, search_radius: int = None) -> Optional[Tuple[int, int]]:
    """
    Encode xy = N as a lattice problem.
    
    We look for integer solutions (x, y) to xy = N near (√N, √N).
    For balanced semiprimes, this is where the solution lies.
    
    The lattice encodes the relation:
    - x·y = N
    - x ≈ √N
    - y ≈ √N
    
    Returns factors if successful, None otherwise.
    
    Note: For balanced semiprimes, this requires checking ~N^(1/4) candidates,
    which is still subexponential but better than factor base enumeration.
    """
    sqrtN = isqrt(N)
    
    if search_radius is None:
        search_radius = int(N ** 0.25) + 10
    
    # Method 1: Direct hyperbola search
    # For x close to √N, check if x divides N
    for delta in range(-search_radius, search_radius + 1):
        x = sqrtN + delta
        if x > 0 and N % x == 0:
            return (x, N // x)
    
    # Method 2: Lattice encoding of hyperbola
    # We want vectors (x, y) such that xy ≈ N and x ≈ y ≈ √N
    
    # Build a 2D lattice where short vectors encode (x - √N, y - √N)
    # with the constraint that xy = N
    
    # This is tricky because the hyperbola xy = N is not a linear subspace
    # We approximate it locally near (√N, √N)
    
    # Tangent line at (√N, √N): y - √N = -√N/√N (x - √N) = -(x - √N)
    # So y + x = 2√N for points on the tangent
    
    # Build lattice:
    B = np.array([
        [1, 0, sqrtN],
        [0, 1, sqrtN],
        [1, 1, 2*sqrtN],
        [0, 0, N]
    ], dtype=np.int64)
    
    B_reduced = lll_reduce(B)
    
    # Check for factors in reduced vectors
    for v in B_reduced:
        # Method: If v encodes (x, y, ...), check if x or y divides N
        for x in v[:2]:
            x_int = abs(int(x))
            if x_int > 1 and x_int < N and N % x_int == 0:
                return (x_int, N // x_int)
    
    return None


# =============================================================================
# APPROACH 5: Higher-Dimensional Embedding (Iterative)
# =============================================================================

def iterative_lattice_factor(N: int, iterations: int = 5) -> Optional[Tuple[int, int]]:
    """
    Iterative approach using partial information.
    
    Start with small factor base and use lattice structure to guide expansion.
    
    Algorithm:
    1. Build small factor base lattice
    2. Analyze reduced basis for "near-miss" vectors
    3. Use near-misses to guide factor base expansion
    4. Repeat
    """
    def generate_primes(n: int):
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = all(candidate % p != 0 for p in primes if p * p <= candidate)
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes
    
    # Start with small factor base
    factor_base_size = 20
    
    for iteration in range(iterations):
        primes = generate_primes(factor_base_size)
        dim = len(primes)
        
        # Build modular lattice
        B = np.zeros((dim + 2, dim + 2), dtype=np.int64)
        
        for i, p in enumerate(primes):
            B[i, i] = p
            B[i, dim] = N % p
        
        B[dim, dim] = 1
        B[dim + 1, dim + 1] = 1
        
        B_reduced = lll_reduce(B)
        
        # Check for factors
        for v in B_reduced:
            g = gcd(abs(int(v.sum())), N)
            if g > 1 and g < N:
                return (g, N // g)
        
        # Analyze "near-miss" vectors
        # These are vectors where the residue N mod p is small
        near_misses = []
        for i, p in enumerate(primes):
            residue = N % p
            if residue < p // 10 or residue > 9 * p // 10:
                # Small or large residue - "near" divisibility
                near_misses.append((p, residue))
        
        # Expand factor base near near-misses
        # This is heuristic - we're looking for primes close to near-miss primes
        if near_misses:
            # Add primes near the near-miss primes
            for p, _ in near_misses[:3]:
                factor_base_size = max(factor_base_size, p + 50)
        else:
            # Just expand
            factor_base_size += 50
        
        # Safety limit
        if factor_base_size > 10000:
            break
    
    return None


# =============================================================================
# APPROACH 6: Dual Lattice Analysis
# =============================================================================

def dual_lattice_factor(N: int, dimension: int = 20) -> Optional[Tuple[int, int]]:
    """
    Analyze the dual lattice of the modular construction.
    
    The dual lattice L* consists of vectors y such that y·x ∈ ℤ for all x ∈ L.
    
    Hypothesis: The dual might encode factorization information differently.
    """
    def generate_primes(n: int):
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = all(candidate % p != 0 for p in primes if p * p <= candidate)
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes
    
    primes = generate_primes(dimension)
    
    # Build primal lattice
    B = np.zeros((dimension + 2, dimension + 2), dtype=np.int64)
    for i, p in enumerate(primes):
        B[i, i] = p
        B[i, dimension] = N % p
    B[dimension, dimension] = 1
    B[dimension + 1, dimension + 1] = 1
    
    # Compute dual lattice: B* = (B^T)^(-1)
    # Actually, we want the lattice generated by B* = (B^{-1})^T
    try:
        B_float = B.astype(np.float64)
        B_dual = np.linalg.inv(B_float).T
        
        # Round to integers (approximate dual)
        B_dual_int = np.round(B_dual * N).astype(np.int64)  # Scale by N
        
        # Reduce dual lattice
        B_dual_reduced = lll_reduce(B_dual_int)
        
        # Check for factors in dual
        for v in B_dual_reduced:
            for x in v:
                x_int = abs(int(x))
                if x_int > 1 and x_int < N and N % x_int == 0:
                    return (x_int, N // x)
    
    except np.linalg.LinAlgError:
        # Singular matrix - lattice is degenerate
        pass
    
    return None


# =============================================================================
# APPROACH: Geometric Constraint Lattice
# =============================================================================

def geometric_constraint_factor(N: int) -> Optional[Tuple[int, int]]:
    """
    Use geometric constraints on p and q.
    
    For balanced semiprime N = pq:
    - p + q = S ≈ 2√N
    - p - q = D (small)
    - p² + q² = S² - 2N
    - (p + q)² - (p - q)² = 4N
    
    Encode these constraints in a lattice.
    """
    sqrtN = isqrt(N)
    
    # We want to find D = p - q
    # If D is small, then p = (S + D)/2 and q = (S - D)/2
    # where S = p + q ≈ 2√N
    
    # For small N, D is often small
    # Try small values of D
    for D in range(1, min(1000, sqrtN)):
        # S² - D² = 4N
        # S = √(4N + D²)
        S_sq = 4 * N + D * D
        S = isqrt(S_sq)
        
        if S * S == S_sq:
            # Found valid S and D
            # p = (S + D) / 2, q = (S - D) / 2
            if (S + D) % 2 == 0:
                p = (S + D) // 2
                q = (S - D) // 2
                if p > 1 and q > 1 and p * q == N:
                    return (min(p, q), max(p, q))
    
    # Alternative: Use lattice to search
    # Build lattice encoding S² - D² = 4N
    B = np.array([
        [1, 0, sqrtN],
        [0, 1, sqrtN],
        [1, 1, 2*sqrtN],
        [1, -1, 0],
        [0, 0, 4*N]
    ], dtype=np.int64)
    
    B_reduced = lll_reduce(B)
    
    for v in B_reduced:
        # Check various combinations
        for x in v:
            x_int = abs(int(x))
            if x_int > 1 and x_int < N and N % x_int == 0:
                return (x_int, N // x_int)
    
    return None


# =============================================================================
# TESTING
# =============================================================================

def test_all_approaches():
    """Test all approaches on small semiprimes."""
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
        (1517, 37, 41),
    ]
    
    approaches = [
        ("Hyperbola", hyperbola_lattice_factor),
        ("Iterative", iterative_lattice_factor),
        ("Dual Lattice", dual_lattice_factor),
        ("Geometric", geometric_constraint_factor),
    ]
    
    print("=" * 70)
    print("Testing Approaches to Avoid Factor Base Enumeration")
    print("=" * 70)
    
    for N, p, q in test_cases:
        print(f"\nN = {N} = {p} × {q}")
        for name, func in approaches:
            result = func(N)
            status = "✓" if result is not None and result[0] * result[1] == N else "✗"
            print(f"  {status} {name:20s}: {result}")


if __name__ == "__main__":
    test_all_approaches()
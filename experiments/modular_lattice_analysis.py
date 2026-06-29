#!/usr/bin/env python3
"""
Investigate the modular lattice construction that successfully found factors.
This is the first working approach - need to understand why it works.
"""

import numpy as np
from math import gcd, isqrt
from typing import Tuple, List, Optional


def generate_primes(n: int) -> List[int]:
    """Generate the first n primes."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = all(candidate % p != 0 for p in primes if p * p <= candidate)
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


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
        if k > 0 and mu[k, k-1] is not None:
            rhs += mu[k, k-1]**2 * np.dot(B_star[k-1], B_star[k-1])
        if lhs <= rhs:
            k += 1
        else:
            B[[k, k-1]] = B[[k-1, k]]
            B_star, mu = gram_schmidt(B)
            k = max(k - 1, 1)
    return B


def modular_lattice_factor(N: int, factor_base_size: int = 20, verbose: bool = True) -> Optional[Tuple[int, int]]:
    """
    The modular lattice construction that found factors.
    
    Construction:
    - Rows i=0..dim-1: [primes[i], 0..0, N % primes[i], 0..0]
    - Last two rows: basis vectors for the "target" dimensions
    
    This encodes the fact that if p divides N, then N ≡ 0 (mod p).
    Short vectors in this lattice may encode linear combinations of primes
    whose product has a specific relationship with N.
    """
    primes = generate_primes(factor_base_size)
    dim = factor_base_size
    
    # Lattice dimension: dim + 2
    B = np.zeros((dim + 2, dim + 2), dtype=np.int64)
    
    for i in range(dim):
        B[i, i] = primes[i]
        B[i, dim] = N % primes[i]
    
    B[dim, dim] = 1
    B[dim + 1, dim + 1] = 1
    
    if verbose:
        print(f"\n=== Modular Lattice Construction ===")
        print(f"N = {N}, factor base size = {factor_base_size}")
        print(f"First 5 primes: {primes[:5]}")
        print(f"Basis shape: {B.shape}")
    
    B_reduced = lll_reduce(B)
    
    if verbose:
        print(f"\nReduced basis - first 5 vectors:")
        for i in range(min(5, len(B_reduced))):
            v = B_reduced[i]
            norm = np.linalg.norm(v)
            print(f"  v[{i}]: norm={norm:.2f}, sum={int(v.sum())}")
    
    # Check all vectors for factorization
    factors_found = []
    for i, v in enumerate(B_reduced):
        # Method 1: GCD of sum with N
        g = gcd(abs(int(v.sum())), N)
        if g > 1 and g < N:
            if verbose:
                print(f"\n  Found factor via GCD(sum, N): {g}")
            factors_found.append((g, N // g))
        
        # Method 2: GCD of individual components with N
        for j, x in enumerate(v):
            g = gcd(abs(int(x)), N)
            if g > 1 and g < N:
                if verbose:
                    print(f"\n  Found factor via GCD(component {j}, N): {g}")
                factors_found.append((g, N // g))
        
        # Method 3: GCD of vector dot products
        # This checks if the vector encodes a multiplicative relationship
        for j in range(len(v)):
            for k in range(j + 1, len(v)):
                g = gcd(abs(int(v[j] * v[k])), N)
                if g > 1 and g < N:
                    factors_found.append((g, N // g))
    
    if factors_found:
        return factors_found[0]
    return None


def analyze_why_it_works(N: int, p: int, q: int, factor_base_size: int = 20):
    """
    Analyze why the modular construction reveals factors.
    """
    print(f"\n{'='*70}")
    print(f"Analysis: N = {N} = {p} × {q}")
    print(f"{'='*70}")
    
    primes = generate_primes(factor_base_size)
    dim = factor_base_size
    
    # Build the lattice
    B = np.zeros((dim + 2, dim + 2), dtype=np.int64)
    
    for i in range(dim):
        B[i, i] = primes[i]
        B[i, dim] = N % primes[i]
    
    B[dim, dim] = 1
    B[dim + 1, dim + 1] = 1
    
    print(f"\nOriginal basis structure:")
    print(f"  Rows 0-{dim-1}: encode primes and N mod primes")
    print(f"  Row {dim}: target basis vector [0,0,...,1,0]")
    print(f"  Row {dim+1}: target basis vector [0,0,...,0,1]")
    
    print(f"\nN mod primes:")
    for i, prime in enumerate(primes[:10]):
        r = N % prime
        print(f"  N mod {prime} = {r}" + (" ← p is in factor base!" if prime == p or prime == q else ""))
    
    # Key insight: if p is in the factor base, N mod p = 0
    if p in primes:
        idx = primes.index(p)
        print(f"\n*** KEY INSIGHT: p = {p} is in factor base at index {idx}")
        print(f"    Row {idx}: B[{idx}] = [{p} at position {idx}, ..., N mod {p} = 0 at position {dim}]")
        print(f"    This row encodes: {p} | N (divisibility)")
    
    if q in primes:
        idx = primes.index(q)
        print(f"\n*** KEY INSIGHT: q = {q} is in factor base at index {idx}")
        print(f"    Row {idx}: B[{idx}] = [{q} at position {idx}, ..., N mod {q} = 0 at position {dim}]")
    
    B_reduced = lll_reduce(B)
    
    print(f"\nReduced basis analysis:")
    print(f"  Looking for vectors where sum has non-trivial GCD with N...")
    
    for i, v in enumerate(B_reduced[:10]):
        s = int(v.sum())
        g = gcd(abs(s), N)
        print(f"  v[{i}]: sum={s}, GCD(|sum|, N)={g}", end="")
        if g > 1 and g < N:
            print(f" ← FACTOR FOUND: {g}")
        else:
            print()


def systematic_test(N_values: List[Tuple[int, int, int]]):
    """
    Systematically test the modular lattice on multiple semiprimes.
    """
    print(f"\n{'='*70}")
    print(f"Systematic Testing")
    print(f"{'='*70}")
    
    results = []
    for N, p, q in N_values:
        factor = modular_lattice_factor(N, factor_base_size=25, verbose=False)
        success = factor is not None
        results.append((N, p, q, success, factor))
        
        status = "✓" if success else "✗"
        factor_str = f"{factor[0]} × {factor[1]}" if factor else "N/A"
        print(f"{status} N={N:6d} ({p}×{q:3d}) → {factor_str:15s}")
    
    success_rate = sum(1 for r in results if r[3]) / len(results)
    print(f"\nSuccess rate: {success_rate:.1%} ({sum(1 for r in results if r[3])}/{len(results)})")
    
    return results


def test_larger_N():
    """
    Test on larger semiprimes to see if this scales.
    """
    print(f"\n{'='*70}")
    print(f"Testing Larger Semiprimes")
    print(f"{'='*70}")
    
    # Generate larger semiprimes
    test_cases = [
        (1147, 31, 37),
        (1517, 37, 41),
        (1643, 31, 53),
        (1739, 37, 47),
        (1829, 31, 59),
        (1927, 41, 47),
        (1961, 37, 53),
        (2021, 43, 47),
        (2147, 43, 499),  # Actually 43 × 499, let me check
        (2491, 47, 53),
    ]
    
    # Verify the factorizations first
    verified_cases = []
    for N, p, q in test_cases:
        if p * q != N:
            # Find actual factors
            for i in range(2, isqrt(N) + 1):
                if N % i == 0:
                    verified_cases.append((N, i, N // i))
                    break
        else:
            verified_cases.append((N, p, q))
    
    return systematic_test(verified_cases)


def investigate_factor_base_requirement():
    """
    Investigate: what's the minimum factor base size needed?
    Does p or q need to be in the factor base?
    """
    print(f"\n{'='*70}")
    print(f"Investigating Factor Base Size Requirements")
    print(f"{'='*70}")
    
    N, p, q = 899, 29, 31  # 29 and 31 are small primes
    
    for fb_size in [5, 10, 15, 20, 25, 30]:
        primes = generate_primes(fb_size)
        p_in_fb = p in primes
        q_in_fb = q in primes
        
        factor = modular_lattice_factor(N, factor_base_size=fb_size, verbose=False)
        success = factor is not None
        
        status = "✓" if success else "✗"
        print(f"{status} factor_base={fb_size:2d}: p_in_fb={p_in_fb}, q_in_fb={q_in_fb} → {factor}")


if __name__ == "__main__":
    # Test cases from earlier
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
        (1517, 37, 41),
    ]
    
    # Analyze why it works
    for N, p, q in test_cases:
        analyze_why_it_works(N, p, q)
    
    # Investigate factor base requirements
    investigate_factor_base_requirement()
    
    # Test larger values
    test_larger_N()
    
    # Systematic test
    systematic_test(test_cases)
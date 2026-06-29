#!/usr/bin/env python3
"""
Analyze lattice structure when factors are known.
Goal: Understand what properties distinguish factor-revealing vectors.
"""

import numpy as np
from math import gcd, isqrt, log2
from typing import Tuple, List, Optional
from itertools import combinations


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


def schnorr_lattice_basis_corrected(N: int, dimension: int) -> Tuple[np.ndarray, List[int]]:
    """
    Construct Schnorr's lattice basis more carefully.
    
    The lattice encodes relationships between:
    - Logarithms of primes
    - The target N
    
    Short vectors may reveal linear combinations that give us information about N's factors.
    """
    primes = generate_primes(dimension)
    n = dimension + 1
    
    B = np.zeros((n + 1, n + 1), dtype=np.int64)
    
    # Encode logarithmic relationships
    for i, p in enumerate(primes):
        B[i, i] = 1
        # We use log2(p) scaled by a large constant to make it integral
        # This encodes that we're looking at multiplicative structure
        B[i, n] = int(log2(p) * 1000)  # Scaling factor
    
    # The target row encodes N
    B[dimension, dimension] = 1
    B[dimension, n] = int(log2(N) * 1000)
    
    # Basis vector for the "target" column
    B[n, n] = 1
    
    return B, primes


def improved_schnorr_lattice(N: int, dimension: int) -> Tuple[np.ndarray, List[int]]:
    """
    Alternative Schnorr construction based on the original paper.
    
    Uses a lattice where:
    - Rows correspond to relations between primes
    - We look for vectors where the linear combination of primes relates to N
    """
    primes = generate_primes(dimension)
    B = np.zeros((dimension + 2, dimension + 2), dtype=np.int64)
    
    # First dimension rows: prime basis
    for i in range(dimension):
        B[i, i] = 1
    
    # Row for N: we want combinations where product of primes^coeffs ≈ N
    # This is harder to encode directly
    
    # Alternative: encode the equation log(p) + log(q) = log(N)
    # But we don't know p, q
    
    # Use the relation: if p divides N, then p is in the factor base
    for i, p in enumerate(primes):
        B[i, dimension] = N % p
        B[i, dimension + 1] = p
    
    B[dimension, dimension] = 1
    B[dimension + 1, dimension + 1] = 1
    
    return B, primes


def cvp_lattice(N: int, dimension: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a lattice for CVP-based factorization.
    
    The idea: embed N in a lattice and look for the closest vector
    that encodes a divisor relationship.
    """
    B = np.zeros((dimension, dimension), dtype=np.int64)
    
    sqrt_N = isqrt(N)
    
    for i in range(dimension - 1):
        B[i, i] = 1
        B[i, dimension - 1] = (sqrt_N + i)
    
    B[dimension - 1, dimension - 1] = N
    
    # Target point
    target = np.zeros(dimension, dtype=np.int64)
    target[dimension - 1] = sqrt_N
    
    return B, target


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


def check_vector_for_factors(v: np.ndarray, N: int, p: int, q: int) -> dict:
    """
    Thoroughly check if a lattice vector encodes factorization info.
    """
    results = {
        "has_p": p in v,
        "has_q": q in v,
        "has_p_or_q": p in v or q in v,
        "gcd_with_N": gcd(abs(int(v.sum())), N),
        "any_component_divides_N": any(x > 1 and x < N and N % x == 0 for x in v),
        "vector_norm": float(np.linalg.norm(v)),
        "vector_sum": int(v.sum()),
        "vector_gcd_all": gcd(gcd(abs(int(v[0])), abs(int(v[1]))) if len(v) > 1 else abs(int(v[0])),
                              gcd(abs(int(v[2])), abs(int(v[3]))) if len(v) > 3 else 1) if len(v) > 3 else 1,
    }
    
    # Check GCDs of components
    for i in range(len(v)):
        g = gcd(abs(int(v[i])), N)
        if g > 1 and g < N:
            results[f"gcd_component_{i}"] = g
    
    # Check GCD of linear combinations
    for i in range(min(5, len(v))):
        for j in range(i + 1, min(5, len(v))):
            g = gcd(abs(int(v[i] + v[j])), N)
            if g > 1 and g < N:
                results[f"gcd_sum_{i}_{j}"] = g
            g = gcd(abs(int(v[i] - v[j])), N)
            if g > 1 and g < N:
                results[f"gcd_diff_{i}_{j}"] = g
    
    return results


def analyze_with_known_factors(N: int, p: int, q: int):
    """
    Deep analysis when factors are known.
    """
    print(f"\n{'='*70}")
    print(f"Analyzing N = {N} = {p} × {q}")
    print(f"{'='*70}")
    
    # Method 1: Corrected Schnorr lattice
    print("\n--- Method 1: Corrected Schnorr Lattice ---")
    B1, primes1 = schnorr_lattice_basis_corrected(N, 20)
    B1_reduced = lll_reduce(B1)
    
    print(f"Original basis shape: {B1.shape}")
    print(f"First 3 rows of reduced basis:")
    for i in range(min(3, len(B1_reduced))):
        print(f"  v[{i}]: norm={np.linalg.norm(B1_reduced[i]):.2f}")
        result = check_vector_for_factors(B1_reduced[i], N, p, q)
        if result["any_component_divides_N"] or result["gcd_with_N"] > 1:
            print(f"       ^ Potential factor found!")
        if any(k.startswith("gcd_") and v > 1 and v < N for k, v in result.items() if isinstance(v, int)):
            print(f"       ^ GCD factor found!")
    
    # Method 2: Improved lattice
    print("\n--- Method 2: Improved Schnorr Lattice ---")
    B2, primes2 = improved_schnorr_lattice(N, 20)
    B2_reduced = lll_reduce(B2)
    
    print(f"First 3 rows of reduced basis:")
    for i in range(min(3, len(B2_reduced))):
        print(f"  v[{i}]: norm={np.linalg.norm(B2_reduced[i]):.2f}")
        result = check_vector_for_factors(B2_reduced[i], N, p, q)
        if result["any_component_divides_N"] or result["gcd_with_N"] > 1:
            print(f"       ^ Potential factor found!")
    
    # Method 3: CVP lattice
    print("\n--- Method 3: CVP Lattice ---")
    B3, target = cvp_lattice(N, 15)
    B3_reduced = lll_reduce(B3)
    
    print(f"First 3 rows of reduced basis:")
    for i in range(min(3, len(B3_reduced))):
        print(f"  v[{i}]: norm={np.linalg.norm(B3_reduced[i]):.2f}, last_component={B3_reduced[i][-1]}")
        result = check_vector_for_factors(B3_reduced[i], N, p, q)
        if result["any_component_divides_N"] or result["gcd_with_N"] > 1:
            print(f"       ^ Potential factor found!")
    
    # Analyze all vectors for GCD relationships
    print("\n--- GCD Analysis Across All Vectors ---")
    for method_name, B_reduced in [
        ("Schnorr corrected", B1_reduced),
        ("Improved", B2_reduced),
        ("CVP", B3_reduced)
    ]:
        print(f"\n{method_name}:")
        found_any = False
        for i, v in enumerate(B_reduced):
            for j in range(len(v)):
                g = gcd(abs(int(v[j])), N)
                if g > 1 and g < N:
                    print(f"  Vector {i}, component {j}: GCD({abs(int(v[j]))}, {N}) = {g}")
                    found_any = True
        if not found_any:
            print("  No GCD factors found in individual components")
        
        # Check sums and differences
        for i in range(min(5, len(B_reduced))):
            for j in range(i + 1, min(5, len(B_reduced))):
                diff = B_reduced[i] - B_reduced[j]
                for k in range(len(diff)):
                    g = gcd(abs(int(diff[k])), N)
                    if g > 1 and g < N:
                        print(f"  v[{i}] - v[{j}], component {k}: GCD = {g}")


def brute_force_lattice_search(N: int, p: int, q: int):
    """
    Try to find any lattice construction that reveals the factors.
    """
    print(f"\n{'='*70}")
    print(f"Brute Force: Searching for factor-revealing lattices")
    print(f"{'='*70}")
    
    # Try various constructions
    for dim in [5, 10, 15, 20]:
        # Construction: encode primes and N modularly
        primes = generate_primes(dim)
        B = np.zeros((dim + 1, dim + 1), dtype=np.int64)
        
        for i in range(dim):
            B[i, i] = 1
            B[i, dim] = primes[i]
        
        B[dim, dim] = N
        
        B_reduced = lll_reduce(B)
        
        # Check all vectors and their combinations
        for i, v in enumerate(B_reduced):
            g = gcd(abs(int(v.sum())), N)
            if g > 1 and g < N:
                print(f"Dim {dim}, vector {i}: GCD(sum, N) = {g}")
            
            for x in v:
                if x > 1 and x < N and N % int(x) == 0:
                    print(f"Dim {dim}, vector {i}: Found factor {x}")
    
    # Try modular constructions
    for dim in [5, 10]:
        primes = generate_primes(dim)
        B = np.zeros((dim + 2, dim + 2), dtype=np.int64)
        
        for i in range(dim):
            B[i, i] = primes[i]
            B[i, dim] = N % primes[i]
        
        B[dim, dim] = 1
        B[dim + 1, dim + 1] = 1
        
        B_reduced = lll_reduce(B)
        
        for i, v in enumerate(B_reduced):
            g = gcd(abs(int(v.sum())), N)
            if g > 1 and g < N:
                print(f"Modular lattice dim {dim}, vector {i}: GCD = {g}")


if __name__ == "__main__":
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
        (1517, 37, 41),
    ]
    
    for N, p, q in test_cases:
        analyze_with_known_factors(N, p, q)
        brute_force_lattice_search(N, p, q)
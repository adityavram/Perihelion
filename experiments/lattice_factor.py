#!/usr/bin/env python3
"""
Lattice Embedding Experiments for Semiprime Factorization
Perihelion Project - Testing lattice-based approaches on small N
"""

import numpy as np
from math import gcd, isqrt, log2, floor
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class FactorizationResult:
    success: bool
    factors: Optional[Tuple[int, int]]
    lattice_dim: int
    method: str
    notes: str


def generate_semiprime(bit_size: int = 10) -> Tuple[int, int, int]:
    """Generate a small semiprime N = p * q for testing."""
    from sympy import nextprime
    p = nextprime(2**(bit_size//2))
    q = nextprime(2**(bit_size//2 + 1))
    N = p * q
    return N, p, q


def schnorr_lattice_basis(N: int, dimension: int) -> Tuple[np.ndarray, int]:
    """
    Construct Schnorr's lattice basis for factoring N.
    
    The lattice encodes relationships between primes and N.
    Short vectors in this lattice may reveal factorization structure.
    """
    B = np.zeros((dimension + 2, dimension + 2), dtype=np.int64)
    
    # First row encodes N
    B[0, 0] = 1
    B[0, -1] = N
    
    # Rows 1 to dimension encode primes
    primes = generate_first_n_primes(dimension)
    for i, p in enumerate(primes):
        B[i + 1, i + 1] = 1
        B[i + 1, -1] = int(log2(p))
    
    # Last row
    B[-1, -1] = 1
    
    return B, dimension + 2


def generate_first_n_primes(n: int) -> List[int]:
    """Generate the first n primes."""
    from sympy import primerange
    return list(primerange(2, n * 20))[:n]


def gram_schmidt_orthogonalization(B: np.ndarray) -> np.ndarray:
    """Compute Gram-Schmidt orthogonalization of lattice basis."""
    n = B.shape[0]
    B_star = np.zeros_like(B, dtype=np.float64)
    
    for i in range(n):
        B_star[i] = B[i].astype(np.float64)
        for j in range(i):
            mu = np.dot(B[i], B_star[j]) / np.dot(B_star[j], B_star[j])
            B_star[i] -= mu * B_star[j]
    
    return B_star


def lll_reduce(B: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """
    LLL lattice basis reduction algorithm.
    
    Returns a reduced basis with short vectors.
    """
    B = B.astype(np.int64).copy()
    n = B.shape[0]
    
    def gram_schmidt(B):
        B_star = np.zeros_like(B, dtype=np.float64)
        mu = np.zeros((n, n), dtype=np.float64)
        
        for i in range(n):
            B_star[i] = B[i].astype(np.float64)
            for j in range(i):
                mu[i, j] = np.dot(B[i], B_star[j]) / np.dot(B_star[j], B_star[j])
                B_star[i] -= mu[i, j] * B_star[j]
        
        return B_star, mu
    
    B_star, mu = gram_schmidt(B)
    
    k = 1
    while k < n:
        # Size reduction
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                B[k] -= round(mu[k, j]) * B[j]
                B_star, mu = gram_schmidt(B)
        
        # Lovász condition
        lhs = delta * np.dot(B_star[k-1], B_star[k-1])
        rhs = np.dot(B_star[k], B_star[k]) + mu[k, k-1]**2 * np.dot(B_star[k-1], B_star[k-1])
        
        if lhs <= rhs:
            k += 1
        else:
            B[[k, k-1]] = B[[k-1, k]]
            B_star, mu = gram_schmidt(B)
            k = max(k - 1, 1)
    
    return B


def extract_factors_from_vector(v: np.ndarray, N: int) -> Optional[Tuple[int, int]]:
    """
    Attempt to extract factors from a lattice vector.
    
    Look for vectors whose components suggest divisibility relationships.
    """
    v = v.astype(np.int64)
    
    # Check if vector directly encodes a factor
    for x in v:
        if x > 1 and x < N and N % x == 0:
            return (x, N // x)
    
    # Check linear combinations
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            g = gcd(abs(v[i]), abs(v[j]))
            if g > 1 and g < N and N % g == 0:
                return (g, N // g)
    
    # Check if vector suggests a small factor
    for x in v:
        if x > 0:
            for d in [2, 3, 5, 7]:
                if x > d and x % d == 0:
                    g = gcd(x, N)
                    if g > 1 and g < N:
                        return (g, N // g)
    
    return None


def factor_via_schnorr_lattice(N: int, max_dim: int = 50) -> FactorizationResult:
    """
    Attempt to factor N using Schnorr's lattice-based approach.
    
    Tests various lattice dimensions and looks for factors in reduced basis vectors.
    """
    for dim in range(5, min(max_dim, int(log2(N)))):
        B, actual_dim = schnorr_lattice_basis(N, dim)
        B_reduced = lll_reduce(B)
        
        for v in B_reduced:
            factors = extract_factors_from_vector(v, N)
            if factors:
                return FactorizationResult(
                    success=True,
                    factors=factors,
                    lattice_dim=actual_dim,
                    method="schnorr_lll",
                    notes=f"Found factors at dimension {actual_dim}"
                )
    
    return FactorizationResult(
        success=False,
        factors=None,
        lattice_dim=max_dim,
        method="schnorr_lll",
        notes="No factors found in tested dimensions"
    )


def alternative_lattice_embedding_v1(N: int) -> np.ndarray:
    """
    Alternative embedding approach 1: Geometric representation.
    
    Embed N in a lattice where coordinates encode modular relationships.
    """
    sqrt_N = isqrt(N)
    dim = max(10, int(log2(N)))
    
    B = np.zeros((dim, dim), dtype=np.int64)
    
    for i in range(dim - 1):
        B[i, i] = 1
        B[i, -1] = (sqrt_N + i) % N
    
    B[-1, -1] = N
    
    return B


def factor_via_alternative_embedding(N: int) -> FactorizationResult:
    """
    Test alternative lattice embeddings for factorization.
    """
    B = alternative_lattice_embedding_v1(N)
    B_reduced = lll_reduce(B)
    
    for v in B_reduced:
        factors = extract_factors_from_vector(v, N)
        if factors:
            return FactorizationResult(
                success=True,
                factors=factors,
                lattice_dim=B.shape[0],
                method="alternative_v1",
                notes="Found factors in alternative embedding"
            )
    
    return FactorizationResult(
        success=False,
        factors=None,
        lattice_dim=B.shape[0],
        method="alternative_v1",
        notes="No factors found"
    )


def trial_factor(N: int, max_trials: int = 1000) -> Optional[Tuple[int, int]]:
    """Naive trial division for verification."""
    for i in range(2, min(isqrt(N) + 1, max_trials)):
        if N % i == 0:
            return (i, N // i)
    return None


def run_experiment(N_values: List[int], methods: List[str] = None) -> dict:
    """
    Run factorization experiments across multiple N values and methods.
    """
    if methods is None:
        methods = ["schnorr_lll", "alternative_v1", "trial"]
    
    results = {}
    
    for N in N_values:
        results[N] = {}
        
        # Verify N is semiprime
        factors = trial_factor(N)
        if factors:
            results[N]["true_factors"] = factors
        else:
            results[N]["true_factors"] = None
            continue
        
        for method in methods:
            if method == "schnorr_lll":
                result = factor_via_schnorr_lattice(N)
            elif method == "alternative_v1":
                result = factor_via_alternative_embedding(N)
            elif method == "trial":
                f = trial_factor(N)
                result = FactorizationResult(
                    success=f is not None,
                    factors=f,
                    lattice_dim=0,
                    method="trial",
                    notes="Naive trial division"
                )
            
            results[N][method] = result
    
    return results


def analyze_lattice_structure(N: int, p: int, q: int) -> dict:
    """
    Analyze the lattice structure when factors are known.
    
    This helps understand what properties the lattice should have.
    """
    analysis = {
        "N": N,
        "factors": (p, q),
        "log2_N": log2(N),
        "factor_ratio": max(p, q) / min(p, q),
    }
    
    # Generate Schnorr lattice
    B, dim = schnorr_lattice_basis(N, 20)
    B_reduced = lll_reduce(B)
    
    # Check if factors appear in reduced basis
    factor_present = False
    for v in B_reduced:
        if p in v or q in v:
            factor_present = True
            break
    
    analysis["factors_in_reduced_lattice"] = factor_present
    analysis["reduced_basis_shape"] = B_reduced.shape
    
    # Compute shortest vector lengths
    lengths = [np.linalg.norm(v) for v in B_reduced]
    analysis["shortest_vector_length"] = min(lengths)
    analysis["vector_length_distribution"] = sorted(lengths)[:5]
    
    return analysis


if __name__ == "__main__":
    # Test with known semiprimes
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
        (1517, 37, 41),
        (4757, 67, 71),
        (9797, 97, 101),
        (11009, 103, 107),
        (62809, 241, 261),
    ]
    
    print("=" * 60)
    print("Perihelion: Lattice Embedding Factorization Experiments")
    print("=" * 60)
    
    for N, p, q in test_cases:
        print(f"\n--- N = {N} = {p} × {q} ---")
        
        # Trial factorization (baseline)
        factors = trial_factor(N)
        print(f"Trial division: {factors}")
        
        # Schnorr lattice
        result = factor_via_schnorr_lattice(N, max_dim=30)
        print(f"Schnorr lattice: {result}")
        
        # Alternative embedding
        result_alt = factor_via_alternative_embedding(N)
        print(f"Alternative embedding: {result_alt}")
        
        # Analyze structure
        analysis = analyze_lattice_structure(N, p, q)
        print(f"Lattice analysis: factors_present={analysis['factors_in_reduced_lattice']}")
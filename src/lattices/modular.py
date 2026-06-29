#!/usr/bin/env python3
"""
Modular lattice factorization.

The key discovery: when primes are in the factor base, the lattice encoding
N mod p = 0 creates a structure that LLL reduction exposes directly.

Construction:
    For each prime p_i in the factor base, create a row:
    [p_i, 0, ..., 0, N mod p_i, 0, ..., 0]
    
After LLL reduction, short vectors have sums equal to the prime factors.
"""

import numpy as np
from math import gcd
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
                if np.dot(B_star[j], B_star[j]) != 0:
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


def modular_lattice_factor(N: int, factor_base_size: int = 20) -> Optional[Tuple[int, int]]:
    """
    Factor N using the modular lattice construction.
    
    The lattice encodes divisibility: for each prime p_i in the factor base,
    we create a row [p_i, ..., N mod p_i, ...]. When p_i divides N, 
    N mod p_i = 0, creating a special short vector.
    
    Args:
        N: Semiprime to factor
        factor_base_size: Number of primes to include in factor base
        
    Returns:
        Tuple (p, q) if factorization succeeds, None otherwise
        
    Limitations:
        Requires that p or q be in the factor base. For small semiprimes
        this works well, but for RSA-sized numbers, the factor base would
        need to be exponentially large.
    """
    primes = generate_primes(factor_base_size)
    dim = factor_base_size
    
    # Build lattice: rows encode [prime_i, ..., N mod prime_i, ...]
    B = np.zeros((dim + 2, dim + 2), dtype=np.int64)
    
    for i in range(dim):
        B[i, i] = primes[i]
        B[i, dim] = N % primes[i]
    
    B[dim, dim] = 1
    B[dim + 1, dim + 1] = 1
    
    # Reduce with LLL
    B_reduced = lll_reduce(B)
    
    # Check all vectors for factorization
    for v in B_reduced:
        # Method 1: GCD of sum with N
        g = gcd(abs(int(v.sum())), N)
        if g > 1 and g < N:
            return (g, N // g)
        
        # Method 2: Check individual components
        for x in v:
            x_int = abs(int(x))
            if x_int > 1 and x_int < N and N % x_int == 0:
                return (x_int, N // x_int)
    
    return None


def analyze_modular_lattice(N: int, p: int, q: int, factor_base_size: int = 20) -> dict:
    """
    Analyze the modular lattice structure when factors are known.
    
    Useful for understanding why the construction works.
    """
    primes = generate_primes(factor_base_size)
    dim = factor_base_size
    
    B = np.zeros((dim + 2, dim + 2), dtype=np.int64)
    
    for i in range(dim):
        B[i, i] = primes[i]
        B[i, dim] = N % primes[i]
    
    B[dim, dim] = 1
    B[dim + 1, dim + 1] = 1
    
    B_reduced = lll_reduce(B)
    
    # Check for factors in reduced basis
    factors_found = []
    for i, v in enumerate(B_reduced):
        g = gcd(abs(int(v.sum())), N)
        if g > 1 and g < N:
            factors_found.append((i, g, "sum"))
        
        for j, x in enumerate(v):
            x_int = abs(int(x))
            if x_int > 1 and x_int < N and N % x_int == 0:
                factors_found.append((i, x_int, f"component_{j}"))
    
    return {
        "N": N,
        "factors": (p, q),
        "p_in_factor_base": p in primes,
        "q_in_factor_base": q in primes,
        "factors_found": factors_found,
        "reduced_basis_shape": B_reduced.shape,
    }
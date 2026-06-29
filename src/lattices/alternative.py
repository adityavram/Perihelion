#!/usr/bin/env python3
"""
Alternative lattice embeddings for integer factorization.

These approaches attempt to factor N without explicit factor base enumeration.
Most are classical methods in disguise, but tested here for comparison.
"""

import numpy as np
from math import gcd, isqrt
from typing import Tuple, Optional


def continued_fraction_factor(N: int, max_iterations: int = 100) -> Optional[Tuple[int, int]]:
    """
    Factor N using continued fractions (CFRAC method).
    
    This is a well-known classical algorithm (subexponential complexity).
    The convergents h_i/k_i of sqrt(N) satisfy h_i² ≡ N·k_i² (mod p) for some
    prime factors p. Computing GCD(h_i² - N·k_i², N) may reveal factors.
    
    Args:
        N: Semiprime to factor
        max_iterations: Maximum convergents to compute
        
    Returns:
        Tuple (p, q) if factorization succeeds, None otherwise
        
    Note:
        This is NOT a new lattice approach - it's the classical CFRAC algorithm
        that happens to use continued fractions.
    """
    sqrtN = isqrt(N)
    
    # Continued fraction state
    m, d, a = 0, 1, sqrtN
    
    # Convergents
    h_prev, h = 1, a
    k_prev, k = 0, 1
    
    for i in range(max_iterations):
        # Next term
        m = d * a - m
        if d == 0:
            break
        d_new = (N - m * m) // d
        if d_new == 0:
            break
        d = d_new
        a = (sqrtN + m) // d
        
        # Next convergent
        h_new = a * h + h_prev
        k_new = a * k + k_prev
        
        h_prev, h = h, h_new
        k_prev, k = k, k_new
        
        # Check for factorization
        val = h * h - N * k * k
        if val != 0:
            g = gcd(abs(val), N)
            if g > 1 and g < N:
                return (g, N // g)
    
    return None


def quadratic_form_factor(N: int, search_range: int = None) -> Optional[Tuple[int, int]]:
    """
    Factor N using quadratic form lattice.
    
    For N = p·q, we have p + q = S and p - q = D.
    The relation x² - y² = N where x ≈ √N, y is small encodes this.
    
    Args:
        N: Semiprime to factor
        search_range: Range for y in x² - y² = N
        
    Returns:
        Tuple (p, q) if factorization succeeds, None otherwise
        
    Note:
        For balanced semiprimes, this is essentially trial division near √N.
    """
    if search_range is None:
        search_range = int(N ** 0.25) + 10
    
    sqrtN = isqrt(N)
    
    for y in range(1, min(search_range, sqrtN)):
        # x² = N + y²
        x_sq = N + y * y
        x = isqrt(x_sq)
        
        if x * x == x_sq:
            # Found: x² - y² = N
            # So (x-y)(x+y) = N
            p = x - y
            q = x + y
            if p > 1 and q > 1 and p * q == N:
                return (p, q)
    
    return None


def simultaneous_approx_factor(N: int, dimension: int = 20) -> Optional[Tuple[int, int]]:
    """
    Factor N using simultaneous Diophantine approximation.
    
    Builds a lattice encoding approximations a ≈ √N where a·b ≈ N.
    
    Args:
        N: Semiprime to factor
        dimension: Lattice dimension
        
    Returns:
        Tuple (p, q) if factorization succeeds, None otherwise
        
    Note:
        This is essentially trial division around √N encoded in a lattice.
    """
    sqrtN = isqrt(N)
    
    # Try values near sqrt(N)
    for offset in range(-dimension // 2, dimension // 2 + 1):
        a = sqrtN + offset
        if a <= 0:
            continue
        
        b = N // a
        if a * b == N:
            return (a, b)
        
        # Also check GCD
        g = gcd(a, N)
        if g > 1 and g < N:
            return (g, N // g)
    
    return None


def trial_division_factor(N: int, max_trials: int = 10000) -> Optional[Tuple[int, int]]:
    """
    Naive trial division (baseline for comparison).
    
    Args:
        N: Semiprime to factor
        max_trials: Maximum number of divisions to try
        
    Returns:
        Tuple (p, q) if factorization succeeds, None otherwise
    """
    for i in range(2, min(isqrt(N) + 1, max_trials)):
        if N % i == 0:
            return (i, N // i)
    return None
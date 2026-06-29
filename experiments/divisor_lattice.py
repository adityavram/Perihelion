#!/usr/bin/env python3
"""
Direction 4: Divisor Lattice Structure

Core Question: Can we build/infer the divisor structure of N without knowing factors?

For N = pq, the divisor lattice is:
    N
   / \
  p   q
   \ /
    1

This is a diamond shape with 4 elements.

Key number-theoretic functions that encode divisor structure:
- d(N) = number of divisors = 4 for N = pq (distinct primes)
- σ(N) = sum of divisors = 1 + p + q + N
- φ(N) = Euler's totient = N - p - q + 1 = (p-1)(q-1)

But these all require factorization to compute exactly.

Question: Can we use approximations/bounds on these functions?
"""

import numpy as np
from math import gcd, isqrt
from typing import Tuple, List, Optional


def divisor_lattice_structure(N: int, p: int = None, q: int = None):
    """
    Analyze the divisor lattice structure of N.
    
    If p and q are known, we can compute everything exactly.
    If not, we use approximations.
    """
    print(f"\n{'='*70}")
    print(f"DIVISOR LATTICE STRUCTURE: N = {N}")
    print(f"{'='*70}")
    
    # Approximations (for balanced semiprimes)
    sqrtN = isqrt(N)
    
    print(f"\nApproximations (assuming balanced semiprime):")
    print(f"  √N ≈ {sqrtN}")
    print(f"  σ(N) ≈ N + 2√N = {N + 2*sqrtN}")
    print(f"  φ(N) ≈ N - 2√N = {N - 2*sqrtN}")
    
    if p and q:
        print(f"\nExact values (factors known):")
        print(f"  p = {p}, q = {q}")
        print(f"  σ(N) = 1 + p + q + N = {1 + p + q + N}")
        print(f"  φ(N) = (p-1)(q-1) = {(p-1)*(q-1)}")
        
        # Approximation errors
        approx_sigma = N + 2*sqrtN
        exact_sigma = 1 + p + q + N
        print(f"  σ approximation error: {abs(approx_sigma - exact_sigma)}")
        
        approx_phi = N - 2*sqrtN
        exact_phi = (p-1)*(q-1)
        print(f"  φ approximation error: {abs(approx_phi - exact_phi)}")
    
    print(f"\nDivisor count:")
    print(f"  d(N) = 4 (for distinct prime semiprime)")
    print(f"  This means: exactly 4 divisors: 1, p, q, N")


def build_divisor_lattice(N: int, p: int = None, q: int = None) -> dict:
    """
    Build the divisor lattice structure.
    
    For N = pq:
    - Elements: {1, p, q, N}
    - Cover relations: 1 < p, 1 < q, p < N, q < N
    
    Returns the Hasse diagram (cover relations).
    """
    if p is None or q is None:
        print("Cannot build exact divisor lattice without factors")
        return None
    
    lattice = {
        'elements': [1, p, q, N],
        'covers': [
            (1, p),
            (1, q),
            (p, N),
            (q, N),
        ],
        'labels': {
            1: '1',
            p: 'p',
            q: 'q',
            N: 'N',
        }
    }
    
    return lattice


def encode_divisor_lattice_in_lattice(N: int, dimension: int = 20):
    """
    Attempt to encode the divisor lattice structure in a higher-dimensional lattice.
    
    Idea: The divisor lattice has 4 elements with specific relations.
    Can we encode these relations geometrically?
    
    Approach: Encode partial order relations as lattice constraints.
    
    For the divisor lattice:
    - 1 divides everything
    - p and q are incomparable
    - N is divisible by everything
    
    The divisibility relations are:
    - 1 | p, 1 | q, p | N, q | N
    - Equivalently: p mod 1 = 0, q mod 1 = 0, N mod p = 0, N mod q = 0
    
    But this is circular: we need p and q to build the encoding!
    """
    print(f"\n{'='*70}")
    print(f"ENCODING DIVISOR LATTICE IN LATTICE")
    print(f"{'='*70}")
    
    sqrtN = isqrt(N)
    
    print(f"\nProblem: To encode the divisor lattice, we need the divisors.")
    print(f"This is circular - we're trying to FIND the divisors.")
    
    print(f"\nWhat we can compute WITHOUT factors:")
    print(f"  - The number of divisors is 4 (if N is product of 2 distinct primes)")
    print(f"  - The divisor sum is σ(N) ≈ N + 2√N (approximation)")
    print(f"  - The divisor product is N (by definition)")
    
    print(f"\nWhat we CANNOT compute:")
    print(f"  - The actual divisors p and q")
    print(f"  - The exact σ(N)")
    print(f"  - The exact φ(N)")
    
    print(f"\nPotential approaches:")
    print(f"  1. Use σ(N) or φ(N) approximations as lattice constraints")
    print(f"  2. Encode the HASSE DIAGRAM structure")
    print(f"  3. Use Möbius function properties")


def mobius_approach(N: int):
    """
    The Möbius function μ(N) encodes divisibility structure.
    
    μ(N) = 1 if N is square-free with even number of prime factors
    μ(N) = -1 if N is square-free with odd number of prime factors
    μ(N) = 0 if N has a squared prime factor
    
    For N = pq (distinct primes): μ(N) = 1
    
    Möbius inversion:
    Σ_{d|N} μ(d) = 0 for N > 1
    
    This gives a constraint on the divisor structure!
    """
    print(f"\n{'='*70}")
    print(f"MÖBIUS FUNCTION APPROACH")
    print(f"{'='*70}")
    
    print(f"\nFor N = {N} (assuming semiprime of distinct primes):")
    print(f"  μ(N) = 1 (square-free, even number of prime factors)")
    print(f"  Σ_{{d|N}} μ(d) = μ(1) + μ(p) + μ(q) + μ(N)")
    print(f"                   = 1 + (-1) + (-1) + 1")
    print(f"                   = 0")
    
    print(f"\nThis is always true for N > 1, regardless of N's value.")
    print(f"It doesn't help us find p and q.")
    
    print(f"\nHowever, the Möbius function relates to:")
    print(f"  - Divisor sums: Σ_{{d|N}} μ(d) f(N/d)")
    print(f"  - Euler's totient: φ(N) = N Σ_{{d|N}} μ(d)/d")
    print(f"  - Prime factorization: log ζ(s) = Σ μ(n)/n^s")


def divisor_sum_approximation_lattice(N: int, dimension: int = 20):
    """
    Use the divisor sum approximation σ(N) ≈ N + 2√N.
    
    Idea: Build a lattice where vectors encode potential divisors,
    and use the constraint that sum of divisors ≈ N + 2√N.
    
    The constraint: p + q ≈ 2√N (for balanced semiprime)
    
    This is what we tried with "geometric constraints" approach.
    """
    print(f"\n{'='*70}")
    print(f"DIVISOR SUM APPROXIMATION LATTICE")
    print(f"{'='*70}")
    
    sqrtN = isqrt(N)
    
    print(f"\nFor balanced semiprime N = pq where p ≈ q ≈ √N:")
    print(f"  p + q ≈ 2√N = {2 * sqrtN}")
    print(f"  σ(N) = 1 + p + q + N ≈ 1 + N + 2√N")
    
    print(f"\nLattice encoding of constraint p + q ≈ 2√N:")
    
    # Build a lattice encoding p + q = S for various S
    # We want to find S = p + q
    
    print(f"  Approach: Search for S near 2√N")
    print(f"  For each candidate S, check if S² - 4N is a perfect square")
    print(f"  If yes, p = (S + √(S² - 4N)) / 2")
    
    # This is Fermat's method - already known
    print(f"\n  This is FERMAT'S METHOD (1670s)")
    print(f"  We've already explored this - it's classical.")
    
    # Try building a lattice anyway
    B = np.zeros((dimension + 1, dimension + 1), dtype=np.int64)
    
    for i in range(dimension):
        # Encode candidate sums S
        S = 2 * sqrtN - dimension // 2 + i
        D_squared = S * S - 4 * N  # Discriminant
        
        B[i, i] = 1
        B[i, dimension] = S
        
        if D_squared > 0:
            sqrt_D = isqrt(D_squared)
            if sqrt_D * sqrt_D == D_squared:
                print(f"  FOUND: S = {S}, D = {sqrt_D}")
                p = (S + sqrt_D) // 2
                q = (S - sqrt_D) // 2
                if p > 0 and q > 0 and p * q == N:
                    print(f"  Factors: {p} × {q}")
                    return p, q
    
    B[dimension, dimension] = 1
    
    print(f"\n  Result: No perfect square discriminant found")
    print(f"  This approach is equivalent to Fermat's method")
    print(f"  It works when p ≈ q but not for unbalanced semiprimes")
    
    return None


def hasse_diagram_encoding(N: int, p: int = None, q: int = None):
    """
    Encode the Hasse diagram of the divisor lattice.
    
    The Hasse diagram is:
        N (top)
       / \
      p   q
       \ /
        1 (bottom)
    
    This has 4 nodes and 4 edges.
    
    Can we encode this structure WITHOUT knowing p and q?
    """
    print(f"\n{'='*70}")
    print(f"HASSE DIAGRAM ENCODING")
    print(f"{'='*70}")
    
    print(f"\nThe Hasse diagram is a 'diamond' graph:")
    print(f"  - 4 nodes: 1, p, q, N")
    print(f"  - 4 edges: (1,p), (1,q), (p,N), (q,N)")
    
    print(f"\nGraph-theoretic properties:")
    print(f"  - Diameter: 2 (max distance between nodes)")
    print(f"  - Height: 2 (longest chain)")
    print(f"  - Width: 2 (max antichain size)")
    print(f"  - Rank: each level has specific elements")
    
    print(f"\nCan we infer this structure from N?")
    print(f"  - Number of divisors: We know it's 4 (for semiprime)")
    print(f"  - Diameter, height, width: Known from structure")
    print(f"  - But: We don't know the actual NODES (p and q)")
    
    print(f"\nThe structure IS the factorization:")
    print(f"  - Knowing the Hasse diagram = knowing p and q")
    print(f"  - So encoding the Hasse diagram requires the answer")
    
    print(f"\nInsight: The structure is UNIVERSAL for all semiprimes")
    print(f"  - Only the LABELS on nodes differ")
    print(f"  - Finding p and q = finding the correct labels")
    
    print(f"\nThis is exactly what the modular lattice does:")
    print(f"  - It searches for which labels make sense")
    print(f"  - 'p' is valid if N mod p = 0")
    print(f"  - The divisibility test is the label validation")


def experiment_summary():
    """
    Summarize experiments with divisor lattice approach.
    """
    print(f"\n{'='*70}")
    print(f"DIVISOR LATTICE APPROACH: SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nKEY INSIGHT:")
    print(f"  The divisor lattice structure is UNIVERSAL for all semiprimes.")
    print(f"  Only the NODE LABELS (p and q) differ.")
    print(f"  Finding p and q = finding the correct labels.")
    
    print(f"\nWHAT WE TRIED:")
    print(f"  1. Approximations (σ(N), φ(N)): Give bounds, not factors")
    print(f"  2. Möbius function: Universal property, doesn't distinguish")
    print(f"  3. Divisor sum constraints: Reduces to Fermat's method")
    print(f"  4. Hasse diagram: Structure is known, labels are the problem")
    
    print(f"\nCONCLUSION:")
    print(f"  Direction 4 (Divisor Lattice) is a DEAD END.")
    print(f"  The structure doesn't help us find factors.")
    print(f"  The structure IS the factorization.")
    
    print(f"\nNEXT DIRECTION:")
    print(f"  Direction 2: Quadratic Residue Patterns")
    print(f"  This encodes information ABOUT factors, not the structure")


if __name__ == "__main__":
    # Test with known semiprime
    N = 143
    p, q = 11, 13
    
    divisor_lattice_structure(N, p, q)
    build_divisor_lattice(N, p, q)
    encode_divisor_lattice_in_lattice(N)
    mobius_approach(N)
    divisor_sum_approximation_lattice(N)
    hasse_diagram_encoding(N, p, q)
    
    print("\n" + "="*70)
    
    # Another test
    N2 = 391
    p2, q2 = 17, 23
    
    print("\n" + "="*70)
    divisor_lattice_structure(N2, p2, q2)
    
    # Summary
    experiment_summary()
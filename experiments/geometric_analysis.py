#!/usr/bin/env python3
"""
Geometric analysis of the modular lattice.

Goal: Understand WHY the lattice construction reveals factors.
What is the geometric structure that makes this work?
"""

import numpy as np
from math import gcd, isqrt
import sys
sys.path.insert(0, '/Users/adiram/Perihelion')


def generate_primes(n: int):
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = all(candidate % p != 0 for p in primes if p * p <= candidate)
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def lll_reduce(B: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """LLL reduction with intermediate tracking."""
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
    swaps = []
    
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
            swaps.append((k-1, k))
            B_star, mu = gram_schmidt(B)
            k = max(k - 1, 1)
    
    return B, swaps


def analyze_lattice_geometry(N: int, factor_base_size: int = 20):
    """
    Deep geometric analysis of the modular lattice.
    """
    primes = generate_primes(factor_base_size)
    dim = factor_base_size
    
    print(f"\n{'='*70}")
    print(f"GEOMETRIC ANALYSIS: N = {N}")
    print(f"{'='*70}")
    
    # Find actual factors for comparison
    actual_factors = []
    for p in primes:
        if N % p == 0:
            actual_factors.append(p)
    
    print(f"\nFactors in factor base: {actual_factors}")
    
    # Build lattice
    B = np.zeros((dim + 2, dim + 2), dtype=np.int64)
    
    print(f"\n1. BASIS CONSTRUCTION")
    print(f"   Dimension: {dim + 2}")
    print(f"   Structure: Rows encode [prime_i, ..., N mod prime_i, ...]")
    
    for i in range(dim):
        B[i, i] = primes[i]
        B[i, dim] = N % primes[i]
    
    B[dim, dim] = 1
    B[dim + 1, dim + 1] = 1
    
    print(f"\n2. ORIGINAL BASIS PROPERTIES")
    
    # Analyze original basis vectors
    norms = [np.linalg.norm(B[i]) for i in range(dim + 2)]
    print(f"   Vector norms (first 10): {[f'{n:.2f}' for n in norms[:10]]}")
    
    # Check for "special" rows (where N mod p_i = 0)
    special_rows = []
    for i in range(dim):
        if B[i, dim] == 0:  # N mod prime_i = 0
            special_rows.append(i)
            print(f"   Row {i}: SPECIAL - prime={primes[i]}, residue=0 (DIVISIBLE)")
    
    # Sparsity analysis
    sparsities = []
    for i in range(dim):
        non_zero = np.count_nonzero(B[i])
        sparsities.append(non_zero)
    
    print(f"\n   Sparsity (non-zero entries per row):")
    print(f"   - Mean: {np.mean(sparsities):.2f}")
    print(f"   - Special rows: {[s for i, s in enumerate(sparsities) if i in special_rows]}")
    
    # Reduce
    B_reduced, swaps = lll_reduce(B)
    
    print(f"\n3. LLL REDUCTION")
    print(f"   Number of swaps: {len(swaps)}")
    
    # Analyze reduced basis
    reduced_norms = [np.linalg.norm(B_reduced[i]) for i in range(dim + 2)]
    print(f"\n   Reduced vector norms (first 10): {[f'{n:.2f}' for n in reduced_norms[:10]]}")
    
    # Find short vectors
    short_threshold = np.mean(reduced_norms) / 2
    short_vectors = [(i, B_reduced[i], reduced_norms[i]) 
                     for i in range(dim + 2) 
                     if reduced_norms[i] < short_threshold]
    
    print(f"\n   Short vectors (norm < {short_threshold:.2f}):")
    for i, v, norm in short_vectors[:5]:
        # Check if this vector encodes a factor
        factor = None
        for x in v:
            x_int = abs(int(x))
            if x_int > 1 and x_int < N and N % x_int == 0:
                factor = x_int
                break
        
        if factor:
            print(f"   - v[{i}]: norm={norm:.2f}, REVEALS FACTOR {factor}")
        else:
            print(f"   - v[{i}]: norm={norm:.2f}")
    
    print(f"\n4. GEOMETRIC STRUCTURE")
    
    # Fundamental domain analysis
    det = np.linalg.det(B[:dim, :dim].astype(np.float64))
    print(f"   Determinant of prime sub-lattice: {det:.2e}")
    
    # Orthogonality analysis
    B_float = B_reduced.astype(np.float64)
    angles = []
    for i in range(min(5, dim)):
        for j in range(i + 1, min(5, dim)):
            cos_angle = np.dot(B_float[i], B_float[j]) / (
                np.linalg.norm(B_float[i]) * np.linalg.norm(B_float[j])
            )
            angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
            angles.append(angle_deg)
    
    print(f"   Angles between reduced vectors (degrees):")
    print(f"   - Mean: {np.mean(angles):.1f}°")
    print(f"   - Min: {np.min(angles):.1f}°")
    print(f"   - Max: {np.max(angles):.1f}°")
    
    # Gram matrix analysis
    gram = B_reduced[:10, :10] @ B_reduced[:10, :10].T
    print(f"\n   Gram matrix (first 5x5) properties:")
    print(f"   - Diagonal (norms²): {np.diag(gram[:5]).astype(int)}")
    
    return B, B_reduced, special_rows


def analyze_why_it_works():
    """
    Geometric intuition for why the modular lattice works.
    """
    print(f"\n{'='*70}")
    print("WHY THE MODULAR LATTICE WORKS: GEOMETRIC INTUITION")
    print(f"{'='*70}")
    
    print("""
    OBSERVATION 1: Sparsity = Shortness
    
    In a lattice, a vector with many zero coordinates is geometrically "short".
    
    Standard row: [prime_i, 0, 0, ..., N mod prime_i, 0, ..., 0]
    - Non-zero entries: 2 (the prime and the residue)
    - Norm: √(prime_i² + (N mod prime_i)²)
    
    Divisor row: [prime_i, 0, 0, ..., 0, 0, ..., 0]  (when prime_i | N)
    - Non-zero entries: 1 (just the prime)
    - Norm: prime_i
    
    KEY INSIGHT: The divisor row is SPARSER and therefore SHORTER.
    
    LLL reduction finds short vectors → finds divisor rows!
    
    ---
    
    OBSERVATION 2: The "Residue Dimension"
    
    The last column (dimension 'dim') encodes "divisibility test result".
    
    When prime_i | N:
    - The entry at [i, dim] becomes 0
    - This "cleans up" the vector
    
    Geometrically: The factor vectors lie in a LOWER-DIMENSIONAL SUBSPACE!
    
    The subspace: {v ∈ L : v[dim] = 0}
    
    LLL reduction tends to find vectors that are close to coordinate axes,
    and vectors in this subspace have one fewer "dimension" to span.
    
    ---
    
    OBSERVATION 3: Orthogonality
    
    Two divisor rows [p, 0, ..., 0] and [q, 0, ..., 0] are ORTHOGONAL.
    
    This creates a nice geometric structure:
    - Factor vectors form an orthogonal basis for a subspace
    - LLL "likes" orthogonal bases
    - The reduction naturally exposes them
    
    ---
    
    OBSERVATION 4: The "Hidden" Structure
    
    In the original basis, factor rows look like:
    [p, 0, 0, ..., 0, 0, ..., 0]
    
    But they're "hidden" among rows like:
    [q, 0, 0, ..., N mod q, 0, ..., 0]
    
    LLL reduction "sorts" the basis by shortness.
    Factor rows (being sparser) rise to the top.
    
    ---
    
    THE GEOMETRIC PICTURE:
    
    Original lattice L:
    - Lives in Z^(k+2)
    - Most vectors have 2 non-zero coordinates
    - Some vectors have 1 non-zero coordinate (the factors)
    
    Reduced lattice:
    - Same lattice, different basis
    - Short vectors first
    - Factor vectors appear early because they're sparse
    
    ---
    
    THE FUNDAMENTAL QUESTION:
    
    Can we construct a lattice where "factor-ness" is encoded differently?
    
    Instead of: sparse vectors = factors
    Try: vectors with property P = factors
    
    Where P is a geometric/algebraic property that doesn't require enumeration.
    """)


def visualize_lattice_structure(N: int, factor_base_size: int = 15):
    """
    Visualize the lattice structure for intuition.
    """
    primes = generate_primes(factor_base_size)
    dim = factor_base_size
    
    # Build lattice
    B = np.zeros((dim + 2, dim + 2), dtype=np.int64)
    for i in range(dim):
        B[i, i] = primes[i]
        B[i, dim] = N % primes[i]
    B[dim, dim] = 1
    B[dim + 1, dim + 1] = 1
    
    # Reduce
    B_reduced, _ = lll_reduce(B)
    
    print(f"\n{'='*70}")
    print(f"VISUALIZATION: N = {N}")
    print(f"{'='*70}\n")
    
    print("ORIGINAL BASIS (non-zero entries only):")
    for i in range(min(10, dim)):
        row = B[i]
        nz = [(j, int(row[j])) for j in range(len(row)) if row[j] != 0]
        prime = primes[i]
        residue = N % prime
        divisible = "✓ DIVISIBLE" if residue == 0 else ""
        print(f"  Row {i:2d} (p={prime:3d}): {nz}  {divisible}")
    
    print(f"\nREDUCED BASIS (first 10 vectors, non-zero entries):")
    for i in range(min(10, dim + 2)):
        row = B_reduced[i]
        nz = [(j, int(row[j])) for j in range(len(row)) if row[j] != 0]
        norm = np.linalg.norm(row)
        
        # Check for factor
        factor = None
        for x in row:
            x_int = abs(int(x))
            if x_int > 1 and x_int < N and N % x_int == 0:
                factor = x_int
                break
        
        factor_str = f" → FACTOR {factor}" if factor else ""
        print(f"  v[{i}]: |v|={norm:6.2f}  {nz[:4]}{'...' if len(nz) > 4 else ''}{factor_str}")


if __name__ == "__main__":
    # Test cases
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
    ]
    
    print("=" * 70)
    print("GEOMETRIC ANALYSIS OF MODULAR LATTICE")
    print("=" * 70)
    
    for N, p, q in test_cases:
        analyze_lattice_geometry(N, factor_base_size=20)
        visualize_lattice_structure(N, factor_base_size=15)
    
    # Intuition
    analyze_why_it_works()
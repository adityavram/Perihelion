#!/usr/bin/env python3
"""
Direction 5.3: Multi-Dimensional Encodings

Core Idea: Previous approaches encode information in low-dimensional lattices.
What if we use higher dimensions to encode multiple relationships simultaneously?

The modular lattice is essentially 2D per prime:
- One dimension for the prime
- One dimension for the residue N mod prime

What happens with:
- 3D encodings: (prime, residue, something_else)
- Tensor structures: encode multiple primes per vector
- Multiplicative relationships: encode p × q = N constraint
"""

import numpy as np
from math import gcd, isqrt, log2
from typing import Tuple, Optional, List


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


def generate_primes(n: int) -> List[int]:
    """Generate first n primes."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = all(candidate % p != 0 for p in primes if p * p <= candidate)
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


# =============================================================================
# APPROACH 1: 3D Encodings - Add a Third Dimension
# =============================================================================

def three_dimensional_lattice(N: int, dimension: int = 20):
    """
    3D encoding: Instead of [prime, residue], use [prime, residue, constraint].
    
    Possible third dimensions:
    - log(prime): encodes size information
    - prime mod something: encodes congruence
    - prime^2: encodes quadratic structure
    - Some function of N and prime
    
    Idea: Multiple constraints might narrow the search space.
    """
    print(f"\n{'='*70}")
    print(f"3D LATTICE ENCODING")
    print(f"{'='*70}")
    
    primes = generate_primes(dimension)
    
    print(f"\nApproach: Encode each prime as [prime, N mod prime, f(prime, N)]")
    print(f"Third dimension options:")
    print(f"  1. log(prime): size information")
    print(f"  2. prime^2: quadratic structure")
    print(f"  3. prime mod (something): congruence class")
    print(f"  4. N // prime: co-factor approximation")
    
    for constraint_type in ['log', 'quadratic', 'mod', 'cofactor']:
        print(f"\n  Testing constraint type: {constraint_type}")
        
        B = np.zeros((dimension + 2, dimension + 3), dtype=np.int64)
        
        for i, p in enumerate(primes):
            B[i, i] = p
            B[i, dimension] = N % p
            
            # Third dimension
            if constraint_type == 'log':
                B[i, dimension + 1] = int(log2(p) * 100)  # Scaled log
            elif constraint_type == 'quadratic':
                B[i, dimension + 1] = p * p
            elif constraint_type == 'mod':
                B[i, dimension + 1] = p % 10  # Congruence class mod 10
            elif constraint_type == 'cofactor':
                B[i, dimension + 1] = N // p  # Integer division
        
        B[dimension, dimension] = 1
        B[dimension + 1, dimension + 1] = 1
        B[dimension, dimension + 1] = 0
        B[dimension + 1, dimension + 2] = 1
        
        # LLL reduce
        B_reduced = lll_reduce(B)
        
        # Check for factors
        found = False
        for v in B_reduced:
            g = gcd(abs(int(v.sum())), N)
            if g > 1 and g < N:
                print(f"    Found factor: {g}")
                found = True
                break
        
        if not found:
            print(f"    No factors found")
    
    return None


# =============================================================================
# APPROACH 2: Tensor Structure - Encode Multiple Primes Per Vector
# =============================================================================

def tensor_lattice(N: int, group_size: int = 3, num_groups: int = 10):
    """
    Tensor structure: Instead of one prime per row, encode groups of primes.
    
    Each row: [p_i, p_j, p_k, N mod p_i, N mod p_j, N mod p_k, ...]
    
    Idea: Multiple primes per row might create constraints that reveal structure.
    
    For N = pq where p and q are in the factor base:
    - Row with (p, q, r) has residues (0, 0, N mod r)
    - This row is "special" - two zeros instead of one
    """
    print(f"\n{'='*70}")
    print(f"TENSOR LATTICE: GROUPED PRIMES")
    print(f"{'='*70}")
    
    primes = generate_primes(group_size * num_groups)
    
    print(f"\nGrouping {len(primes)} primes into groups of {group_size}")
    print(f"Each row encodes {group_size} primes simultaneously")
    
    # Build lattice with grouped primes
    B = np.zeros((num_groups + 2, group_size * 2 + 2), dtype=np.int64)
    
    for g in range(num_groups):
        # Get group of primes
        group = primes[g * group_size : (g + 1) * group_size]
        
        # Encode: [p_1, p_2, p_3, N mod p_1, N mod p_2, N mod p_3]
        for i, p in enumerate(group):
            B[g, i] = p
            B[g, group_size + i] = N % p
        
        # Last two dimensions for padding
        B[g, group_size * 2] = sum(N % p for p in group)  # Sum of residues
        B[g, group_size * 2 + 1] = sum(p for p in group)  # Sum of primes
    
    B[num_groups, group_size * 2] = 1
    B[num_groups + 1, group_size * 2 + 1] = 1
    
    print(f"\nLattice shape: {B.shape}")
    print(f"Looking for groups where multiple residues are 0")
    
    # LLL reduce
    B_reduced = lll_reduce(B)
    
    # Check for factors in reduced vectors
    for v in B_reduced:
        # Check sum
        g = gcd(abs(int(v.sum())), N)
        if g > 1 and g < N:
            print(f"  Found via sum: {g}")
            return g, N // g
        
        # Check individual components
        for x in v[:group_size]:
            x_int = abs(int(x))
            if x_int > 1 and x_int < N and N % x_int == 0:
                print(f"  Found via component: {x_int}")
                return x_int, N // x_int
    
    print(f"  No factors found")
    return None


# =============================================================================
# APPROACH 3: Multiplicative Constraint Lattice
# =============================================================================

def multiplicative_constraint_lattice(N: int, dimension: int = 20):
    """
    Encode the constraint p × q = N in a lattice.
    
    For semiprime N, we have:
    - p × q = N
    - p + q ≈ 2√N (for balanced)
    - p² + q² = (p + q)² - 2N
    
    Idea: Build a lattice encoding these algebraic relationships.
    
    Construction:
    - Row 1: encode p
    - Row 2: encode q
    - Row 3: encode p × q
    - Row 4: encode p + q
    - etc.
    
    We don't know p and q, but we can use variables and search.
    """
    print(f"\n{'='*70}")
    print(f"MULTIPLICATIVE CONSTRAINT LATTICE")
    print(f"{'='*70}")
    
    sqrtN = isqrt(N)
    
    print(f"\nN = {N}, √N ≈ {sqrtN}")
    print(f"For balanced semiprime: p ≈ q ≈ {sqrtN}")
    print(f"Constraints:")
    print(f"  p × q = N")
    print(f"  p + q ≈ {2 * sqrtN}")
    
    # We'll try different values of S = p + q
    # and check if S² - 4N is a perfect square
    
    print(f"\nApproach: Search for S such that S² - 4N is a perfect square")
    
    for S_offset in range(-dimension, dimension + 1):
        S = 2 * sqrtN + S_offset
        discriminant = S * S - 4 * N
        
        if discriminant >= 0:
            sqrt_disc = isqrt(discriminant)
            if sqrt_disc * sqrt_disc == discriminant:
                # Found valid discriminant
                p = (S + sqrt_disc) // 2
                q = (S - sqrt_disc) // 2
                
                if p > 0 and q > 0 and p * q == N:
                    print(f"  Found: S = {S}, discriminant = {discriminant}")
                    print(f"  p = {p}, q = {q}")
                    return p, q
    
    # Lattice approach: encode the search
    print(f"\nEncoding search in lattice:")
    
    B = np.zeros((dimension, dimension + 2), dtype=np.int64)
    
    for i in range(dimension):
        S_candidate = 2 * sqrtN - dimension + 2 * i
        D = S_candidate * S_candidate - 4 * N
        
        B[i, i] = S_candidate
        B[i, dimension] = D
        B[i, dimension + 1] = N
    
    B_reduced = lll_reduce(B)
    
    # Look for vectors with small discriminant
    for v in B_reduced:
        D = abs(int(v[dimension]))
        sqrt_D = isqrt(D)
        if sqrt_D * sqrt_D == D and D > 0:
            S = abs(int(v[0]))
            if S > 0:
                p = (S + sqrt_D) // 2
                q = (S - sqrt_D) // 2
                if p > 0 and q > 0 and p * q == N:
                    print(f"  Found via lattice: p = {p}, q = {q}")
                    return p, q
    
    print(f"  No factors found")
    print(f"  Note: This is equivalent to Fermat's method")
    return None


# =============================================================================
# APPROACH 4: Product Lattice - Encode Multiple Products
# =============================================================================

def product_lattice(N: int, dimension: int = 15):
    """
    Encode multiple products simultaneously.
    
    For each prime p_i, consider:
    - p_i × (N // p_i) = N (approximately)
    - The actual divisor q satisfies p_i × q_i ≈ N where q_i ≈ N / p_i
    
    Build lattice encoding:
    [p_i, N // p_i, N mod p_i, ...]
    
    For true factors p, q:
    - N mod p = 0, N mod q = 0
    - p × q = N exactly
    - Both appear in the factor base
    """
    print(f"\n{'='*70}")
    print(f"PRODUCT LATTICE")
    print(f"{'='*70}")
    
    primes = generate_primes(dimension)
    
    print(f"\nEncoding: [prime_i, N // prime_i, N mod prime_i, ...]")
    
    B = np.zeros((dimension + 2, dimension + 4), dtype=np.int64)
    
    for i, p in enumerate(primes):
        B[i, i] = p
        B[i, dimension] = N // p  # Approximate co-factor
        B[i, dimension + 1] = N % p  # Residue
        B[i, dimension + 2] = p * (N // p)  # Product approximation
        B[i, dimension + 3] = N - p * (N // p)  # Error
    
    B[dimension, dimension] = 1
    B[dimension + 1, dimension + 1] = 1
    
    print(f"Lattice shape: {B.shape}")
    
    # For factors in factor base, look for:
    # - Small residue (N mod p = 0)
    # - Exact product (p × (N // p) = N)
    
    B_reduced = lll_reduce(B)
    
    for v in B_reduced:
        # Check residue column
        residue = abs(int(v[dimension + 1]))
        if residue == 0:
            # Found a divisor!
            for x in v[:dimension]:
                x_int = abs(int(x))
                if x_int > 1 and x_int < N and N % x_int == 0:
                    print(f"  Found factor: {x_int}")
                    return x_int, N // x_int
        
        # Check product error
        error = abs(int(v[dimension + 3]))
        if error == 0:
            # Exact product found
            for x in v[:dimension]:
                x_int = abs(int(x))
                if x_int > 1 and x_int < N and N % x_int == 0:
                    print(f"  Found via exact product: {x_int}")
                    return x_int, N // x_int
    
    print(f"  No factors found")
    return None


# =============================================================================
# APPROACH 5: Recursive/Refinement Lattice
# =============================================================================

def recursive_lattice(N: int, iterations: int = 3, dimension: int = 15):
    """
    Build lattices recursively, using output of one as input to next.
    
    Idea: Each iteration refines the search space.
    
    Iteration 1: Broad search
    Iteration 2: Focus on promising regions
    Iteration 3: Zoom in on candidates
    """
    print(f"\n{'='*70}")
    print(f"RECURSIVE/REFINEMENT LATTICE")
    print(f"{'='*70}")
    
    sqrtN = isqrt(N)
    
    for iteration in range(iterations):
        print(f"\nIteration {iteration + 1}:")
        
        # Search range narrows each iteration
        search_range = dimension // (iteration + 1)
        candidates = list(range(sqrtN - search_range, sqrtN + search_range + 1))
        
        print(f"  Searching range [{sqrtN - search_range}, {sqrtN + search_range}]")
        
        for candidate in candidates:
            if candidate > 1 and N % candidate == 0:
                print(f"  Found factor: {candidate}")
                return candidate, N // candidate
        
        # Refine: use information from this iteration to narrow next
        # (For now, just narrowing the range)
    
    print(f"  No factors found in {iterations} iterations")
    print(f"  Note: This is essentially trial division around √N")
    return None


# =============================================================================
# APPROACH 6: Constraint Satisfaction Lattice
# =============================================================================

def constraint_satisfaction_lattice(N: int, dimension: int = 20):
    """
    Encode factorization as constraint satisfaction.
    
    We want p, q such that:
    1. p × q = N
    2. p and q are integers
    3. p ≈ q ≈ √N (for balanced)
    4. p, q > 1
    
    Encode as lattice where:
    - Each dimension represents a "variable"
    - Constraints are encoded in linear relationships
    """
    print(f"\n{'='*70}")
    print(f"CONSTRAINT SATISFACTION LATTICE")
    print(f"{'='*70}")
    
    sqrtN = isqrt(N)
    
    print(f"\nConstraints:")
    print(f"  p × q = {N}")
    print(f"  p + q = S (unknown, but S ≈ {2 * sqrtN})")
    print(f"  p - q = D (unknown, but D is small for balanced)")
    
    # Encode in a 4D lattice: [p, q, p+q, p-q]
    # We don't know p, q, but we can encode relationships
    
    print(f"\nEncoding constraint: (p+q)² - (p-q)² = 4N")
    print(f"Equivalently: S² - D² = 4N")
    
    # Build lattice for S and D
    B = np.zeros((dimension, 4), dtype=np.int64)
    
    for i in range(dimension):
        # Try different values of D (difference)
        D_candidate = i - dimension // 2
        S_squared = 4 * N + D_candidate * D_candidate
        
        if S_squared >= 0:
            S_candidate = isqrt(S_squared)
            
            B[i, 0] = D_candidate
            B[i, 1] = S_candidate
            B[i, 2] = S_squared - S_candidate * S_candidate  # Error in S²
            B[i, 3] = N
    
    print(f"\nSearching for D where 4N + D² is a perfect square...")
    
    for D in range(1, min(dimension, sqrtN)):
        S_squared = 4 * N + D * D
        S = isqrt(S_squared)
        
        if S * S == S_squared:
            # Found valid D and S
            p = (S + D) // 2
            q = (S - D) // 2
            
            if p > 0 and q > 0 and p * q == N:
                print(f"  Found: D = {D}, S = {S}")
                print(f"  p = {p}, q = {q}")
                return p, q
    
    print(f"  No factors found")
    print(f"  This is still Fermat's method in constraint form")
    return None


# =============================================================================
# TESTING
# =============================================================================

def test_all_approaches(N: int, p: int, q: int):
    """Test all multi-dimensional approaches."""
    print(f"\n{'='*70}")
    print(f"TESTING MULTI-DIMENSIONAL APPROACHES")
    print(f"N = {N} = {p} × {q}")
    print(f"{'='*70}")
    
    approaches = [
        ("3D Lattice", three_dimensional_lattice),
        ("Tensor Lattice", tensor_lattice),
        ("Multiplicative Constraint", multiplicative_constraint_lattice),
        ("Product Lattice", product_lattice),
        ("Recursive Lattice", recursive_lattice),
        ("Constraint Satisfaction", constraint_satisfaction_lattice),
    ]
    
    results = []
    for name, func in approaches:
        print(f"\n{'='*70}")
        result = func(N)
        results.append((name, result))
    
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    
    for name, result in results:
        if result:
            print(f"  {name:30s}: ✓ Found {result}")
        else:
            print(f"  {name:30s}: ✗ No factors")
    
    return results


if __name__ == "__main__":
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
    ]
    
    for N, p, q in test_cases:
        test_all_approaches(N, p, q)
        print("\n" + "="*70 + "\n")
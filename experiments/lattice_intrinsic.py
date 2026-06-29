#!/usr/bin/env python3
"""
Direction 1: Lattice-Intrinsic Properties.

The key question from Direction 1 exploration:
Can we construct a lattice where factor-related vectors have special norms
WITHOUT explicitly checking divisibility?

The Sparsity Principle shows that factor vectors are sparse (1 non-zero coordinate).
Can we construct a lattice where this sparsity appears naturally?

Approach:
1. Construct generic lattices based on N (no factor base)
2. Analyze whether factor information appears in:
   - Norm distribution of reduced basis vectors
   - Geometric properties (angles, distances)
   - Algebraic structure (dual lattice, etc.)
"""

from math import gcd, isqrt, sqrt
import random


def construct_modular_lattice(N: int, factor_base: list) -> list:
    """
    Standard modular lattice construction (requires factor base).
    This is the KNOWN working approach.
    """
    k = len(factor_base)
    
    # Basis vectors
    basis = []
    
    # First k rows: [p_i, 0, ..., N mod p_i, ..., 0]
    for i, p in enumerate(factor_base):
        row = [0] * (k + 1)
        row[0] = p
        residue = N % p
        if residue == 0:
            row[i + 1] = 1
        basis.append(row)
    
    # Last row: [N, 1, 1, ..., 1]
    row = [N] + [1] * k
    basis.append(row)
    
    return basis


def construct_generic_lattice_v1(N: int, dimension: int = 10) -> list:
    """
    Generic lattice construction WITHOUT factor base.
    
    Uses N's structure alone:
    - N itself as one coordinate
    - Powers of small integers
    - Fibonacci-like sequences
    """
    basis = []
    
    # Row 1: N as the "anchor"
    basis.append([N] + [0] * (dimension - 1))
    
    # Rows 2-k: Various functions of N
    for i in range(1, dimension):
        # Powers
        if i == 1:
            row = [N**2] + [0] * (dimension - 1)
        # Fibonacci-like
        elif i == 2:
            row = [N, N] + [0] * (dimension - 2)
        # Polynomial values
        else:
            x = i
            row = [N * x + x**2 - N] + [0] * (dimension - 1)
        
        basis.append(row)
    
    return basis


def construct_generic_lattice_v2(N: int, k: int = 10) -> list:
    """
    Alternative: Use N's algebraic properties.
    
    Key insight: If p | N, then N ≡ 0 (mod p).
    Can we create structure where p "emerges" from N's properties?
    """
    basis = []
    
    # Use continued fraction convergents to √N
    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N:
        # N is a perfect square - special case
        return [[N, 0], [0, 1]]
    
    # Continued fraction for √N
    m = 0
    d = 1
    a0 = sqrt_N
    a = a0
    
    h_prev, h = 1, a0
    k_prev, k = 0, 1
    
    convergents = []
    
    for _ in range(k):
        m = d * a - m
        d = (N - m * m) // d
        a = (a0 + m) // d
        
        h_new = a * h + h_prev
        k_new = a * k + k_prev
        
        h_prev, h = h, h_new
        k_prev, k = k, k_new
        
        convergents.append((h, k, h * h - N * k * k))
    
    # Construct basis from convergents
    basis = [[N, 0, 0]]
    for h, k, val in convergents[:k]:
        # Row: [h, k, val] where h² - N·k² = val
        if val != 0:
            g = gcd(abs(val), N)
            basis.append([h, k, g if g > 1 and g < N else 0])
    
    return basis


def analyze_lattice_properties(N: int, basis: list, description: str):
    """
    Analyze a lattice for factor-related properties.
    """
    print(f"\n{'='*70}")
    print(f"LATTICE ANALYSIS: {description}")
    print(f"N = {N}")
    print(f"{'='*70}")
    
    print(f"\nBasis ({len(basis)} vectors, dimension {len(basis[0])}):")
    for i, row in enumerate(basis[:5]):
        print(f"  {i}: {row}")
    if len(basis) > 5:
        print(f"  ... ({len(basis) - 5} more)")
    
    # Analyze norm distribution
    norms = [sum(x**2 for x in row) for row in basis]
    print(f"\nNorm distribution:")
    print(f"  Min: {min(norms)}")
    print(f"  Max: {max(norms)}")
    print(f"  Mean: {sum(norms)/len(norms):.1f}")
    
    # Check for factor information
    factors_found = []
    for row in basis:
        for x in row:
            if x > 1 and x < N and N % x == 0:
                factors_found.append(x)
    
    if factors_found:
        print(f"\nFactors found in basis: {set(factors_found)}")
    else:
        print(f"\nNo factors found in basis vectors")
    
    # Check GCDs
    gcds_found = []
    for row in basis:
        g = gcd(row[0], N)
        if 1 < g < N:
            gcds_found.append(g)
    
    if gcds_found:
        print(f"GCDs with N: {set(gcds_found)}")
    
    return basis


def norm_distribution_experiment():
    """
    Experiment: Do factor-related vectors have special norms in generic lattices?
    """
    print("="*70)
    print("NORM DISTRIBUTION EXPERIMENT")
    print("="*70)
    print("""
Question: In a generic lattice constructed from N (no factor base),
do factor-related vectors have special norms?

If yes: We could use norm distribution to find factors.
If no: Factor information requires explicit divisibility encoding.
""")
    
    # Test on small semiprimes
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
    ]
    
    for N, p, q in test_cases:
        print(f"\n{'='*70}")
        print(f"N = {N} = {p} × {q}")
        print(f"{'='*70}")
        
        # Construct lattice v1
        basis1 = construct_generic_lattice_v1(N, dimension=10)
        analyze_lattice_properties(N, basis1, "Generic v1 (N, N², etc.)")
        
        # Construct lattice v2 (continued fractions)
        basis2 = construct_generic_lattice_v2(N, k=10)
        analyze_lattice_properties(N, basis2, "Generic v2 (CF convergents)")


def the_key_question():
    """
    The key question for Direction 1.
    """
    print("\n" + "="*70)
    print("THE KEY QUESTION")
    print("="*70)
    print("""
From the Sparsity Principle (KEY-FINDINGS.md):

Factor vectors are sparse (1 non-zero coordinate) while non-factors
are dense (2+ non-zero coordinates). This sparsity is CREATED by
checking divisibility.

Question: Can we create sparsity WITHOUT checking divisibility?

Approaches:
1. Use N's intrinsic properties (N, N², √N convergents, etc.)
2. Hope that factor-related structure appears naturally

Results from experiments:
  - Generic lattice v1: No factors found, no special norms
  - Generic lattice v2 (CF): Factors might appear in GCDs
  
The continued fraction approach (v2) is interesting:
  - Convergents h/k to √N give h² - N·k² = small values
  - Small values are likely to share factors with N
  - This is CFRAC! Already subexponential, not polynomial.

The fundamental issue:
  - ANY construction using N alone produces "generic" structure
  - Factor-specific structure requires factor knowledge
  - The only way to inject factor info is divisibility checking

Conclusion so far: No generic construction has found factors.
But the experiment is worth doing thoroughly.
""")


if __name__ == "__main__":
    norm_distribution_experiment()
    the_key_question()
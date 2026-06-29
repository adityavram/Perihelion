#!/usr/bin/env python3
"""
Direction 2: Quadratic Residue Patterns

Core Idea: The Legendre symbol (a/N) encodes factor information.

For N = pq:
  (a/N) = (a/p) · (a/q)

Key Property: If (a/p) ≠ (a/q), then gcd(a^((N-1)/2) ± 1, N) reveals a factor.

This is the basis of Miller-Rabin primality test and other factorization methods.

Question: Can we use QR patterns in a lattice?
"""

import numpy as np
from math import gcd, isqrt
from typing import Tuple, List, Optional


def legendre_symbol(a: int, p: int) -> int:
    """
    Compute the Legendre symbol (a/p).
    
    Returns:
        1 if a is a quadratic residue mod p (and a ≢ 0 mod p)
        -1 if a is a non-residue mod p
        0 if a ≡ 0 mod p
    """
    if a % p == 0:
        return 0
    
    # Use Euler's criterion: (a/p) = a^((p-1)/2) mod p
    result = pow(a, (p - 1) // 2, p)
    
    # Convert to {-1, 0, 1}
    if result == p - 1:
        return -1
    return result


def jacobi_symbol(a: int, n: int) -> int:
    """
    Compute the Jacobi symbol (a/n), the generalization of Legendre to composite n.
    
    For n = p₁^e₁ · p₂^e₂ · ..., the Jacobi symbol is the product of Legendre symbols.
    """
    if gcd(a, n) > 1:
        return 0
    
    # Jacobi symbol computation using quadratic reciprocity
    a = a % n
    result = 1
    
    while a > 0:
        # Remove factors of 2
        while a % 2 == 0:
            a //= 2
            if n % 8 in [3, 5]:
                result *= -1
        
        # Quadratic reciprocity
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result *= -1
        
        a %= n
    
    if n == 1:
        return result
    return 0


def analyze_qr_patterns(N: int, p: int = None, q: int = None, num_bases: int = 50):
    """
    Analyze quadratic residue patterns for N.
    
    For semiprimes, QR patterns can distinguish p from q.
    """
    print(f"\n{'='*70}")
    print(f"QUADRATIC RESIDUE ANALYSIS: N = {N}")
    print(f"{'='*70}")
    
    # Sample bases
    bases = list(range(1, min(num_bases + 1, N)))
    
    # Compute Jacobi symbols
    jacobi_values = [(a, jacobi_symbol(a, N)) for a in bases]
    
    # Count QR types
    qr_count = sum(1 for _, j in jacobi_values if j == 1)
    nqr_count = sum(1 for _, j in jacobi_values if j == -1)
    zero_count = sum(1 for _, j in jacobi_values if j == 0)
    
    print(f"\nJacobi symbols for bases 1 to {num_bases}:")
    print(f"  (a/N) =  1: {qr_count} values (quadratic residues)")
    print(f"  (a/N) = -1: {nqr_count} values (non-residues)")
    print(f"  (a/N) =  0: {zero_count} values (gcd > 1)")
    
    if p and q:
        print(f"\nFactorization: N = {p} × {q}")
        print(f"\nLegendre symbols (a/p) and (a/q) for comparison:")
        
        different_qr = []
        for a in bases[:20]:
            leg_p = legendre_symbol(a, p)
            leg_q = legendre_symbol(a, q)
            if leg_p != leg_q:
                different_qr.append((a, leg_p, leg_q))
        
        print(f"\n  Bases where (a/p) ≠ (a/q): {len(different_qr)} out of first 20")
        print(f"  First 5 examples:")
        for a, lp, lq in different_qr[:5]:
            print(f"    a = {a}: (a/p) = {lp}, (a/q) = {lq}")
            # Check if gcd reveals factor
            power = pow(a, (N - 1) // 2, N)
            if power == N - 1:  # -1 mod N
                power = -1
            print(f"      a^((N-1)/2) mod N = {power}")
            if power != 1 and power != -1:
                g1 = gcd(power - 1, N)
                g2 = gcd(power + 1, N)
                print(f"      gcd(a^((N-1)/2) - 1, N) = {g1}")
                print(f"      gcd(a^((N-1)/2) + 1, N) = {g2}")
    
    return jacobi_values, different_qr if p and q else []


def qr_based_factorization(N: int, max_tries: int = 100) -> Optional[Tuple[int, int]]:
    """
    Try to factor N using QR-based method.
    
    Find a such that (a/p) ≠ (a/q), then compute gcd(a^((N-1)/2) ± 1, N).
    
    This is the basis of the Miller-Rabin test.
    """
    print(f"\n{'='*70}")
    print(f"QR-BASED FACTORIZATION ATTEMPT")
    print(f"{'='*70}")
    
    for a in range(2, max_tries + 2):
        # Compute a^((N-1)/2) mod N
        power = pow(a, (N - 1) // 2, N)
        
        # Check if power reveals factors
        if power != 1 and power != N - 1:
            # This is the case where (a/p) ≠ (a/q)
            g1 = gcd(power - 1, N)
            g2 = gcd(power + 1, N)
            
            if g1 > 1 and g1 < N:
                return (g1, N // g1)
            if g2 > 1 and g2 < N:
                return (g2, N // g2)
    
    return None


def qr_pattern_lattice(N: int, dimension: int = 20):
    """
    Encode QR patterns in a lattice.
    
    Idea: For each base a, we have Jacobi symbol (a/N).
    Can we use the pattern of Jacobi symbols to constrain factors?
    
    Key insight: The Jacobi symbol is MULTIPLICATIVE:
    (ab/N) = (a/N) · (b/N)
    
    This means QR patterns have algebraic structure.
    """
    print(f"\n{'='*70}")
    print(f"QR PATTERN LATTICE")
    print(f"{'='*70}")
    
    # Collect Jacobi symbols for various bases
    bases = list(range(1, dimension + 1))
    jacobi_vals = [jacobi_symbol(a, N) for a in bases]
    
    print(f"\nJacobi symbol pattern (first {dimension} bases):")
    print(f"  {jacobi_vals[:20]}")
    
    # Count patterns
    qr_indices = [i for i, j in enumerate(jacobi_vals) if j == 1]
    nqr_indices = [i for i, j in enumerate(jacobi_vals) if j == -1]
    
    print(f"\n  QR bases (first 10): {qr_indices[:10]}")
    print(f"  Non-QR bases (first 10): {nqr_indices[:10]}")
    
    # Build lattice encoding QR patterns
    # Row i: [base_i, ..., (base_i)^((N-1)/2) mod N, ...]
    
    # Actually, Jacobi symbols are already a "signature" of N
    # Can we use this signature directly?
    
    print(f"\nQR signature of N:")
    print(f"  Jacobi symbol pattern: {''.join(['+' if j == 1 else '-' if j == -1 else '0' for j in jacobi_vals[:20]])}")
    
    # The pattern is determined by the factors
    # Can we reverse-engineer factors from the pattern?
    
    print(f"\nRelationship to factors:")
    print(f"  (a/N) = (a/p) · (a/q)")
    print(f"  The Jacobi symbol pattern encodes the PRODUCT of Legendre symbols")
    print(f"  We want to find p, q such that (a/p) · (a/q) matches the observed pattern")
    
    # Build a lattice from QR relationships
    # For quadratic reciprocity: (a/N) depends on a mod p and a mod q
    
    B = np.zeros((dimension, dimension + 1), dtype=np.int64)
    
    for i, a in enumerate(bases):
        B[i, i] = a
        B[i, dimension] = jacobi_vals[i]  # +1 or -1
    
    print(f"\nLattice encoding QR patterns:")
    print(f"  Dimension: {dimension}")
    print(f"  Each row: [base, ..., Jacobi symbol]")
    
    # This lattice encodes the QR pattern, but how do we extract factors?
    
    # The Jacobi symbol (a/N) tells us whether a is QR mod BOTH p and q,
    # or NEITHER, but doesn't distinguish between (1,-1) and (-1,1)
    
    print(f"\nKey insight:")
    print(f"  Jacobi symbol (a/N) = 1 means:")
    print(f"    - (a/p) = (a/q) = 1   (QR mod both)")
    print(f"    OR (a/p) = (a/q) = -1 (non-QR mod both)")
    print(f"  Jacobi symbol (a/N) = -1 means:")
    print(f"    - (a/p) = 1, (a/q) = -1 OR (a/p) = -1, (a/q) = 1")
    print(f"    - One is QR, one is not")
    
    print(f"\nTo find factors:")
    print(f"  We need to find a where (a/p) ≠ (a/q)")
    print(f"  This is exactly what Miller-Rabin does")
    print(f"  But we need to TEST each a (enumeration)")
    
    return B


def higher_power_qr(N: int, k: int, p: int = None, q: int = None):
    """
    Explore higher power residues: (a/N)_k for k > 2.
    
    For k-th power residues:
    - (a/p)_k = 1 if a is a k-th power residue mod p
    - This gives more information than quadratic residues
    
    For quadratic (k=2), we have the Legendre symbol.
    For quartic (k=4), cubic (k=3), etc., we get more refined information.
    
    The user suspected we might need k > 2 to get useful information.
    """
    print(f"\n{'='*70}")
    print(f"HIGHER POWER RESIDUES (k={k})")
    print(f"{'='*70}")
    
    print(f"\nFor k-th power residues:")
    print(f"  (a/p)_k = 1 if there exists x such that x^k ≡ a (mod p)")
    print(f"  (a/p)_k takes values in the k-th roots of unity mod p")
    
    if p:
        print(f"\nFor p = {p}:")
        print(f"  Primitive k-th root of unity mod p:")
        
        # For prime p, the multiplicative group (Z/pZ)* is cyclic of order p-1
        # k-th power residues exist iff gcd(k, p-1) | k
        
        if gcd(k, p - 1) == k or k % gcd(k, p - 1) == 0:
            print(f"    k divides p-1 = {p-1}")
            print(f"    k-th power residues exist")
            
            # Find a primitive root
            primitive_root = find_primitive_root(p)
            if primitive_root:
                print(f"    Primitive root: {primitive_root}")
                print(f"    k-th roots of unity mod p: {primitive_root}^((p-1)/k), ...")
        else:
            print(f"    k does not divide p-1 = {p-1}")
            print(f"    All residues are k-th power residues")
    
    print(f"\nHigher power residue tests:")
    print(f"  For k=4 (quartic residues):")
    print(f"    - More refined classification of residues")
    print(f"    - But: only works if gcd(k, p-1) and gcd(k, q-1) differ")
    
    if p and q:
        print(f"\n  p-1 = {p-1}, q-1 = {q-1}")
        print(f"  gcd(4, p-1) = {gcd(4, p-1)}, gcd(4, q-1) = {gcd(4, q-1)}")
        
        if gcd(4, p-1) != gcd(4, q-1):
            print(f"  Different! Quartic residues might help.")
        else:
            print(f"  Same. Quartic residues give same pattern for p and q.")


def find_primitive_root(p: int) -> Optional[int]:
    """Find a primitive root mod p (generator of (Z/pZ)*)."""
    if p == 2:
        return 1
    
    # Factor p - 1
    factors = []
    n = p - 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            if d not in factors:
                factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    
    # Find primitive root
    for g in range(2, p):
        is_primitive = all(pow(g, (p - 1) // f, p) != 1 for f in factors)
        if is_primitive:
            return g
    
    return None


def analyze_qr_lattice_potential(N: int, p: int = None, q: int = None):
    """
    Analyze whether QR patterns can help with factorization.
    """
    print(f"\n{'='*70}")
    print(f"QR LATTICE POTENTIAL ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nWHAT QR PATTERNS ENCODE:")
    print(f"  The Jacobi symbol (a/N) = (a/p) · (a/q)")
    print(f"  This is the PRODUCT of Legendre symbols")
    
    print(f"\nWHAT WE WANT:")
    print(f"  Find p, q such that the QR pattern matches")
    
    print(f"\nTHE PROBLEM:")
    print(f"  The Jacobi symbol is a 2-to-1 map:")
    print(f"    (a/p) · (a/q) = 1 if both are QR or both are non-QR")
    print(f"    (a/p) · (a/q) = -1 if one is QR, one is non-QR")
    print(f"  We cannot distinguish (1,1) from (-1,-1)")
    print(f"  We cannot distinguish (1,-1) from (-1,1)")
    
    print(f"\nHOW FACTORIZATION WORKS:")
    print(f"  Find a where (a/p) ≠ (a/q)")
    print(f"  Then gcd(a^((N-1)/2) ± 1, N) reveals a factor")
    print(f"  This requires TESTING different values of a")
    
    print(f"\nLATTICE ENCODING POSSIBILITY:")
    print(f"  Can we encode the constraint '(a/p) ≠ (a/q)'?")
    print(f"  NO - this requires knowing p and q")
    
    print(f"\nALTERNATIVE:")
    print(f"  Use higher power residues (k > 2)")
    print(f"  For k=4, (a/p)_4 gives more information")
    print(f"  But: requires gcd(k, p-1) ≠ gcd(k, q-1)")
    
    if p and q:
        print(f"\nFOR N = {N} = {p} × {q}:")
        print(f"  p-1 = {p-1}")
        print(f"  q-1 = {q-1}")
        print(f"  p-1 and q-1 differ by {abs((p-1) - (q-1))}")
        
        # Check if different prime factorizations
        factors_p1 = prime_factors(p - 1)
        factors_q1 = prime_factors(q - 1)
        
        print(f"  p-1 factors: {factors_p1}")
        print(f"  q-1 factors: {factors_q1}")
        
        if factors_p1 != factors_q1:
            print(f"  Different factorizations!")
            print(f"  Higher power residues might work.")
        else:
            print(f"  Same factorizations.")
            print(f"  Higher power residues give same patterns.")


def prime_factors(n: int) -> List[int]:
    """Find prime factorization of n."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return sorted(factors)


def experiment_summary():
    """Summarize QR pattern experiments."""
    print(f"\n{'='*70}")
    print(f"QR PATTERN ANALYSIS: SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nKEY FINDINGS:")
    print(f"  1. Jacobi symbol (a/N) encodes QR pattern multiplicatively")
    print(f"  2. This gives a 'signature' of N")
    print(f"  3. But: signature doesn't directly reveal factors")
    print(f"  4. Miller-Rabin finds factors by TESTING each a")
    
    print(f"\nLATTICE CONNECTION:")
    print(f"  - QR patterns have algebraic structure (multiplicative)")
    print(f"  - Could encode in lattice: (ab/N) = (a/N) · (b/N)")
    print(f"  - But: still requires enumeration to find distinguishing a")
    
    print(f"\nHIGHER POWER RESIDUES (k > 2):")
    print(f"  - More refined classification")
    print(f"  - Only helps if gcd(k, p-1) ≠ gcd(k, q-1)")
    print(f"  - Still requires enumeration or clever choice of k")
    
    print(f"\nCONCLUSION:")
    print(f"  Direction 2 is NOT promising for lattice-based factorization.")
    print(f"  QR patterns encode factor information, but extraction requires enumeration.")
    print(f"  Higher power residues don't fundamentally change this.")
    
    print(f"\nNEXT DIRECTION TO EXPLORE:")
    print(f"  Direction 3: Order-Finding (Shor's classical equivalent?)")


if __name__ == "__main__":
    # Test with known semiprime
    N = 143
    p, q = 11, 13
    
    analyze_qr_patterns(N, p, q, num_bases=30)
    qr_based_factorization(N)
    qr_pattern_lattice(N, dimension=20)
    higher_power_qr(N, k=4, p=p, q=q)
    analyze_qr_lattice_potential(N, p, q)
    
    print("\n" + "="*70)
    
    # Test with another
    N2 = 391
    p2, q2 = 17, 23
    
    print("\n" + "="*70)
    analyze_qr_patterns(N2, p2, q2, num_bases=30)
    higher_power_qr(N2, k=4, p=p2, q=q2)
    
    # Summary
    experiment_summary()
#!/usr/bin/env python3
"""
Direction 3: Order-Finding via Lattices

Shor's Algorithm (Quantum):
1. Find order r of a mod N (smallest r such that a^r ≡ 1 mod N)
2. If r is even and a^(r/2) ≢ -1 mod N, then gcd(a^(r/2) ± 1, N) gives factors
3. Quantum: Uses QFT to find r in polynomial time

Classical Challenge:
- Order-finding is believed to be hard classically
- No known polynomial-time algorithm

Question: Can lattices help find orders?

Connections:
- Order is a period of the sequence {a^i mod N}
- Finding periods in sequences has lattice connections
- LLL can sometimes find short vectors that reveal periods
"""

import numpy as np
from math import gcd, isqrt
from typing import Tuple, Optional, List


def find_order_classical(a: int, N: int, max_r: int = None) -> Optional[int]:
    """
    Find the order of a modulo N using classical brute force.
    
    Order r: smallest positive integer such that a^r ≡ 1 (mod N)
    
    This is exponential time in the number of bits of N.
    """
    if max_r is None:
        max_r = N  # Order divides φ(N) ≤ N
    
    power = 1
    for r in range(1, max_r + 1):
        power = (power * a) % N
        if power == 1:
            return r
    
    return None


def order_to_factor(a: int, r: int, N: int) -> Optional[Tuple[int, int]]:
    """
    Given the order r of a mod N, try to extract a factor.
    
    Works when:
    - r is even
    - a^(r/2) ≢ -1 (mod N)
    
    Then gcd(a^(r/2) ± 1, N) gives a factor.
    """
    if r % 2 != 0:
        return None
    
    half_power = pow(a, r // 2, N)
    
    if half_power == N - 1 or half_power == 1:
        return None
    
    g1 = gcd(half_power - 1, N)
    g2 = gcd(half_power + 1, N)
    
    if g1 > 1 and g1 < N:
        return (g1, N // g1)
    if g2 > 1 and g2 < N:
        return (g2, N // g2)
    
    return None


def sequence_lattice_order(a: int, N: int, length: int = 50) -> Optional[int]:
    """
    Encode the sequence {a^i mod N} in a lattice and try to find the period.
    
    Idea: The sequence is periodic with period = order.
    Can we find this period using lattice reduction?
    
    Construction: Build a lattice where consecutive powers encode periodicity.
    """
    # Generate sequence
    sequence = [pow(a, i, N) for i in range(length)]
    
    # Look for period by checking if sequence[i] == sequence[0] for small i
    for period in range(1, length // 2):
        if all(sequence[i] == sequence[i % period] for i in range(min(period * 3, length))):
            return period
    
    return None


def build_multiplicative_order_lattice(a: int, N: int, dimension: int = 20):
    """
    Build a lattice encoding the multiplicative structure.
    
    For order r: a^r ≡ 1 (mod N)
    
    This means: a^r - 1 ≡ 0 (mod N)
    So: N | (a^r - 1)
    
    Idea: Encode the condition "N divides a^r - 1" in a lattice.
    
    Construction:
    - Each row encodes a power of a
    - Look for linear combinations where N divides the result
    """
    print(f"\n{'='*70}")
    print(f"MULTIPLICATIVE ORDER LATTICE")
    print(f"{'='*70}")
    
    print(f"\nGoal: Find order r of {a} mod {N}")
    print(f"Order r: smallest r such that {a}^r ≡ 1 (mod {N})")
    
    # Generate powers
    powers = [pow(a, i, N) for i in range(dimension)]
    
    print(f"\nPowers of {a} mod {N} (first {dimension}):")
    print(f"  {powers[:min(20, dimension)]}")
    
    # Build lattice: [power_i, a^i - 1, N]
    # We want to find where a^i - 1 is divisible by N
    
    B = np.zeros((dimension, 3), dtype=np.int64)
    
    for i in range(dimension):
        B[i, 0] = powers[i]
        B[i, 1] = pow(a, i) - 1  # a^i - 1 (full, not mod N)
        B[i, 2] = N
    
    print(f"\nLattice shape: {B.shape}")
    print(f"Row i: [a^i mod N, a^i - 1, N]")
    
    # Check for orders
    print(f"\nSearching for order r such that N | (a^r - 1):")
    for i in range(1, dimension):
        if powers[i] == 1:
            print(f"  Found: r = {i}")
            return i
    
    print(f"  No small order found (r > {dimension})")
    
    # The lattice approach: look for short vectors where linear combinations
    # give multiples of N
    
    return None


def diffsequence_lattice(a: int, N: int, length: int = 30) -> Optional[int]:
    """
    Use the difference sequence to find periods.
    
    If a^r ≡ 1 (mod N), then the sequence {a^i mod N} has period r.
    
    The difference sequence d[i] = a^(i+1) - a^i mod N also has period r.
    
    Encode these differences in a lattice and look for periodicity.
    """
    # Generate sequence
    sequence = [pow(a, i, N) for i in range(length)]
    
    # Difference sequence
    diffs = [(sequence[i+1] - sequence[i]) % N for i in range(length - 1)]
    
    # Build lattice of differences
    dim = len(diffs)
    B = np.zeros((dim, dim), dtype=np.int64)
    
    for i in range(dim):
        B[i, i] = diffs[i]
    
    # LLL reduce
    B_reduced = lll_reduce(B)
    
    # Look for short vectors with consistent patterns
    for i, v in enumerate(B_reduced[:10]):
        if np.count_nonzero(v) <= dim // 2:
            # Sparse vector might indicate period
            nonzero_indices = np.nonzero(v)[0]
            if len(nonzero_indices) >= 2:
                potential_period = nonzero_indices[1] - nonzero_indices[0]
                if potential_period > 0:
                    # Check if this is actually the period
                    if all(sequence[j] == sequence[j % potential_period] 
                           for j in range(min(potential_period * 3, length))):
                        return potential_period
    
    return None


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


def shor_classical_attempt(N: int, max_a: int = 100, max_r: int = 1000) -> Optional[Tuple[int, int]]:
    """
    Classical attempt at Shor's algorithm: try different bases and find orders.
    
    This is still exponential, but demonstrates the principle.
    """
    print(f"\n{'='*70}")
    print(f"SHOR-STYLE CLASSICAL ATTEMPT")
    print(f"{'='*70}")
    
    print(f"\nTrying to factor N = {N} using order-finding:")
    
    for a in range(2, max_a):
        if gcd(a, N) != 1:
            # Found a factor directly
            if gcd(a, N) > 1 and gcd(a, N) < N:
                print(f"  Found factor via gcd: {gcd(a, N)}")
                return (gcd(a, N), N // gcd(a, N))
            continue
        
        # Find order
        r = find_order_classical(a, N, max_r)
        
        if r is None:
            continue
        
        # Try to extract factor
        factors = order_to_factor(a, r, N)
        
        if factors:
            print(f"  Base a = {a}, order r = {r}")
            print(f"  {a}^{r//2} mod {N} = {pow(a, r//2, N)}")
            print(f"  Factors: {factors[0]} × {factors[1]}")
            return factors
    
    print(f"  No factor found in {max_a} attempts")
    return None


def classical_vs_quantum_comparison(N: int, p: int = None, q: int = None):
    """
    Compare what Shor's quantum algorithm does vs classical approaches.
    """
    print(f"\n{'='*70}")
    print(f"CLASSICAL vs QUANTUM COMPARISON for N = {N}")
    print(f"{'='*70}")
    
    if p and q:
        print(f"\nFactorization: N = {p} × {q}")
    
    print(f"\nShor's Quantum Algorithm:")
    print(f"  1. Create superposition: (1/√N) Σ|x⟩ for x = 1 to N")
    print(f"  2. Apply modular exponentiation: |x⟩ → |x, a^x mod N⟩")
    print(f"  3. Measure second register, collapse to |x₀ + kr⟩ for some x₀")
    print(f"  4. Apply QFT: peaks at multiples of 1/r")
    print(f"  5. Classical post-processing to find r")
    print(f"  Complexity: O((log N)³) quantum gates")
    
    print(f"\nClassical Challenge:")
    print(f"  Step 4 (QFT) reveals the period r efficiently")
    print(f"  Classically, finding r requires:")
    print(f"    - Brute force: O(r) operations")
    print(f"    - Could be O(N) in worst case")
    
    print(f"\nLattice Connection:")
    print(f"  Period-finding: finding r such that sequence repeats")
    print(f"  LLL can find periods in some cases")
    print(f"  But: requires encoding the sequence in a lattice")
    
    print(f"\nKey Question:")
    print(f"  Can we encode the period-finding problem in a lattice")
    print(f"  such that LLL reveals the period r?")


def analyze_period_finding_lattice():
    """
    Analyze whether period-finding can be encoded in lattices.
    """
    print(f"\n{'='*70}")
    print(f"PERIOD-FINDING IN LATTICES: THEORETICAL ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nPeriod-finding is a well-studied problem.")
    print(f"\nKnown Classical Algorithms:")
    print(f"  1. Brute force: O(period)")
    print(f"  2. Baby-step giant-step: O(√period)")
    print(f"  3. Pollard's rho: O(√period) expected")
    print(f"  4. Using discrete log algorithms: subexponential")
    
    print(f"\nLattice Approaches to Period-Finding:")
    print(f"  1. Encode sequence as lattice vectors")
    print(f"  2. Look for linear relations that reveal period")
    print(f"  3. Use LLL to find short vectors")
    
    print(f"\nChallenge:")
    print(f"  Period could be very large (up to φ(N) ~ N)")
    print(f"  Encoding requires O(period) space in general")
    print(f"  No known polynomial-size lattice encoding")
    
    print(f"\nConnection to Hidden Subgroup Problem (HSP):")
    print(f"  Period-finding is a special case of HSP")
    print(f"  Quantum: QFT solves HSP in polynomial time")
    print(f"  Classical: No polynomial-time algorithm known")
    print(f"  Shor's breakthrough: QFT on cyclic group finds period")
    
    print(f"\nLattice vs HSP:")
    print(f"  LLL can sometimes solve HSP for certain groups")
    print(f"  But: cyclic group HSP (period-finding) is hard")
    print(f"  No known lattice method that works in general")


def experiment_summary():
    """
    Summarize order-finding experiments.
    """
    print(f"\n{'='*70}")
    print(f"ORDER-FINDING: SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nWHAT SHOR DOES:")
    print(f"  Uses quantum Fourier transform to find period in O((log N)³)")
    
    print(f"\nWHY IT WORKS QUANTUMLY:")
    print(f"  1. Superposition allows parallel computation")
    print(f"  2. QFT reveals period via constructive interference")
    print(f"  3. Quantum parallelism + interference = polynomial time")
    
    print(f"\nCLASSICAL DIFFICULTY:")
    print(f"  1. No known way to find period in polynomial time")
    print(f"  2. Baby-step giant-step: O(√N)")
    print(f"  3. Lattice methods: No polynomial-size encoding known")
    
    print(f"\nLATTICE CONNECTION:")
    print(f"  Period-finding is HSP for cyclic group")
    print(f"  HSP can sometimes be solved by lattice methods")
    print(f"  But: Cyclic group HSP (period-finding) is hard")
    
    print(f"\nATTEMPTS WE MADE:")
    print(f"  1. Sequence lattice: Encodes powers of a")
    print(f"     - Requires O(r) space where r is period")
    print(f"     - No polynomial-size encoding")
    
    print(f"  2. Difference lattice: Encodes differences")
    print(f"     - Still requires O(r) space")
    print(f"     - LLL doesn't reveal period in general")
    
    print(f"\nCONCLUSION:")
    print(f"  Direction 3 (Order-Finding) is a DEAD END for classical lattices.")
    print(f"  Period-finding requires quantum parallelism or exponential time.")
    print(f"  No known polynomial-size lattice encoding exists.")
    
    print(f"\nWHY THIS MATTERS:")
    print(f"  Shor's algorithm works because quantum mechanics allows")
    print(f"  something lattices cannot: exponential parallelism with")
    print(f"  polynomial interference.")
    print(f"  ")
    print(f"  The lattice would need to encode ALL powers a^i mod N")
    print(f"  simultaneously, which requires exponential space.")
    print(f"  ")
    print(f"  Quantum: superposition = polynomial space encoding exponential info")
    print(f"  Classical lattice: exponential space needed")


if __name__ == "__main__":
    # Test cases
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
    ]
    
    print("=" * 70)
    print("DIRECTION 3: ORDER-FINDING VIA LATTICES")
    print("=" * 70)
    
    for N, p, q in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST: N = {N} = {p} × {q}")
        print(f"{'='*70}")
        
        # Classical Shor attempt
        shor_classical_attempt(N, max_a=20, max_r=100)
    
    # Theoretical analysis
    classical_vs_quantum_comparison(143, 11, 13)
    analyze_period_finding_lattice()
    
    # Summary
    experiment_summary()
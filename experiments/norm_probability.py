#!/usr/bin/env python3
"""
Factor Discovery via Norm Computation in Z[√d].

BREAKTHROUGH: We can find factors by computing norms!

The norm N(a + b√d) = a² - d·b² is computable directly.
gcd(|N(α)|, N) might reveal factors.

No circular dependency - we don't need to compute √d mod N!

Question: How likely are we to find such (a, b)?
"""

from math import gcd, isqrt
from typing import List, Tuple, Optional
import random


def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    result = pow(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else 1


def jacobi_symbol(a: int, n: int) -> int:
    """Jacobi symbol (a/n)."""
    if gcd(a, n) != 1:
        return 0
    
    result = 1
    a = a % n
    
    while a != 0:
        while a % 2 == 0:
            a //= 2
            n_mod_8 = n % 8
            if n_mod_8 in [3, 5]:
                result = -result
        
        a, n = n, a
        
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        
        a = a % n
    
    return result if n == 1 else 0


def norm(a: int, b: int, d: int) -> int:
    """Norm of a + b√d: N = a² - d·b²."""
    return a * a - d * b * b


def find_factor_via_norm(N: int, d: int, max_a: int = None, max_b: int = None) -> Optional[Tuple[int, int, int]]:
    """
    Find a factor by searching for elements with factor-norms.
    
    Returns (a, b, factor) if found, None otherwise.
    """
    if max_a is None:
        max_a = int(N ** 0.25) + 10  # O(√p) where p ≈ √N
    
    if max_b is None:
        max_b = int(N ** 0.25) + 10
    
    for b in range(1, max_b):
        for a in range(-max_a, max_a + 1):
            n = norm(a, b, d)
            if n != 0:
                g = gcd(abs(n), N)
                if 1 < g < N:
                    return (a, b, g)
    
    return None


def find_factor_via_norm_random(N: int, d: int, max_tries: int = 10000) -> Optional[Tuple[int, int, int]]:
    """
    Find a factor by random search for elements with factor-norms.
    """
    for _ in range(max_tries):
        a = random.randint(-N, N)
        b = random.randint(1, int(N ** 0.5))
        
        n = norm(a, b, d)
        if n != 0:
            g = gcd(abs(n), N)
            if 1 < g < N:
                return (a, b, g)
    
    return None


def probability_analysis(N: int, p: int, q: int, d: int):
    """
    Analyze the probability of finding a factor-revealing element.
    """
    print("="*70)
    print(f"PROBABILITY ANALYSIS")
    print(f"N = {N} = {p} × {q}, d = {d}")
    print("="*70)
    
    leg_p = legendre_symbol(d, p)
    leg_q = legendre_symbol(d, q)
    
    print(f"\nLegendre symbols:")
    print(f"  ({d}/{p}) = {leg_p} {'→ splits' if leg_p == 1 else '→ inert'}")
    print(f"  ({d}/{q}) = {leg_q} {'→ splits' if leg_q == 1 else '→ inert'}")
    print(f"  ({d}/{N}) = {jacobi_symbol(d, N)}")
    
    # Search for elements with factor-norms
    max_a = int(N ** 0.25) + 20
    max_b = int(N ** 0.25) + 20
    
    print(f"\nSearching for a ∈ [-{max_a}, {max_a}], b ∈ [1, {max_b}):")
    
    count_p = 0  # gcd(norm, N) = p
    count_q = 0  # gcd(norm, N) = q
    count_N = 0  # gcd(norm, N) = N (norm = 0 mod N)
    count_total = 0
    
    found_p = None
    found_q = None
    
    for b in range(1, max_b):
        for a in range(-max_a, max_a + 1):
            count_total += 1
            n = norm(a, b, d)
            
            if n != 0:
                g = gcd(abs(n), N)
                if g == p:
                    count_p += 1
                    if found_p is None:
                        found_p = (a, b)
                elif g == q:
                    count_q += 1
                    if found_q is None:
                        found_q = (a, b)
                elif g == N:
                    count_N += 1
    
    total_elements = (2 * max_a + 1) * (max_b - 1)
    
    print(f"\nResults:")
    print(f"  Total elements searched: {count_total}")
    print(f"  Elements with gcd(norm, N) = {p}: {count_p} ({100*count_p/count_total:.2f}%)")
    print(f"  Elements with gcd(norm, N) = {q}: {count_q} ({100*count_q/count_total:.2f}%)")
    print(f"  Elements with gcd(norm, N) = N: {count_N} ({100*count_N/count_total:.2f}%)")
    print(f"  Elements revealing a factor: {count_p + count_q} ({100*(count_p+count_q)/count_total:.2f}%)")
    
    if found_p:
        n_p = norm(found_p[0], found_p[1], d)
        print(f"\nFirst element with norm having gcd = {p}:")
        print(f"  α = {found_p[0]} + {found_p[1]}√{d}")
        print(f"  N(α) = {n_p}")
        print(f"  |N(α)| = {abs(n_p)}")
        print(f"  gcd(|N(α)|, N) = {p}")
    
    if found_q:
        n_q = norm(found_q[0], found_q[1], d)
        print(f"\nFirst element with norm having gcd = {q}:")
        print(f"  α = {found_q[0]} + {found_q[1]}√{d}")
        print(f"  N(α) = {n_q}")
        print(f"  |N(α)| = {abs(n_q)}")
        print(f"  gcd(|N(α)|, N) = {q}")
    
    # Theoretical analysis
    print(f"\n{'='*70}")
    print("THEORETICAL ANALYSIS")
    print(f"{'='*70}")
    
    if leg_p == 1 and leg_q == 1:
        print(f"\nCase 1: d is QR mod both p and q")
        print(f"  Elements with norm = p exist (solve a² - d·b² = p)")
        print(f"  Elements with norm = q exist (solve a² - d·b² = q)")
        print(f"  Probability depends on density of solutions")
        print(f"  Roughly: O(1/p) of elements have norm divisible by p")
        print(f"  Roughly: O(1/q) of elements have norm divisible by q")
        print(f"  Combined: O(1/p + 1/q) = O(1/min(p,q))")
    
    elif (leg_p == 1 and leg_q == -1) or (leg_p == -1 and leg_q == 1):
        print(f"\nCase 2: d is QR mod one factor, NR mod the other")
        print(f"  If (d/p) = 1, (d/q) = -1:")
        print(f"    Elements with norm divisible by p exist")
        print(f"    Elements with norm divisible by q do NOT exist (inert)")
        print(f"  gcd(|norm|, N) = p for many elements")
        print(f"  Probability: higher than case 1!")
        
    else:
        print(f"\nCase 3: d is NR mod both p and q")
        print(f"  No elements with norm divisible by p or q")
        print(f"  (unless norm has small factors that share with p or q)")
        print(f"  Probability: low, depends on chance divisibility")


def theoretical_probability():
    """
    Theoretical probability of finding a factor-revealing element.
    """
    print("\n" + "="*70)
    print("THEORETICAL PROBABILITY ANALYSIS")
    print("="*70)
    print("""
For N = pq, searching for (a, b) such that gcd(|a² - d·b²|, N) > 1:

Case 1: d is QR mod both p and q
  - Elements with norm = p or p·k exist
  - Elements with norm = q or q·k exist
  - Probability of finding such (a, b): O(1/p + 1/q)
  - For RSA-size N: VERY LOW (p, q are ~2^512 for RSA-1024)

Case 2: d is QR mod one factor, NR mod the other
  - Elements with norm divisible by the QR-factor
  - No elements with norm divisible by the NR-factor
  - Probability: O(1/max(p,q)) for the QR-factor
  - Still VERY LOW for RSA-size N

Case 3: d is NR mod both factors
  - Norms have small factors by chance
  - Probability that norm shares factor with N: O(1/√N)
  - Essentially: need norm ≈ N for chance divisibility

The problem: SEARCH SPACE IS TOO LARGE!

For N ≈ 2^1024 (RSA-1024):
  - p, q ≈ 2^512
  - Probability of finding (a, b) with factor-norm: O(1/2^512)
  - Expected number of trials: O(2^512)
  - This is WORSE than trial division!

Wait... this can't be right. Let me reconsider.

Actually:
  - We're searching for (a, b) such that gcd(|norm|, N) > 1
  - This happens when norm ≡ 0 (mod p) or norm ≡ 0 (mod q)
  - For a fixed d, norm = a² - d·b²
  
  If d is QR mod p (i.e., (d/p) = 1):
    - norm ≡ 0 (mod p) ⟺ a² ≡ d·b² (mod p)
    - ⟺ (a/b)² ≡ d (mod p) ⟺ (a/b) is a square root of d mod p
    - This has solutions! About 2/p of all (a, b) pairs satisfy this
  
  So probability for d QR mod p: O(1/p)
  And for d QR mod q: O(1/q)

Combined probability when both QR: O(1/p + 1/q)

For RSA-1024: O(1/2^512) which is negligible.

So this approach DOES NOT WORK for RSA-size numbers!

The fundamental issue:
  - We can COMPUTE norms efficiently (polynomial)
  - But the probability of finding factor-norm is exponential in log N
  - Expected number of trials: O(min(p, q)) ≈ O(√N)
  - Same as trial division!

The breakthrough was illusory. We found factor-norms for small N, but for large N, the search space is too large.

However, there might still be a way to improve:
  1. Choose d strategically
  2. Use continued fractions (CFRAC does this!)
  3. Use lattice reduction on the norm form
""")


def compare_with_cfrac():
    """
    Compare with CFRAC approach.
    """
    print("\n" + "="*70)
    print("COMPARISON WITH CFRAC")
    print("="*70)
    print("""
CFRAC (Continued Fraction Factorization):
  - Uses continued fractions of √(kN) for various k
  - Convergents give |a² - kN·b²| < 2√(kN)
  - Small values are more likely to be smooth
  - Smooth values lead to factorization

Our approach:
  - Uses norm form a² - d·b²
  - We want gcd(|a² - d·b²|, N) > 1
  - But random (a, b) have probability O(1/√N) of working

The connection:
  - CFRAC finds (a, b) with |a² - kN·b²| small
  - Small values are likely to have small prime factors
  - Accumulating enough small factors leads to factorization
  - This is SUBEXPONENTIAL but not polynomial

Our approach vs CFRAC:
  - We want |a² - d·b²| to share factors with N
  - This requires a² ≡ d·b² (mod p) for some p | N
  - Random (a, b): probability O(1/p)
  - CFRAC: biased (a, b) from convergents, higher chance

So our approach is a variant of CFRAC with d instead of kN.
But we're not using continued fractions to bias the search!

Conclusion:
  - Our "breakthrough" is essentially the first step of CFRAC
  - CFRAC improves by using convergents
  - Still subexponential, not polynomial
""")


if __name__ == "__main__":
    # Test on small semiprimes
    test_cases = [
        (143, 11, 13, 3),  # Both split
        (391, 17, 23, 2),  # Both split
        (899, 29, 31, 5),  # Both split
        (143, 11, 13, 2),  # Both inert
        (143, 11, 13, 5),  # Split for p only
    ]
    
    for N, p, q, d in test_cases:
        probability_analysis(N, p, q, d)
        print("\n")
    
    theoretical_probability()
    compare_with_cfrac()
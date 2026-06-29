#!/usr/bin/env python3
"""
Miller-Rabin Analogy for Ideals in Z[√d]/(N).

Key Question: Can we find an analog of Miller-Rabin that works in Z[√d]/(N)?

Miller-Rabin works because:
  1. a^(N-1) mod N is always computable (no circular dependency)
  2. For composite N, the result differs from prime N
  3. The difference reveals factors via gcd(a^((N-1)/2) ± 1, N)

Can we find similar structure in Z[√d]/(N)?
"""

from math import gcd, isqrt
from typing import Tuple, Optional, List


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


def miller_rabin_test(N: int, a: int) -> Tuple[bool, Optional[int]]:
    """
    Miller-Rabin test for primality.
    
    Returns (is_probable_prime, factor_if_composite).
    
    The key insight: We compute a^(N-1) mod N without knowing factors.
    If N is composite, the computation reveals factors in some cases.
    """
    # Write N-1 = 2^s * d with d odd
    s = 0
    d = N - 1
    while d % 2 == 0:
        d //= 2
        s += 1
    
    # Compute a^d mod N
    x = pow(a, d, N)
    
    if x == 1 or x == N - 1:
        return (True, None)
    
    # Square repeatedly
    for r in range(s - 1):
        x = (x * x) % N
        if x == N - 1:
            return (True, None)
        
        # KEY INSIGHT: If x ≠ ±1 but x² ≡ 1, then gcd(x - 1, N) is a factor
        if x == 1:
            # This should not happen in standard Miller-Rabin
            return (False, None)
    
    # N is composite, but we didn't find a factor
    return (False, None)


def miller_rabin_factor(N: int, max_tries: int = 100) -> Optional[int]:
    """
    Try to find a factor using Miller-Rabin witnesses.
    
    This is NOT efficient for factoring, but demonstrates the principle.
    """
    for a in range(2, min(max_tries + 2, N)):
        if gcd(a, N) > 1:
            return gcd(a, N)
        
        # Write N-1 = 2^s * d
        s = 0
        d = N - 1
        while d % 2 == 0:
            d //= 2
            s += 1
        
        x = pow(a, d, N)
        
        if x == 1 or x == N - 1:
            continue
        
        for r in range(s - 1):
            x_prev = x
            x = (x * x) % N
            
            if x == 1:
                # Found a factor!
                return gcd(x_prev - 1, N)
            
            if x == N - 1:
                break
    
    return None


def norm_in_quadratic_field(a: int, b: int, d: int) -> int:
    """Norm of a + b√d in Z[√d]: N = a² - d·b²."""
    return a * a - d * b * b


def power_in_quadratic_field(a: int, b: int, d: int, k: int, N: int) -> Tuple[int, int]:
    """
    Compute (a + b√d)^k mod N.
    
    Returns (coeff_0, coeff_1) where result = coeff_0 + coeff_1 * √d.
    
    Uses binary exponentiation in the ring Z[√d]/(N).
    """
    # Start with (a, b) representing a + b√d
    result_0, result_1 = 1, 0  # Identity element = 1
    base_0, base_1 = a % N, b % N
    
    while k > 0:
        if k % 2 == 1:
            # Multiply result by base
            # (r0 + r1√d) * (b0 + b1√d) = r0*b0 + r0*b1√d + r1*b0√d + r1*b1*d
            #                             = (r0*b0 + r1*b1*d) + (r0*b1 + r1*b0)√d
            new_0 = (result_0 * base_0 + result_1 * base_1 * d) % N
            new_1 = (result_0 * base_1 + result_1 * base_0) % N
            result_0, result_1 = new_0, new_1
        
        # Square the base
        # (b0 + b1√d)² = b0² + 2*b0*b1√d + b1²*d
        #              = (b0² + b1²*d) + 2*b0*b1√d
        new_base_0 = (base_0 * base_0 + base_1 * base_1 * d) % N
        new_base_1 = (2 * base_0 * base_1) % N
        base_0, base_1 = new_base_0, new_base_1
        
        k //= 2
    
    return (result_0, result_1)


def quadratic_field_order_test(N: int, d: int, a: int, b: int) -> Optional[int]:
    """
    Find the order of (a + b√d) in (Z[√d]/(N))*.
    
    This is analogous to finding the order of a in (Z/(N))*.
    
    For N = pq, the order is lcm(order mod p, order mod q).
    
    If the order differs from expected, might reveal factors.
    """
    # Compute norm
    norm = norm_in_quadratic_field(a, b, d)
    
    if gcd(abs(norm), N) > 1 and gcd(abs(norm), N) < N:
        return gcd(abs(norm), N)
    
    # Try to find order by brute force (exponential!)
    # In practice, we'd need a better method
    
    result_0, result_1 = 1, 0  # Identity
    curr_0, curr_1 = a % N, b % N
    
    for order in range(1, N):
        # Multiply by base
        new_0 = (result_0 * curr_0 + result_1 * curr_1 * d) % N
        new_1 = (result_0 * curr_1 + result_1 * curr_0) % N
        result_0, result_1 = new_0, new_1
        
        if result_0 == 1 and result_1 == 0:
            return order
    
    return None


def quadratic_field_miller_rabin_analogy(N: int, d: int, a: int, b: int) -> Tuple[bool, Optional[int]]:
    """
    Analogy of Miller-Rabin for Z[√d]/(N).
    
    The norm N(a + b√d) = a² - d·b².
    
    Key idea: If (a + b√d)^(N-1) ≢ 1 (mod N) in some sense, might reveal factors.
    
    But: In Z[√d]/(N), the "order" is more complex.
    
    Alternative: Use the norm map.
    - N((a + b√d)^k) = N(a + b√d)^k mod N
    - This is just integer exponentiation
    - Maybe we can use structure in the ring itself?
    """
    # Compute (a + b√d)^(N-1) mod N
    result_0, result_1 = power_in_quadratic_field(a, b, d, N - 1, N)
    
    # For the result to be "1", we need result_0 ≡ 1 and result_1 ≡ 0
    # If this doesn't hold, N might be composite
    
    # More interesting: If result_1 ≢ 0, there's structure
    if result_1 != 0:
        # The √d coefficient is non-zero
        # This might indicate something about factor structure
        
        # Check gcd with coefficients
        g = gcd(result_1, N)
        if 1 < g < N:
            return (False, g)
        
        g = gcd(result_0 - 1, N)
        if 1 < g < N:
            return (False, g)
    
    # If result_1 = 0 and result_0 = 1, the element has "order dividing N-1"
    # This is analogous to a^(N-1) ≡ 1 (mod N) in Miller-Rabin
    
    return (result_0 == 1 and result_1 == 0, None)


def analyze_quadratic_field_structure(N: int, p: int, q: int, d: int):
    """
    Analyze the structure of Z[√d]/(N) for N = pq.
    
    Key question: Can we detect compositeness from ring operations?
    """
    print("="*70)
    print(f"QUADRATIC FIELD MILLER-RABIN ANALOGY")
    print(f"N = {N} = {p} × {q}, d = {d}")
    print("="*70)
    
    # Check if d is QR mod p and q
    leg_p = legendre_symbol(d, p)
    leg_q = legendre_symbol(d, q)
    
    print(f"\nLegendre symbols:")
    print(f"  ({d}/{p}) = {leg_p}")
    print(f"  ({d}/{q}) = {leg_q}")
    print(f"  ({d}/{N}) = {jacobi_symbol(d, N)}")
    
    # Case 1: Both split
    if leg_p == 1 and leg_q == 1:
        print(f"\nBoth p and q SPLIT in Z[√{d}]:")
        print(f"  Z[√{d}]/(p) ≅ Z/(p) (since √d exists mod p)")
        print(f"  Z[√{d}]/(q) ≅ Z/(q) (since √d exists mod q)")
        print(f"  Z[√{d}]/(N) ≅ Z[√{d}]/(p) × Z[√{d}]/(q) ≅ Z/(p) × Z/(q)")
        print(f"  Structure: Same as Z/(N)!")
        
    # Case 2: One splits, one inert
    elif (leg_p == 1 and leg_q == -1) or (leg_p == -1 and leg_q == 1):
        print(f"\nOne splits, one inert:")
        if leg_p == 1:
            print(f"  Z[√{d}]/(p) ≅ Z/(p)")
            print(f"  Z[√{d}]/(q) is a field extension (degree 2)")
        else:
            print(f"  Z[√{d}]/(p) is a field extension (degree 2)")
            print(f"  Z[√{d}]/(q) ≅ Z/(q)")
        print(f"  Z[√{d}]/(N) ≅ Z/(p) × GF(q²)")
        print(f"  Structure: Different from Z/(N)!")
        print(f"  THIS ASYMMETRY MIGHT BE DETECTABLE!")
        
    # Case 3: Both inert
    else:
        print(f"\nBoth inert:")
        print(f"  Z[√{d}]/(p) is a field extension (degree 2)")
        print(f"  Z[√{d}]/(q) is a field extension (degree 2)")
        print(f"  Z[√{d}]/(N) ≅ GF(p²) × GF(q²)")
        print(f"  Structure: Both are field extensions")
    
    # Try to find "witnesses" in Z[√d]/(N)
    print(f"\nSearching for witnesses in Z[√{d}]/(N):")
    
    found_factors = []
    
    for a in range(1, min(50, N)):
        for b in range(1, min(10, N)):
            if gcd(a, N) > 1 and gcd(a, N) < N:
                found_factors.append((a, b, gcd(a, N)))
                continue
            
            # Check norm
            norm = norm_in_quadratic_field(a, b, d)
            g = gcd(abs(norm), N)
            if 1 < g < N:
                found_factors.append((a, b, g))
                continue
            
            # Compute (a + b√d)^(N-1) mod N
            result_0, result_1 = power_in_quadratic_field(a, b, d, N - 1, N)
            
            # Check if result differs from expected
            if result_1 != 0:
                g = gcd(result_1, N)
                if 1 < g < N:
                    found_factors.append((a, b, g))
            
            g = gcd(result_0 - 1, N)
            if 1 < g < N:
                found_factors.append((a, b, g))
    
    if found_factors:
        print(f"\n  Found {len(found_factors)} potential factor-revealing elements:")
        for a, b, factor in found_factors[:5]:
            norm = norm_in_quadratic_field(a, b, d)
            print(f"    α = {a} + {b}√{d}: N(α) = {norm}, gcd = {factor}")
            result_0, result_1 = power_in_quadratic_field(a, b, d, N - 1, N)
            print(f"      α^(N-1) mod N = {result_0} + {result_1}√{d}")
    else:
        print(f"\n  No factor-revealing elements found in range")
    
    # Miller-Rabin analogy
    print(f"\n{'='*70}")
    print("MILLER-RABIN ANALOGY")
    print(f"{'='*70}")
    print("""
Miller-Rabin for Z/(N):
  - Works in ring Z/(N)
  - Computes a^(N-1) mod N (always defined)
  - Composite N might give a ≠ 1
  - gcd(a^(N-1)/2 ± 1, N) reveals factors

For Z[√d]/(N):
  - Works in ring Z[√d]/(N)
  - Computes (a + b√d)^(N-1) mod N (always defined)
  - Result = (r₀ + r₁√d) mod N
  
  If d is QR mod both p and q:
    - √d ≡ r_p (mod p) and √d ≡ r_q (mod q)
    - The computation is "consistent" across factors
  
  If d is QR mod one but not the other:
    - The computation has different behavior mod p vs mod q
    - This creates an "inconsistency" that might be detectable
  
Key insight: The √d coefficient in the result!
  - For prime modulus where √d exists: r₁ = 0 in the result
  - For composite modulus: r₁ might be non-zero
  - gcd(r₁, N) might reveal factors

This is analogous to:
  - Miller-Rabin: a^((N-1)/2) mod N
  - Here: (a + b√d)^(N-1) mod N
""")


def explore_witness_elements(N: int, p: int, q: int, d: int):
    """
    Find elements (a + b√d) that reveal factors.
    """
    print(f"\n{'='*70}")
    print(f"WITNESS ELEMENTS IN Z[√{d}]/({N})")
    print(f"{'='*70}")
    
    # Elements that reveal factors via norm
    print("\nElements with factor-norms:")
    for b in range(1, min(20, N)):
        for a in range(-20, 21):
            norm = norm_in_quadratic_field(a, b, d)
            if norm != 0:
                g = gcd(abs(norm), N)
                if 1 < g < N:
                    print(f"  ({a} + {b}√{d}): N = {norm}, gcd(N, {N}) = {g}")
    
    # Elements that reveal factors via power computation
    print("\nElements where (a + b√d)^(N-1) mod N reveals factors:")
    for a in range(1, min(20, N)):
        for b in range(1, min(10, N)):
            result_0, result_1 = power_in_quadratic_field(a, b, d, N - 1, N)
            
            # Check various gcds
            g1 = gcd(result_0 - 1, N)
            g2 = gcd(result_1, N)
            g3 = gcd(result_0, N)
            
            if 1 < g1 < N:
                print(f"  ({a} + {b}√{d})^(N-1) = {result_0} + {result_1}√{d}")
                print(f"    gcd(result_0 - 1, N) = {g1} ← FACTOR")
            
            if 1 < g2 < N:
                print(f"  ({a} + {b}√{d})^(N-1) = {result_0} + {result_1}√{d}")
                print(f"    gcd(result_1, N) = {g2} ← FACTOR")


if __name__ == "__main__":
    test_cases = [
        (143, 11, 13, 3),  # Both split
        (143, 11, 13, 2),  # One inert (2 is QR mod 11, not mod 13)
        (391, 17, 23, 2),  # Both split
        (899, 29, 31, 5),  # Both split
    ]
    
    for N, p, q, d in test_cases:
        analyze_quadratic_field_structure(N, p, q, d)
        print("\n")
    
    # Explore witness elements for N=143, d=3 (both split)
    explore_witness_elements(143, 11, 13, 3)
    
    # Explore witness elements for N=143, d=2 (one inert)
    explore_witness_elements(143, 11, 13, 2)
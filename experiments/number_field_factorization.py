#!/usr/bin/env python3
"""
Number Field Factorization

Core Idea: Embed N = pq in a number field where factorization is different.

In Z: N = pq (unique factorization)
In Z[√d]: N might factor as (a + b√d)(a - b√d) or differently as ideals

If we can find a, b, d such that:
- a² - db² = pq
- The factorization in Z[√d] is computable
- We can recover p, q from the decomposition

Then we've factored N via higher-dimensional structure.
"""

from math import isqrt, gcd
from typing import Optional, Tuple, List


def factor_in_quadratic_field(N: int, max_d: int = 100) -> Optional[Tuple[int, int]]:
    """
    Try to factor N by embedding in quadratic field Q(√d).
    
    In Z[√d], we look for solutions to:
        a² - db² = pq
    
    If found, we have N = (a + b√d)(a - b√d)
    
    This factors N as a norm from Z[√d].
    """
    print(f"\n{'='*70}")
    print(f"QUADRATIC FIELD FACTORIZATION")
    print(f"N = {N}")
    print(f"{'='*70}")
    
    # Try different squarefree d
    for d in range(2, max_d):
        # Check if d is squarefree
        if not is_squarefree(d):
            continue
        
        # We want to solve: a² - db² = pq
        # This is a norm equation: N(a + b√d) = pq
        
        # Try small b values
        for b in range(1, isqrt(N // d) + 1):
            # a² = pq + db²
            a_squared = N + d * b * b
            
            if not is_perfect_square(a_squared):
                continue
            
            a = isqrt(a_squared)
            
            # Check: a² - db² = pq
            if a * a - d * b * b == N:
                print(f"\nFound solution in Z[√{d}]:")
                print(f"  a = {a}, b = {b}, d = {d}")
                print(f"  a² - {d}·b² = {a*a} - {d}·{b*b} = {a*a - d*b*b} = {N}")
                print(f"  Factorization: {N} = ({a} + {b}√{d})({a} - {b}√{d})")
                
                # Now: can we extract p, q from this?
                # In Z[√d], this gives us a factorization
                # But we need to see if it reveals p and q
                
                # Try to find integer factors from this
                # The norm of (a + b√d) is N
                # We need to find elements of norm p and q
                
                # One approach: check if a ± b√d has prime norm
                # Another: find gcds with integers
                
                # For now, let's see if a or b gives factors
                g1 = gcd(a, N)
                g2 = gcd(b, N)
                
                if g1 > 1 and g1 < N:
                    print(f"  Extracted factor via gcd(a, N) = {g1}")
                    return (g1, N // g1)
                
                if g2 > 1 and g2 < N:
                    print(f"  Extracted factor via gcd(b, N) = {g2}")
                    return (g2, N // g2)
    
    print(f"\nNo quadratic field factorization found for d < {max_d}")
    return None


def is_squarefree(n: int) -> bool:
    """Check if n is squarefree (no square divisor > 1)."""
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True


def is_perfect_square(n: int) -> bool:
    """Check if n is a perfect square."""
    s = isqrt(n)
    return s * s == n


def ideal_factorization(N: int, p: int = None, q: int = None):
    """
    Study how the ideal (N) factors in Z[√d].
    
    For a prime p:
    - In Z[√d], the ideal (p) might split, ramify, or remain prime
    - Splitting depends on the Legendre symbol (d/p)
    
    If (p) = P₁·P₂ in Z[√d], then (pq) = P₁·P₂·Q₁·Q₂ (or similar)
    """
    print(f"\n{'='*70}")
    print(f"IDEAL FACTORIZATION IN QUADRATIC FIELDS")
    print(f"N = {N}")
    print(f"{'='*70}")
    
    if p is None or q is None:
        print("Need factors to study ideal decomposition")
        return
    
    print(f"\nStudying how (p) and (q) behave in Z[√d] for various d:")
    
    for d in range(2, 50):
        if not is_squarefree(d):
            continue
        
        # Legendre symbols tell us if prime splits
        leg_p = legendre_symbol(d, p)
        leg_q = legendre_symbol(d, q)
        
        # Behavior:
        # (d/p) = 1: (p) splits as P₁·P₂
        # (d/p) = 0: (p) ramifies as P²
        # (d/p) = -1: (p) remains prime
        
        behavior_p = {1: "splits", 0: "ramifies", -1: "inert"}[leg_p]
        behavior_q = {1: "splits", 0: "ramifies", -1: "inert"}[leg_q]
        
        if leg_p == 1 or leg_q == 1:  # At least one splits
            print(f"\n  d = {d}:")
            print(f"    (p) = ({p}) {behavior_p} in Z[√{d}]")
            print(f"    (q) = ({q}) {behavior_q} in Z[√{d}]")
            
            if leg_p == 1 and leg_q == 1:
                print(f"    Both split! (N) = P₁·P₂·Q₁·Q₂")
                print(f"    This might help extract p and q from ideal structure")
    
    print(f"\nNote: Ideal factorization is well-understood")
    print(f"Question: Can we compute it without knowing p and q?")


def legendre_symbol(a: int, p: int) -> int:
    """Compute Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    
    # Use Euler's criterion
    result = pow(a, (p - 1) // 2, p)
    
    if result == p - 1:
        return -1
    return result


def ring_of_integers_elements(N: int, d: int, bound: int = 100):
    """
    Explore elements in Z[√d] and their norms.
    
    Elements are of form: a + b√d where a, b ∈ Z
    Norm: N(a + b√d) = a² - db²
    
    We want elements with norm = N or factors of N.
    """
    print(f"\n{'='*70}")
    print(f"ELEMENTS IN Z[√{d}] WITH SMALL NORM")
    print(f"{'='*70}")
    
    elements_with_norm_N = []
    elements_with_norm_factors = []
    
    for a in range(-bound, bound + 1):
        for b in range(-bound, bound + 1):
            if a == 0 and b == 0:
                continue
            
            norm = a * a - d * b * b
            
            if norm == N:
                elements_with_norm_N.append((a, b, norm))
            elif abs(norm) > 1 and abs(norm) < N:
                # Check if norm divides N
                if N % abs(norm) == 0:
                    elements_with_norm_factors.append((a, b, norm))
    
    print(f"\nElements with norm = {N}:")
    if elements_with_norm_N:
        for a, b, norm in elements_with_norm_N[:10]:
            print(f"  {a} + {b}√{d} has norm {norm}")
    else:
        print(f"  None found in range")
    
    print(f"\nElements with norm dividing {N}:")
    if elements_with_norm_factors:
        for a, b, norm in elements_with_norm_factors[:10]:
            print(f"  {a} + {b}√{d} has norm {norm}, divides {N}")
    else:
        print(f"  None found in range")
    
    return elements_with_norm_N, elements_with_norm_factors


def norm_based_factorization(N: int, p: int = None, q: int = None):
    """
    Try to factor N using norms from quadratic fields.
    
    Key insight: If we find elements α, β in Z[√d] with:
    - N(α) = p (or divides p)
    - N(β) = q (or divides q)
    
    Then we've "factored" N in the extension, which might reveal integer factors.
    """
    print(f"\n{'='*70}")
    print(f"NORM-BASED FACTORIZATION")
    print(f"{'='*70}")
    
    print(f"\nIdea: Find elements in Z[√d] with norms = factors of N")
    print(f"This would give us factorization in the extension")
    
    # For each squarefree d, look for elements with norm = p or q
    # If we can find such elements without knowing p, q, we win
    
    # But: we need to know what norm to look for
    # The norm we want is the factor itself
    
    # Observation: If we find α with N(α) = k, then k | N(α·β) for any β
    # So if we can find elements with norms dividing N...
    
    print(f"\nProblem: We don't know what norms to look for")
    print(f"We'd need to search for all elements with norms = divisors of N")
    print(f"But divisors of N = {N} are: 1, {p if p else '?'}, {q if q else '?'}, N")
    print(f"Finding elements with these norms requires knowing the factors")
    
    print(f"\nHowever:")
    print(f"If we find α with N(α) = N, then α has norm {N}")
    print(f"The element α might have useful algebraic properties")
    print(f"Even without knowing factors, α could be factorized in Z[√d]")
    print(f"And that factorization might project to Z factors")
    
    return None


def test_quadratic_field_approach():
    """Test the quadratic field approach on small semiprimes."""
    
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
    ]
    
    for N, p, q in test_cases:
        print(f"\n{'='*70}")
        print(f"TESTING N = {N} = {p} × {q}")
        print(f"{'='*70}")
        
        # Try quadratic field factorization
        result = factor_in_quadratic_field(N, max_d=100)
        
        if result:
            print(f"\n✓ SUCCESS: Found factors {result}")
        else:
            print(f"\n✗ FAILED: No quadratic field factorization found")
        
        # Study ideal factorization
        ideal_factorization(N, p, q)
        
        # Explore elements
        ring_of_integers_elements(N, d=5, bound=20)
        
        # Norm-based approach
        norm_based_factorization(N, p, q)


if __name__ == "__main__":
    print("=" * 70)
    print("HIGHER-DIMENSIONAL FACTORIZATION")
    print("Number Field Approach")
    print("=" * 70)
    
    test_quadratic_field_approach()
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}")
    
    print("""
The number field approach connects to deep algebraic number theory:

1. QUADRATIC FIELDS:
   - Every integer has a norm in Z[√d]
   - Factorization in Z[√d] can differ from Z
   - But: computing factorization in Z[√d] might be as hard as in Z

2. IDEAL THEORY:
   - Ideals always have unique factorization
   - The ideal (pq) factors as product of prime ideals
   - Prime ideals above p and q carry information
   - But: computing ideal factorization requires knowing p and q

3. THE FUNDAMENTAL ISSUE:
   - To factor in Z[√d], we need to know the factors
   - Or search for elements with specific norms
   - This is the same enumeration problem we've been avoiding

4. POTENTIAL PATHS:
   - Find d where (pq) has simple ideal factorization
   - Use class group structure to encode factors
   - Use unit group to create new factorizations
   - Connect to Pell's equation and continued fractions

5. CONNECTIONS TO KNOWN METHODS:
   - Continued fraction factorization (CFRAC) already uses quadratic fields
   - Quadratic sieve uses norm forms
   - These are subexponential, not polynomial

NEXT STEPS:
- Explore class group computation
- Study units and regulators
- Look for special d where factorization is easy
- Connect to continued fractions more deeply
""")
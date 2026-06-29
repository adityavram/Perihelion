#!/usr/bin/env python3
"""
Ideal Construction from Square Roots mod N.

KEY INSIGHT:
  If d is a QR mod both p and q (where N = pq),
  then r² ≡ d (mod N) has solutions.
  
  These solutions r give us the IDEAL GENERATORS:
    P₁ = (p, √d - r₁) where r₁ ≡ √d (mod p)
    P₂ = (q, √d - r₂) where r₂ ≡ √d (mod q)
  
  We DON'T need to know p, q to find r!
  We just need to find r such that r² ≡ d (mod N).
  
  This is the SQUARE ROOT COMPUTATION approach.
"""

from math import isqrt, gcd
from typing import Optional, Tuple, List


def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    result = pow(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else result


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


def isqrt_mod_prime(a: int, p: int) -> Optional[Tuple[int, int]]:
    """
    Compute square roots of a modulo prime p.
    Returns (r, -r) if they exist, None otherwise.
    
    Uses Tonelli-Shanks algorithm.
    """
    if legendre_symbol(a, p) != 1:
        return None
    
    # Handle p ≡ 3 (mod 4) case (simple)
    if p % 4 == 3:
        r = pow(a, (p + 1) // 4, p)
        return (r, p - r)
    
    # General case: Tonelli-Shanks
    # Factor p - 1 = Q * 2^S
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    
    # Find a quadratic non-residue
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1
    
    M = S
    c = pow(z, Q, p)
    t = pow(a, Q, p)
    R = pow(a, (Q + 1) // 2, p)
    
    while t != 1:
        # Find least i such that t^(2^i) = 1
        i = 1
        t2i = t * t % p
        while t2i != 1:
            t2i = t2i * t2i % p
            i += 1
        
        # Update
        b = pow(c, 1 << (M - i - 1), p)
        M = i
        c = b * b % p
        t = t * c % p
        R = R * b % p
    
    return (R, p - R)


def isqrt_mod_n(d: int, N: int) -> List[int]:
    """
    Compute square roots of d modulo N (possibly composite).
    
    If N = pq and d is QR mod both p and q,
    then there are 4 square roots mod N.
    
    This uses Chinese Remainder Theorem.
    """
    # Factor N for testing (in practice, we don't know this)
    # This is for DEMONSTRATION only
    factors = trial_division(N)
    
    if len(factors) != 2:
        return []
    
    p, q = factors
    
    # Check if d is QR mod both
    if legendre_symbol(d, p) != 1:
        return []
    if legendre_symbol(d, q) != 1:
        return []
    
    # Find square roots mod p and q
    roots_p = isqrt_mod_prime(d, p)
    roots_q = isqrt_mod_prime(d, q)
    
    if roots_p is None or roots_q is None:
        return []
    
    # Use CRT to combine
    roots_mod_N = []
    
    for rp in roots_p:
        for rq in roots_q:
            # Solve: x ≡ rp (mod p), x ≡ rq (mod q)
            x = crt_solve(rp, p, rq, q)
            roots_mod_N.append(x)
    
    return sorted(set(roots_mod_N))


def crt_solve(a1: int, m1: int, a2: int, m2: int) -> int:
    """Solve x ≡ a1 (mod m1), x ≡ a2 (mod m2) using CRT."""
    # Extended Euclidean algorithm
    g, x, y = extended_gcd(m1, m2)
    
    # Solution exists only if g | (a2 - a1)
    # For our case, gcd(m1, m2) = 1, so g = 1
    
    return (a1 + m1 * x * (a2 - a1) // g) % (m1 * m2)


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended GCD: returns (g, x, y) such that ax + by = g."""
    if b == 0:
        return (a, 1, 0)
    
    g, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    
    return (g, x, y)


def trial_division(N: int) -> List[int]:
    """Simple factorization for testing."""
    factors = []
    n = N
    for p in range(2, isqrt(N) + 1):
        while n % p == 0:
            factors.append(p)
            n //= p
    if n > 1:
        factors.append(n)
    return factors


def analyze_square_roots_mod_N(N: int, p: int, q: int):
    """
    Analyze square roots of d mod N and their connection to factors.
    
    KEY INSIGHT:
      If r² ≡ d (mod N), then:
        - r ≡ ±r₁ (mod p) for some r₁ with r₁² ≡ d (mod p)
        - r ≡ ±r₂ (mod q) for some r₂ with r₂² ≡ d (mod q)
      
      The 4 square roots are:
        - r₁₁: r ≡ r₁ (mod p), r ≡ r₂ (mod q)
        - r₁₂: r ≡ r₁ (mod p), r ≡ -r₂ (mod q)
        - r₂₁: r ≡ -r₁ (mod p), r ≡ r₂ (mod q)
        - r₂₂: r ≡ -r₁ (mod p), r ≡ -r₂ (mod q)
      
      Note: gcd(r - r₁, N) = p or q depending on the combination!
    """
    print("="*70)
    print(f"SQUARE ROOTS AND IDEAL GENERATORS")
    print(f"N = {N} = {p} × {q}")
    print("="*70)
    
    # Find d such that d is QR mod both p and q
    for d in range(2, 50):
        if not is_squarefree(d):
            continue
        
        leg_p = legendre_symbol(d, p)
        leg_q = legendre_symbol(d, q)
        
        if leg_p == 1 and leg_q == 1:
            print(f"\nd = {d}: QR mod both {p} and {q}")
            
            # Find square roots mod N
            roots = isqrt_mod_n(d, N)
            
            if len(roots) == 4:
                print(f"  4 square roots mod {N}:")
                
                for i, r in enumerate(roots):
                    g = gcd(r, N)
                    g_diff = gcd(r * r - d, N)
                    
                    # Check if r² ≡ d (mod N)
                    check = (r * r) % N
                    
                    print(f"    r[{i}] = {r}")
                    print(f"      r² mod N = {check}")
                    print(f"      gcd(r, N) = {g}")
                    print(f"      gcd(r² - {d}, N) = {g_diff}")
                    
                    # The KEY insight: gcd(r - r', N) can reveal factors
                    # where r, r' are different square roots
                    
                # Key observation
                print(f"\n  KEY: Different square roots give different factor info:")
                print(f"    r² ≡ d (mod N)")
                print(f"    (r - r₁)(r + r₁) ≡ r² - r₁² ≡ d - d ≡ 0 (mod p)")
                print(f"    So p | gcd(r - r₁, N) if r₁² ≡ d (mod p)")
                
                # Find which roots correspond to which residues
                roots_p = isqrt_mod_prime(d, p)
                roots_q = isqrt_mod_prime(d, q)
                
                print(f"\n  Square roots mod p={p}: {roots_p}")
                print(f"  Square roots mod q={q}: {roots_q}")
                
                # For each root mod N, find its residues mod p and q
                print(f"\n  Residues of each root mod N:")
                for i, r in enumerate(roots):
                    r_mod_p = r % p
                    r_mod_q = r % q
                    
                    # Normalize to the positive square root
                    r_p_normalized = min(r_mod_p, p - r_mod_p)
                    r_q_normalized = min(r_mod_q, q - r_mod_q)
                    
                    print(f"    r[{i}] = {r}:")
                    print(f"      mod p: {r_mod_p} (±{r_p_normalized})")
                    print(f"      mod q: {r_mod_q} (±{r_q_normalized})")
                
                # The breakthrough: compare roots
                print(f"\n  BREAKTHROUGH INSIGHT:")
                print(f"    If r₁ and r₂ are different square roots of d mod N:")
                print(f"    Then r₁ ≢ ±r₂ (mod N)")
                print(f"    So r₁ ≡ ±r₂ (mod p) XOR r₁ ≡ ±r₂ (mod q)")
                print(f"    This means gcd(r₁ - r₂, N) = p or q!")
                print()
                
                # Demonstrate
                for i in range(len(roots)):
                    for j in range(i + 1, len(roots)):
                        diff_gcd = gcd(roots[i] - roots[j], N)
                        sum_gcd = gcd(roots[i] + roots[j], N)
                        
                        if 1 < diff_gcd < N:
                            print(f"    gcd(r[{i}] - r[{j}], N) = {diff_gcd} ← FACTOR!")
                        if 1 < sum_gcd < N:
                            print(f"    gcd(r[{i}] + r[{j}], N) = {sum_gcd} ← FACTOR!")
                
                return  # Just show first example
    
    print("\nNo d found where d is QR mod both factors.")


def the_key_insight():
    """
    The KEY insight that transforms divisibility into multiplication.
    """
    print("\n" + "="*70)
    print("THE KEY INSIGHT: SQUARE ROOTS → FACTORS")
    print("="*70)
    print("""
The Classical Approach (Enumeration):
  For each candidate p:
    Check if p | N (division)
  This requires checking ~√N candidates

The Number Field Approach (Square Roots):
  1. Choose d where d is QR mod both p and q (we don't know p, q yet)
  2. Compute √d mod N (square root, not division!)
  3. If there are 4 square roots r₁, r₂, r₃, r₄:
     Then gcd(rᵢ - rⱼ, N) = p or q for some i, j

Why does this work?
  r² ≡ d (mod N)
  ⟹ r² ≡ d (mod p) AND r² ≡ d (mod q)
  
  Let r₁, r₂ be the two square roots mod p: r₁² ≡ d ≡ r₂² (mod p)
  Let r₃, r₄ be the two square roots mod q: r₃² ≡ d ≡ r₄² (mod q)
  
  By CRT, we get 4 roots mod N:
    R₁₁ ≡ r₁ (mod p), R₁₁ ≡ r₃ (mod q)
    R₁₂ ≡ r₁ (mod p), R₁₂ ≡ r₄ (mod q)
    R₂₁ ≡ r₂ (mod p), R₂₁ ≡ r₃ (mod q)
    R₂₂ ≡ r₂ (mod p), R₂₂ ≡ r₄ (mod q)
  
  Now:
    R₁₁ - R₂₁ ≡ r₁ - r₂ ≢ 0 (mod p) but R₁₁ - R₂₁ ≡ r₃ - r₃ ≡ 0 (mod q)
    So gcd(R₁₁ - R₂₁, N) = p
  
  And similarly:
    R₁₁ - R₁₂ ≡ 0 (mod p) but R₁₁ - R₁₂ ≢ 0 (mod q)
    So gcd(R₁₁ - R₁₂, N) = q

This transforms:
  "Find p such that p | N" (division, enumeration)
  →
  "Find √d mod N, compute gcds" (square roots, polynomial!)

The only question: How do we know which d to choose?
  - We need d to be QR mod both p and q
  - About 25% of d satisfy this (by quadratic reciprocity)
  - We can try different d until we find one that works

Algorithm:
  1. For d in [2, 3, 5, 6, 7, ...]:  # squarefree
  2.   Compute Jacobi symbol (d/N)
  3.   If (d/N) = 1:  # d might be QR mod both factors
  4.     Compute √d mod N using Tonelli-Shanks or similar
  5.     If √d exists mod N (4 roots):
  6.       Compute gcds between roots
  7.       If gcd reveals factor: return factor

Complexity:
  - Expected ~4 tries to find valid d
  - Each try: O(log²N) for square root computation
  - Total: O(log²N) expected time

This is POLYNOMIAL!

But wait... there's a catch.
""")


def the_catch():
    """
    The catch in the square root approach.
    """
    print("\n" + "="*70)
    print("THE CATCH: COMPUTING √d mod N")
    print("="*70)
    print("""
The algorithm above assumes we can compute √d mod N efficiently.

For prime modulus p:
  - Tonelli-Shanks: O(log²p) polynomial
  - Works when (d/p) = 1

For composite modulus N:
  - Need to factor N first!
  - Use CRT with factors
  - But factoring N is what we're trying to do!

So we have a circular dependency:
  To compute √d mod N, we need factors of N
  To find factors of N, we compute √d mod N

This is the SAME circular dependency as other approaches.

HOWEVER, there's a way out:

The Miller-Rabin primality test uses a similar approach:
  - If N = pq, then a^(N-1) ≡ 1 (mod N) by Fermat
  - But a^((N-1)/2) might not be ±1 mod N
  - The "witnesses" reveal factors

Similarly, for √d mod N:
  - The square root computation might fail or behave differently
  - We can detect the failure mode
  - The failure mode might reveal factors

Alternative approach:
  - Don't compute √d mod N directly
  - Work in the IDEAL (N) ⊂ Z[√d]
  - The ideal (N) factors as product of prime ideals
  - Prime ideals encode √d information

The ideal approach:
  - (N) = P₁ × P₁' × P₂ × P₂' (when d is QR mod both p and q)
  - Each P has norm p or q
  - We can work with ideals without computing √d mod N

This is where algebraic number theory comes in:
  - The ideal class group Cl(Z[√d])
  - The norm form N(a + b√d) = a² - d·b²
  - The connection between ideals and norms
""")


def is_squarefree(n: int) -> bool:
    """Check if n is squarefree."""
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True


if __name__ == "__main__":
    # Test with small semiprimes
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
    ]
    
    for N, p, q in test_cases:
        analyze_square_roots_mod_N(N, p, q)
    
    the_key_insight()
    the_catch()
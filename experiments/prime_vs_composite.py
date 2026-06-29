#!/usr/bin/env python3
"""
Critical Analysis: Consistency test for prime vs composite N.

Key finding: For composite N, consistency fraction = 0% for all tested d.

Question: What about prime N?
If consistent fraction is 100% for prime N, we have a polynomial-time primality test!
"""

from math import gcd
import random


def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    result = pow(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else 1


def power_in_quadratic_field(a: int, b: int, d: int, k: int, N: int):
    """Compute (a + b√d)^k mod N."""
    if k == 0:
        return (1, 0)
    
    result_0, result_1 = 1, 0
    base_0, base_1 = a % N, b % N
    
    while k > 0:
        if k % 2 == 1:
            new_0 = (result_0 * base_0 + result_1 * base_1 * d) % N
            new_1 = (result_0 * base_1 + result_1 * base_0) % N
            result_0, result_1 = new_0, new_1
        
        new_base_0 = (base_0 * base_0 + base_1 * base_1 * d) % N
        new_base_1 = (2 * base_0 * base_1) % N
        base_0, base_1 = new_base_0, new_base_1
        
        k //= 2
    
    return (result_0, result_1)


def test_prime_vs_composite():
    """
    Test consistency fraction for prime vs composite N.
    
    Key question:
      - Prime N: Is consistency fraction = 100%?
      - Composite N: Is consistency fraction = 0%?
      
    If so, we have a polynomial-time primality test!
    """
    print("="*70)
    print("PRIME VS COMPOSITE CONSISTENCY TEST")
    print("="*70)
    
    # Test prime N
    primes = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    print("\nTesting PRIME N:")
    print("N\td\tConsistent\tTotal")
    print("-" * 40)
    
    for N in primes:
        # Choose d such that (d/N) = 1 (QR mod N)
        for d in range(2, 20):
            if legendre_symbol(d, N) == 1:
                count_consistent = 0
                count_total = 0
                
                for a in range(1, min(15, N)):
                    for b in range(1, min(15, N)):
                        count_total += 1
                        r0, r1 = power_in_quadratic_field(a, b, d, N - 1, N)
                        
                        if r1 == 0 and r0 == 1:
                            count_consistent += 1
                
                frac = count_consistent / count_total if count_total > 0 else 0
                print(f"{N}\t{d}\t{count_consistent}\t\t{count_total}")
                break
    
    # Test composite N
    composites = [143, 391, 899, 1517, 3127]  # Semiprimes
    
    print("\n\nTesting COMPOSITE N:")
    print("N\t\td\tConsistent\tTotal")
    print("-" * 40)
    
    for N in composites:
        # Choose d such that (d/N) = 1 (Jacobi = 1)
        for d in range(2, 20):
            jacobi = jacobi_symbol(d, N)
            if jacobi == 1:
                count_consistent = 0
                count_total = 0
                
                for a in range(1, min(15, N)):
                    for b in range(1, min(15, N)):
                        if gcd(a, N) > 1:
                            continue
                        count_total += 1
                        r0, r1 = power_in_quadratic_field(a, b, d, N - 1, N)
                        
                        if r1 == 0 and r0 == 1:
                            count_consistent += 1
                
                print(f"{N}\t\t{d}\t{count_consistent}\t\t{count_total}")
                break


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


def the_critical_question():
    """
    The critical question: Can we distinguish prime from composite in polynomial time?
    """
    print("\n" + "="*70)
    print("THE CRITICAL QUESTION")
    print("="*70)
    print("""
For Miller-Rabin:
  - Prime N: a^(N-1) ≡ 1 (mod N) for ALL a coprime to N
  - Composite N: a^(N-1) ≢ 1 (mod N) for at least 75% of a
  - Test: Single exponentiation per a
  - Polynomial time: O(log³N) per test

For Z[√d]/(N):
  - We found: Composite N has 0% consistency in tests
  - Question: What about prime N?
  
If:
  - Prime N: (a + b√d)^(N-1) ≡ 1 + 0√d (mod N) for all a, b
  - Composite N: (a + b√d)^(N-1) ≢ 1 + 0√d (mod N) for all a, b

Then we have:
  - Polynomial-time primality test!
  - BUT: This doesn't factor N, only tests primality
  
Wait... let me check if this is even true for prime N.

Theorem: For prime p and (d/p) = 1, Z[√d]/(p) ≅ Z/(p) × Z/(p)
  - In this case, (a + b√d)^(p-1) should give (1, 1) or something else
  - NOT necessarily 1 + 0√d

Let me compute more carefully...
""")


def test_prime_n_detailed():
    """
    Detailed test for prime N.
    """
    print("\n" + "="*70)
    print("DETAILED TEST FOR PRIME N")
    print("="*70)
    
    # Test N = 13 with d = 3 ((3/13) = 1)
    N = 13
    d = 3
    
    print(f"\nN = {N} (prime), d = {d} (({d}/{N}) = {legendre_symbol(d, N)})")
    print("\n(a + b√d)^(N-1) mod N:")
    print("a\tb\tr₀\tr₁")
    print("-" * 30)
    
    for a in range(1, 5):
        for b in range(1, 5):
            r0, r1 = power_in_quadratic_field(a, b, d, N - 1, N)
            print(f"{a}\t{b}\t{r0}\t{r1}")
    
    # Test N = 17 with d = 2 ((2/17) = 1)
    N = 17
    d = 2
    
    print(f"\nN = {N} (prime), d = {d} (({d}/{N}) = {legendre_symbol(d, N)})")
    print("\n(a + b√d)^(N-1) mod N:")
    print("a\tb\tr₀\tr₁")
    print("-" * 30)
    
    for a in range(1, 5):
        for b in range(1, 5):
            r0, r1 = power_in_quadratic_field(a, b, d, N - 1, N)
            print(f"{a}\t{b}\t{r0}\t{r1}")
    
    # Test with (d/p) = -1
    N = 13
    d = 2  # (2/13) = -1
    
    print(f"\nN = {N} (prime), d = {d} (({d}/{N}) = {legendre_symbol(d, N)})")
    print("Note: d is NR mod N, so Z[√d]/(N) is a field extension")
    print("\n(a + b√d)^(N-1) mod N:")
    print("a\tb\tr₀\tr₁")
    print("-" * 30)
    
    for a in range(1, 5):
        for b in range(1, 5):
            r0, r1 = power_in_quadratic_field(a, b, d, N - 1, N)
            print(f"{a}\t{b}\t{r0}\t{r1}")


if __name__ == "__main__":
    test_prime_vs_composite()
    the_critical_question()
    test_prime_n_detailed()
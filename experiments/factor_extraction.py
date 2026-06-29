#!/usr/bin/env python3
"""
Can the result reveal factors?

For composite N, (a + b√d)^(N-1) ≢ (1, 0).
The result is (r₀, r₁) where r₁ ≠ 0 or r₀ ≠ 1.

Can gcd(r₁, N) or gcd(r₀ - 1, N) reveal factors?
"""

from math import gcd


def legendre_symbol(a: int, p: int) -> int:
    if a % p == 0:
        return 0
    result = pow(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else 1


def power_in_quadratic_field(a: int, b: int, d: int, k: int, N: int):
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


def can_result_reveal_factors(N: int, p: int, q: int):
    """
    Test if (r₀, r₁) from (a + b√d)^(N-1) can reveal factors.
    """
    print("="*70)
    print(f"CAN RESULT REVEAL FACTORS?")
    print(f"N = {N} = {p} × {q}")
    print("="*70)
    
    # Choose d with various Legendre symbol combinations
    for d in range(2, 10):
        leg_p = legendre_symbol(d, p)
        leg_q = legendre_symbol(d, q)
        
        if leg_p == 0 or leg_q == 0:
            continue
        
        print(f"\nd = {d}: ({d}/{p}) = {leg_p}, ({d}/{q}) = {leg_q}")
        
        count_factor_via_r0 = 0
        count_factor_via_r1 = 0
        count_total = 0
        
        # Test various (a, b)
        for a in range(1, min(15, p)):
            for b in range(1, min(15, q)):
                if gcd(a, N) > 1:
                    continue
                
                r0, r1 = power_in_quadratic_field(a, b, d, N - 1, N)
                
                count_total += 1
                
                # Try to reveal factors
                g0 = gcd(r0 - 1, N)
                g1 = gcd(r1, N)
                
                if 1 < g0 < N:
                    count_factor_via_r0 += 1
                
                if 1 < g1 < N:
                    count_factor_via_r1 += 1
        
        print(f"  Tested {count_total} elements")
        print(f"  Factors via gcd(r₀ - 1, N): {count_factor_via_r0} ({100*count_factor_via_r0/count_total:.1f}%)")
        print(f"  Factors via gcd(r₁, N): {count_factor_via_r1} ({100*count_factor_via_r1/count_total:.1f}%)")


def miller_rabin_analogy():
    """
    Miller-Rabin reveals factors via gcd(a^((N-1)/2) ± 1, N).
    Can we do something similar?
    """
    print("\n" + "="*70)
    print("MILLER-RABIN ANALOGY")
    print("="*70)
    print("""
Miller-Rabin:
  - For composite N = pq with p ≠ q:
    - a^(N-1) ≠ 1 (mod N) for most a
    - Write N-1 = 2^s · d with d odd
    - Compute a^d, a^(2d), a^(4d), ...
    - If a^(2^r · d) ≠ ±1 but a^(2^(r+1) · d) = 1:
      - gcd(a^(2^r · d) - 1, N) reveals a factor!

Analogous in Z[√d]/(N):
  - (a + b√d)^(N-1) ≠ (1, 0) for composite N
  - Can we extract a factor from the result (r₀, r₁)?
  
Options:
  1. gcd(r₀ - 1, N)
  2. gcd(r₁, N)
  3. gcd(r₀ ± r₁, N)
  4. Some combination

Let's test these...
""")


def test_various_extractions(N: int, p: int, q: int):
    """
    Test various ways to extract factors from (r₀, r₁).
    """
    print("\n" + "="*70)
    print(f"TESTING VARIOUS FACTOR EXTRACTIONS")
    print(f"N = {N} = {p} × {q}")
    print("="*70)
    
    for d in range(2, 5):
        leg_p = legendre_symbol(d, p)
        leg_q = legendre_symbol(d, q)
        
        print(f"\nd = {d} (Legendre: {leg_p}, {leg_q})")
        print("Testing: gcd(r₀-1,N), gcd(r₁,N), gcd(r₀+r₁,N), gcd(r₀-r₁,N)")
        
        successes = [0, 0, 0, 0]
        count = 0
        
        for a in range(1, min(10, p)):
            for b in range(1, min(10, q)):
                if gcd(a, N) > 1:
                    continue
                
                count += 1
                r0, r1 = power_in_quadratic_field(a, b, d, N - 1, N)
                
                g0 = gcd(r0 - 1, N)
                g1 = gcd(r1, N)
                g2 = gcd(r0 + r1, N)
                g3 = gcd(abs(r0 - r1), N)
                
                if 1 < g0 < N:
                    successes[0] += 1
                if 1 < g1 < N:
                    successes[1] += 1
                if 1 < g2 < N:
                    successes[2] += 1
                if 1 < g3 < N:
                    successes[3] += 1
        
        print(f"  Success rates:")
        print(f"    gcd(r₀ - 1, N): {100*successes[0]/count:.1f}%")
        print(f"    gcd(r₁, N):     {100*successes[1]/count:.1f}%")
        print(f"    gcd(r₀ + r₁, N): {100*successes[2]/count:.1f}%")
        print(f"    gcd(r₀ - r₁, N): {100*successes[3]/count:.1f}%")


if __name__ == "__main__":
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
    ]
    
    miller_rabin_analogy()
    
    for N, p, q in test_cases:
        can_result_reveal_factors(N, p, q)
    
    for N, p, q in test_cases:
        test_various_extractions(N, p, q)
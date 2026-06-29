#!/usr/bin/env python3
"""
Find ACTUAL solutions to a² - d·b² = kp.

The issue: Solutions might have |a|, |b| much larger than √p.
"""

from math import isqrt, gcd


def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    result = pow(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else 1


def find_solutions(p: int, d: int, max_search: int = 10000) -> list:
    """
    Find all solutions to a² - d·b² = k·p for various k.
    
    Returns list of (a, b, k, norm) tuples.
    """
    solutions = []
    
    for b in range(1, max_search):
        for a in range(-max_search, max_search + 1):
            norm = a * a - d * b * b
            if norm != 0 and norm % p == 0:
                k = norm // p
                solutions.append((a, b, k, norm))
                
                if len(solutions) >= 10:  # First 10 solutions
                    return solutions
    
    return solutions


def analyze_solution_sizes():
    """
    Analyze the sizes of solutions to a² - d·b² = kp.
    """
    print("="*70)
    print("SOLUTION SIZE ANALYSIS")
    print("="*70)
    print("""
Finding solutions to a² - d·b² = k·p for various primes p.
Checking how large |a| and |b| need to be.
""")
    
    test_cases = [
        (11, 3),   # d=3 is QR mod 11
        (13, 3),   # d=3 is QR mod 13
        (17, 2),   # d=2 is QR mod 17
        (23, 2),   # d=2 is QR mod 23
        (29, 5),   # d=5 is QR mod 29
        (31, 5),   # d=5 is QR mod 31
        (37, 4),   # d=4 is QR mod 37 (actually 4 is always QR)
        (41, 4),   # d=4 is QR mod 41
        (53, 2),   # d=2 is QR mod 53
        (59, 2),   # d=2 is QR mod 59
        (67, 2),   # d=2 is QR mod 67
        (71, 2),   # d=2 is QR mod 71
        (79, 3),   # d=3 is QR mod 79
        (83, 3),   # d=3 is QR mod 83
        (97, 5),   # d=5 is QR mod 97
        (101, 5),  # d=5 is QR mod 101
    ]
    
    for p, d in test_cases:
        leg = legendre_symbol(d, p)
        if leg != 1:
            print(f"\np={p}, d={d}: NOT QR, skipping")
            continue
        
        solutions = find_solutions(p, d, max_search=500)
        
        print(f"\np={p}, d={d} (QR):")
        print(f"  First 5 solutions:")
        
        for a, b, k, norm in solutions[:5]:
            print(f"    a={a:6d}, b={b:4d}, norm={norm:8d} = {k}·{p}")
        
        if solutions:
            min_a = min(abs(a) for a, b, k, norm in solutions[:10])
            max_a = max(abs(a) for a, b, k, norm in solutions[:10])
            min_b = min(b for a, b, k, norm in solutions[:10])
            max_b = max(b for a, b, k, norm in solutions[:10])
            
            print(f"  Size range: |a| ∈ [{min_a}, {max_a}], b ∈ [{min_b}, {max_b}]")
            print(f"  √p ≈ {isqrt(p)}, max |a| / √p ≈ {max_a / isqrt(p):.2f}")


def fundamental_solution_analysis():
    """
    The fundamental solutions are the smallest.
    """
    print("\n" + "="*70)
    print("FUNDAMENTAL SOLUTION ANALYSIS")
    print("="*70)
    print("""
For a² - d·b² = k·p, the fundamental solutions are the smallest.

In Z[√d], units satisfy a² - d·b² = ±1.
All other solutions are obtained by multiplying by units.

For d > 0 (indefinite), units grow exponentially.

So the sizes of fundamental solutions determine the complexity.
""")
    
    # Find fundamental solutions for various (p, d)
    test_cases = [
        (11, 3), (13, 3), (17, 2), (23, 2), (29, 5),
        (31, 5), (37, 4), (41, 4), (53, 2), (59, 2),
    ]
    
    print("\nPrime\td\t√p\tMin |a|\tMin b\tRatio")
    print("-" * 50)
    
    for p, d in test_cases:
        if legendre_symbol(d, p) != 1:
            continue
        
        # Find first solution with minimal |a| and b
        best_a = None
        best_b = None
        
        for b in range(1, p):
            for a in range(-p, p + 1):
                norm = a * a - d * b * b
                if norm % p == 0 and norm != 0:
                    if best_a is None or abs(a) < abs(best_a):
                        best_a = a
                        best_b = b
        
        if best_a is not None:
            ratio = abs(best_a) / isqrt(p)
            print(f"{p}\t{d}\t{isqrt(p)}\t{abs(best_a)}\t{best_b}\t{ratio:.2f}")


def the_fundamental_problem():
    """
    The fundamental problem: solutions grow with p.
    """
    print("\n" + "="*70)
    print("THE FUNDAMENTAL PROBLEM")
    print("="*70)
    print("""
Observation: Fundamental solutions for a² - d·b² = k·p have:
  - |a| ≈ O(p) for some solutions
  - Even the minimal |a| grows roughly linearly with p

Why?
  - The norm equation a² - d·b² = k·p
  - For b = 1: a² = d + k·p, so a ≈ √(d + k·p)
  - For k = 1: a ≈ √(d + p) ≈ √p
  
Wait, that's √p, not p!

Let me reconsider...

Actually, for b=1 and k such that d + k·p is a perfect square:
  - a² = d + k·p
  - We need d + k·p to be a square
  
For d=2, p=17:
  - k=1: 2 + 17 = 19 (not square)
  - k=2: 2 + 34 = 36 = 6² ← square!
  - So a=6, b=1, norm = 36 - 2 = 34 = 2·17 ✓

For d=2, p=53:
  - Need k such that 2 + 53k is a square
  - k=2: 2 + 106 = 108 (not square)
  - k=6: 2 + 318 = 320 (not square)
  - k=8: 2 + 424 = 426 (not square)
  - ... need to check many k
  
The problem: Finding k such that d + k·p is a square is like finding
square roots mod p, which requires O(p) operations in the worst case.

Actually, by quadratic reciprocity:
  - d + k·p is square for some k with 0 < k < p/2
  - Expected k is O(p), but we need to search O(p) values
  
This brings us back to O(p) = O(√N) complexity!

So the "breakthrough" of finding factor-norms is NOT polynomial-time.

The complexity is:
  - Search range for a: O(√p)
  - Search range for b: O(√p)
  - But the solutions we want have k that may require O(p) trials to find!
  
Actually, wait. For b=1:
  - a² = d + k·p
  - a² ≡ d (mod p)
  - This has 2 solutions mod p if (d/p) = 1
  - So a ≡ ±√d (mod p)
  
By Tonelli-Shanks, we can find √d mod p in O(log²p) time!

So for b=1:
  - Find √d mod p: O(log²p)
  - Then a = √d + k·p for various k
  - The smallest positive a is when k = floor(√d / p)
  - This gives a ≈ √d if √d < p
  
Wait, that's not right either...

Let me think more carefully:

If r² ≡ d (mod p) and (d/p) = 1:
  - Then r is a square root of d mod p
  - The smallest such r is O(√p) (by number theory)
  
Actually, the smallest r such that r² ≡ d (mod p) can be as small as O(√p).

No wait, that's not true. The smallest r is just the smallest positive
integer such that r² - d is divisible by p, which could be anywhere from
1 to p-1.

Hmm, let me just compute the actual smallest solutions for various p.
""")


if __name__ == "__main__":
    analyze_solution_sizes()
    fundamental_solution_analysis()
    the_fundamental_problem()
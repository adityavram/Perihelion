#!/usr/bin/env python3
"""
Find the SMALLEST solutions to a² - d·b² = kp.

Key insight: Small a values require large b values!
"""

from math import isqrt, gcd


def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    result = pow(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else 1


def find_smallest_solutions(p: int, d: int, max_b: int = 1000) -> list:
    """
    Find smallest solutions to a² - d·b² = kp.
    
    Returns list of (a, b, k, norm) sorted by |norm|.
    """
    solutions = []
    
    for b in range(1, max_b + 1):
        for a in range(-isqrt(d * b * b + 100 * p), isqrt(d * b * b + 100 * p) + 1):
            norm = a * a - d * b * b
            if norm != 0 and norm % p == 0:
                k = norm // p
                solutions.append((a, b, k, norm))
    
    # Sort by |a| (smallest first)
    solutions.sort(key=lambda x: abs(x[0]))
    
    return solutions


def analyze_smallest_solutions():
    """
    Analyze the smallest solutions for various primes.
    """
    print("="*70)
    print("SMALLEST SOLUTION ANALYSIS")
    print("="*70)
    print("""
Finding the smallest solutions to a² - d·b² = k·p.

Key question: How large do b and |a| need to be?
""")
    
    test_cases = [
        (11, 3), (13, 3), (17, 2), (23, 2), (29, 5),
        (31, 5), (37, 4), (41, 4), (71, 2), (83, 3),
        (101, 5),
    ]
    
    print("\nPrime\td\tMin |a|\tMin b\tMax |a|\tMax b\t√p")
    print("-" * 60)
    
    for p, d in test_cases:
        if legendre_symbol(d, p) != 1:
            continue
        
        solutions = find_smallest_solutions(p, d, max_b=100)
        
        if solutions:
            # Get the solution with smallest |a|
            sol_min_a = solutions[0]
            
            # Get solutions with various sizes
            all_a = [abs(a) for a, b, k, norm in solutions]
            all_b = [b for a, b, k, norm in solutions]
            
            print(f"{p}\t{d}\t{abs(sol_min_a[0])}\t{sol_min_a[1]}\t{max(all_a)}\t{max(all_b)}\t{isqrt(p)}")
            
            # Show the smallest few solutions
            print(f"  First 3 solutions:")
            for a, b, k, norm in solutions[:3]:
                print(f"    a={a:4d}, b={b:3d}, norm={norm:8d} = {k}·{p}")


def proper_search_range():
    """
    Determine the correct search range for finding factor-norm elements.
    """
    print("\n" + "="*70)
    print("PROPER SEARCH RANGE")
    print("="*70)
    print("""
For a² - d·b² = k·p, the smallest solutions satisfy:

If |a| is small:
  - a² ≈ d·b²
  - |a| ≈ √d · b
  - So |a| ≈ O(b)

If |a| ≈ b, then:
  - a² - d·b² = k·p
  - (1-d)·b² ≈ k·p
  - So b ≈ √(k·p / |1-d|)

For small k and d not close to 1:
  - b ≈ √(O(p)) ≈ O(√p)
  - |a| ≈ O(√p)

So the search range should be O(√p) for BOTH a and b!

Expected complexity: O(√p × √p) = O(p) = O(√N)

This is the SAME as trial division!

UNLESS... there's a way to find small solutions faster.

Let me verify with experiments:
""")
    
    test_cases = [
        (11, 13, 3),
        (17, 23, 2),
        (29, 31, 5),
        (37, 41, 4),
    ]
    
    for p, q, d in test_cases:
        N = p * q
        
        # Search in range √p
        sqrt_p = isqrt(p) + 1
        
        print(f"\nN = {N} = {p} × {q}, d = {d}")
        print(f"Search range: a, b ∈ [1, {sqrt_p}]")
        
        count = 0
        first_found = None
        
        for b in range(1, sqrt_p + 1):
            for a in range(-sqrt_p, sqrt_p + 1):
                norm = a * a - d * b * b
                if norm != 0:
                    g = gcd(abs(norm), N)
                    if 1 < g < N:
                        count += 1
                        if first_found is None:
                            first_found = (a, b, norm, g)
        
        total = (2 * sqrt_p + 1) * sqrt_p
        prob = count / total if total > 0 else 0
        
        print(f"  Factor-revealing elements: {count} / {total} = {100*prob:.2f}%")
        if first_found:
            print(f"  First: a={first_found[0]}, b={first_found[1]}, norm={first_found[2]}, gcd={first_found[3]}")
        
        # Now search in larger range
        larger_range = max(10, 2 * sqrt_p)
        print(f"\nSearch range: a, b ∈ [1, {larger_range}]")
        
        count2 = 0
        for b in range(1, larger_range + 1):
            for a in range(-larger_range, larger_range + 1):
                norm = a * a - d * b * b
                if norm != 0:
                    g = gcd(abs(norm), N)
                    if 1 < g < N:
                        count2 += 1
        
        total2 = (2 * larger_range + 1) * larger_range
        prob2 = count2 / total2 if total2 > 0 else 0
        
        print(f"  Factor-revealing elements: {count2} / {total2} = {100*prob2:.2f}%")


def the_ultimate_question():
    """
    Can we do better than O(√p) search?
    """
    print("\n" + "="*70)
    print("THE ULTIMATE QUESTION")
    print("="*70)
    print("""
The complexity analysis:

1. Finding factor-norm elements requires searching O(√p × √p) = O(p) pairs (a, b).

2. For balanced semiprimes: p ≈ √N, so complexity is O(√N).

3. This is the SAME as trial division!

4. BUT: Trial division checks O(√N / ln √N) primes, while we check O(√N) pairs.

5. Our method has a worse constant factor, but:

6. We're computing norms (polynomial), not checking primality.

7. More importantly: Can we find the solutions FASTER than search?

Key insight from continued fractions (CFRAC):

CFRAC finds (a, b) with |a² - d·b²| < 2√(dN) by using convergents to √(d/d'N).

For d' = 1, this gives |a² - d·b²| < 2√(dN).

If we set d such that the convergents give norms sharing factors with N...

This is EXACTLY what CFRAC does! It's subexponential, not polynomial.

So our "breakthrough" is actually a rediscovery of CFRAC's key step!

The question remains: Can we do BETTER than CFRAC?

CFRAC's complexity: L[N] = exp(√(log N log log N))

Our norm search: O(√N) = O(exp(log N / 2))

For large N: L[N] << √N

So CFRAC is BETTER than our direct search!

The next question: Is there a polynomial-time variant?
""")


if __name__ == "__main__":
    analyze_smallest_solutions()
    proper_search_range()
    the_ultimate_question()
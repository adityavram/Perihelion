#!/usr/bin/env python3
"""
Direction 9: Miller-Rabin Analogy for Z[√d]/(N).

Key question: Can we define "witnesses" in Z[√d]/(N) analogous to Miller-Rabin?

Miller-Rabin structure:
  - Computes a^(N-1) mod N (polynomial)
  - For composite N, result differs from prime N
  - Witnesses reveal factors via gcd(a^((N-1)/2) ± 1, N)

Our structure:
  - Compute (a + b√d)^(N-1) mod N (polynomial)
  - Result = r₀ + r₁√d
  - Can r₁ ≠ 0 or other properties reveal factors?

The critical difference:
  - Miller-Rabin: a^(N-1) is ALWAYS defined for any a
  - Our case: (a + b√d)^(N-1) is defined, but do we get "witnesses"?
"""

from math import gcd, isqrt
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


def power_in_quadratic_field(a: int, b: int, d: int, k: int, N: int):
    """
    Compute (a + b√d)^k mod N.
    
    Returns (r₀, r₁) where result = r₀ + r₁√d.
    
    Uses binary exponentiation in the ring Z[√d]/(N).
    """
    if k == 0:
        return (1, 0)
    
    result_0, result_1 = 1, 0  # Identity
    base_0, base_1 = a % N, b % N
    
    while k > 0:
        if k % 2 == 1:
            # Multiply result by base
            # (r₀ + r₁√d)(b₀ + b₁√d) = r₀b₀ + r₀b₁√d + r₁b₀√d + r₁b₁d
            #                         = (r₀b₀ + r₁b₁d) + (r₀b₁ + r₁b₀)√d
            new_0 = (result_0 * base_0 + result_1 * base_1 * d) % N
            new_1 = (result_0 * base_1 + result_1 * base_0) % N
            result_0, result_1 = new_0, new_1
        
        # Square the base
        # (b₀ + b₁√d)² = b₀² + 2b₀b₁√d + b₁²d
        new_base_0 = (base_0 * base_0 + base_1 * base_1 * d) % N
        new_base_1 = (2 * base_0 * base_1) % N
        base_0, base_1 = new_base_0, new_base_1
        
        k //= 2
    
    return (result_0, result_1)


def norm_in_quadratic_field(a: int, b: int, d: int) -> int:
    """Norm of a + b√d: N = a² - d·b²."""
    return a * a - d * b * b


def miller_rabin_witnesses(N: int, a: int) -> tuple:
    """
    Standard Miller-Rabin test.
    
    Returns (is_probable_prime, factor_if_found).
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
        x_prev = x
        x = (x * x) % N
        
        if x == 1:
            # Found a factor!
            return (False, gcd(x_prev - 1, N))
        
        if x == N - 1:
            return (True, None)
    
    return (False, None)


def quadratic_field_witnesses(N: int, d: int, a: int, b: int) -> tuple:
    """
    Quadratic field "witness" test analogous to Miller-Rabin.
    
    Computes (a + b√d)^(N-1) mod N and checks for factor-revealing properties.
    
    Returns (is_consistent, factor_if_found, result).
    """
    # Compute (a + b√d)^(N-1) mod N
    r0, r1 = power_in_quadratic_field(a, b, d, N - 1, N)
    
    # Check various gcds
    g0 = gcd(r0 - 1, N) if r0 != 1 else N
    g1 = gcd(r1, N) if r1 != 0 else N
    g2 = gcd(r0, N)
    g3 = gcd(abs(norm_in_quadratic_field(a, b, d)), N)
    
    # Check if any gcd reveals a factor
    for g in [g0, g1, g2, g3]:
        if 1 < g < N:
            return (False, g, (r0, r1))
    
    # Check if result is "consistent" (r₁ = 0 and r₀ = 1)
    if r1 == 0 and r0 == 1:
        return (True, None, (r0, r1))
    
    # Result differs from expected
    return (False, None, (r0, r1))


def analyze_witness_probability(N: int, p: int, q: int):
    """
    Analyze witness probability in Z[√d]/(N).
    
    Key question: What fraction of (a, b, d) are "witnesses"?
    """
    print("="*70)
    print(f"WITNESS PROBABILITY ANALYSIS")
    print(f"N = {N} = {p} × {q}")
    print("="*70)
    
    # Choose d such that (d/p) and (d/q) have various combinations
    cases = [
        ("Both QR", None),  # Will find such d
        ("One QR, one NR", None),
        ("Both NR", None),
    ]
    
    # Find appropriate d values
    d_values = {}
    for d in range(2, 50):
        leg_p = legendre_symbol(d, p)
        leg_q = legendre_symbol(d, q)
        
        if leg_p == 1 and leg_q == 1 and "Both QR" not in d_values:
            d_values["Both QR"] = d
        elif leg_p == 1 and leg_q == -1 and "One QR, one NR" not in d_values:
            d_values["One QR, one NR"] = d
        elif leg_p == -1 and leg_q == -1 and "Both NR" not in d_values:
            d_values["Both NR"] = d
    
    for case_name, d in d_values.items():
        print(f"\n{case_name}: d = {d}")
        print(f"  Legendre symbols: ({d}/{p}) = {legendre_symbol(d, p)}, ({d}/{q}) = {legendre_symbol(d, q)}")
        
        # Count witnesses
        count_consistent = 0
        count_factor_revealing = 0
        count_total = 0
        witnesses_found = []
        
        # Test range proportional to p
        test_range = min(20, p)
        
        for a in range(1, test_range + 1):
            for b in range(1, test_range + 1):
                if gcd(a, N) > 1:
                    continue
                
                count_total += 1
                is_consistent, factor, result = quadratic_field_witnesses(N, d, a, b)
                
                if factor:
                    count_factor_revealing += 1
                    if len(witnesses_found) < 5:
                        witnesses_found.append((a, b, factor, result))
                
                if is_consistent:
                    count_consistent += 1
        
        if count_total > 0:
            prob_consistent = count_consistent / count_total
            prob_factor = count_factor_revealing / count_total
            
            print(f"  Tested {count_total} elements (a, b) in range [1, {test_range}]")
            print(f"  Consistent (r₀=1, r₁=0): {count_consistent} ({100*prob_consistent:.1f}%)")
            print(f"  Factor-revealing: {count_factor_revealing} ({100*prob_factor:.1f}%)")
            
            if witnesses_found:
                print(f"  First few witnesses:")
                for a, b, factor, (r0, r1) in witnesses_found[:3]:
                    print(f"    (a={a}, b={b}): gcd = {factor}, result = {r0} + {r1}√{d}")


def miller_rabin_comparison(N: int):
    """
    Compare with standard Miller-Rabin witness probability.
    """
    print(f"\n{'='*70}")
    print("MILLER-RABIN COMPARISON")
    print(f"N = {N}")
    print("="*70)
    
    # Miller-Rabin has at least 75% witness probability for composite N
    print("\nMiller-Rabin properties:")
    print("  - Computes a^(N-1) mod N (polynomial)")
    print("  - Witness probability ≥ 75% for composite N")
    print("  - At most 25% of a give 'consistent' (a^(N-1) ≡ 1 mod N)")
    
    # Count Miller-Rabin witnesses
    count_mr_witnesses = 0
    count_total = min(50, N - 2)
    
    for a in range(2, count_total + 2):
        is_prime, factor = miller_rabin_witnesses(N, a)
        if not is_prime:
            count_mr_witnesses += 1
    
    prob_mr_witness = count_mr_witnesses / count_total
    print(f"\n  Miller-Rabin witnesses in range [2, {count_total+1}]: {count_mr_witnesses} ({100*prob_mr_witness:.1f}%)")
    
    print("\nQuadratic field witness properties:")
    print("  - Computes (a + b√d)^(N-1) mod N (polynomial)")
    print("  - Witness probability unknown - this is what we're testing!")
    print("  - 'Consistent' means result = 1 + 0√d")


def the_key_question():
    """
    The key question: Is there a polynomial-time test in Z[√d]/(N)?
    """
    print("\n" + "="*70)
    print("THE KEY QUESTION")
    print("="*70)
    print("""
Miller-Rabin works because:
  1. a^(N-1) mod N is ALWAYS computable (no circular dependency)
  2. For composite N = pq, a^(N-1) mod p ≠ a^(N-1) mod q in general
  3. This creates "witnesses" that reveal factors
  4. At least 75% of random a are witnesses

For Z[√d]/(N):
  1. (a + b√d)^(N-1) mod N is ALWAYS computable (polynomial)
  2. For composite N = pq, the result differs mod p and mod q
  3. Can this create "witnesses"?

The difference:
  - Miller-Rabin: Single element a, single computation a^(N-1)
  - Quadratic field: Tuple (a, b, d), more complex structure

Question: What's the "witness" definition?
  Option 1: gcd(|r₁|, N) > 1
  Option 2: gcd(r₀ - 1, N) > 1
  Option 3: gcd(|N(a+b√d)|, N) > 1
  Option 4: r₁ ≠ 0 when (d/N) = 1

From experiments, Option 3 (norm-based) has probability ~10-30% for small N.
This requires searching O(1/probability) elements = O(√p) for larger N.

The critical insight:
  - Miller-Rabin doesn't require SEARCH
  - It tests a FIXED a^(N-1) for random a
  - The test is: "Is a^(N-1) ≡ 1 (mod N)?"
  
  - Our norm-based test requires FINDING elements with factor-norms
  - This requires SEARCH proportional to O(√p)

Can we define a test that doesn't require search?

The asymmetry approach:
  - If (d/p) ≠ (d/q), then Z[√d]/(p) and Z[√d]/(q) have different structure
  - One might be a field extension (degree 2), the other a product of fields
  - Can we DETECT this asymmetry?

Let's test this...
""")


def test_asymmetry_detection():
    """
    Test if we can detect (d/p) ≠ (d/q) without knowing p, q.
    """
    print("\n" + "="*70)
    print("ASYMMETRY DETECTION")
    print("="*70)
    print("""
Idea: If (d/p) ≠ (d/q), then:
  - Z[√d]/(p) is a field extension (degree 2) if (d/p) = -1
  - Z[√d]/(p) ≅ Z/(p) × Z/(p) if (d/p) = 1
  - Z[√d]/(q) has opposite structure

By CRT, Z[√d]/(N) ≅ Z[√d]/(p) × Z[√d]/(q)

Can we detect which case we're in without knowing p, q?
""")
    
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
    ]
    
    for N, p, q in test_cases:
        print(f"\nN = {N} = {p} × {q}")
        
        # Find d with (d/p) ≠ (d/q)
        for d in range(2, 50):
            leg_p = legendre_symbol(d, p)
            leg_q = legendre_symbol(d, q)
            
            if leg_p != leg_q:
                print(f"\n  d = {d}: ({d}/{p}) = {leg_p}, ({d}/{q}) = {leg_q}")
                
                # The Jacobi symbol tells us something
                jac = jacobi_symbol(d, N)
                print(f"  Jacobi symbol ({d}/{N}) = {jac}")
                print(f"  This is {leg_p} × {leg_q} = {leg_p * leg_q}")
                
                # Test a few elements
                print(f"  Testing elements in Z[√{d}]/({N}):")
                
                for a in range(1, 10):
                    b = 1
                    r0, r1 = power_in_quadratic_field(a, b, d, N - 1, N)
                    
                    # Check if r₁ ≠ 0 indicates asymmetry
                    if r1 != 0:
                        print(f"    (a={a}, b={b})^(N-1) = {r0} + {r1}√{d}")
                        print(f"      r₁ ≠ 0: might indicate asymmetry?")
                        break
                
                break


def polynomial_time_test():
    """
    The critical question: Is there a polynomial-time test?
    """
    print("\n" + "="*70)
    print("POLYNOMIAL-TIME TEST?")
    print("="*70)
    print("""
Miller-Rabin is polynomial because:
  1. Pick random a ∈ [2, N-2]
  2. Compute a^(N-1) mod N (polynomial: O(log³N) with modular exponentiation)
  3. Check if result ≡ 1 (mod N) and other conditions
  4. If not consistent, N is composite

Key: NO SEARCH REQUIRED. Test is deterministic once a is chosen.

For quadratic fields:
  1. Pick random (a, b, d)?
  2. Compute (a + b√d)^(N-1) mod N (polynomial)
  3. Check what condition?
  4. If condition fails, N is composite?

Problem: What condition to check?
  - r₁ = 0 and r₀ = 1? This is the "consistent" case
  - But many (a, b, d) give r₁ ≠ 0 even for prime N!

The asymmetry approach:
  - If (d/N) = -1, then (d/p) ≠ (d/q) for some factorization
  - This creates structural asymmetry
  - But can we DETECT it?

Let's compute what fraction of (a, b, d) give consistent results...
""")


def compute_consistency_fraction(N: int, p: int, q: int, max_range: int = 20):
    """
    Compute what fraction of (a, b, d) give consistent results.
    """
    print(f"\n{'='*70}")
    print(f"CONSISTENCY FRACTION FOR N = {N}")
    print(f"{'='*70}")
    
    # Test for various d
    d_results = []
    
    for d in range(2, min(20, N)):
        if gcd(d, N) > 1:
            continue
        
        leg_p = legendre_symbol(d, p)
        leg_q = legendre_symbol(d, q)
        jac = jacobi_symbol(d, N)
        
        count_consistent = 0
        count_total = 0
        
        for a in range(1, max_range + 1):
            for b in range(1, max_range + 1):
                if gcd(a, N) > 1:
                    continue
                
                count_total += 1
                r0, r1 = power_in_quadratic_field(a, b, d, N - 1, N)
                
                if r1 == 0 and r0 == 1:
                    count_consistent += 1
        
        frac = count_consistent / count_total if count_total > 0 else 0
        
        d_results.append((d, leg_p, leg_q, jac, frac))
    
    # Sort by Legendre symbol combinations
    both_qr = [r for r in d_results if r[1] == 1 and r[2] == 1]
    one_qr = [r for r in d_results if r[1] != r[2]]
    both_nr = [r for r in d_results if r[1] == -1 and r[2] == -1]
    
    print("\nBoth QR (Legendre = 1 for both factors):")
    for d, leg_p, leg_q, jac, frac in both_qr[:5]:
        print(f"  d = {d}: consistent fraction = {frac:.3f}")
    
    print("\nOne QR, one NR:")
    for d, leg_p, leg_q, jac, frac in one_qr[:5]:
        print(f"  d = {d}: consistent fraction = {frac:.3f}")
    
    print("\nBoth NR (Legendre = -1 for both factors):")
    for d, leg_p, leg_q, jac, frac in both_nr[:5]:
        print(f"  d = {d}: consistent fraction = {frac:.3f}")
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("""
If consistent fraction is:
  - ~1 for prime N
  - Significantly < 1 for composite N
  
Then we might have a Miller-Rabin-like test!

But: The fraction depends on d's Legendre symbols.
  - For both QR: Z[√d]/(p) and Z[√d]/(q) both split, similar structure
  - For both NR: Z[√d]/(p) and Z[√d]/(q) both field extensions, similar structure
  - For one QR, one NR: Different structure!

The asymmetry case (one QR, one NR) is most promising for detection.
""")


if __name__ == "__main__":
    # Test on small semiprimes
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
    ]
    
    for N, p, q in test_cases:
        analyze_witness_probability(N, p, q)
        miller_rabin_comparison(N)
    
    the_key_question()
    test_asymmetry_detection()
    
    # Compute consistency fractions
    compute_consistency_fraction(143, 11, 13)
    compute_consistency_fraction(391, 17, 23)
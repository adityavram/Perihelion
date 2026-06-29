#!/usr/bin/env python3
"""
Deep dive into the number field breakthrough.

Key question: Can we find elements with factor-norms efficiently?

This connects to:
1. Continued fractions (CFRAC already subexponential)
2. Class group computation
3. Unit group structure
4. Norm form theory
"""

from math import isqrt, gcd, sqrt
from typing import Optional, Tuple, List


def continued_fraction_sqrt(N: int, max_convergents: int = 50):
    """
    Continued fraction expansion of sqrt(N).
    
    CFRAC uses this: convergents give solutions to a² - db² = k
    for small k, and GCD(k, N) can reveal factors.
    
    Our question: Do convergents give elements with factor-norms?
    """
    sqrtN = isqrt(N)
    if sqrtN * sqrtN == N:
        return sqrtN, []  # Perfect square
    
    # Continued fraction algorithm
    m = 0
    d = 1
    a0 = sqrtN
    a = a0
    
    convergents = []
    h_prev, h = 1, a0
    k_prev, k = 0, 1
    
    for i in range(max_convergents):
        m = d * a - m
        d = (N - m * m) // d
        a = (sqrtN + m) // d
        
        h_new = a * h + h_prev
        k_new = a * k + k_prev
        
        h_prev, h = h, h_new
        k_prev, k = k, k_new
        
        convergents.append((h, k, h * h - N * k * k))
        
        # Check if this convergent gives a factor
        h_sq_minus_N_k_sq = h * h - N * k * k
        
        if h_sq_minus_N_k_sq != 0:
            g = gcd(abs(h_sq_minus_N_k_sq), N)
            if g > 1 and g < N:
                # Found a factor!
                return sqrtN, (convergents, g, N // g, i)
    
    return sqrtN, (convergents, None, None, -1)


def analyze_convergents_for_factors(N: int, p: int, q: int):
    """
    Analyze convergents of sqrt(N) to see if they connect to factor-norms.
    
    Convergents give (h, k) such that |h² - N·k²| is small.
    
    Question: Do these give elements with factor-norms?
    """
    print(f"\n{'='*70}")
    print(f"CONVERGENTS AND FACTOR-NORMS")
    print(f"N = {N} = {p} × {q}")
    print(f"{'='*70}")
    
    sqrtN, result = continued_fraction_sqrt(N, 50)
    
    if isinstance(result, tuple) and len(result) == 4:
        convergents, factor1, factor2, idx = result
        
        print(f"\nConvergents that revealed factors:")
        print(f"  Found at index {idx}: GCD = {factor1}")
        print(f"  Factors: {factor1} × {factor2}")
        
        # Analyze the convergent
        h, k, val = convergents[idx]
        print(f"\n  Convergent {idx}: h={h}, k={k}")
        print(f"  h² - N·k² = {val}")
        print(f"  |val| = {abs(val)}")
        
        # This convergent gives element (h + k√1) with norm h² - k²? No...
        # Actually: In Z[√N], element is h + k√N
        # But sqrt(N) isn't in Z[√d] for d = N...
        
        print(f"\nNote: CFRAC uses convergents differently")
        print(f"  It finds h² - N·k² with small |val|")
        print(f"  Then GCD(val, N) can reveal factors")
        print(f"  This is subexponential, not polynomial")
        
        return factor1, factor2
    
    return None


def norm_equation_solutions(N: int, d: int, max_b: int = 100):
    """
    Find all solutions to a² - d·b² = N in the range b ≤ max_b.
    
    This is the norm equation: elements α ∈ Z[√d] with N(α) = N.
    
    Returns list of (a, b) pairs.
    """
    solutions = []
    
    for b in range(0, max_b + 1):
        # a² = N + d·b²
        a_sq = N + d * b * b
        
        if a_sq >= 0:
            a = isqrt(a_sq)
            if a * a == a_sq:
                solutions.append((a, b))
                if a != 0:
                    solutions.append((-a, b))
    
    return solutions


def find_factor_norm_elements(N: int, p: int, q: int, max_d: int = 50, max_b: int = 50):
    """
    Find elements in Z[√d] with norms equal to factors of N.
    
    For N = pq, we want elements α ∈ Z[√d] such that:
    - N(α) = p or N(α) = q or N(α) = ±1 or N(α) = ±N
    
    Question: Can we find these without enumerating all (d, b)?
    """
    print(f"\n{'='*70}")
    print(f"SEARCHING FOR ELEMENTS WITH FACTOR-NORMS")
    print(f"N = {N} = {p} × {q}")
    print(f"{'='*70}")
    
    # First, let's see where these elements exist
    print(f"\nFinding d where elements with norm p={p} or q={q} exist:")
    
    for d in range(2, max_d):
        # Check if d is squarefree
        if not is_squarefree(d):
            continue
        
        # Find elements with norm = p
        sols_p = norm_equation_solutions(p, d, max_b)
        sols_q = norm_equation_solutions(q, d, max_b)
        sols_N = norm_equation_solutions(N, d, max_b)
        
        if sols_p or sols_q:
            print(f"\n  d = {d}:")
            if sols_p:
                print(f"    Elements with norm {p}: {sols_p[:5]}")
            if sols_q:
                print(f"    Elements with norm {q}: {sols_q[:5]}")
            if sols_N:
                print(f"    Elements with norm {N}: {sols_N[:5]}")
            
            # Check if any a shares gcd with N
            for a, b in sols_p + sols_q + sols_N:
                g = gcd(abs(a), N)
                if g > 1 and g < N:
                    print(f"      → gcd(a, N) = {g} ✓ A FACTOR!")
    
    # The pattern: Elements with norm p or q exist in MANY quadratic fields
    print(f"\n" + "="*70)
    print(f"OBSERVATION:")
    print(f"  Elements with norm p or q exist in many Z[√d]")
    print(f"  The question is: Can we find them without enumerating d?")
    print(f"\n  Pattern: If d ≡ ? (mod p) then elements with norm p exist")
    print(f"  This is connected to Legendre symbol (d/p)")


def is_squarefree(n: int) -> bool:
    """Check if n is squarefree."""
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True


def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    result = pow(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else result


def analyze_legendre_pattern(N: int, p: int, q: int):
    """
    Analyze when elements with norm p exist.
    
    Key theorem: a² - d·b² = p has solution iff (d/p) ≠ -1
    (i.e., d is a QR mod p, or p divides d)
    
    This connects to: When does (d/p) ≠ -1?
    """
    print(f"\n{'='*70}")
    print(f"LEGENDRE SYMBOL PATTERN")
    print(f"For N = {N} = {p} × {q}")
    print(f"{'='*70}")
    
    print(f"\nTheorem: a² - d·b² = p has solution iff (d/p) ≠ -1")
    print(f"  (d/p) = 1: d is QR mod p → solutions exist")
    print(f"  (d/p) = 0: p divides d → solutions exist (trivial)")
    print(f"  (d/p) = -1: d is NR mod p → NO solutions")
    
    print(f"\nFor p = {p}:")
    print(f"  We need (d/{p}) ≠ -1, i.e., d is QR mod {p}")
    
    qr_mod_p = []
    nr_mod_p = []
    
    for d in range(2, 50):
        if not is_squarefree(d):
            continue
        
        leg = legendre_symbol(d, p)
        if leg == 1:
            qr_mod_p.append(d)
        elif leg == -1:
            nr_mod_p.append(d)
    
    print(f"  d with (d/{p}) = 1 (QR): {qr_mod_p}")
    print(f"  d with (d/{p}) = -1 (NR): {nr_mod_p}")
    
    print(f"\nFor q = {q}:")
    qr_mod_q = []
    nr_mod_q = []
    
    for d in range(2, 50):
        if not is_squarefree(d):
            continue
        
        leg = legendre_symbol(d, q)
        if leg == 1:
            qr_mod_q.append(d)
        elif leg == -1:
            nr_mod_q.append(d)
    
    print(f"  d with (d/{q}) = 1 (QR): {qr_mod_q}")
    print(f"  d with (d/{q}) = -1 (NR): {nr_mod_q}")
    
    print(f"\n" + "="*70)
    print(f"KEY INSIGHT:")
    print(f"  For p = {p}: {len(qr_mod_p)} values of d give (d/p) = 1")
    print(f"  For q = {q}: {len(qr_mod_q)} values of d give (d/{q}) = 1")
    print(f"  For BOTH: d must satisfy both conditions")
    print(f"  This constrains d to specific congruence classes")
    
    # Find d that are QR for both
    both_qr = set(qr_mod_p) & set(qr_mod_q)
    print(f"\n  d that are QR for both p and q: {sorted(both_qr)}")
    
    if both_qr:
        print(f"  Number of such d in range: {len(both_qr)}")
        print(f"  In Z[{min(both_qr)}], elements with norms p AND q exist!")
    
    return both_qr


def the_fundamental_question():
    """
    The fundamental question we need to answer.
    """
    print(f"\n{'='*70}")
    print(f"THE FUNDAMENTAL QUESTION")
    print(f"{'='*70}")
    
    print("""
We've found:
  ✓ Elements with norm p exist in Z[√d] for many d
  ✓ Elements with norm q exist in Z[√d] for many d
  ✓ Elements with norm N exist in Z[√d] for many d
  ✓ Some of these elements a + b√d have gcd(a, N) = p or q

The enumeration:
  for d in range(2, max_d):
    for b in range(1, bound):
      check if a² - d·b² divides N

The question:
  Can we find the right (d, b) without enumeration?

Possible approaches:
  1. Legendre symbol: (d/p) ≠ -1 constrains d
     But: Only tells us existence, not how to find (a, b)
  
  2. Class group: Cl(Z[√d]) might encode factor info
     But: Computing class group is hard
  
  3. Continued fractions: CFRAC already uses this
     But: Subexponential, not polynomial
  
  4. Unit group: Fundamental units might help
     But: Finding units requires solving Pell's equation

  5. Special d: Is there always a "good" d?
     Question: For any N, is there d with special properties?

What makes this different from previous approaches:
  - Working in ring extension Z[√d], not Z
  - Factor-norms appear naturally
  - Connected to deep number theory
  - CFRAC is subexponential, maybe we can do better?

The subexponential complexity of CFRAC comes from:
  - Finding convergents: O(√N) operations
  - Each convergent gives small h² - N·k²
  - GCD of small values with N reveals factors

Can we improve? Unknown. But this is the most promising direction yet.
""")


if __name__ == "__main__":
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
    ]
    
    for N, p, q in test_cases:
        print(f"\n{'='*70}")
        print(f"ANALYZING N = {N} = {p} × {q}")
        print(f"{'='*70}")
        
        # Analyze convergents
        analyze_convergents_for_factors(N, p, q)
        
        # Find factor-norm elements
        find_factor_norm_elements(N, p, q, max_d=30, max_b=30)
        
        # Legendre symbol pattern
        analyze_legendre_pattern(N, p, q)
    
    # The fundamental question
    the_fundamental_question()
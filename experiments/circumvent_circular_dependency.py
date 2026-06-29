#!/usr/bin/env python3
"""
Working Around the Circular Dependency.

The Catch: Computing √d mod N requires factoring N first.

The Way Out: Work in the RING Z[√d] instead of mod N.

Key Idea:
  - In Z[√d], elements are a + b√d
  - We can find elements with specific norms WITHOUT computing √d mod N
  - The norm N(a + b√d) = a² - d·b²
  
  If we find α ∈ Z[√d] with N(α) = k where gcd(k, N) = p or q,
  then gcd(α, (N)) reveals factors!

This avoids the circular dependency:
  - No need to compute √d mod N
  - We search for elements with specific norms
  - The search might be polynomial for special d
"""

from math import isqrt, gcd
from typing import List, Tuple, Optional


def find_elements_with_norm_dividing_N(N: int, d: int, bound: int = 100) -> List[Tuple[int, int, int]]:
    """
    Find elements α = a + b√d in Z[√d] with N(α) | N.
    
    N(α) = a² - d·b²
    
    Returns list of (a, b, norm) tuples.
    """
    elements = []
    
    for b in range(0, bound):
        # a² = N(α) + d·b²
        # We want N(α) to divide N
        
        # For small b, check all possible a
        for a in range(-bound, bound + 1):
            norm = a * a - d * b * b
            
            if norm != 0 and N % abs(norm) == 0:
                # This norm divides N!
                g = gcd(abs(a), N)
                
                if 1 < g < N:
                    elements.append((a, b, norm, g))
    
    return elements


def the_key_question():
    """
    The key question: Can we find these elements efficiently?
    
    We're searching for (a, b) such that a² - d·b² divides N.
    
    This is the same as finding (a, b) such that:
      a² ≡ d·b² (mod p) or a² ≡ d·b² (mod q)
    
    In other words: (a/b)² ≡ d (mod p) or (a/b)² ≡ d (mod q)
    
    So a/b is a square root of d mod p (or mod q).
    """
    print("="*70)
    print("THE KEY QUESTION")
    print("="*70)
    print("""
We're searching for (a, b) such that a² - d·b² divides N.

This is equivalent to: a² ≡ d·b² (mod N)
Which means: (a/b)² ≡ d (mod N) if b is invertible mod N

So we're looking for elements a + b√d where:
  (a/b) is a square root of d mod p (or mod q)

But this brings us back to square roots!

Alternative: Use the NORM FORM directly

In Z[√d]:
  - Elements α = a + b√d have norm N(α) = a² - d·b²
  - The norm is multiplicative: N(αβ) = N(α)N(β)
  - If N(α) = k, then α divides k in the norm sense

For N = pq:
  - We want elements α with N(α) = p or q
  - Such elements exist if d is QR mod p or mod q
  - We can find them by solving a² - d·b² = p (or q)

BUT: This requires solving a Diophantine equation.
      For each d, this is a norm equation.
      Solutions exist if d is QR mod p.

The question: How many (a, b) pairs do we need to check?

For a given d:
  - a² - d·b² = p has solutions iff (d/p) = 1
  - About 50% of d satisfy this for a given p
  - For both p and q: about 25% of d

So expected number of d to try: 4

For each d:
  - We need to find (a, b) with a² - d·b² dividing N
  - The smallest solutions have |a|, |b| = O(√p)

So the search is polynomial in √p, not √N!

Wait... this is still subexponential for large N = pq where p ≈ q ≈ √N.
""")


def analyze_search_space(N: int, p: int, q: int, d: int):
    """
    Analyze the search space for elements with factor-norms.
    """
    print("\n" + "="*70)
    print(f"SEARCH SPACE ANALYSIS")
    print(f"N = {N} = {p} × {q}, d = {d}")
    print("="*70)
    
    # Find solutions to a² - d·b² = p
    print(f"\nSolutions to a² - {d}·b² = {p}:")
    sols_p = []
    for b in range(0, int(p**0.5) + 10):
        a_sq = p + d * b * b
        if a_sq >= 0:
            a = isqrt(a_sq)
            if a * a == a_sq:
                sols_p.append((a, b))
                print(f"  a = {a}, b = {b}, norm = {a*a - d*b*b}")
    
    # Find solutions to a² - d·b² = q
    print(f"\nSolutions to a² - {d}·b² = {q}:")
    sols_q = []
    for b in range(0, int(q**0.5) + 10):
        a_sq = q + d * b * b
        if a_sq >= 0:
            a = isqrt(a_sq)
            if a * a == a_sq:
                sols_q.append((a, b))
                print(f"  a = {a}, b = {b}, norm = {a*a - d*b*b}")
    
    # Find solutions to a² - d·b² = N
    print(f"\nSolutions to a² - {d}·b² = {N}:")
    sols_N = []
    for b in range(0, int(N**0.5) + 10):
        a_sq = N + d * b * b
        if a_sq >= 0:
            a = isqrt(a_sq)
            if a * a == a_sq:
                sols_N.append((a, b))
                print(f"  a = {a}, b = {b}, norm = {a*a - d*b*b}")
    
    print(f"\nAnalysis:")
    print(f"  Solutions for p={p}: {len(sols_p)}")
    print(f"  Solutions for q={q}: {len(sols_q)}")
    print(f"  Solutions for N={N}: {len(sols_N)}")
    
    # Check gcd(a, N) for each solution
    print(f"\nGCD analysis:")
    for a, b in sols_p + sols_q + sols_N:
        g = gcd(abs(a), N)
        if 1 < g < N:
            print(f"  ({a}, {b}): gcd(|a|, N) = {g} ← FACTOR!")
    
    # Size of solutions
    if sols_p:
        max_a = max(abs(a) for a, b in sols_p)
        max_b = max(b for a, b in sols_p)
        print(f"\n  For p={p}: max |a| = {max_a}, max b = {max_b}")
        print(f"  Search space: O(√p) = O({int(p**0.5)})")
    
    if sols_q:
        max_a = max(abs(a) for a, b in sols_q)
        max_b = max(b for a, b in sols_q)
        print(f"  For q={q}: max |a| = {max_a}, max b = {max_b}")
        print(f"  Search space: O(√q) = O({int(q**0.5)})")


def the_fundamental_problem():
    """
    The fundamental problem: Search space is still exponential in log N.
    """
    print("\n" + "="*70)
    print("THE FUNDAMENTAL PROBLEM")
    print("="*70)
    print("""
We've reduced the problem from:
  - Enumerating ~√N candidates (trial division)
  
To:
  - Enumerating ~√p candidates (where p is a factor)

For N = pq with p ≈ q ≈ √N:
  - √p ≈ √√N = N^(1/4)
  - This is still exponential in log N

Example for RSA-2048:
  - p ≈ 2^1024
  - √p ≈ 2^512
  - Still WAY too large!

So we haven't made the problem polynomial.

HOWEVER, there are approaches that might work:

1. Continued Fractions (CFRAC):
   - Find convergents to √(dN)
   - These give a² - d·b² = k for small k
   - If k shares factors with N, we find them
   - CFRAC is subexponential: L[N] = exp(√(log N log log N))
   - Better than √N but not polynomial

2. Number Field Sieve (NFS):
   - Work in Z[α] where α is an algebraic number
   - Find smooth elements
   - Similar complexity to CFRAC

3. Special Structure:
   - For special N (e.g., N = p^k for known k), faster methods
   - For special d (e.g., class number 1), simpler structure

4. Quantum (Shor's Algorithm):
   - Uses quantum superposition to find periods
   - Polynomial in log N
   - But requires quantum computer

The question remains: Is there a classical polynomial algorithm?

Our investigation shows:
  - Number fields provide rich structure
  - Square roots reveal factors
  - But computing square roots mod N requires knowing factors
  
The circular dependency is fundamental:
  - To find factors, we need structure (ideals, norms, etc.)
  - To compute structure, we need factors
  
This is why factoring is believed to be hard.
But belief is not proof. The search continues.
""")


def potential_breakthrough():
    """
    A potential breakthrough: The Miller-Rabin connection.
    """
    print("\n" + "="*70)
    print("POTENTIAL BREAKTHROUGH: THE MILLER-RABIN CONNECTION")
    print("="*70)
    print("""
Miller-Rabin primality testing uses a similar circular dependency:

  - Write N-1 = 2^s · d (d odd)
  - For random a, compute a^d mod N
  - If a^d ≢ ±1 (mod N), square repeatedly
  - If we find a^(d·2^r) ≡ 1 but a^(d·2^(r-1)) ≢ ±1 (mod N)
    Then N is composite and gcd(a^(d·2^(r-1)) - 1, N) is a factor!

This works WITHOUT knowing factors in advance.

The key: Modular exponentiation reveals structure.

Similarly, for square roots:
  - Can we compute something about √d mod N WITHOUT computing √d mod N?
  
Ideas:
  1. Work with the IDEAL (N) ⊂ Z[√d] directly
     - (N) factors as P₁ × P₁' × P₂ × P₂' when both p, q split
     - The ideals P₁, P₁', P₂, P₂' are generated by:
       P₁ = (p, √d - r₁) where r₁² ≡ d (mod p)
       P₁' = (p, √d + r₁)
       P₂ = (q, √d - r₂) where r₂² ≡ d (mod q)
       P₂' = (q, √d + r₂)
     
  2. The ideal (N, √d - r) for some r with r² ≡ d (mod N) is:
     - If r² ≡ d (mod N), then (N, √d - r) factors
     - The factors have norms p and q
     
  3. We can work with the ideal (N, √d - r) WITHOUT computing r explicitly
     - The ideal is defined by the equation r² ≡ d (mod N)
     - Operations on ideals are polynomial
     - The structure might reveal factors

This is deep algebraic number theory territory.
The question: Can ideal operations reveal factor structure efficiently?

For class number 1 fields (PIDs):
  - Every ideal is principal
  - (N, √d - r) = (α) for some α
  - N(α) = ±p or ±q (one of the factors!)
  - We can find α by solving N(α) | N

For higher class number fields:
  - Ideals form a class group
  - The class group structure might encode factors
  - Computing class groups is subexponential
  - But maybe we can use structure without full computation?

This is the frontier. Unexplored territory.
""")


def the_unexplored_territory():
    """
    Unexplored territory: Can ideal structure reveal factors efficiently?
    """
    print("\n" + "="*70)
    print("UNEXPLORED TERRITORY: IDEAL STRUCTURE AND FACTORIZATION")
    print("="*70)
    print("""
Current status:
  - We know ideals encode factor structure
  - We know square roots mod N would reveal factors
  - We know computing square roots requires factors
  
The unexplored question:
  - Can we work with ideals abstractly?
  - Can we perform operations on ideals that reveal factors?
  - Can we use class group structure efficiently?

Potential approaches:
  
  1. Class Field Theory:
     - The Hilbert class field H of Q(√d) has Galois group Cl(Q(√d))
     - The extension H/Q encodes ideal structure
     - Can we use this structure?
  
  2. Complex Multiplication:
     - Special values of elliptic functions give algebraic numbers
     - These encode ideal class group information
     - Can we extract factor information?
  
  3. Ideals as Modules:
     - Ideals are Z-modules of rank 2
     - Module operations are polynomial
     - Can we find submodules corresponding to factors?
  
  4. Reduction mod N:
     - The ideal (N) in Z[√d] can be studied abstractly
     - Its prime ideal factors have norms p and q
     - Can we compute prime ideal factors without factoring N?

The key insight:
  - We're looking for polynomial-time operations
  - That take (N, d) as input
  - And output (p, q) or structure that reveals them

This is a research frontier. The answer might be:
  - Yes: There exists a polynomial algorithm (major breakthrough)
  - No: Factoring is inherently hard classically
  - Maybe: Special cases admit efficient solutions

The investigation continues...
""")


if __name__ == "__main__":
    test_cases = [
        (143, 11, 13, 3),
        (143, 11, 13, 5),
        (391, 17, 23, 2),
        (899, 29, 31, 5),
    ]
    
    the_key_question()
    
    for N, p, q, d in test_cases:
        analyze_search_space(N, p, q, d)
    
    the_fundamental_problem()
    potential_breakthrough()
    the_unexplored_territory()
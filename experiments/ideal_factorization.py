#!/usr/bin/env python3
"""
Ideal Factorization in Quadratic Fields

Goal: Transform divisibility problem into multiplication problem.

In Z[√d]:
  - Divisibility in Z: "Does p divide N?" (hard)
  - Multiplication in Z[√d]: "How does (N) factor as ideals?" (easier?)

The ideal (N) = p₁ᵉ¹ × p₂ᵉ² × ... factors uniquely.
If we can compute this efficiently, factors emerge.

Key: Prime p SPLIT/INERT/RAMIFY in Z[√d] based on (d/p).
"""

from math import isqrt, gcd
from typing import List, Tuple, Optional


def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p). Returns 1 (QR), -1 (NR), 0 (divisible)."""
    if a % p == 0:
        return 0
    result = pow(a, (p - 1) // 2, p)
    return -1 if result == p - 1 else result


def prime_behavior_in_quadratic_field(p: int, d: int) -> str:
    """
    How does prime p behave in Z[√d]?
    
    Based on Legendre symbol (d/p):
      - (d/p) = 1: p SPLITS as P × P'
      - (d/p) = -1: p is INERT (remains prime)
      - (d/p) = 0: p RAMIFIES as P²
    """
    leg = legendre_symbol(d, p)
    if leg == 1:
        return "SPLIT"
    elif leg == -1:
        return "INERT"
    else:
        return "RAMIFY"


def find_splitting_primes_for_N(N: int, max_d: int = 100) -> List[Tuple[int, str]]:
    """
    For N = pq, find d where both p and q SPLIT.
    
    If both p and q split in Z[√d], then:
      - (p) = P₁ × P₁' (two ideals)
      - (q) = P₂ × P₂' (two ideals)
      - (N) = P₁P₁'P₂P₂' (four ideals)
    
    The ideals P₁, P₁', P₂, P₂' encode factor information.
    We can compute ideal multiplication without knowing p, q!
    """
    # Factor N (we know p, q for testing)
    factors = trial_division(N)
    if len(factors) != 2:
        return []
    
    p, q = factors
    
    splitting_ds = []
    
    for d in range(2, max_d):
        if not is_squarefree(d):
            continue
        
        p_behavior = prime_behavior_in_quadratic_field(p, d)
        q_behavior = prime_behavior_in_quadratic_field(q, d)
        
        if p_behavior == "SPLIT" and q_behavior == "SPLIT":
            # Both split! Ideal factorization:
            # (N) = P₁P₁'P₂P₂'
            splitting_ds.append((d, f"both split: ({p}) splits, ({q}) splits"))
    
    return splitting_ds


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


def is_squarefree(n: int) -> bool:
    """Check if n is squarefree."""
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True


def compute_ideal_norm_in_quadratic_field(N: int, d: int) -> List[Tuple[int, int]]:
    """
    In Z[√d], elements α = a + b√d have norm N(α) = a² - d·b².
    
    The ideal (N) factors as a product of prime ideals.
    
    If p|N and p splits in Z[√d]:
      (p) = P × P' where P ≠ P'
      There exist elements α, β with N(α)=p and N(β) divides p²
    """
    # Find all elements with norm dividing N
    elements_with_small_norm = []
    
    for b in range(0, 100):
        for a in range(-200, 200):
            norm = a * a - d * b * b
            
            if norm != 0 and N % abs(norm) == 0:
                g = gcd(abs(a), N)
                if g > 1 and g < N:
                    elements_with_small_norm.append((a, b, norm, g))
    
    return elements_with_small_norm


def ideal_theory_approach(N: int):
    """
    THE KEY INSIGHT:
    
    In Z, we ask: "What divides N?" (enumeration required)
    In Z[√d], we ask: "How does (N) factor?" (ideal multiplication)
    
    Ideal factorization is UNIQUE:
      (N) = P₁ᵉ¹ × P₂ᵉ² × ...
    
    The exponents eᵢ and the ideals Pᵢ encode factor information.
    
    Question: Can we compute the ideal factorization efficiently?
    
    Subexponential algorithms exist for:
      - Computing class group Cl(Z[√d])
      - Computing unit group Z[√d]*
      - Computing ideal class of (N)
    
    But: These are subexponential, not polynomial.
    
    HOWEVER: There might be SPECIAL d where this is easy!
    
    Special cases:
      - d = -1 (Gaussian integers): class number 1, easy units
      - d = -3 (Eisenstein integers): class number 1
      - Other d with small class number
      - d with special structure (cyclotomic, etc.)
    """
    print("="*70)
    print("IDEAL THEORY APPROACH")
    print("="*70)
    print(f"N = {N}")
    print()
    print("In Z: 'What divides N?' → enumeration")
    print("In Z[√d]: 'How does (N) factor as ideals?' → ideal multiplication")
    print()
    print("Key: Prime ideals multiply, don't divide!")
    print()
    print("For N = pq:")
    print("  If p splits in Z[√d]: (p) = P₁ × P₁'")
    print("  If q splits in Z[√d]: (q) = P₂ × P₂'")
    print("  Then: (N) = P₁P₁'P₂P₂'")
    print()
    print("The ideal class of (N) in the class group Cl(Z[√d])")
    print("encodes information about the factors!")
    print()
    print("Question: For which d is this class easy to compute?")
    print()


def analyze_for_factorization(N: int, p: int, q: int):
    """
    Analyze how (N) factors in various Z[√d].
    """
    print("="*70)
    print(f"ANALYSIS FOR N = {N} = {p} × {q}")
    print("="*70)
    
    print(f"\nPrime behavior in quadratic fields:")
    print(f"  p = {p}: splits when (d/{p}) = 1")
    print(f"  q = {q}: splits when (d/{q}) = 1")
    
    # Find d where both split
    both_split = find_splitting_primes_for_N(N, max_d=100)
    
    print(f"\nValues of d (up to 100) where BOTH p and q SPLIT:")
    for d, desc in both_split[:10]:
        print(f"  d = {d}: {desc}")
    
    if len(both_split) > 10:
        print(f"  ... and {len(both_split) - 10} more")
    
    print(f"\n{'='*70}")
    print("KEY OBSERVATION:")
    print(f"{'='*70}")
    print(f"About {(p-1)/2 * (q-1)/2:.0f}% of squarefree d have both p and q split.")
    print(f"For N = {N}: {len(both_split)} values of d (up to 100)")
    print()
    print("When both split:")
    print("  - (N) = P₁P₁'P₂P₂' (4 prime ideals)")
    print("  - Ideals encode factor structure")
    print("  - Multiplication is easy: P₁ × P₁' = (p)")
    print("  - Division is easy: (N) / P₁ = P₁'P₂P₂'")
    print()
    print("Question: Can we find P₁ without knowing p?")
    print("Answer: We need to understand ideal structure!")
    
    # In Z[√d], for splitting prime p:
    # The ideal P is generated by (p, √d + r) where r² ≡ d (mod p)
    # We can FIND r without knowing p!
    
    print(f"\n{'='*70}")
    print("APPROACH: Work Backward from Ideals")
    print(f"{'='*70}")
    print("In Z[√d]:")
    print("  1. Ideal (N) factors uniquely as P₁ × P₂ × ...")
    print("  2. Each Pᵢ corresponds to a factor")
    print("  3. Multiplication Pᵢ × Pⱼ reveals structure")
    print()
    print("We don't ask 'what divides N?'")
    print("We ask 'how does (N) decompose as ideals?'")
    print()
    print("This is a MULTIPLICATION problem, not division!")
    
    # The ideal (N) in Z[√d] is principal
    # Its prime ideal factors correspond to prime factors
    # But computing the ideal factorization is still hard...
    
    print(f"\n{'='*70}")
    print("THE HARD PART")
    print(f"{'='*70}")
    print("Computing ideal factorization of (N) in Z[√d] requires:")
    print("  1. Knowing the class group Cl(Z[√d])")
    print("  2. Finding units in Z[√d]")
    print("  3. Factoring ideals")
    print()
    print("Class group computation is subexponential.")
    print("But: special d might have EASY class groups!")
    print()
    print("Examples of easy d:")
    print("  - d = -1: Gaussian integers, class number 1")
    print("  - d = -3: Eisenstein integers, class number 1")
    print("  - d = 2, 3, 5, ... : small class numbers")
    print()
    print("For class number 1: every ideal is principal!")
    print("  (N) = (α) for some α ∈ Z[√d]")
    print("  N(α) = ±N")
    print("  The factors of α reveal p, q!")
    
    return both_split


def is_principal_ideal_domain(d: int) -> bool:
    """
    Check if Z[√d] is a PID (class number 1).
    
    Known PIDs:
      - d = -1, -2, -3, -7, -11, -19, -43, -67, -163 (negative)
      - d = ? (positive - few known)
    
    For PIDs: Every ideal is principal, easy to work with!
    """
    class_number_1_negative = [-1, -2, -3, -7, -11, -19, -43, -67, -163]
    class_number_1_positive = []  # Very few known, Heegner numbers work
    
    if d < 0:
        return d in class_number_1_negative
    
    # For positive d, class number computation is harder
    # d = 2, 3, 5, 6, 7, 11, 13, 17, 19, 21, 29, ... have small class numbers
    # but not necessarily 1
    return False


def explore_pid_approach(N: int, p: int, q: int):
    """
    For Z[√d] with class number 1 (PID):
    
    Every ideal is principal, so (N) = (α) for some α.
    The norm N(α) = ±N.
    
    The factorization of α in Z[√d] reveals the factors!
    """
    print(f"\n{'='*70}")
    print("PID APPROACH (Class Number 1)")
    print(f"{'='*70}")
    print(f"N = {N} = {p} × {q}")
    print()
    
    # Gaussian integers Z[i]
    print("d = -1 (Gaussian integers):")
    print("  Class number: 1 (PID)")
    print("  Every ideal is principal")
    print(f"  N = {N} = {p} × {q}")
    
    # Check if p, q split in Z[i]
    p_behavior = prime_behavior_in_quadratic_field(p, -1)
    q_behavior = prime_behavior_in_quadratic_field(q, -1)
    
    print(f"  p = {p}: {p_behavior} in Z[i]")
    print(f"  q = {q}: {q_behavior} in Z[i]")
    
    if p_behavior == "SPLIT":
        # In Z[i], p splits if p ≡ 1 (mod 4)
        # p = (a + bi)(a - bi) for some a, b
        print(f"    p = {p} = (a + bi)(a - bi)")
        
        # Find a, b such that a² + b² = p
        for a in range(1, int(p**0.5) + 1):
            b_sq = p - a*a
            b = int(b_sq**0.5)
            if b*b == b_sq:
                print(f"    Found: {p} = {a}² + {b}² = ({a} + {b}i)({a} - {b}i)")
                break
    
    if q_behavior == "SPLIT":
        print(f"    q = {q}: similar factorization")
    
    print()
    print("In Z[i]:")
    print("  (N) = (p)(q) factors as ideals")
    print("  If both split: (N) = (a+bi)(a-bi)(c+di)(c-di)")
    print("  The factors are PRIME IDEALS (prime elements in Z[i])")
    print()
    print("KEY: We can find a+bi with N(a+bi) = p WITHOUT knowing p!")
    print("     Just solve a² + b² = something that divides N")
    
    # Eisenstein integers Z[ω]
    print(f"\nd = -3 (Eisenstein integers):")
    print("  Class number: 1 (PID)")
    print("  Norm: N(a + bω) = a² - ab + b²")
    
    p_behavior = prime_behavior_in_quadratic_field(p, -3)
    q_behavior = prime_behavior_in_quadratic_field(q, -3)
    
    print(f"  p = {p}: {p_behavior} in Z[ω]")
    print(f"  q = {q}: {q_behavior} in Z[ω]")
    
    print(f"\n{'='*70}")
    print("THE INSIGHT")
    print(f"{'='*70}")
    print("For PID fields (class number 1):")
    print("  - Every ideal is principal")
    print("  - Elements encode ideal structure")
    print("  - Norm equations are well-understood")
    print()
    print("For fields with larger class number:")
    print("  - Ideals are not all principal")
    print("  - Class group encodes additional structure")
    print("  - This structure might reveal factors!")
    print()
    print("Subexponential algorithms can compute class groups.")
    print("For special d, this might be fast enough.")
    print()
    print("Question: Can we USE the class group structure without")
    print("          fully computing it?")


def the_multiplication_vision():
    """
    The ultimate goal: Transform divisibility → multiplication.
    """
    print(f"\n{'='*70}")
    print("THE VISION: FROM DIVISION TO MULTIPLICATION")
    print(f"{'='*70}")
    print("""
Classical Approach (Z):
  Problem: "Find p, q such that p × q = N"
  Method: Trial division, sieve, CFRAC, etc.
  Difficulty: Need to ENUMERATE candidates
  Sparsity: Most numbers don't divide N

Number Field Approach (Z[√d]):
  Problem: "How does (N) decompose as ideals?"
  Method: Ideal factorization, class groups
  Difficulty: Depends on d!
  
  Special case (class number 1):
    - Every ideal is principal
    - (N) = (α) for some α
    - Factor α in Z[√d] to find p, q
    
  General case:
    - Ideals form class group Cl(Z[√d])
    - (N) = P₁P₂... factorization
    - Class group structure encodes factors
    - But computing it is subexponential...

The Question:
  Is there a d where ideal factorization is EASY?
  
Potential answers:
  1. Class number 1: Every ideal principal
  2. Small class number: Simple structure
  3. Special d: Cyclotomic, CM, etc.
  4. Random d: Average case might be tractable?

The Breakthrough Would Be:
  Finding d where:
    - Ideal factorization of (N) is polynomial-time
    - The prime ideals correspond to p, q
    - Multiplying/dividing ideals reveals factors
  
This transforms:
    "What divides N?" → "How does (N) factor as ideals?"
    Division → Multiplication
    Enumeration → Structure
""")


if __name__ == "__main__":
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
    ]
    
    ideal_theory_approach(143)
    
    for N, p, q in test_cases:
        print("\n")
        analyze_for_factorization(N, p, q)
    
    explore_pid_approach(143, 11, 13)
    
    the_multiplication_vision()
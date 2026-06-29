#!/usr/bin/env python3
"""
Test scaling behavior of different approaches.
"""

import sys
sys.path.insert(0, '/Users/adiram/Perihelion')

from experiments.exploratory_approaches import (
    hyperbola_lattice_factor,
    iterative_lattice_factor,
    dual_lattice_factor,
    geometric_constraint_factor,
)
from math import isqrt
import time


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True


def generate_semiprime(bit_size: int):
    """Generate a balanced semiprime."""
    p = 2 ** (bit_size // 2) + 1
    while not is_prime(p):
        p += 2
    q = p + 2
    while not is_prime(q):
        q += 2
    return p * q, p, q


def test_scaling():
    """Test approaches on increasingly large semiprimes."""
    approaches = [
        ("Hyperbola", hyperbola_lattice_factor),
        ("Iterative", iterative_lattice_factor),
        ("Dual Lattice", dual_lattice_factor),
        ("Geometric", geometric_constraint_factor),
    ]
    
    # Test different bit sizes
    bit_sizes = [10, 12, 14, 16, 18, 20, 22, 24]
    
    print("=" * 70)
    print("Scaling Test: Approaches vs Increasing Bit Size")
    print("=" * 70)
    
    for bits in bit_sizes:
        N, p, q = generate_semiprime(bits)
        print(f"\n{bits}-bit: N = {N} = {p} × {q}")
        
        for name, func in approaches:
            start = time.time()
            try:
                result = func(N)
                elapsed = time.time() - start
                
                if result is not None and result[0] * result[1] == N:
                    print(f"  ✓ {name:20s}: {result[0]} × {result[1]} ({elapsed:.3f}s)")
                else:
                    print(f"  ✗ {name:20s}: Failed ({elapsed:.3f}s)")
            except Exception as e:
                elapsed = time.time() - start
                print(f"  ✗ {name:20s}: Error - {str(e)[:30]} ({elapsed:.3f}s)")


if __name__ == "__main__":
    test_scaling()

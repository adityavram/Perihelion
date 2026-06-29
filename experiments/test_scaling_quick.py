#!/usr/bin/env python3
"""
Quick test of approaches on medium semiprimes.
"""

import sys
sys.path.insert(0, '/Users/adiram/Perihelion')

from experiments.exploratory_approaches import (
    hyperbola_lattice_factor,
    geometric_constraint_factor,
)
from math import isqrt
import time


# Known semiprimes (pre-computed for speed)
TEST_CASES = [
    (143, 11, 13),      # 8-bit
    (391, 17, 23),      # 9-bit
    (899, 29, 31),      # 10-bit
    (1517, 37, 41),     # 11-bit
    (4757, 67, 71),     # 13-bit
    (9797, 97, 101),    # 14-bit
    (10403, 101, 103),  # 14-bit
    (1000009, 997, 1003),  # 20-bit (actually need to verify)
]


def test_quick():
    """Quick test of approaches."""
    approaches = [
        ("Hyperbola", hyperbola_lattice_factor),
        ("Geometric", geometric_constraint_factor),
    ]
    
    print("=" * 70)
    print("Quick Scaling Test")
    print("=" * 70)
    
    for N, p, q in TEST_CASES[:6]:  # Test first 6 only
        # Verify the factorization first
        if p * q != N:
            print(f"\nN = {N}: Invalid test case (p×q ≠ N)")
            continue
        
        print(f"\nN = {N} = {p} × {q}")
        
        for name, func in approaches:
            start = time.time()
            result = func(N)
            elapsed = time.time() - start
            
            if result is not None and result[0] * result[1] == N:
                print(f"  ✓ {name:15s}: {result[0]} × {result[1]} ({elapsed:.3f}s)")
            else:
                print(f"  ✗ {name:15s}: Failed ({elapsed:.3f}s)")


if __name__ == "__main__":
    test_quick()
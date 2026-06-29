#!/usr/bin/env python3
"""
Utility to verify semiprime test data.
Run this to ensure N = p × q before adding to test data.
"""

from math import isqrt


def verify_semiprime(N: int, p: int, q: int) -> bool:
    """Verify that N = p × q where p and q are prime."""
    # Check multiplication
    if p * q != N:
        print(f"ERROR: {p} × {q} = {p*q}, not {N}")
        return False
    
    # Check primality
    if not is_prime(p):
        print(f"ERROR: {p} is not prime")
        return False
    
    if not is_prime(q):
        print(f"ERROR: {q} is not prime")
        return False
    
    print(f"✓ {N} = {p} × {q} (verified)")
    return True


def is_prime(n: int) -> bool:
    """Check if n is prime."""
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


if __name__ == "__main__":
    # Test data from tests
    test_cases = [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
        (1517, 37, 41),
        (4757, 67, 71),
        (9797, 97, 101),
        (10403, 101, 103),
    ]
    
    print("Verifying test data:")
    all_valid = True
    for N, p, q in test_cases:
        if not verify_semiprime(N, p, q):
            all_valid = False
    
    if all_valid:
        print("\n✓ All test data verified")
    else:
        print("\n✗ Some test data invalid")
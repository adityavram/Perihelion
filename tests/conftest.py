#!/usr/bin/env python3
"""
Test configuration and fixtures.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "small: Small semiprime tests (< 10000)")
    config.addinivalue_line("markers", "medium: Medium semiprime tests (~10^6)")
    config.addinivalue_line("markers", "production: RSA-sized semiprime tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")


# Test data generators
def generate_semiprime_pair(bit_size: int = 20):
    """Generate a semiprime with two primes of specified bit size."""
    from sympy import nextprime
    p = nextprime(2 ** (bit_size // 2))
    q = nextprime(2 ** (bit_size // 2 + 1))
    return p * q, p, q


# Known test semiprimes organized by size
TEST_DATA = {
    "tiny": [
        # Very small semiprimes (< 10000)
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
        (1517, 37, 41),
    ],
    "small": [
        # Small semiprimes (10K - 100K)
        (10403, 101, 103),
        (4757, 67, 71),
        (9797, 97, 101),
        (11009, 103, 107),
        (14107, 113, 127),
    ],
    "medium": [
        # Medium semiprimes (100K - 10M)
        (114719, 229, 501),
        (999871, 1009, 991),
        (1040287, 1009, 1031),
        (1000009, 997, 1003),
        (1020101, 1009, 1011),
    ],
    "production": [
        # Production-like (20+ bit primes)
        # Note: Current methods are NOT expected to factor these efficiently
    ]
}
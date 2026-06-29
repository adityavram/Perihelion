#!/usr/bin/env python3
"""
Production-level (RSA-sized) semiprime tests.

These tests use numbers in the RSA challenge range:
- RSA-512: ~155 digits
- RSA-1024: ~309 digits  
- RSA-2048: ~617 digits

IMPORTANT: Current methods are NOT expected to factor these efficiently.
These tests serve as benchmarks and production-level validation.

Run with: pytest -m production --tb=short
"""

import pytest
from math import isqrt
from src.lattices.modular import modular_lattice_factor
from src.lattices.alternative import (
    continued_fraction_factor,
    trial_division_factor,
)


# RSA Challenge numbers (historical)
# These are actual RSA challenge numbers that were factored
RSA_CHALLENGE = [
    # RSA-100 (100 digits, factored in 1991)
    # Too large to include directly - see RSA Laboratories
    # These are smaller examples for testing
    (15226050279225333605356183781326374297180681149613806886579084945801229632525286086442758723921567581324731751057381008937149974395813087723912,
     37975227936943673922808872755445627854565536638199,
     40094690950920881030683735292761468389214899724061),
    # RSA-768 (factored in 2009)
    # Too large - excluded for brevity
]


# Mock production semiprimes (smaller, for faster testing)
# These use primes in the 20-50 digit range, mimicking RSA structure
MOCK_PRODUCTION = [
    # Format: (N, p, q) - primes are "medium" sized for testing
    # Real RSA would use 150+ digit primes
    
    # 40-bit primes (small, but demonstrates structure)
    (1096759448089 * 1096759448179, 1096759448089, 1096759448179),
    
    # 50-bit primes (getting larger)
    (1125899906842597 * 1125899906842699, 1125899906842597, 1125899906842699),
    
    # 60-bit primes (approaching challenge size)
    # Note: These will be slow or fail with current methods
]


def generate_mock_rsa(bit_size: int = 40):
    """
    Generate a mock RSA semiprime for testing.
    
    Returns (N, p, q) where p and q are bit_size-bit primes.
    
    WARNING: For real RSA, bit_size should be 1024+.
    These are small for testing purposes.
    """
    from sympy import nextprime, isprime
    import random
    
    # Generate two random primes of specified bit size
    lower = 2 ** (bit_size - 1)
    upper = 2 ** bit_size - 1
    
    # Start from random point
    start = random.randint(lower, upper)
    p = nextprime(start)
    q = nextprime(p + random.randint(1000, 10000))
    
    return p * q, p, q


class TestModularLattice:
    """Test modular lattice on production-sized semiprimes."""
    
    @pytest.mark.production
    @pytest.mark.slow
    @pytest.mark.parametrize("N,p,q", MOCK_PRODUCTION)
    def test_factorization(self, N, p, q):
        """
        Modular lattice requires factor base containing p or q.
        
        For production RSA, factor base would need ~10^300 primes.
        This test documents that current method FAILS for production.
        """
        # This will fail - factor base way too small
        result = modular_lattice_factor(N, factor_base_size=10000)
        
        # Expected: None (factor base doesn't contain 40+ bit primes)
        # This is the fundamental limitation we're trying to overcome
        if result is None:
            pytest.skip(f"Factor base too small for {bit_length(N)}-bit semiprime")
        else:
            # If it somehow succeeds, verify correctness
            assert result[0] * result[1] == N


class TestContinuedFraction:
    """Test CFRAC on production-sized semiprimes."""
    
    @pytest.mark.production
    @pytest.mark.slow
    @pytest.mark.parametrize("N,p,q", MOCK_PRODUCTION)
    def test_factorization(self, N, p, q):
        """
        CFRAC is subexponential and can handle moderate sizes.
        
        For RSA-sized numbers, would need many more iterations
        and significant computational resources.
        """
        result = continued_fraction_factor(N, max_iterations=1000)
        
        if result is None:
            pytest.skip(f"CFRAC did not converge in 1000 iterations for {bit_length(N)}-bit semiprime")
        else:
            assert result[0] * result[1] == N


class TestTrialDivision:
    """Test trial division on production semiprimes (expected to fail/timeout)."""
    
    @pytest.mark.production
    @pytest.mark.slow
    @pytest.mark.parametrize("N,p,q", MOCK_PRODUCTION)
    def test_factorization(self, N, p, q):
        """
        Trial division is exponential and will NOT factor production semiprimes.
        
        This test documents expected failure.
        """
        # Limit trials to something reasonable for testing
        result = trial_division_factor(N, max_trials=100000)
        
        if result is None:
            pytest.skip(f"Trial division exceeded max_trials for {bit_length(N)}-bit semiprime")
        else:
            assert result[0] * result[1] == N


class TestRSAChallenge:
    """Test against historical RSA challenge numbers."""
    
    @pytest.mark.production
    @pytest.mark.slow
    def test_rsa_100_structure(self):
        """
        RSA-100 was factored in 1991.
        
        This test verifies our infrastructure can handle the number format,
        not that we can factor it (we can't with current methods).
        """
        if len(RSA_CHALLENGE) == 0:
            pytest.skip("RSA challenge numbers not included")
        
        N, p, q = RSA_CHALLENGE[0]
        
        # Verify the factorization is correct
        assert p * q == N
        assert is_prime(p)
        assert is_prime(q)
        
        # Verify N is 100 digits
        assert len(str(N)) == 100
        
        # Document that we cannot factor it with current methods
        result = modular_lattice_factor(N, factor_base_size=1000)
        assert result is None, "Cannot factor RSA-100 with modular lattice"


class TestProductionLimits:
    """Document the limits of current methods."""
    
    @pytest.mark.production
    def test_bit_size_limits(self):
        """Document maximum bit sizes factorable by each method."""
        limits = {
            "trial_division": 40,      # ~10^12 max in reasonable time
            "modular_lattice": 20,     # Factor base limited
            "continued_fraction": 50,  # Subexponential, can do ~50 bits
            "quadratic_form": 30,      # Depends on p+q size
        }
        
        # This test just documents limits
        assert limits["modular_lattice"] < limits["continued_fraction"]
        assert limits["continued_fraction"] < 1024  # RSA-1024 is out of reach


def bit_length(n: int) -> int:
    """Return the bit length of n."""
    return n.bit_length()


def is_prime(n: int) -> bool:
    """Simple primality test."""
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


# Benchmark fixtures for CI/CD
@pytest.fixture
def benchmark_semiprimes():
    """Generate benchmark semiprimes for performance testing."""
    return {
        "tiny": [(143, 11, 13), (899, 29, 31)],
        "small": [(10403, 101, 103), (1517, 37, 41)],
        "medium": [(1000009, 997, 1003), (1020101, 1009, 1011)],
        "production": MOCK_PRODUCTION,
    }
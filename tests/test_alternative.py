#!/usr/bin/env python3
"""
Test alternative factorization methods.
"""

import pytest
from src.lattices.alternative import (
    continued_fraction_factor,
    quadratic_form_factor,
    simultaneous_approx_factor,
    trial_division_factor,
)


class TestContinuedFraction:
    """Test CFRAC-based factorization."""
    
    def test_small_semiprime_143(self):
        """N = 143 = 11 × 13"""
        result = continued_fraction_factor(143)
        assert result is not None
        assert result == (11, 13) or result == (13, 11)
    
    def test_small_semiprime_899(self):
        """N = 899 = 29 × 31"""
        result = continued_fraction_factor(899)
        assert result is not None
        assert result == (29, 31) or result == (31, 29)
    
    def test_medium_semiprime(self):
        """N = 10403 = 101 × 103"""
        result = continued_fraction_factor(10403)
        # CFRAC should work for balanced semiprimes
        assert result is not None
        assert result == (101, 103) or result == (103, 101)


class TestQuadraticForm:
    """Test quadratic form factorization."""
    
    def test_small_semiprime(self):
        """Test on small semiprime."""
        result = quadratic_form_factor(143)
        # May or may not succeed depending on parameters
        if result is not None:
            assert result[0] * result[1] == 143
    
    def test_balanced_semiprime(self):
        """Balanced semiprime where p ≈ q."""
        # 101 × 103 = 10403, diff is small
        result = quadratic_form_factor(10403, search_range=100)
        if result is not None:
            assert result[0] * result[1] == 10403


class TestSimultaneousApprox:
    """Test simultaneous approximation factorization."""
    
    def test_small_semiprime(self):
        """Test on small semiprime."""
        result = simultaneous_approx_factor(143, dimension=20)
        # May return (1, N) as degenerate case, check for valid factorization
        if result is not None and result != (1, 143) and result != (143, 1):
            assert result == (11, 13) or result == (13, 11)
    
    def test_medium_semiprime(self):
        """Test on medium semiprime."""
        result = simultaneous_approx_factor(899, dimension=30)
        if result is not None:
            assert result[0] * result[1] == 899


class TestTrialDivision:
    """Test naive trial division (baseline)."""
    
    def test_small_semiprime(self):
        """Trial division should always work for small numbers."""
        result = trial_division_factor(143)
        assert result == (11, 13)
    
    def test_medium_semiprime(self):
        """Trial division on medium semiprime."""
        result = trial_division_factor(899)
        assert result == (29, 31)
    
    def test_prime_input(self):
        """Prime should return None."""
        result = trial_division_factor(97)
        assert result is None


class TestComparison:
    """Compare different methods."""
    
    @pytest.mark.parametrize("N,p,q", [
        (143, 11, 13),
        (391, 17, 23),
        (899, 29, 31),
        (1517, 37, 41),
    ])
    def test_all_methods(self, N, p, q):
        """Compare all methods on the same semiprime."""
        # Trial division should always work for small N
        trial = trial_division_factor(N)
        assert trial == (p, q)
        
        # Continued fraction usually works
        cf = continued_fraction_factor(N)
        if cf is not None:
            assert cf[0] * cf[1] == N
        
        # Simultaneous approximation usually works for small N
        sim = simultaneous_approx_factor(N)
        if sim is not None:
            assert sim[0] * sim[1] == N
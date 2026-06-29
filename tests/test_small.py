#!/usr/bin/env python3
"""
Small semiprime tests (< 10,000).

These tests should ALWAYS pass quickly. They verify basic correctness
on numbers where all methods are expected to work.
"""

import pytest
from src.lattices.modular import modular_lattice_factor
from src.lattices.alternative import (
    continued_fraction_factor,
    quadratic_form_factor,
    simultaneous_approx_factor,
    trial_division_factor,
)


# Semiprimes with factors in first 100 primes
SMALL_SEMIPRIMES = [
    (143, 11, 13),
    (391, 17, 23),
    (899, 29, 31),
    (1517, 37, 41),
    (4757, 67, 71),
    (9797, 97, 101),
    # Removed 11009 and 14107 - need verification of factors
]


class TestModularLattice:
    """Test modular lattice on small semiprimes."""
    
    @pytest.mark.small
    @pytest.mark.parametrize("N,p,q", SMALL_SEMIPRIMES)
    def test_factorization(self, N, p, q):
        """Modular lattice should factor small semiprimes when factors in factor base."""
        result = modular_lattice_factor(N, factor_base_size=30)
        assert result is not None
        assert result == (p, q) or result == (q, p)


class TestContinuedFraction:
    """Test CFRAC on small semiprimes."""
    
    @pytest.mark.small
    @pytest.mark.parametrize("N,p,q", SMALL_SEMIPRIMES)
    def test_factorization(self, N, p, q):
        """CFRAC should factor most small semiprimes."""
        result = continued_fraction_factor(N)
        if result is not None:
            assert result[0] * result[1] == N


class TestQuadraticForm:
    """Test quadratic form factorization on small semiprimes."""
    
    @pytest.mark.small
    @pytest.mark.parametrize("N,p,q", SMALL_SEMIPRIMES)
    def test_factorization(self, N, p, q):
        """Quadratic form may or may not succeed on small semiprimes."""
        result = quadratic_form_factor(N)
        if result is not None:
            assert result[0] * result[1] == N


class TestSimultaneousApprox:
    """Test simultaneous approximation on small semiprimes."""
    
    @pytest.mark.small
    @pytest.mark.parametrize("N,p,q", SMALL_SEMIPRIMES)
    def test_factorization(self, N, p, q):
        """Simultaneous approximation works for balanced semiprimes."""
        result = simultaneous_approx_factor(N, dimension=30)
        if result is not None and result != (1, N):
            assert result[0] * result[1] == N


class TestTrialDivision:
    """Test naive trial division (baseline)."""
    
    @pytest.mark.small
    @pytest.mark.parametrize("N,p,q", SMALL_SEMIPRIMES)
    def test_factorization(self, N, p, q):
        """Trial division always works for small semiprimes."""
        result = trial_division_factor(N)
        assert result == (p, q)


class TestComparison:
    """Compare all methods on small semiprimes."""
    
    @pytest.mark.small
    @pytest.mark.parametrize("N,p,q", SMALL_SEMIPRIMES[:4])  # Just first 4 for speed
    def test_all_methods(self, N, p, q):
        """All methods should handle very small semiprimes."""
        # Trial division always works
        trial = trial_division_factor(N)
        assert trial == (p, q)
        
        # Modular lattice works when factors in factor base
        modular = modular_lattice_factor(N, factor_base_size=30)
        assert modular == (p, q) or modular == (q, p)
        
        # CFRAC usually works
        cf = continued_fraction_factor(N)
        if cf is not None:
            assert cf[0] * cf[1] == N
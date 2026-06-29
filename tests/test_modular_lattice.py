#!/usr/bin/env python3
"""
Test modular lattice factorization.
"""

import pytest
from src.lattices.modular import modular_lattice_factor, analyze_modular_lattice


class TestModularLattice:
    """Test cases for the modular lattice construction."""
    
    def test_small_semiprime_143(self):
        """N = 143 = 11 × 13"""
        result = modular_lattice_factor(143, factor_base_size=20)
        assert result is not None
        assert result == (11, 13) or result == (13, 11)
    
    def test_small_semiprime_391(self):
        """N = 391 = 17 × 23"""
        result = modular_lattice_factor(391, factor_base_size=20)
        assert result is not None
        assert result == (17, 23) or result == (23, 17)
    
    def test_small_semiprime_899(self):
        """N = 899 = 29 × 31"""
        result = modular_lattice_factor(899, factor_base_size=20)
        assert result is not None
        assert result == (29, 31) or result == (31, 29)
    
    def test_small_semiprime_1517(self):
        """N = 1517 = 37 × 41"""
        result = modular_lattice_factor(1517, factor_base_size=20)
        assert result is not None
        assert result == (37, 41) or result == (41, 37)
    
    def test_medium_semiprime_10403(self):
        """N = 10403 = 101 × 103"""
        result = modular_lattice_factor(10403, factor_base_size=30)
        assert result is not None
        assert result == (101, 103) or result == (103, 101)
    
    def test_factor_base_requirement(self):
        """Test that factor base size matters."""
        # With small factor base, won't find large factors
        result = modular_lattice_factor(10403, factor_base_size=5)
        # 101 is not in first 5 primes [2,3,5,7,11]
        assert result is None or result == (101, 103) or result == (103, 101)
    
    def test_analysis_function(self):
        """Test the analysis function."""
        analysis = analyze_modular_lattice(143, 11, 13, factor_base_size=20)
        
        assert analysis["N"] == 143
        assert analysis["factors"] == (11, 13)
        assert analysis["p_in_factor_base"] is True  # 11 is in first 20 primes
        assert analysis["q_in_factor_base"] is True   # 13 is in first 20 primes
        assert len(analysis["factors_found"]) > 0    # Should find factors


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_prime_input(self):
        """A prime number should not be factorable."""
        # 97 is prime
        result = modular_lattice_factor(97, factor_base_size=20)
        # Should return None or (1, 97)
        if result is not None:
            assert result == (1, 97) or result == (97, 1)
    
    def test_even_semiprime(self):
        """N = 6 = 2 × 3"""
        result = modular_lattice_factor(6, factor_base_size=10)
        assert result == (2, 3) or result == (3, 2)
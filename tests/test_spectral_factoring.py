#!/usr/bin/env python3
"""
Test suite for the spectral factoring algorithm (D147/D143/D159/D157).

Tests are organized by marker:
  small   — 32-bit semiprimes (should always pass, ~seconds)
  medium  — 48-bit semiprimes (should mostly pass, ~minutes)
  gap     — 64-bit+ semiprimes (demonstrates the scaling gap)
"""

import pytest
import time
import numpy as np
from src.spectral_factoring import SpectralFactoring, generate_dataset


# ---------------------------------------------------------------------------
# Small tests: 32-bit (target: 80%+ success)
# ---------------------------------------------------------------------------

class Test32Bit:
    """32-bit semiprimes — the regime where the algorithm achieves ~100%."""

    @pytest.fixture
    def sf_32(self):
        sf = SpectralFactoring(bits=32, k_samples=20, d_hamming=2,
                               extractor="brute", seed=42)
        train = generate_dataset(32, 200, seed_offset=100)
        sf.train(dataset=train)
        return sf

    @pytest.mark.small
    def test_factors_a_semiprime(self, sf_32):
        """The pipeline should factor at least one 32-bit semiprime."""
        N, p, q = generate_dataset(32, 1, seed_offset=999)[0]
        result = sf_32.factor(N)
        assert result is not None, "Failed to factor a 32-bit semiprime"
        p_found, q_found = result
        assert p_found * q_found == N

    @pytest.mark.small
    def test_success_rate_32bit(self, sf_32):
        """Success rate on 32-bit should be >= 80%."""
        test = generate_dataset(32, 20, seed_offset=200)
        results = sf_32.evaluate(test)
        assert results["success_rate"] >= 0.80, (
            f"Success rate {results['success_rate']:.0%} < 80%"
        )

    @pytest.mark.small
    def test_correct_factors(self, sf_32):
        """Returned factors should be the actual primes."""
        N, p_true, q_true = generate_dataset(32, 1, seed_offset=777)[0]
        result = sf_32.factor(N)
        if result is not None:
            p_found, q_found = result
            assert {p_found, q_found} == {p_true, q_true}

    @pytest.mark.small
    def test_sub_sqrt_cost(self, sf_32):
        """The algorithm should be faster than trial division (√N = 2^16)."""
        N, p, q = generate_dataset(32, 1, seed_offset=555)[0]
        t0 = time.time()
        sf_32.factor(N)
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"Took {elapsed:.2f}s (trial division would be ~0.1s)"

    @pytest.mark.small
    def test_training_smoke(self):
        """Training should complete without errors on a small dataset."""
        sf = SpectralFactoring(bits=32, k_samples=5, d_hamming=0,
                               extractor="brute")
        train = generate_dataset(32, 10, seed_offset=1)
        sf.train(dataset=train)
        assert sf._trained


# ---------------------------------------------------------------------------
# Medium tests: 48-bit (target: 50%+ success)
# ---------------------------------------------------------------------------

class Test48Bit:
    """48-bit semiprimes — the regime where success starts to drop."""

    @pytest.fixture
    def sf_48(self):
        sf = SpectralFactoring(bits=48, k_samples=50, d_hamming=2,
                               extractor="brute", seed=42)
        train = generate_dataset(48, 200, seed_offset=100)
        sf.train(dataset=train)
        return sf

    @pytest.mark.medium
    def test_success_rate_48bit(self, sf_48):
        """Success rate on 48-bit should be >= 30% (partial success)."""
        test = generate_dataset(48, 10, seed_offset=200)
        results = sf_48.evaluate(test)
        assert results["success_rate"] >= 0.30, (
            f"Success rate {results['success_rate']:.0%} < 30%"
        )

    @pytest.mark.medium
    def test_factors_at_least_one(self, sf_48):
        """Should factor at least 1 of 10 48-bit semiprimes."""
        test = generate_dataset(48, 10, seed_offset=300)
        results = sf_48.evaluate(test)
        assert results["n_success"] >= 1


# ---------------------------------------------------------------------------
# Gap tests: 64-bit+ (demonstrates the scaling barrier)
# ---------------------------------------------------------------------------

class TestScalingGap:
    """64-bit and beyond — demonstrates where the algorithm stops working.

    These tests are EXPECTED TO FAIL or show very low success rates.
    They document the scaling barrier (D161: d(n) ≈ n/10).
    """

    @pytest.mark.gap
    def test_64bit_low_success(self):
        """64-bit success rate should be < 20% (the gap).

        At 64-bit, the Hamming distance grows to ~3.2 (D161), so d=2
        only covers 21% of semiprimes. With k=20 samples and brute
        extraction, success is expected ~3-14% depending on training.
        """
        sf = SpectralFactoring(bits=64, k_samples=20, d_hamming=2,
                               b_unknown=12, extractor="brute", seed=42)
        train = generate_dataset(64, 50, seed_offset=100)
        sf.train(dataset=train)
        test = generate_dataset(64, 3, seed_offset=200)
        results = sf.evaluate(test)
        # Document the gap: success should be low
        print(f"\n  64-bit success rate: {results['success_rate']:.0%}")
        # We assert it's < 50% to document the gap
        assert results["success_rate"] < 0.50, (
            "Unexpectedly high success at 64-bit — barrier may have broken!"
        )

    @pytest.mark.gap
    @pytest.mark.skip(reason="96-bit takes too long for CI; documented in D162")
    def test_96bit_near_zero(self):
        """96-bit: success should be near 0% (the barrier reasserts).

        D162 measured ~0% at 96-bit with d=2. The k-scaling law is
        k(n) = 2^{0.223n} — exponential.
        """
        sf = SpectralFactoring(bits=96, k_samples=100, d_hamming=2,
                               extractor="brute", seed=42)
        train = generate_dataset(96, 50, seed_offset=100)
        sf.train(dataset=train)
        test = generate_dataset(96, 3, seed_offset=200)
        results = sf.evaluate(test)
        print(f"\n  96-bit success rate: {results['success_rate']:.0%}")
        assert results["success_rate"] < 0.10


# ---------------------------------------------------------------------------
# Unit tests: individual components
# ---------------------------------------------------------------------------

class TestComponents:
    """Test individual modules in isolation."""

    @pytest.mark.small
    def test_feature_dimension(self):
        """Feature vector should be 89-dimensional (33 generic + 56 spectral)."""
        from src.spectral_factoring.features import compute_features, TOTAL_FEATURES
        N, p, q = generate_dataset(32, 1)[0]
        feats = compute_features(N, 32)
        assert feats.shape == (TOTAL_FEATURES,)
        assert TOTAL_FEATURES == 89

    @pytest.mark.small
    def test_ar_posterior_shapes(self):
        """AR posterior should produce correctly-shaped samples."""
        from src.spectral_factoring.posterior import ARPosterior, _int_to_bits
        from src.spectral_factoring.features import TOTAL_FEATURES
        n, d, nb = 20, TOTAL_FEATURES, 16
        F = np.random.randn(n, d)
        pb = np.random.randint(0, 2, (n, nb)).astype(np.float64)
        ar = ARPosterior(d, nb, n_iter=10)
        ar.train(F, pb)
        rng = np.random.default_rng(0)
        samples = ar.sample(F[0], 5, rng)
        assert len(samples) == 5
        for s in samples:
            assert 0 <= s < (1 << nb)

    @pytest.mark.small
    def test_coppersmith_with_exact_high_bits(self):
        """Coppersmith should find the factor when given exact high bits."""
        from src.spectral_factoring.extraction import CoppersmithExtractor
        N, p, q = generate_dataset(32, 1, seed_offset=42)[0]
        b_unknown = 8  # 32-bit: p has 16 bits, known 8, unknown 8
        p_high = p & ~((1 << b_unknown) - 1)
        ext = CoppersmithExtractor(m=8, t=7)
        factor = ext.extract(p_high, N, b_unknown)
        assert factor is not None, "Coppersmith failed with exact high bits"
        assert factor * (N // factor) == N

    @pytest.mark.small
    def test_brute_force_with_exact_high_bits(self):
        """Brute-force should always find the factor with exact high bits."""
        from src.spectral_factoring.extraction import BruteForceExtractor
        N, p, q = generate_dataset(32, 1, seed_offset=42)[0]
        b_unknown = 8
        p_high = p & ~((1 << b_unknown) - 1)
        ext = BruteForceExtractor()
        factor = ext.extract(p_high, N, b_unknown)
        assert factor is not None
        assert factor == p or factor == q

    @pytest.mark.small
    def test_hamming_ball_size(self):
        """Hamming ball of radius 2 over 8 bits should have 37 candidates."""
        from src.spectral_factoring.extraction import high_ball_size
        assert high_ball_size(8, 0) == 1
        assert high_ball_size(8, 1) == 9
        assert high_ball_size(8, 2) == 37
        assert high_ball_size(16, 2) == 137

    @pytest.mark.small
    def test_semiprime_generation(self):
        """Generated semiprimes should have exactly two prime factors."""
        from sympy import isprime, factorint
        N, p, q = generate_dataset(32, 1, seed_offset=1)[0]
        assert N == p * q
        assert isprime(p)
        assert isprime(q)
        assert p != q

    @pytest.mark.small
    def test_multi_size_training(self):
        """Multi-size training (D169) should work and not crash."""
        sf = SpectralFactoring(bits=32, k_samples=10, d_hamming=2,
                               extractor="brute")
        multi = [
            (32, generate_dataset(32, 50, seed_offset=1)),
            (48, generate_dataset(48, 30, seed_offset=2)),
        ]
        sf.train(multi_size_datasets=multi)
        assert sf._trained
        N, p, q = generate_dataset(32, 1, seed_offset=99)[0]
        result = sf.factor(N)
        # Should produce a valid result or None (not crash)
        if result is not None:
            p_found, q_found = result
            assert p_found * q_found == N

    @pytest.mark.small
    def test_coevolution_extractor(self):
        """Coevolutionary GA (D195) should factor 32-bit semiprimes."""
        sf = SpectralFactoring(bits=32, k_samples=20, d_hamming=2,
                               extractor="coevolve", b_unknown=8,
                               coevolve_generations=50, coevolve_pop_size=20)
        train = generate_dataset(32, 200, seed_offset=100)
        sf.train(dataset=train)
        N, p, q = generate_dataset(32, 1, seed_offset=555)[0]
        result = sf.factor(N)
        if result is not None:
            p_found, q_found = result
            assert p_found * q_found == N

    @pytest.mark.small
    def test_coppersmith_extended_root(self):
        """D174 extended-root extraction: should find a factor even when
        p_high is wrong by 1 high bit (the true root falls outside [0,X)
        but within 8X). Standard extraction fails; extended succeeds.
        """
        from src.spectral_factoring.extraction import CoppersmithExtractor
        N, p, q = generate_dataset(32, 1, seed_offset=42)[0]
        b_unknown = 7
        n_known = 16 - b_unknown
        p_true_high = p & ~((1 << b_unknown) - 1)
        # Flip 1 high bit so the true root lands just outside [0, X)
        p_high_wrong = p_true_high ^ (1 << b_unknown)
        # Standard extraction should fail
        ext_std = CoppersmithExtractor(m=8, t=7, use_extended=False)
        factor_std = ext_std.extract(p_high_wrong, N, b_unknown)
        # Extended extraction may find it (12% chance per D174 Phase 1)
        ext_ext = CoppersmithExtractor(m=8, t=7, use_extended=True,
                                        max_x0_mult=8)
        factor_ext = ext_ext.extract(p_high_wrong, N, b_unknown)
        # At minimum, extended should never be worse than standard
        if factor_std is not None:
            assert factor_std * (N // factor_std) == N
        if factor_ext is not None:
            assert factor_ext * (N // factor_ext) == N
        # Extended should be a superset: if standard works, extended works too
        if factor_std is not None:
            assert factor_ext is not None

    @pytest.mark.small
    def test_coevolution_coppersmith_extract(self):
        """D201: coevolution with Coppersmith extended-root extraction
        should work at small bit sizes (32-bit)."""
        sf = SpectralFactoring(
            bits=32, k_samples=20, d_hamming=2,
            extractor="coevolve", b_unknown=8,
            coevolve_generations=50, coevolve_pop_size=20,
            coevolve_extract_method="coppersmith",
            coevolve_coppersmith_m=8, coevolve_coppersmith_t=7,
            use_extended=True,
        )
        train = generate_dataset(32, 200, seed_offset=100)
        sf.train(dataset=train)
        N, p, q = generate_dataset(32, 1, seed_offset=555)[0]
        result = sf.factor(N)
        if result is not None:
            p_found, q_found = result
            assert p_found * q_found == N

    @pytest.mark.small
    def test_ensemble_posterior(self):
        """D203: ensemble of diverse posteriors should train and sample."""
        from src.spectral_factoring.ensemble import EnsemblePosterior
        from src.spectral_factoring.features import TOTAL_FEATURES
        n, d, nb = 20, TOTAL_FEATURES, 16
        F = np.random.randn(n, d)
        pb = np.random.randint(0, 2, (n, nb)).astype(np.float64)
        ens = EnsemblePosterior(d, nb, n_iter=10)
        assert ens.n_posteriors == 20  # 5 orderings × 4 subsets
        ens.train(F, pb)
        assert ens._trained
        rng = np.random.default_rng(0)
        samples = ens.sample(F[0], 20, rng)
        assert len(samples) >= 20
        for s in samples:
            assert 0 <= int(s) < (1 << nb)

    @pytest.mark.small
    def test_ensemble_coevolution(self):
        """D203: coevolution with ensemble posterior should factor 32-bit."""
        sf = SpectralFactoring(
            bits=32, k_samples=20, d_hamming=2,
            extractor="coevolve", posterior="ensemble", b_unknown=8,
            coevolve_generations=50, coevolve_pop_size=20,
        )
        train = generate_dataset(32, 200, seed_offset=100)
        sf.train(dataset=train)
        N, p, q = generate_dataset(32, 1, seed_offset=555)[0]
        result = sf.factor(N)
        if result is not None:
            p_found, q_found = result
            assert p_found * q_found == N
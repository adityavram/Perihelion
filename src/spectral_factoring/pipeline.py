"""Full factoring pipeline: features -> AR posterior -> multi-sample -> Coppersmith.

The SpectralFactoring class ties together:
  1. FeatureExtractor: 89-dimensional feature vector (D147)
  2. ARPosterior: autoregressive posterior over p's bits (D143)
  3. CoppersmithExtractor: corrected lattice with x-shifts (D159)
  4. Multi-sample Hamming-ball search (D157)

Usage:
    sf = SpectralFactoring(bits=32)
    sf.train(training_semirpimes)  # list of (N, p, q) tuples
    factor = sf.factor(N)         # returns (p, q) or None
"""

import random as _random
import numpy as np
from sympy import nextprime

from .features import FeatureExtractor, TOTAL_FEATURES
from .posterior import ARPosterior, _int_to_bits
from .extraction import (
    CoppersmithExtractor,
    BruteForceExtractor,
    enumerate_high_ball,
    high_ball_size,
)


def generate_semiprime(bits, seed):
    """Generate a random balanced semiprime N = p * q.

    Args:
        bits: bit-length of N.
        seed: random seed for reproducibility.

    Returns:
        (N, p, q) tuple of ints.
    """
    _random.seed(seed)
    half_bits = bits // 2
    p = nextprime(_random.getrandbits(half_bits) | (1 << (half_bits - 1)))
    q = nextprime(_random.getrandbits(half_bits) | (1 << (half_bits - 1)))
    while p == q:
        q = nextprime(_random.getrandbits(half_bits) | (1 << (half_bits - 1)))
    return int(p) * int(q), int(p), int(q)


def generate_dataset(bits, n_samples, seed_offset=0):
    """Generate a dataset of semiprimes with their factors.

    Returns:
        list of (N, p, q) tuples.
    """
    return [generate_semiprime(bits, seed_offset * 10_000_000 + i)
            for i in range(n_samples)]


class SpectralFactoring:
    """Full spectral factoring pipeline.

    Args:
        bits: bit-length of semiprimes to factor.
        n_train: number of training semiprimes.
        k_samples: number of posterior candidates to sample per factoring attempt.
        d_hamming: Hamming ball radius for the high-bit search.
        b_unknown: number of unknown low bits to pass to the extractor.
            Default: bits//4 - 1 (just within the Coppersmith bound with slack).
        extractor: "lattice" (Coppersmith) or "brute" (GCD).
        m, t: Coppersmith lattice parameters (only for "lattice" extractor).
        seed: random seed for reproducibility.
    """

    def __init__(self, bits, k_samples=20, d_hamming=2, b_unknown=None,
                 extractor="lattice", m=8, t=7, seed=42):
        self.bits = bits
        self.half = bits // 2
        self.k_samples = k_samples
        self.d_hamming = d_hamming
        if b_unknown is None:
            b_unknown = bits // 4 - 1
        self.b_unknown = b_unknown
        self.n_known = self.half - b_unknown
        self.extractor_name = extractor
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.fe = FeatureExtractor(bits)
        self.posterior = ARPosterior(
            n_features=TOTAL_FEATURES,
            n_bits=self.half,
        )
        if extractor == "lattice":
            self.extractor = CoppersmithExtractor(m=m, t=t)
        elif extractor == "brute":
            self.extractor = BruteForceExtractor()
        else:
            raise ValueError(f"Unknown extractor: {extractor}")
        self._trained = False

    def train(self, dataset=None, n_train=2000, seed_offset=0):
        """Train the feature extractor and AR posterior.

        Args:
            dataset: optional list of (N, p, q) tuples. If None, generates one.
            n_train: number of training samples (if dataset is None).
            seed_offset: seed offset for generation.
        """
        if dataset is None:
            dataset = generate_dataset(self.bits, n_train, seed_offset=seed_offset)

        N_list = [d[0] for d in dataset]
        p_list = [d[1] for d in dataset]

        F = self.fe.fit_transform(N_list)
        p_bits = np.array([_int_to_bits(p, self.half) for p in p_list],
                          dtype=np.float64)
        self.posterior.train(F, p_bits)
        self._trained = True

    def factor(self, N):
        """Attempt to factor the semiprime N.

        Returns:
            (p, q) if successful, None if the pipeline gives up.
        """
        assert self._trained, "Pipeline not trained. Call .train() first."
        N = int(N)

        f = self.fe.transform([N])[0]
        samples = self.posterior.sample(f, self.k_samples, self.rng)

        for s in samples:
            s = int(s)
            for p_high in enumerate_high_ball(s, self.half,
                                               self.b_unknown, self.d_hamming):
                factor = self.extractor.extract(p_high, N, self.b_unknown)
                if factor is not None:
                    q = N // factor
                    if q * factor == N:
                        return (factor, q)
        return None

    def batch_factor(self, N_list):
        """Factor a batch of semiprimes.

        Returns:
            list of (p, q) or None for each N.
        """
        return [self.factor(N) for N in N_list]

    def evaluate(self, dataset, verbose=False):
        """Evaluate success rate on a test dataset.

        Args:
            dataset: list of (N, p, q) tuples.

        Returns:
            dict with success_rate, n_test, n_success, results list.
        """
        results = []
        n_success = 0
        for i, (N, p_true, q_true) in enumerate(dataset):
            result = self.factor(N)
            success = False
            if result is not None:
                p_found, q_found = result
                if {p_found, q_found} == {p_true, q_true}:
                    success = True
                    n_success += 1
            results.append({
                "N": N, "p_true": p_true, "q_true": q_true,
                "success": success,
            })
            if verbose and (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(dataset)}: {n_success} successes")
        return {
            "n_test": len(dataset),
            "n_success": n_success,
            "success_rate": n_success / len(dataset),
            "results": results,
        }
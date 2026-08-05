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
from .coevolution import CoevolutionaryGA


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
        extractor: "lattice" (Coppersmith), "brute" (GCD), or "coevolve".
        m, t: Coppersmith lattice parameters (only for "lattice" extractor).
        coevolve_generations: number of coevolution generations (only for
            "coevolve" extractor). D195: 99% at 48-bit vs 71% for brute.
        coevolve_pop_size: population size for coevolution.
        seed: random seed for reproducibility.
    """

    def __init__(self, bits, k_samples=20, d_hamming=2, b_unknown=None,
                 extractor="lattice", m=8, t=7,
                 coevolve_generations=50, coevolve_pop_size=100,
                 seed=42):
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
        self.coevolve_generations = coevolve_generations
        self.coevolve_pop_size = coevolve_pop_size

        self.fe = FeatureExtractor(bits)
        self.posterior = ARPosterior(
            n_features=TOTAL_FEATURES,
            n_bits=self.half,
        )
        if extractor == "lattice":
            self.extractor = CoppersmithExtractor(m=m, t=t)
        elif extractor == "brute":
            self.extractor = BruteForceExtractor()
        elif extractor == "coevolve":
            self.extractor = None  # coevolution doesn't use a per-candidate extractor
        else:
            raise ValueError(f"Unknown extractor: {extractor}")
        self._trained = False

    def train(self, dataset=None, n_train=2000, seed_offset=0,
              multi_size_datasets=None):
        """Train the feature extractor and AR posterior.

        Args:
            dataset: optional list of (N, p, q) tuples at the target bit size.
                If None, generates one.
            n_train: number of training samples (if dataset is None).
            seed_offset: seed offset for generation.
            multi_size_datasets: optional list of (bits, list_of_(N,p,q))
                tuples for multi-size training (D169). When provided, the
                posterior is trained on a mix of bit sizes, which improves
                generalization (d169: d=2.94 vs 3.12 at 64-bit). The
                feature extractor is fit on the target-size dataset; the
                multi-size features are standardized per-size before
                combining.
        """
        if multi_size_datasets is not None:
            return self._train_multi_size(multi_size_datasets)

        if dataset is None:
            dataset = generate_dataset(self.bits, n_train, seed_offset=seed_offset)

        N_list = [d[0] for d in dataset]
        p_list = [d[1] for d in dataset]

        F = self.fe.fit_transform(N_list)
        p_bits = np.array([_int_to_bits(p, self.half) for p in p_list],
                          dtype=np.float64)
        self.posterior.train(F, p_bits)
        self._trained = True

    def _train_multi_size(self, multi_size_datasets):
        """Train on a mix of bit sizes (D169 transfer learning).

        Each size's features are standardized independently (z-score per
        size), then concatenated. The AR posterior learns size-invariant
        patterns. D169 showed this improves d at 64-bit from 3.12 to 2.94.

        Args:
            multi_size_datasets: list of (bits, dataset) where dataset is
                a list of (N, p, q) tuples. The target size must be included.
        """
        all_F = []
        all_p_bits = []
        for bits_i, dataset_i in multi_size_datasets:
            half_i = bits_i // 2
            N_list = [d[0] for d in dataset_i]
            p_list = [d[1] for d in dataset_i]
            fe_i = FeatureExtractor(bits_i)
            F_i = fe_i.fit_transform(N_list)
            all_F.append(F_i)
            pb_i = np.array([_int_to_bits(p, half_i) for p in p_list],
                            dtype=np.float64)
            if half_i < self.half:
                pad = np.full((len(pb_i), self.half - half_i), 0.5)
                pb_i = np.hstack([pb_i, pad])
            elif half_i > self.half:
                pb_i = pb_i[:, :self.half]
            all_p_bits.append(pb_i)
        F_combined = np.vstack(all_F)
        pb_combined = np.vstack(all_p_bits)
        self.fe.fit_transform([d[0] for d in multi_size_datasets[0][1]])
        self.posterior.train(F_combined, pb_combined)
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

        if self.extractor_name == "coevolve":
            return self._factor_coevolve(N, samples)

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

    def _factor_coevolve(self, N, samples):
        """Factor using coevolutionary GA (D195).

        Initializes both populations from posterior samples, then
        coevolves the high bits of p and q via the constraint p*q ≈ N.
        Achieves 99% at 48-bit vs 71% for brute-force (D195).
        """
        init_p = [int(s) for s in samples]
        init_q = [int(N // s) if s > 0 else 1 for s in samples]
        ga = CoevolutionaryGA(
            N, self.half, self.b_unknown,
            pop_size=min(self.coevolve_pop_size, len(init_p)),
            seed=int(self.rng.integers(0, 2**31)),
        )
        factor, _ = ga.run(init_p, init_q, generations=self.coevolve_generations)
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
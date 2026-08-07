"""Ensemble of diverse AR posteriors (D203).

Trains N independent posteriors with different bit orderings and feature
subsets, then takes the UNION of all samples. D203 showed this reduces
d_high from 3.15 to 1.35 at 64-bit, giving 70% success (+15pp over single).

The key insight: different bit orderings (MSB-first, LSB-first, even-odd,
random) and feature subsets (spectral-only, arithmetic-only, mod-residues)
produce uncorrelated predictions. The ensemble's best sample is much closer
to the true p than any single posterior's best. "Diversity beats strength."

D203 result: 70% at 64-bit (vs 55% single posterior, same test set).
"""

import numpy as np

from .posterior import ARPosterior, _int_to_bits


DEFAULT_BIT_ORDERINGS = [
    "msb_first",
    "lsb_first",
    "even_odd",
    "random_42",
    "random_99",
]

DEFAULT_FEATURE_SUBSETS = {
    "all": list(range(89)),
    "spectral": list(range(33, 89)),
    "arithmetic": list(range(33)),
    "mod_residues": list(range(16)),
}


def _get_bit_ordering(name, n_bits, seed=None):
    """Return a permutation of [0, n_bits) for the given ordering name."""
    if name == "msb_first":
        return list(range(n_bits))
    elif name == "lsb_first":
        return list(reversed(range(n_bits)))
    elif name == "even_odd":
        return list(range(0, n_bits, 2)) + list(range(1, n_bits, 2))
    elif name.startswith("random_"):
        s = int(name.split("_")[1])
        rng = np.random.default_rng(s)
        return rng.permutation(n_bits).tolist()
    else:
        raise ValueError(f"Unknown bit ordering: {name}")


class EnsemblePosterior:
    """Ensemble of diverse AR posteriors (D203).

    Trains N_posteriors independent AR posteriors with different bit
    orderings and feature subsets. At sample time, draws from all
    posteriors and returns the union of samples (in original bit space).

    Args:
        n_features: dimension of the full feature vector (89).
        n_bits: number of bits in p (half the semiprime bit-size).
        bit_orderings: list of ordering names (default: 5 orderings).
        feature_subsets: dict of name -> feature indices (default: 4 subsets).
        n_iter, lr, reg: AR posterior training params.
    """

    def __init__(self, n_features, n_bits,
                 bit_orderings=None, feature_subsets=None,
                 n_iter=300, lr=0.1, reg=1e-3):
        self.n_features = n_features
        self.n_bits = n_bits
        self.n_iter = n_iter
        self.lr = lr
        self.reg = reg
        if bit_orderings is None:
            bit_orderings = list(DEFAULT_BIT_ORDERINGS)
        if feature_subsets is None:
            feature_subsets = dict(DEFAULT_FEATURE_SUBSETS)
        # Generate configs: all combinations of ordering × subset
        self.configs = []
        for oi, oname in enumerate(bit_orderings):
            order = _get_bit_ordering(oname, n_bits)
            for fname, fidx in feature_subsets.items():
                self.configs.append({
                    "id": f"o{oi}_{fname}",
                    "bit_order": order,
                    "feature_idx": list(fidx),
                    "n_features_sub": len(fidx),
                })
        self.posteriors = []
        self._trained = False

    @property
    def n_posteriors(self):
        return len(self.configs)

    def train(self, F, p_bits):
        """Train all posteriors in the ensemble.

        Args:
            F: (n_samples, n_features) standardized full feature matrix.
            p_bits: (n_samples, n_bits) bit matrix for p (MSB-first).
        """
        n, d = F.shape
        n_bits = p_bits.shape[1]
        self.posteriors = []
        for config in self.configs:
            fidx = config["feature_idx"]
            order = config["bit_order"]
            # Feature subset
            F_sub = F[:, fidx]
            # Reorder bits: position j in reordered space = bit order[j]
            pb_sub = np.zeros_like(p_bits)
            for j, bit_idx in enumerate(order):
                pb_sub[:, j] = p_bits[:, bit_idx]
            # Train AR posterior on the subset
            post = ARPosterior(
                n_features=len(fidx), n_bits=n_bits,
                n_iter=self.n_iter, lr=self.lr, reg=self.reg,
            )
            post.train(F_sub, pb_sub)
            self.posteriors.append((post, config))
        self._trained = True

    def sample(self, feature_vec, n_samples, rng):
        """Sample from ALL posteriors, returning union in original bit space.

        Each posterior gets n_samples // n_posteriors samples (with rounding
        up to ensure at least 1 per posterior). The total may be slightly
        more than n_samples.

        Args:
            feature_vec: (n_features,) full standardized features.
            n_samples: total desired samples (split across posteriors).
            rng: numpy random Generator.

        Returns:
            numpy array of sampled p values in original bit space.
        """
        assert self._trained, "Ensemble not trained"
        per_post = max(1, n_samples // self.n_posteriors)
        all_samples = []
        for post, config in self.posteriors:
            fidx = config["feature_idx"]
            order = config["bit_order"]
            f_sub = feature_vec[fidx]
            samples_bits = post.sample(f_sub, per_post, rng)
            # Convert from reordered bit space to original bit space
            for s in samples_bits:
                s = int(s)
                orig = 0
                for j in range(self.n_bits):
                    bit_j = (s >> j) & 1
                    orig |= (bit_j << order[j])
                all_samples.append(orig)
        return np.array(all_samples, dtype=object)
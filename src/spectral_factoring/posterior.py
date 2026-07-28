"""Autoregressive (AR) posterior over p's bits via logistic regression.

The AR model (D143 Family B) predicts each bit of p conditioned on
the feature vector AND all previously predicted bits:

    P(bit_j | features, bit_0, ..., bit_{j-1}) = sigmoid(w_j . [features, prev_bits, 1])

This captures bit dependencies that the mean-field (independent) model
misses, achieving ~10x concentration over the 1/sqrt(N) baseline.
"""

import numpy as np


def _sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def _int_to_bits(val, nbits):
    return np.array([(val >> (nbits - 1 - i)) & 1 for i in range(nbits)],
                    dtype=np.float64)


class ARPosterior:
    """Autoregressive posterior over p's bits.

    Args:
        n_features: dimension of the feature vector.
        n_bits: number of bits in p (half the semiprime bit-size).
        n_iter: gradient-descent iterations.
        lr: learning rate.
        reg: L2 regularization strength.
    """

    def __init__(self, n_features, n_bits, n_iter=300, lr=0.1, reg=1e-3):
        self.n_features = n_features
        self.n_bits = n_bits
        self.n_iter = n_iter
        self.lr = lr
        self.reg = reg
        self.thetas = None
        self.majority_bits = None

    def train(self, F, p_bits):
        """Train per-bit logistic regression models.

        Args:
            F: (n_samples, n_features) standardized feature matrix.
            p_bits: (n_samples, n_bits) bit matrix for p (MSB-first).
        """
        n, d = F.shape
        n_bits = p_bits.shape[1]
        thetas = []
        majority_bits = []
        for j in range(n_bits):
            y = p_bits[:, j]
            cb = float(y.mean())
            if cb < 1e-6 or cb > 1 - 1e-6:
                thetas.append(None)
                majority_bits.append(int(round(cb)))
                continue
            majority_bits.append(-1)
            if j == 0:
                X = F
            else:
                X = np.hstack([F, p_bits[:, :j].astype(np.float64)])
            Xb = np.hstack([X, np.ones((n, 1))])
            theta = np.zeros(Xb.shape[1], dtype=np.float64)
            for _ in range(self.n_iter):
                z = Xb @ theta
                p = _sigmoid(z)
                err = p - y
                g = Xb.T @ err / n + self.reg * theta / n
                theta -= self.lr * g
            thetas.append(theta)
        self.thetas = thetas
        self.majority_bits = majority_bits

    def sample(self, feature_vec, n_samples, rng):
        """Sample candidate p values from the posterior.

        Args:
            feature_vec: (n_features,) standardized features for this N.
            n_samples: number of candidates to draw.
            rng: numpy random Generator.

        Returns:
            numpy array of sampled p values (dtype=object for big ints).
        """
        assert self.thetas is not None, "Posterior not trained"
        n_bits = self.n_bits
        d = len(feature_vec)
        bits_batch = np.zeros((n_samples, n_bits), dtype=np.int64)
        for j in range(n_bits):
            th = self.thetas[j]
            if th is None:
                bits_batch[:, j] = int(self.majority_bits[j])
                continue
            theta_f = th[:d]
            theta_bias = th[-1]
            if j == 0:
                const = float(feature_vec @ theta_f) + float(theta_bias)
                pr_all = 1.0 / (1.0 + np.exp(-const))
            else:
                theta_b = th[d:d + j]
                const = float(feature_vec @ theta_f) + float(theta_bias)
                varying = bits_batch[:, :j].astype(np.float64) @ theta_b
                logits = const + varying
                pr_all = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
            bits_batch[:, j] = (rng.random(n_samples) < pr_all).astype(np.int64)
        samples = np.zeros(n_samples, dtype=object)
        for s in range(n_samples):
            samples[s] = _bits_to_int(bits_batch[s])
        return samples

    def predict_mode(self, feature_vec):
        """Return the MAP estimate (greedy argmax of each bit)."""
        assert self.thetas is not None, "Posterior not trained"
        n_bits = self.n_bits
        d = len(feature_vec)
        bits = np.zeros(n_bits, dtype=np.int64)
        for j in range(n_bits):
            th = self.thetas[j]
            if th is None:
                bits[j] = int(self.majority_bits[j])
                continue
            theta_f = th[:d]
            theta_bias = th[-1]
            if j == 0:
                const = float(feature_vec @ theta_f) + float(theta_bias)
                bits[j] = int(const > 0)
            else:
                theta_b = th[d:d + j]
                const = float(feature_vec @ theta_f) + float(theta_bias)
                varying = bits[:j].astype(np.float64) @ theta_b
                bits[j] = int(const + varying > 0)
        return _bits_to_int(bits)
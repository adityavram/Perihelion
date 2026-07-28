"""Feature extraction: 8 spectral sequences + 30 arithmetic features."""

import math
import numpy as np
from math import gcd, isqrt, log2, floor


SEQ_L = 256

PRIMES_LIST = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
N_GENERIC_FEATURES = 33
N_SPECTRAL_PER_SEQ = 7
N_SEQUENCES = 8
TOTAL_FEATURES = N_GENERIC_FEATURES + N_SEQUENCES * N_SPECTRAL_PER_SEQ  # 89


# ---------------------------------------------------------------------------
# Sequence generators — 8 diverse algebraic maps over Z/NZ
# ---------------------------------------------------------------------------

def seq_pollard_rho(N, L=SEQ_L):
    seq = np.empty(L, dtype=object)
    x = 2
    for k in range(L):
        seq[k] = x
        x = (x * x + 1) % N
    return seq


def seq_cubic(N, L=SEQ_L):
    seq = np.empty(L, dtype=object)
    x = 2
    for k in range(L):
        seq[k] = x
        x = (x * x * x + 1) % N
    return seq


def seq_lcg(N, L=SEQ_L):
    seq = np.empty(L, dtype=object)
    a = isqrt(N)
    c = bin(N).count("1")
    x = 2
    for k in range(L):
        seq[k] = x
        x = (a * x + c) % N
    return seq


def seq_exponential(N, L=SEQ_L):
    seq = np.empty(L, dtype=object)
    x = 2
    for k in range(L):
        seq[k] = x
        x = int(pow(2, int(x), N))
    return seq


def seq_fibonacci(N, L=SEQ_L):
    seq = np.empty(L, dtype=object)
    a, b = 1, 1
    for k in range(L):
        seq[k] = a
        a, b = b, (a + b) % N
    return seq


def seq_arnold_cat(N, L=SEQ_L):
    seq = np.empty(L, dtype=object)
    x, y = 2, 1
    for k in range(L):
        seq[k] = x
        x, y = (2 * x + y) % N, (x + y) % N
    return seq


def seq_disclog_walk(N, L=SEQ_L):
    seq = np.empty(L, dtype=object)
    x = 2
    for k in range(L):
        seq[k] = x
        x = int(pow(3, int(x), N))
    return seq


def seq_mobius(N, L=SEQ_L):
    seq = np.empty(L, dtype=object)
    x = 2
    for k in range(L):
        seq[k] = x
        num = (x * x + 1) % N
        den = (x + 1) % N
        g = gcd(den, N)
        if g == 1:
            den_inv = int(pow(den, -1, N))
            x = (num * den_inv) % N
        else:
            x = (num * (den + 1)) % N
    return seq


SEQ_FUNCS = [
    seq_pollard_rho,
    seq_cubic,
    seq_lcg,
    seq_exponential,
    seq_fibonacci,
    seq_arnold_cat,
    seq_disclog_walk,
    seq_mobius,
]


# ---------------------------------------------------------------------------
# Spectral features per sequence
# ---------------------------------------------------------------------------

def _lempel_ziv_complexity(arr):
    signs = (arr > 0).astype(np.int8)
    s = "".join(str(int(b)) for b in signs)
    n = len(s)
    if n == 0:
        return 0.0
    i = 0
    c = 0
    while i < n:
        k = 1
        while i + k <= n:
            sub = s[i:i + k]
            if sub in s[:i + k - 1]:
                k += 1
            else:
                break
        c += 1
        i += k
    denom = (np.log2(n)) if n > 1 else 1.0
    return float(c / denom) if denom > 0 else 0.0


def _spectral_features(seq_ints):
    L = len(seq_ints)
    arr = np.asarray(seq_ints, dtype=np.float64)
    arr = arr - arr.mean()
    if np.std(arr) < 1e-12:
        return np.zeros(N_SPECTRAL_PER_SEQ, dtype=np.float64)
    fft = np.fft.rfft(arr)
    psd = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(L, d=1.0)
    psd_nodc = psd[1:]
    freqs_nodc = freqs[1:]
    total_power = float(np.sum(psd_nodc))
    if total_power <= 0:
        peak_freq = 0.0
    else:
        peak_idx = int(np.argmax(psd_nodc))
        peak_freq = float(freqs_nodc[peak_idx])
    low_mask = freqs_nodc < 0.1
    high_mask = freqs_nodc > 0.4
    low_power = float(np.sum(psd_nodc[low_mask])) if low_mask.any() else 0.0
    high_power = float(np.sum(psd_nodc[high_mask])) if high_mask.any() else 0.0

    def autocorr(lag):
        if lag >= L:
            return 0.0
        a = arr[:L - lag]
        b = arr[lag:]
        denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
        if denom < 1e-12:
            return 0.0
        return float(np.sum(a * b) / denom)

    max_lag = min(L // 2, 128)
    acs = np.empty(max_lag + 1, dtype=np.float64)
    acs[0] = 1.0
    for lag in range(1, max_lag + 1):
        acs[lag] = autocorr(lag)
    first_zero = 0
    for lag in range(1, max_lag):
        if acs[lag - 1] > 0 and acs[lag] <= 0:
            first_zero = lag
            break
    if max_lag > 2:
        peak_lag_idx = int(np.argmax(acs[2:])) + 2
    else:
        peak_lag_idx = 0
    lz = _lempel_ziv_complexity(arr)
    signs = (arr > 0).astype(np.int8)
    runs = []
    cur = signs[0]
    cnt = 1
    for s in signs[1:]:
        if s == cur:
            cnt += 1
        else:
            runs.append(cnt)
            cur = s
            cnt = 1
    runs.append(cnt)
    runlen_mean = float(np.mean(runs)) if runs else 0.0
    return np.array([
        float(peak_lag_idx), peak_freq, low_power,
        high_power, lz, runlen_mean, float(first_zero),
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Generic (arithmetic) features
# ---------------------------------------------------------------------------

def _compute_generic_features(N, bits):
    N = int(N)
    feats = []
    for M in PRIMES_LIST:
        feats.append(float(N % M))
    feats.append(float(N.bit_length()))
    feats.append(float(bin(N).count("1")))
    feats.append(float(sum(int(c) for c in str(N))))
    feats.append(float(sum(int(c, 16) for c in format(N, 'x'))))
    if N > 0:
        lf = log2(N) - floor(log2(N))
    else:
        lf = 0.0
    feats.append(float(lf))
    sqrtN = isqrt(N)
    feats.append(float(N % sqrtN if sqrtN > 0 else 0.0))
    sqrtN_ceil = sqrtN + 1
    feats.append(float(N % sqrtN_ceil if sqrtN_ceil > 0 else 0.0))
    cbrt = int(round(N ** (1.0 / 3.0)))
    feats.append(float(N % cbrt if cbrt > 0 else 0.0))
    feats.append(float(sqrtN))
    feats.append(float(cbrt))
    feats.append(float(N & 0xF))
    feats.append(float(N & 0xFF))
    feats.append(float(N & 0xFFFF))
    half = bits // 2
    low = N & ((1 << half) - 1)
    high = N >> half
    feats.append(float(low ^ high))
    bit_seq = [(N >> i) & 1 for i in range(bits - 1, -1, -1)]
    bit_arr = np.array(bit_seq, dtype=np.float64) * 2.0 - 1.0
    fft_mag = np.abs(np.fft.rfft(bit_arr))
    for k in range(1, 4):
        if k < len(fft_mag):
            feats.append(float(fft_mag[k]))
        else:
            feats.append(0.0)
    return np.array(feats, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_features(N, bits):
    """Compute the full 89-dimensional feature vector for semiprime N.

    Args:
        N: the semiprime (int).
        bits: the bit-length of N (e.g. 32, 48, 64).

    Returns:
        numpy array of shape (89,) — 33 generic + 56 spectral (8 seqs * 7 features).
    """
    N = int(N)
    generic = _compute_generic_features(N, bits)
    spectral = []
    for fn in SEQ_FUNCS:
        seq = fn(N, SEQ_L)
        spectral.append(_spectral_features(seq))
    return np.concatenate([generic] + spectral)


def standardize(F, mean=None, std=None):
    if mean is None:
        mean = F.mean(axis=0)
        std = F.std(axis=0) + 1e-9
    return ((F - mean) / std, mean, std)


class FeatureExtractor:
    """Manages feature extraction and standardization across a dataset."""

    def __init__(self, bits):
        self.bits = bits
        self.mean = None
        self.std = None

    def fit_transform(self, N_list):
        n = len(N_list)
        F = np.zeros((n, TOTAL_FEATURES), dtype=np.float64)
        for i, N in enumerate(N_list):
            F[i] = compute_features(N, self.bits)
        F, self.mean, self.std = standardize(F)
        return F

    def transform(self, N_list):
        n = len(N_list)
        F = np.zeros((n, TOTAL_FEATURES), dtype=np.float64)
        for i, N in enumerate(N_list):
            F[i] = compute_features(N, self.bits)
        return (F - self.mean) / (self.std + 1e-9)
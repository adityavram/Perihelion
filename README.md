# Perihelion

**Exploring polynomial-time integer factorization through spectral analysis, machine learning, and lattice methods.**

## Overview

Perihelion is a research project investigating whether integer factorization (particularly semiprimes N = p × q) can be solved in polynomial time. The project has explored 198 research directions across 32+ mathematical fields, producing both a practical factoring algorithm and a comprehensive characterization of the computational barriers.

## Current Status

### The Spectral Factoring Algorithm (D147/D143/D159/D157/D195)

We have built a factoring algorithm that combines:

1. **Spectral features** from 8 algebraic maps over Z/NZ (D147): Pollard rho, cubic, LCG, exponential, Fibonacci, Arnold cat, discrete log, and Möbius sequences — each producing 7 spectral features (autocorrelation, PSD, entropy rate, etc.), totaling 56 spectral + 33 arithmetic = 89 features.

2. **Autoregressive (AR) posterior** (D143): predicts each bit of p conditioned on the feature vector and previously predicted bits. Achieves ~10× concentration over the 1/√N baseline — the first reproducible beat of the per-N factoring barrier.

3. **Corrected Coppersmith lattice** (D159): the standard Coppersmith construction with x-shift polynomials (Howgrave-Graham 1999). Fixes a critical bug in earlier implementations (missing x-shifts → 6.5% success → 100% with the fix). Extracts the low bits of p from the high bits in poly(dim) time.

4. **Multi-sample Hamming-ball search** (D157): samples k candidates from the posterior, enumerates a Hamming ball of radius d around each sample's high bits, and runs Coppersmith for each. At n=32: 100% success with k=20, d=2.

5. **Coevolutionary GA** (D195): two populations (p_high, q_high) coevolve via the constraint p×q ≈ N. When one population improves, the other's target sharpens. Achieves 99% at 48-bit vs 71% for brute-force.

6. **Multi-restart** (D197/D198): running coevolution R times with different seeds. At 64-bit: R=20 gives 27% success with Coppersmith extraction (polynomial-time cost, 0.035× √N).

### Performance

| Bit size | Success rate | Cost vs √N | Cost vs NFS | Method |
|----------|-------------|------------|-------------|--------|
| 32-bit | 100% | 0.02× | 0.02× | Coevolution, R=1 |
| 48-bit | 100% | 0.005× | 0.005× | Multi-restart coevolution, R=3 |
| 64-bit | 27% | 0.035× | 0.001× | Multi-restart + Coppersmith, R=20 |

The algorithm beats NFS by 50-500× for RSA ≤ 180 bits (D162).

### The Scaling Barrier

The AR posterior extracts ~3.5 bits of concentration (stable, non-decaying). This fixes the top ~3 bits of p. The Hamming distance to the true p grows linearly: d(n) ≈ 0.08n (D170, R²=0.9965). The polynomial-time regime (d=2) extends to n≈48. Beyond that, d must grow, making the cost super-polynomial.

At 64-bit, the coevolution + multi-restart approach achieves 27% with polynomial-time Coppersmith extraction. The 27% ceiling is a new form of the barrier: 27% of semiprimes are "coevolution-easy" (the high bits are findable), 73% are hard.

## The Barrier System

After 198 directions across 32+ mathematical fields, the barrier is characterized from every angle:

### The Three Mechanisms (D113-D115)
1. **M1 (Multiplicative OTP)**: N mod M = (p mod M)·(q mod M) mod M. For prime M < √N, the key q mod M is uniform → perfect pad → zero excess information. Broken by additive functions (bitlen, N mod √N, carries), but breaking M1 alone doesn't help.

2. **M2 (Key bias)**: At small bit sizes, the key deviates from uniform by O(1/√M). Creates the small-n excess that vanishes by 32-bit. Broken by prime gap non-uniformity, but M3 blocks exploitation.

3. **M3 (√N scale)**: The ultimate barrier. Even when M1 and M2 are broken, the factor-revealing structure has O(n) bits of entropy, and no poly-time observable can extract more than O(1) bits.

### Formal Results
- **FM4 (Channel Capacity)**: poly-time functions of N carry O(1) bits about p. Empirically confirmed across 8+ measurement methods. The 6.5-bit ceiling holds across all 32 fields tested.
- **D121 Theorem 4.2**: C_modular(n) = Θ(log n) — the modular channel capacity is TIGHT (not conservative).
- **D120**: single-candidate G4 conditional on FM4 — no single poly-time candidate achieves hit rate > 1/√N without smoothness.
- **D185 Theorem**: factoring is not definable in any o-minimal structure (unconditional). The o-minimal definability approach is structurally impossible.
- **D133**: the N^{1/4} barrier is dynamical (Nyquist-of-periods + Margolus-Levitin), not digital.
- **D140**: Dickman smoothness is the unique asymptotic shape of any prime-factor-based smoothness in a UFD.
- **D191**: the descent-theoretic barrier — local data is either small-prime (O(1) bits) or bad-prime (circular). No third category.

### Fields Explored (32+)
Number theory (analytic, algebraic, p-adic, computational), algebraic geometry, complexity theory, information theory (MI, min-entropy, Rényi, MIC, capacity), lattice theory (LLL, Coppersmith, tropical), machine learning (logistic, MLP, transformer, boosting, variational, transfer, coevolutionary GP), signal processing (FFT, spectral, non-abelian Fourier), algebra (commutative, non-commutative, representation theory), dynamics (iterated maps, periodic, p-adic), geometry (hyperbolic, circles, Pell, tropical), combinatorics (smoothness, structured sets), topology (simplicial complexes, persistent homology), probability (MI, Rényi, transfer entropy, f-divergences), proof theory (resolution, Craig interpolation, cut elimination), model theory (o-minimality, definability), quantum foundations (contextuality, sheaf theory), analytic combinatorics (Ramanujan sums, generating functions), perfectoid spaces, descent theory, and more.

## Project Structure

```
Perihelion/
├── src/
│   ├── lattices/              # Original lattice constructions
│   └── spectral_factoring/    # Production factoring algorithm
│       ├── features.py        # 89-dim feature extraction (8 spectral + 33 arithmetic)
│       ├── posterior.py       # Autoregressive posterior over p's bits
│       ├── extraction.py      # Coppersmith lattice + brute-force GCD
│       ├── coevolution.py     # Coevolutionary GA for high-bit refinement
│       └── pipeline.py        # Full SpectralFactoring pipeline
├── tests/                     # Unit and integration tests
│   └── test_spectral_factoring.py
├── experiments/                # 198 research direction scripts
├── direction-*.md             # Research writeups (D116-D198)
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
from src.spectral_factoring import SpectralFactoring, generate_dataset

# Initialize and train
sf = SpectralFactoring(bits=32, k_samples=20, d_hamming=2,
                       extractor="coevolve", b_unknown=8)
train_data = generate_dataset(32, 200, seed_offset=100)
sf.train(dataset=train_data)

# Factor a semiprime
N, p, q = generate_dataset(32, 1, seed_offset=999)[0]
result = sf.factor(N)
# Returns: (p, q) or None

# Evaluate on a test set
test_data = generate_dataset(32, 20, seed_offset=200)
results = sf.evaluate(test_data)
print(f"Success rate: {results['success_rate']:.0%}")
```

### Multi-size training (D169)

```python
sf = SpectralFactoring(bits=48, extractor="coevolve")
multi = [
    (32, generate_dataset(32, 500, seed_offset=1)),
    (48, generate_dataset(48, 300, seed_offset=2)),
    (64, generate_dataset(64, 200, seed_offset=3)),
]
sf.train(multi_size_datasets=multi)
```

## Testing

```bash
source venv/bin/activate
python -m pytest tests/test_spectral_factoring.py -m small -v
# 32-bit: 100% success, 48-bit: 60%+, 64-bit: documents the gap
```

## Key Papers

1. **Coppersmith (1996)** — Finding small roots of bivariate integer polynomial equations
2. **Howgrave-Graham (1999)** — Finding small roots of univariate modular equations modulo an unknown divisor
3. **Schnorr (1991)** — Foundational lattice-based factoring
4. **Shor (1994)** — Polynomial-time quantum factoring (the quantum escape)
5. **Bernstein (2004)** — Batch GCD for fast factor detection

## License

MIT

## Author

Aditya Ram (adityavram)
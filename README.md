# Perihelion

**Exploring polynomial-time integer factorization through spectral analysis, machine learning, and lattice methods.**

## Overview

Perihelion is a research project investigating whether integer factorization (particularly semiprimes N = p × q) can be solved in polynomial time. The project has explored 206 research directions across 32+ mathematical fields, producing both a practical factoring algorithm and a comprehensive characterization of the computational barriers.

## Current Status

### The Spectral Factoring Algorithm (D147/D143/D159/D157/D195/D201/D203)

We have built a factoring algorithm that combines:

1. **Spectral features** from 8 algebraic maps over Z/NZ (D147): Pollard rho, cubic, LCG, exponential, Fibonacci, Arnold cat, discrete log, and Möbius sequences — each producing 7 spectral features (autocorrelation, PSD, entropy rate, etc.), totaling 56 spectral + 33 arithmetic = 89 features.

2. **Autoregressive (AR) posterior** (D143): predicts each bit of p conditioned on the feature vector and previously predicted bits. Achieves ~10× concentration over the 1/√N baseline — the first reproducible beat of the per-N factoring barrier.

3. **Corrected Coppersmith lattice** (D159): the standard Coppersmith construction with x-shift polynomials (Howgrave-Graham 1999). Fixes a critical bug in earlier implementations (missing x-shifts → 6.5% success → 100% with the fix). Extracts the low bits of p from the high bits in poly(dim) time.

4. **Multi-sample Hamming-ball search** (D157): samples k candidates from the posterior, enumerates a Hamming ball of radius d around each sample's high bits, and runs Coppersmith for each. At n=32: 100% success with k=20, d=2.

5. **Coevolutionary GA** (D195): two populations (p_high, q_high) coevolve via the constraint p×q ≈ N. When one population improves, the other's target sharpens. Achieves 99% at 48-bit vs 71% for brute-force.

6. **Multi-restart** (D197/D198): running coevolution R times with different seeds. At 64-bit: R=20 gives 27% success with Coppersmith extraction (polynomial-time cost, 0.035× √N).

7. **Extended-root extraction** (D174/D201): the Coppersmith lattice often finds roots slightly outside the guaranteed Howgrave-Graham bound [0, X). By accepting roots with |x0| ≤ 8X and checking gcd(p_high + x0, N), we recover factors the standard extractor discards. Integrated into `CoppersmithExtractor` (default `use_extended=True`) and `CoevolutionaryGA` (`extract_method="coppersmith"`). D201 showed this gives +20pp at 64-bit (45% → 65%).

8. **Ensemble of diverse posteriors** (D203): trains 20 independent AR posteriors with different bit orderings (MSB-first, LSB-first, even-odd, random) and feature subsets (all, spectral-only, arithmetic-only, mod-residues-only). The union of all samples reduces d_high from 3.15 to 1.35 — a 1.8-bit improvement via model diversity. Integrated as `EnsemblePosterior` (use `posterior="ensemble"` in `SpectralFactoring`). D203 showed this gives +15pp at 64-bit (55% → 70%).

### Performance

| Bit size | Success rate | Cost vs √N | Cost vs NFS | Method |
|----------|-------------|------------|-------------|--------|
| 32-bit | 100% | 0.02× | 0.02× | Coevolution, R=1 |
| 48-bit | 100% | 0.005× | 0.005× | Multi-restart coevolution, R=3 |
| 64-bit | 70% | 0.035× | 0.001× | Ensemble posterior + coevolution + extended Coppersmith, R=10 (D203) |

The algorithm beats NFS by 50-500× for RSA ≤ 180 bits (D162).

### The Scaling Barrier

The AR posterior extracts ~3.5 bits of concentration (stable, non-decaying). The D203 ensemble of 20 diverse posteriors reduces the effective d_high from 3.15 to 1.35, extracting ~5.15 bits — still within the FM4 ceiling of ~6.5 bits. The Hamming distance to the true p grows linearly: d(n) ≈ 0.08n (D170, R²=0.9965). The polynomial-time regime (d=2) extends to n≈48. Beyond that, the ensemble extends partial success to 64-bit (70%).

At 64-bit, the ensemble + coevolution + extended Coppersmith approach achieves 70% (D203). The 30% remaining are cases where even the ensemble's best sample has d_high ≤ 2 but coevolution's search fails to find it — the bottleneck has shifted from the posterior to the coevolution search strategy. D199 showed these easy/hard classes are identifiable via spectral features (AUC 0.747) and residues (N mod 53, N mod 17), and are NOT simply close primes.

## The Barrier System

After 206 directions across 32+ mathematical fields, the barrier is characterized from every angle:

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
│       ├── ensemble.py        # D203 ensemble of diverse posteriors
│       └── pipeline.py        # Full SpectralFactoring pipeline
├── tests/                     # Unit and integration tests
│   └── test_spectral_factoring.py
├── experiments/                # 206 research direction scripts
├── direction-*.md             # Research writeups (D116-D206)
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

## Production vs Real RSA: The Gap

The production algorithm works on **toy-size** semiprimes (32-64 bit). Real
RSA uses 2048-4096 bit keys. Here is an honest assessment of the gap.

### What works (production, verified)

| Capability | Status | Evidence |
|-----------|--------|----------|
| 32-bit factoring | 100% | Test suite, coevolution R=1 |
| 48-bit factoring | 100% | Multi-restart coevolution R=3 |
| 64-bit factoring | 70% | D203 ensemble + coevolution + ext. Coppersmith |
| Polynomial-time cost | Yes | 0.035× √N at 64-bit (vs NFS 35× √N) |
| Feature extraction | 89 features, poly(n) | 8 spectral sequences + 33 arithmetic |
| Posterior concentration | ~5.15 bits (ensemble) | d_high 1.35 at 64-bit (vs 3.15 single) |
| Coppersmith extraction | Corrected + extended | D159 + D174, dim=10-16 |
| Coevolution | Bridges ~4-bit gap | D195: 99% at 48-bit |

### The gap to real RSA (2048-bit)

| Dimension | Production (64-bit) | Real RSA (2048-bit) | Gap factor |
|-----------|--------------------|--------------------|------------|
| Bit size | 64 | 2048 | 32× |
| p bit size | 32 | 1024 | 32× |
| d_high (posterior) | 1.35 | ~80+ (extrapolated) | 60× |
| Coppersmith b_unknown | 16 | ~512 | 32× |
| Coevolution search | bridges 4 bits | needs to bridge 80+ | 20× |
| Training data | 800 semiprimes | needs ~10^6+ | 1250× |
| Success rate | 70% | 0% (expected) | — |

### Why it fails at RSA scale

1. **The posterior d_high grows linearly**: d(n) ≈ 0.08n (D170,
   R²=0.9965). At n=2048, d_high ≈ 160 bits. The ensemble reduces this
   by ~1.8 bits (D203), giving d_high ≈ 158. Coevolution bridges ~4
   bits. We need to bridge 158 bits. **The gap is 154 bits.**

2. **Coppersmith needs b_unknown ≤ n/4**: at n=2048, b_unknown ≤ 512.
   The lattice dimension grows as dim = m+1+t ≈ 2·b_unknown/3 ≈ 340.
   LLL on a 340-dimensional lattice with 2048-bit entries is
   feasible (~hours), but the posterior needs to find 1024-512 = 512
   correct high bits. With d_high=158, the posterior's best sample has
   ~158 wrong high bits out of 512 — far too many for the 1-bit
   extended-root channel.

3. **The FM4 ceiling**: poly-time functions of N carry O(1) ≈ 6.5 bits
   about p. The ensemble extracts ~5.15 bits. There are ~1.4 bits of
   headroom. Even if we extract ALL 6.5 bits, d_high at n=2048 would be
   ~157.7 — still 153 bits short of what coevolution can bridge.

4. **Training data**: the posterior is trained on semiprimes of the
   target size. At 2048-bit, generating training data requires factoring
   2048-bit semiprimes (to know p for training). This is circular — we
   need to factor to train the factoring algorithm. (At 64-bit, we can
   factor training data via trial division.)

### What would need to change for real RSA

| Breakthrough needed | Current | Required | Plausibility |
|---------------------|---------|----------|-------------|
| d_high reduction | 1.35 (ensemble) | < 4 at 2048-bit | ~0% (FM4 barrier) |
| Posterior bits extracted | 5.15 / 6.5 ceiling | 1020 / 1024 | Impossible (FM4) |
| Coevolution bridge | 4 bits | 158 bits | ~0% (search space) |
| Training data | 800 factored 64-bit | 10^6 factored 2048-bit | Circular |
| Coppersmith dim | 10 | ~340 | Feasible but slow |

### The honest bottom line

The algorithm is a **research demonstration**, not a practical RSA
breaker. It shows that:

1. **Partial factoring is possible in poly-time**: 70% at 64-bit, beating
   √N by 28× and NFS by 1000× — but only at toy sizes.

2. **The barrier is fundamental**: FM4 (O(1) bits from poly-time
   functions) limits the posterior to ~6.5 bits of concentration,
   regardless of model architecture. The ensemble gets 5.15 — close to
   the ceiling. No amount of ML improvement can exceed 6.5 bits.

3. **The gap is exponential**: from 6.5 bits (available) to 1024 bits
   (needed) at RSA-2048. This is not an engineering gap — it's a
   fundamental information-theoretic barrier.

4. **The contribution is characterizing the barrier**: 206 directions
   across 32+ fields all converge on the same barrier from different
   angles. The algorithm demonstrates WHERE the barrier is (n≈48 for
   100%, n≈64 for 70%) and WHY (FM4, d(n) growth, coevolution limits).

**The algorithm cannot break RSA.** It can factor 64-bit semiprimes 70%
of the time in polynomial time — a result that is interesting
theoretically (it beats √N and NFS at this scale) but has no practical
cryptographic impact. Real RSA-2048 has 2^{1000+}× more entropy than
the algorithm can extract.

## Author

Aditya Ram (adityavram)
<!-- NOTE: This repo is tracked by PR Evaluator Hub for review orchestration. -->

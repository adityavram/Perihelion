# Perihelion

**Exploring polynomial-time integer factorization through higher-dimensional lattice structures.**

## Overview

Perihelion is a research project investigating whether integer factorization (particularly semiprimes N = p × q) can be solved in polynomial time using lattice-based and higher-dimensional algebraic structures.

### Core Hypothesis

Classical factorization algorithms optimize division in ℝ (the reals), treating primes as scalars in a search problem. This project explores whether divisibility relationships can be encoded in higher-dimensional structures (lattices, vector spaces, rings, topological spaces) where they become geometrically or algebraically accessible.

## Current Status

### Breakthrough: Modular Lattice Construction

We discovered that a **modular lattice encoding** successfully factors semiprimes when the prime factors are in the factor base:

```python
# Construction: [prime_i, ..., N mod prime_i, ...]
# After LLL reduction: vector sums reveal factors
```

**Success rate**: 100% when factors are in factor base

**Key insight**: Divisibility (`N mod p = 0`) creates a special lattice structure that LLL reduction exposes directly.

**Limitation**: Requires factor base containing the factors. For RSA-2048, would need ~10³⁰⁰ primes — not scalable.

### Alternative Embeddings Tested

| Method | Success Rate | What It Is |
|--------|--------------|------------|
| Continued Fraction | 78% | CFRAC (known classical method) |
| Simultaneous Approx | 78% | Trial division near √N |
| Elliptic-Inspired | 44% | Trial division |
| Quadratic Form v2 | 33% | Search for p+q |

**Finding**: Most "successful" approaches are classical methods in disguise. The modular lattice is genuinely novel.

## Project Structure

```
Perihelion/
├── src/
│   ├── lattices/          # Lattice constructions
│   ├── factorization/     # Factorization algorithms
│   └── utils/             # Helper functions
├── tests/                 # Unit and integration tests
├── experiments/           # Research notebooks and scripts
├── docs/                  # Research notes and paper summaries
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
from src.lattices.modular import modular_lattice_factor
from src.lattices.alternative import continued_fraction_factor

# Modular lattice (requires factor base)
result = modular_lattice_factor(N=143, factor_base_size=20)
# Returns: (11, 13)

# Continued fraction (CFRAC method)
result = continued_fraction_factor(N=143)
# Returns: (11, 13)
```

## Research Notes

Key findings are documented in the Obsidian vault at `~/Obsidian/Perihelion/`:
- `MISSION.md` - Project goals and hypotheses
- `hypotheses.md` - Working hypotheses (H1-H4)
- `lattice-research-findings.md` - Literature review
- `breakthrough-modular-lattice.md` - Key discovery
- `paper-summaries.md` - Plain-English summaries of relevant papers

## Key Papers

1. **Schnorr (1991)** - Foundational lattice-based factoring
2. **Coppersmith Optimality (2016)** - Proves lattice bounds are tight
3. **Gao et al. (2025)** - State-of-art "second vector" approach
4. **Al-Hasso (2025)** - Probabilistic computing for CVP
5. **Ajani & Bright (2024)** - SAT + lattice hybrid

See `docs/paper-summaries.md` for detailed summaries.

## License

MIT

## Author

Aditya Ram (adityavram)
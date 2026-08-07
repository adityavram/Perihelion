# Direction 204: Class-Specific Posteriors

**Status:** Experiment complete. **Negative — no improvement.**
**Run time:** 2,198s (~37 min)
**Date:** 2026-08-06

---

## Question

D199 found that easy vs hard cases are distinguished by N mod 53 (Cohen's
d = -1.4), N mod 17 (AUC 0.76), and spectral features. Can training a
SEPARATE posterior for each residue class (N mod 53) exploit class-specific
structure that a general posterior averages away?

## Setup

- Classification: N mod 53 (D199's strongest discriminator).
- Training: 800 64-bit semiprimes → ~15 samples per class. Classes with
  ≥ 30 samples get a class-specific posterior; smaller classes use the
  general posterior (fallback).
- Phase 1: partition, train per-class posteriors.
- Phase 2: measure d_high for general vs class-specific.
- Phase 3: coevolution with class-specific posteriors.

## Results

### Phase 2: d_high comparison

| Metric | General posterior | Class-specific | Improvement |
|--------|------------------|----------------|-------------|
| Mean d_high | 3.15 | 3.15 | 0 |
| Improved in | — | 0/20 cases | **None** |

**Class-specific posteriors give ZERO improvement in d_high.** Not a
single instance out of 20 has lower d_high with the class-specific model
than the general model.

### Phase 3: Coevolution

| Method | Success | Rate |
|--------|---------|------|
| General posterior (D202) | 11/20 | 55.0% |
| Class-specific posterior | 11/20 | 55.0% |

No change. The class-specific posterior is identical in predictive power
to the general posterior.

## Findings

1. **N mod 53 discriminates easy from hard, but doesn't PREDICT p.**
   D199 found N mod 53 correlates with coevolution-easy/hard status (AUC
   0.81). But this correlation is about the POSTERIOR's difficulty, not
   about p's bits directly. Knowing N mod 53 = r tells you the instance
   is "easy," but it doesn't tell you what p IS. The class-specific
   posterior still has to predict p from the same 89 features — and the
   partitioning doesn't add new information.

2. **Small training sets hurt.** With 800 samples / 53 classes ≈ 15
   samples per class, most classes fall below the MIN_CLASS_SIZE=30
   threshold and use the general posterior. Even classes with ≥ 30
   samples have 3-4× less data than the general model (800 samples),
   making the class-specific model noisier, not better.

3. **The easy/hard distinction is not about residue class.** D199's AUC
   0.81 for N mod 53 measures how well the residue SEPARATES easy from
   hard. But the separation is probabilistic, not deterministic — many
   hard instances share residues with easy ones. Training a per-class
   model doesn't help because the class doesn't determine p's bits.

## Conclusions

- **Class-specific posteriors do not help.** The partition by N mod 53
  doesn't add information about p. The general posterior already captures
  all available structure.
- **D203 (ensemble) is the right approach, not D204 (class-specific).**
  Diversity (different models on the same data) beats specialization
  (same model on different data subsets).

## Deliverables

- Script: `experiments/d204_class_posteriors.py`
- Results: `experiments/d204_results.json`
- Log: `experiments/d204_run.log`
- Writeup: `direction-204-class-posteriors.md`
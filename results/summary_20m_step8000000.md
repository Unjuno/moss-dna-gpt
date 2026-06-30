# DNA-GPT 20M — Evaluation Summary (Step 8,000,000)

## Model

| Property | Value |
|---|---|
| Architecture | Decoder-only Transformer (GPT) |
| Parameters | 21,691,008 |
| Context length | 1,024 |
| Training steps | 8,000,000 |
| Training data | *Physcomitrium patens* GCA_000002425.3 / Phypa_V5 |
| Split policy | Sequence-level |

## Results

Evaluation on held-out sequence-level test split (177,723,392 tokens):

| Model | Bits/base | Nats/base |
|---|---:|---:|
| Markov order 0 | 1.92504 | 1.33434 |
| Markov order 1 | 1.90902 | 1.32323 |
| Markov order 5 | 1.88614 | 1.30738 |
| **DNA-GPT 20M (step 8M)** | **1.41665** | **0.98195** |

## Claim

> On a sequence-level held-out split of *Physcomitrium patens*, the 20M-parameter DNA-GPT next-base prediction model outperformed 0th, 1st, and 5th order Markov baselines in bits/base.

## Files

- `results/eval_markov_20m_step8000000.json` — machine-readable JSON
- `results/figures/bits_per_base_20m_step8000000.png` — comparison bar chart

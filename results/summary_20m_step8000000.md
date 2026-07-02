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
| Training windows | 681,613 |
| Test windows | 173,558 |
| Test tokens | 177,723,392 |

## Results

Evaluation on held-out sequence-level test split:

| Model | Bits/base | Nats/base | Improvement vs baseline |
|---|---:|---:|---:|
| Markov order 0 | 1.92504 | 1.33434 | — |
| Markov order 1 | 1.90902 | 1.32323 | — |
| Markov order 2 | 1.90315 | 1.31917 | — |
| Markov order 3 | 1.89569 | 1.31399 | — |
| Dinucleotide shuffled (order 5) | 1.90728 | 1.32203 | — |
| K3 shuffled (order 5) | 1.90049 | 1.31732 | — |
| IMM (interpolated 0..5) | 1.88614 | 1.30738 | — |
| Markov order 5 | 1.88614 | 1.30738 | — |
| **DNA-GPT 20M (step 8M)** | **1.41665** | **0.98195** | **24.9% vs IMM** |

## Claim

> On a sequence-level held-out split of *Physcomitrium patens*, the 20M-parameter DNA-GPT next-base prediction model achieves **1.41665 bits/base**, outperforming the strongest Markov baseline (Interpolated Markov Model, orders 0–5) by **24.9%** in bits/base.

## Files

- `results/eval_markov_20m_step8000000.json` — canonical machine-readable results
- `results/eval_markov_20m_full.json` — full raw output with all baselines
- `results/eval_markov_20m_lc_filtered.json` — low-complexity filtered evaluation
- `results/figures/bits_per_base_20m_step8000000.png` — comparison bar chart

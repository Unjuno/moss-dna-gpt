# Evaluation results

## 20M-parameter DNA-GPT (step 8,000,000)

Evaluation on a held-out **sequence-level** test split of *Physcomitrium patens* (Phypa_V5) using 177,723,392 tokens.

| Model | Bits/base | Nats/base |
|---|---:|---:|
| Markov order 0 | 1.92504 | 1.33434 |
| Markov order 1 | 1.90902 | 1.32323 |
| Markov order 5 | 1.88614 | 1.30738 |
| **DNA-GPT 20M** | **1.41665** | **0.98195** |

This result supports only the following claim:

> On a sequence-level held-out split of *Physcomitrium patens*, the 20M-parameter DNA-GPT next-base prediction model outperformed 0th, 1st, and 5th order Markov baselines in bits/base.

### Important caveats

- Bits/base values depend on evaluation settings (number of batches, batch size, sequence split policy).
- **Curve vs final discrepancy.** The intermediate evaluation curve (stored in `repo/moss-dna-gpt-20m-patens/eval_curve.json`) was computed with `eval_batches=200`. The canonical final evaluation at step 8,000,000 (stored in `results/eval_markov_20m_step8000000.json`) used a different number of evaluation batches (`eval_batches=50`). These were produced under different evaluation settings and are **not directly interchangeable**. The canonical release claim uses `results/eval_markov_20m_step8000000.json` which reports **1.41665 bits/base**.
- Lower bits/base than Markov baselines does **not** prove functional understanding, SNP effect prediction, or cross-species generalization.

### Source data

- `results/eval_markov_20m_step8000000.json` — machine-readable JSON with full evaluation details.
- `repo/moss-dna-gpt-20m-patens/eval_curve.json` — intermediate evaluation curve across training.
- `repo/moss-dna-gpt-20m-patens/eval_markov_step8000000.json` — canonical Markov + DNA-GPT comparison at step 8M.

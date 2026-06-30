---
license: mit
tags:
- biology
- genomics
- dna
- pytorch
- causal-lm
- moss
- next-base-prediction
- dna-language-model
- physcomitrium-patens
language:
- en
library_name: pytorch
pipeline_tag: text-generation
datasets:
- physcomitrium-patens-genome
---

# moss-dna-gpt-20m-patens

Experimental 20M-parameter decoder-only Transformer trained for DNA next-base prediction on *Physcomitrium patens* (moss) genomic sequence.

**Key result:** The model achieves **1.29 bits/base**, significantly outperforming the 5th-order Markov baseline at **1.886 bits/base** (~31% improvement).

## Model description

- **Architecture:** Decoder-only Transformer (GPT)
- **Parameters:** 21,691,008
- **Context length:** 1,024 bases
- **Vocabulary:** A, C, G, T, N + special tokens (BOS, EOS, PAD, UNK)
- **Tokenizer:** Single-nucleotide (no k-mer or BPE)
- **Training steps:** 8,000,000
- **Optimizer:** AdamW (lr=3e-4, weight_decay=0.1)
- **Hardware:** Single GPU
- **Training time:** ~2 weeks

## Intended use

- Research on DNA sequence patterns and next-base prediction
- Comparing neural language models against classical Markov baselines
- Educational demonstrations of small-scale genomic language modeling

## Not intended for

- SNP effect or variant pathogenicity prediction
- Gene function or phenotype prediction
- Environmental adaptation prediction
- Clinical, agricultural, or ecological decision-making
- Cross-species generalization (trained on moss only)

## Quickstart

```python
import torch
from safetensors.torch import load_file

# Load the model from this repository
# (requires the moss_dna_gpt package from github.com/Unjuno/moss-dna-gpt)
from moss_dna_gpt.model import GPT, GPTConfig
from moss_dna_gpt.tokenizer import DnaTokenizer
from moss_dna_gpt.sampling import sample

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load config
config = GPTConfig(
    vocab_size=9, block_size=1024, n_layer=12,
    n_head=8, n_embd=384, dropout=0.1, bias=True,
)

# Load weights
model = GPT(config).to(device)
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)
model.eval()

# Generate
tokenizer = DnaTokenizer()
prefix = "ACGTACGTACGT"
ids = tokenizer.encode(prefix, unknown="n")
idx = torch.tensor([ids], dtype=torch.long, device=device)
allowed = [tokenizer.stoi[b] for b in tokenizer.dna_tokens]
out = sample(model, idx, max_new_tokens=128, temperature=0.8, top_k=4, allowed_token_ids=allowed)
print(tokenizer.decode(out[0].tolist(), skip_special=True))
```

## Evaluation results

Evaluation on held-out sequence-level test windows:

| Model | Bits/base | Nats/base |
|---|---|---:|
| Markov order 0 | 1.925 | 1.334 |
| Markov order 1 | 1.909 | 1.323 |
| Markov order 5 | 1.886 | 1.307 |
| **DNA-GPT 20M (step 8M)** | **1.291** | **0.895** |

The learning curve shows DNA-GPT surpassing all Markov baselines within the first 750K steps:

![Learning curve](learning_curve.png)

## Files

| File | Description |
|---|---|
| `model.safetensors` | Model weights in safetensors format |
| `config.json` | Model configuration |
| `metadata.json` | Training metadata |
| `loss.csv` | Training/validation loss log |
| `eval_curve.json` | Evaluation curve (bits/base vs step) |
| `learning_curve.png` | Publication-quality learning curve figure |

## Training data

- **Organism:** *Physcomitrium patens* ecotype Gransden 2004
- **Assembly:** GCA_000002425.3 / Phypa_V5
- **Source:** NCBI GenBank genomic FASTA
- **Window size:** 1,024
- **Stride:** 512
- **Split:** Sequence-level split (train/val/test)
- Raw FASTA and processed shards are not distributed.

## Citation

```bibtex
@software{moss_dna_gpt_2026,
  author = {Unjuno},
  title = {moss-dna-gpt: Minimal DNA Language Model for Moss Genome},
  year = {2026},
  url = {https://github.com/Unjuno/moss-dna-gpt}
}
```

## Related models

- [DNABERT](https://github.com/jerryji1993/DNABERT) — BERT for DNA
- [Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer) — Large-scale genomic LM
- [HyenaDNA](https://github.com/HazyResearch/hyena-dna) — Long-context DNA model
- [Evo 2](https://github.com/arcinstitute/evo2) — Genome-scale generative model
- [Carbon](https://github.com/huggingface/carbon) — Efficient eukaryotic DNA LM

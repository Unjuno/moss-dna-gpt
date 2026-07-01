# Streamlit demo

## How to run

```bash
# Install with app dependencies
pip install -e ".[app]"

# Run the Streamlit UI
streamlit run apps/dna_chat.py -- --checkpoint <path-to-ckpt> --device auto
```

## Example commands

**Demo with the 50-step quick checkpoint:**

```bash
streamlit run apps/dna_chat.py -- \
    --checkpoint runs/physcomitrium_patens_quick_256/ckpt_step_50.pt \
    --device cpu
```

**Demo with the trained 20M checkpoint (.pt format):**

```bash
streamlit run apps/dna_chat.py -- \
    --checkpoint repo/moss-dna-gpt-20m-patens/ckpt_step_8000000.pt \
    --device auto
```

**Demo with the Hugging Face safetensors release:**

```bash
streamlit run apps/dna_chat.py -- \
    --checkpoint repo/moss-dna-gpt-20m-patens/model.safetensors \
    --config repo/moss-dna-gpt-20m-patens/config.json \
    --device auto
```

## Required checkpoint path

The `--checkpoint` argument points to a local checkpoint file — either a `.pt` training checkpoint or a `.safetensors` inference checkpoint. Checkpoints are **not stored in Git**; you must either:

- Train a model yourself with `scripts/train.py`
- Download a pre-trained checkpoint from Hugging Face (see `docs/huggingface_release.md`)

## What the output means

The model generates a continuation of the input DNA prefix, base by base. Each base is sampled from the model's predicted probability distribution over A, C, G, T, and N. The output is purely a **next-base prediction** — the model predicts the next most likely DNA base given the preceding context.

The sidebar shows:
- **Temperature** — controls randomness: lower = more deterministic, higher = more diverse
- **Top-k** — limits sampling to the k most likely tokens
- **Sliding window mode** — for generating longer sequences in chunks
- **Max new bases** — total number of bases to generate

## What the output does NOT mean biologically

- The model does **not** predict gene function, SNP effects, or phenotype.
- The model does **not** identify regulatory elements, splice sites, or other functional genomic features.
- The model does **not** represent evolutionary conservation or selective pressure.
- Lower perplexity than Markov baselines is a **statistical language modeling result** — it does not imply biological understanding.
- The model was trained on a single moss genome and does **not** generalize to other species.

## Recommended demo prefix examples

| Prefix | Notes |
|---|---|
| `ACGT` | Typical starting sequence |
| `ATATAT` | Simple repeat — model may continue the pattern |
| `NNNN` | All N — tests how the model handles ambiguous bases |
| `GCCGCCATGGCCGAGCTCGAGCTCGAG` | GC-rich region resembling coding sequence context |

## Safety

Only load checkpoints from trusted sources. PyTorch `.pt` files may execute unsafe pickle payloads on older PyTorch versions. Prefer `.safetensors` format (used for the Hugging Face release) as it is pickle-free and safer for inference.

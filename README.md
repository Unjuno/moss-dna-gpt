# moss-dna-gpt

Minimal research scaffold for pretraining a small decoder-only Transformer (GPT) on a single moss genome (*Physcomitrium patens*) and comparing held-out next-base prediction against Markov baselines.

## What this project is

- A reproducible research demo for DNA next-base prediction
- A comparison between a small neural language model and classical Markov baselines on a single genome
- An educational example of genomic language modeling at minimal scale

## What this project is NOT

- A production-ready variant effect predictor
- A gene function or phenotype prediction tool
- A clinical, agricultural, or ecological decision-making system
- A cross-species genomic foundation model
- A finished research result — it is an exploratory scaffold

## Result

On a **sequence-level held-out test split** of *Physcomitrium patens*, the 20M-parameter DNA-GPT outperforms 0th, 1st, and 5th order Markov baselines in bits/base:

| Model | Bits/base |
|---|---:|
| Markov order 0 | 1.92504 |
| Markov order 1 | 1.90902 |
| Markov order 5 | 1.88614 |
| **DNA-GPT 20M (step 8M)** | **1.41665** |

See [`docs/results.md`](docs/results.md) for details and [`results/summary_20m_step8000000.md`](results/summary_20m_step8000000.md) for the canonical result statement.

## Repository structure

```
moss-dna-gpt/
├── moss_dna_gpt/          # Core library package
├── apps/                  # Streamlit demo application
├── scripts/               # Training, evaluation, plotting, data preparation
├── configs/               # Training configuration YAML files
├── tests/                 # Pytest test suite
├── docs/                  # Documentation
├── results/               # Evaluation result artifacts (Git-safe)
├── repo/                  # Hugging Face model card and metadata
├── data/                  # NOT IN GIT — raw FASTA and processed shards
└── runs/                  # NOT IN GIT — training checkpoints and logs
```

## Installation

```bash
git clone https://github.com/Unjuno/moss-dna-gpt.git
cd moss-dna-gpt
python -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install -e .

# For Streamlit demo
pip install -e ".[app]"

# For development (testing)
pip install -e ".[dev]"

# For Hugging Face publishing
pip install -e ".[hf]"

# All extras
pip install -e ".[all]"
```

## Dummy smoke test

Run the entire pipeline on a tiny dummy FASTA file (no GPU, no data download, <1 minute):

```bash
python scripts/inspect_fasta.py tests/fixtures/dummy.fa
python scripts/prepare_windows.py tests/fixtures/dummy.fa \
    --out-dir data/processed_dummy \
    --block-size 32 \
    --stride 16 \
    --max-n-rate 1.0 \
    --max-windows 4
python scripts/train.py \
    --train-path data/processed_dummy/train.txt \
    --val-path data/processed_dummy/val.txt \
    --run-dir runs/dummy \
    --max-steps 10 \
    --batch-size 2 \
    --n-layer 1 \
    --n-head 1 \
    --n-embd 32 \
    --block-size 32 \
    --device cpu
python scripts/generate.py \
    --checkpoint runs/dummy/ckpt_step_10.pt \
    --prefix ACGT \
    --max-new-tokens 32 \
    --device cpu
python scripts/eval_markov.py \
    --train-path data/processed_dummy/train.txt \
    --test-path data/processed_dummy/test.txt \
    --checkpoint runs/dummy/ckpt_step_10.pt \
    --device cpu
```

## Real FASTA workflow

```bash
# 1. Fetch the moss genome
python scripts/fetch_genome.py

# 2. Prepare windows (sequence-level split for honest evaluation)
python scripts/prepare_windows.py data/raw/genome.fa.gz \
    --out-dir data/processed \
    --block-size 1024 \
    --stride 512 \
    --max-n-rate 0.2 \
    --shard \
    --shard-size 100000 \
    --split-policy sequence

# 3. Train
python scripts/train.py \
    --config configs/train_20m_1024_sequence_2weeks.yaml \
    --streaming
```

## Sequence-level split warning

This project uses **sequence-level splitting**: each FASTA sequence entry is assigned entirely to one split (train, validation, or test). This prevents the model from seeing nearly identical windows from the same genomic region in both training and evaluation.

If you use random splitting (each window assigned independently), evaluation metrics will be **optimistically biased** due to data leakage.

## Markov baseline comparison

```bash
python scripts/eval_markov.py \
    --train-path data/processed/physcomitrium_patens_5m_1024_sequence/train \
    --test-path data/processed/physcomitrium_patens_5m_1024_sequence/test \
    --checkpoint runs/physcomitrium_patens_20m_1024_sequence_2weeks/ckpt_step_8000000.pt \
    --device auto \
    --streaming
```

## Streamlit demo

```bash
pip install -e ".[app]"
# With the Hugging Face safetensors release (recommended):
streamlit run apps/dna_chat.py -- \
    --checkpoint repo/moss-dna-gpt-20m-patens/model.safetensors \
    --config repo/moss-dna-gpt-20m-patens/config.json \
    --device auto
# Or with a local .pt training checkpoint:
# streamlit run apps/dna_chat.py -- \
#     --checkpoint runs/physcomitrium_patens_quick_256/ckpt_step_50.pt \
#     --device cpu
```

See [`docs/demo.md`](docs/demo.md) for detailed usage, example prefixes, and biological limitations.

## Test

```bash
pytest
```

## Plot results

```bash
# Markov + DNA-GPT comparison bar chart
python scripts/plot_results.py

# Training loss curve
python scripts/plot_loss.py \
    --loss-csv repo/moss-dna-gpt-20m-patens/loss.csv \
    --out results/figures/loss_20m.png

# Learning curve
python scripts/plot_learning_curve.py \
    --eval-json repo/moss-dna-gpt-20m-patens/eval_curve.json \
    --out results/figures/learning_curve_20m.png
```

## Hugging Face model

The released 20M checkpoint is available on the Hugging Face Hub:

[https://huggingface.co/Unjuno/moss-dna-gpt-20m-patens](https://huggingface.co/Unjuno/moss-dna-gpt-20m-patens)

The model card is maintained at [`repo/moss-dna-gpt-20m-patens/README.md`](repo/moss-dna-gpt-20m-patens/README.md).  
Model weights, config, evaluation results, and loss curves are published to Hugging Face — they are not stored in this Git repository.

## Data and checkpoint NOT in Git

The following are **not stored in this repository**:

- Raw FASTA files (`data/`)
- Processed dataset shards (`data/`)
- Training checkpoints and logs (`runs/`)
- Any `.pt`, `.pth`, `.ckpt`, or `.safetensors` files

These are either generated locally by the user or downloaded from Hugging Face.

## Limitations

See the full [limitations document](docs/limitations.md). Key points:

- **Next-base prediction only** — not SNP effect, gene function, or phenotype prediction.
- **Single species** — trained only on *Physcomitrium patens*.
- **Small scale** — 21.7M parameters vs 40B+ in modern genomic foundation models.
- **Statistical result** — lower bits/base than Markov baselines does not prove biological understanding.

## Safety

Only load checkpoints from trusted sources. PyTorch `.pt` files may execute unsafe pickle payloads on older versions. Prefer `.safetensors` format (Hugging Face release) for inference.

## Related work

DNA language models and genomic foundation models:

| Model | Year | Architecture | Key idea |
|---|---|---|---|
| **DNABERT** | 2021 | BERT encoder | First DNA LM, k-mer tokenization, 512 bp |
| **DNABERT-2** | 2023 | BERT encoder | BPE tokenization, multi-species |
| **Nucleotide Transformer** | 2023 | BERT encoder | 850 species, up to 2.5B params |
| **HyenaDNA** | 2023 | Decoder (Hyena) | 1M token context, single-nucleotide |
| **DNAGPT** | 2023 | GPT decoder | Multi-task pre-training |
| **Caduceus** | 2024 | SSM (Mamba) | Bidirectional, RC-equivariant |
| **GROVER** | 2024 | BERT encoder | BPE for human genome |
| **Evo** | 2024 | StripedHyena | Genome-scale generative model |
| **MambaDNA** | 2024 | SSM (Mamba) | Bidirectional, single-nucleotide |
| **Evo 2** | 2025 | StripedHyena | 40B params, all domains of life |
| **Omni-DNA** | 2025 | GPT decoder | Multi-task, 20M-1B params |
| **Carbon** | 2026 | Llama decoder | 3B/8B params, eukaryotic-focused |

## Citation

```bibtex
@software{moss_dna_gpt_2026,
  author = {Unjuno},
  title = {moss-dna-gpt: Minimal DNA Language Model for Moss Genome},
  year = {2026},
  url = {https://github.com/Unjuno/moss-dna-gpt}
}
```

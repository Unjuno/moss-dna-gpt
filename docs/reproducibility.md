# Reproducibility guide

## Prerequisites

```bash
git clone https://github.com/Unjuno/moss-dna-gpt.git
cd moss-dna-gpt
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## Dummy smoke test (no GPU, no data download)

```bash
# Inspect dummy FASTA
python scripts/inspect_fasta.py tests/fixtures/dummy.fa

# Prepare windows
python scripts/prepare_windows.py tests/fixtures/dummy.fa \
    --out-dir data/processed_dummy \
    --block-size 32 \
    --stride 16 \
    --max-n-rate 1.0 \
    --max-windows 4

# Train a tiny model
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

# Generate
python scripts/generate.py \
    --checkpoint runs/dummy/ckpt_step_10.pt \
    --prefix ACGT \
    --max-new-tokens 32 \
    --device cpu

# Evaluate against Markov baselines
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

# 2. Prepare windows (sequence-level split)
python scripts/prepare_windows.py data/raw/genome.fa.gz \
    --out-dir data/processed \
    --block-size 1024 \
    --stride 512 \
    --max-n-rate 0.2 \
    --shard \
    --shard-size 100000

# 3. Train the 20M model
python scripts/train.py \
    --config configs/train_20m_1024_sequence_2weeks.yaml \
    --streaming \
    --max-steps 8000000
```

## Sequence split dataset preparation

The `prepare_windows.py` script assigns each FASTA sequence entry entirely to one split:

- Sequences 0–79% → training
- Sequences 80–89% → validation
- Sequences 90–100% → test

This ensures no overlapping windows between splits.

## Leakage overlap audit

To verify that training and test windows do not share sequence content, check the sequence IDs in the generated shards:

```bash
# Extract unique sequence IDs from training and test shards
head -1 data/processed/train/*.txt | grep '>' | sort -u > train_seqs.txt
head -1 data/processed/test/*.txt | grep '>' | sort -u > test_seqs.txt

# Check for overlap
comm -12 train_seqs.txt test_seqs.txt
```

No intersection means clean sequence-level split. If any sequence ID appears in both, the split has leakage.

## Resume training

```bash
python scripts/train.py \
    --config configs/train_20m_1024_sequence_2weeks.yaml \
    --streaming \
    --resume-from runs/physcomitrium_patens_20m_1024_sequence_2weeks/ckpt_step_4000000.pt
```

## Evaluate Markov baselines + DNA-GPT

```bash
python scripts/eval_markov.py \
    --train-path data/processed/physcomitrium_patens_5m_1024_sequence/train \
    --test-path data/processed/physcomitrium_patens_5m_1024_sequence/test \
    --checkpoint runs/physcomitrium_patens_20m_1024_sequence_2weeks/ckpt_step_8000000.pt \
    --device auto \
    --streaming
```

## Plot results

```bash
# Markov + DNA-GPT comparison bar chart
python scripts/plot_results.py

# Training loss curve
python scripts/plot_loss.py \
    --loss-csv repo/moss-dna-gpt-20m-patens/loss.csv \
    --out results/figures/loss_20m.png

# Learning curve from eval_curve.json
python scripts/plot_learning_curve.py \
    --eval-json repo/moss-dna-gpt-20m-patens/eval_curve.json \
    --out results/figures/learning_curve_20m.png
```

## All-checkpoint evaluation curve

```bash
python scripts/eval_all_checkpoints.py \
    --run-dir runs/physcomitrium_patens_20m_1024_sequence_2weeks \
    --test-path data/processed/physcomitrium_patens_5m_1024_sequence/test \
    --device auto \
    --streaming
```

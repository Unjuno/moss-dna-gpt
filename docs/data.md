# Data

## Genomic source

- **Organism:** *Physcomitrium patens* ecotype Gransden 2004
- **Assembly:** GCA_000002425.3 / Phypa_V5
- **Source:** NCBI GenBank genomic FASTA
- **Download:** `scripts/fetch_genome.py`

The raw FASTA file is **not stored in Git**. It must be downloaded manually or via the provided scripts:

```bash
python scripts/fetch_genome.py
```

## Processed dataset

Training windows were generated with:

```bash
python scripts/prepare_windows.py data/raw/genome.fa.gz \
    --out-dir data/processed \
    --block-size 1024 \
    --stride 512 \
    --max-n-rate 0.2 \
    --shard \
    --shard-size 100000
```

- **Window size:** 1,024 bases
- **Stride:** 512 bases
- **Max N-rate:** 20%
- **Split policy:** Sequence-level (train/val/test) — each sequence is assigned entirely to one split to avoid leakage.

Processed shards are **not stored in Git**. They are generated locally by the user.

## Sequence-level split warning

A **sequence-level split** means that every window from a given FASTA sequence entry is assigned to the same split (train, validation, or test). This prevents the model from seeing nearly identical windows from the same genomic region in both training and evaluation.

If you use a random split (each window assigned independently), nearby windows from the same sequence will appear in both training and test sets, producing **optimistically biased** evaluation metrics. Always use sequence-level splits for honest evaluation.

The 20M model was evaluated on a sequence-level held-out test split.

# MVP scope

## Included

- One-species FASTA inspection.
- Fixed-length non-cross-contig windows.
- `A C G T N` character tokenizer plus special tokens.
- Small decoder-only Transformer for next-base prediction.
- 0th, 1st, and 5th-order Markov cross-entropy baselines.
- Dummy FASTA tests and smoke pipeline.

## Excluded in the initial version

- Own sequence collection.
- Real SNP analysis.
- Environmental adaptation prediction.
- Multi-species training.
- FASTA/GFF3/checkpoint/processed-dataset versioning in Git.

## Scientific target

The first falsifiable target is narrow:

> A 5M-class DNA-GPT trained on one moss genome FASTA should obtain lower held-out next-base cross entropy than simple Markov baselines under the same train/test window split.

This is not evidence of biological function understanding. It is only a minimal sanity check that the neural baseline learns non-trivial sequence regularities beyond fixed-order Markov models.

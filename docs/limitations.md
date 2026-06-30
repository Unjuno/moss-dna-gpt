# Limitations

## Biological limitations

This model is a **decoder-only Transformer trained for DNA next-base prediction on a single moss genome**. It has severe and deliberate limitations:

- **No SNP effect prediction.** The model predicts the next base given surrounding context. A single-nucleotide variant may change the predicted probability of the next base, but this does **not** indicate functional impact, pathogenicity, or evolutionary conservation.

- **No gene function prediction.** The model has no concept of gene structure, coding regions, regulatory elements, or biological function.

- **No phenotype prediction.** The model cannot predict organismal traits, development, or environmental responses from DNA sequence.

- **No environmental adaptation prediction.** The model was trained on a single genome assembly and has no information about population genetics, selective pressures, or environmental conditions.

- **No cross-species generalization.** The model was trained exclusively on *Physcomitrium patens* and is not expected to produce meaningful results on other species.

- **No structural variant interpretation.** The fixed context window (1,024 bases) limits the model's ability to capture long-range genomic dependencies such as chromatin conformation, regulatory loops, or large structural variants.

## Computational limitations

- **Small scale.** At 21.7M parameters, this model is orders of magnitude smaller than modern genomic foundation models (e.g., Evo 2 at 40B, Nucleotide Transformer at 2.5B).
- **Single-species.** Training on one genome limits the model's ability to learn universal genomic patterns.
- **Context length.** 1,024 bases is short relative to gene and regulatory element scales.

## Evaluation limitations

- **Sequence-level split** reduces data leakage but does not eliminate it entirely — nearby genomic regions have similar sequence content.
- **Bits/base comparison against Markov baselines** shows that a neural language model can capture longer-range dependencies than fixed-order Markov models. This is a **statistical result**, not a biological one.
- **Final evaluation at step 8M used different eval batch size** than the intermediate curve points, causing minor numerical differences.

## Safety

Only load checkpoints from trusted sources. PyTorch `.pt` files may execute unsafe pickle payloads on older versions.

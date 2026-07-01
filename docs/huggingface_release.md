# Hugging Face release guide

## Suggested model name

`moss-dna-gpt-20m-patens`

## Files to upload

| File | Source | Description |
|---|---|---|
| `model.safetensors` | Converted from `.pt` | Safe weights-only format (recommended for inference) |
| `config.json` | `repo/moss-dna-gpt-20m-patens/config.json` | Model hyperparameters |
| `loss.csv` | `repo/moss-dna-gpt-20m-patens/loss.csv` | Training/validation loss log |
| `eval_markov_step8000000.json` | `repo/moss-dna-gpt-20m-patens/eval_markov_step8000000.json` | Markov baseline comparison results |
| `eval_curve.json` | `repo/moss-dna-gpt-20m-patens/eval_curve.json` | Evaluation curve (bits/base vs step) |
| `learning_curve.png` | `repo/moss-dna-gpt-20m-patens/learning_curve.png` | Publication-quality learning curve figure |
| `README.md` | `repo/moss-dna-gpt-20m-patens/README.md` | Model card (HF renders this automatically) |

> **Prefer `model.safetensors` over `.pt`.** The `.pt` checkpoint (PyTorch pickle) may execute unsafe code on older PyTorch versions. Upload only the converted `model.safetensors` for the main model weights.

## Files NOT to upload

- Raw FASTA files (not public data distribution)
- Processed train/val/test shards (can be regenerated)
- Training logs beyond the loss CSV
- Any files containing paths or data from your local machine

## Conversion to safetensors

```bash
python scripts/convert_to_safetensors.py \
    --checkpoint runs/physcomitrium_patens_20m_1024_sequence_2weeks/ckpt_step_8000000.pt \
    --out-dir repo/moss-dna-gpt-20m-patens
```

This produces `model.safetensors` in the output directory.

## Upload command

```bash
pip install -e ".[hf]"

python scripts/publish_to_hf.py \
    --repo-id your-username/moss-dna-gpt-20m-patens \
    --local-dir repo/moss-dna-gpt-20m-patens \
    --token hf_xxxx
```

### Optional flags

- `--private` — create a private repository
- `--dry-run` — print what would be uploaded without uploading

### Dry-run example

```bash
python scripts/publish_to_hf.py \
    --repo-id your-username/moss-dna-gpt-20m-patens \
    --local-dir repo/moss-dna-gpt-20m-patens \
    --dry-run
```

## Verify published files

```bash
pip install -e ".[hf]"

hf download Unjuno/moss-dna-gpt-20m-patens \
    model.safetensors config.json metadata.json README.md \
    --local-dir hf_check
ls -lh hf_check/
```

## Model card limitations section

The Hugging Face model card must include the following limitations prominently:

> **Research use only.** This model performs next-base prediction on DNA sequences. It is not intended for:
> - SNP effect or variant pathogenicity prediction
> - Gene function or phenotype prediction
> - Environmental adaptation prediction
> - Clinical, agricultural, or ecological decision-making
> - Cross-species generalization (trained on moss only)
>
> Lower bits/base than Markov baselines is a statistical language modeling result and does not imply biological understanding.

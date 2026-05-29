# moss-dna-gpt

Minimal research scaffold for pretraining a 5M-class DNA-GPT on one moss genome FASTA and comparing held-out next-base prediction against Markov baselines.

## Hard constraints

- Do **not** commit real genome FASTA, GFF3, checkpoints, or processed datasets.
- `data/` and `runs/` are ignored by `.gitignore`.
- Initial version does **not** handle real SNP analysis.
- Initial version does not collect own sequences, predict environmental adaptation, or train across multiple species.
- First target: train on one FASTA and check whether DNA-GPT beats 0th/1st/5th-order Markov baselines.

## Install

```bash
git clone https://github.com/Unjuno/moss-dna-gpt.git
cd moss-dna-gpt
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## GPU / device check

Training uses `--device auto` by default in the real-data quickstart. If PyTorch can see CUDA, the scripts select `cuda`; otherwise they fall back to `cpu`.

```bash
python scripts/check_device.py
```

If `cuda_available` is `false` on a machine with an NVIDIA GPU, install the CUDA-enabled PyTorch wheel that matches your driver from the official PyTorch selector.

## Fastest real-data quickstart

This fetches one default moss genome target, prepares windows under `data/processed/...`, and starts a short quick training run. Real FASTA and checkpoints stay under `data/` and `runs/`, which are ignored by Git.

```bash
python scripts/run_real_quickstart.py --profile quick --device auto
```

For the 5M-class MVP target:

```bash
python scripts/run_real_quickstart.py --profile 5m --device auto --max-steps 1000
```

After training, the script prints the exact `generate.py` and `eval_markov.py` commands for the produced checkpoint.

## Test

```bash
pytest
```

## Dummy smoke pipeline

```bash
python scripts/inspect_fasta.py tests/fixtures/dummy.fa
python scripts/prepare_windows.py tests/fixtures/dummy.fa --out-dir data/processed_dummy --block-size 32 --stride 16 --max-n-rate 1.0
python scripts/train.py --train-path data/processed_dummy/train.txt --val-path data/processed_dummy/val.txt --run-dir runs/dummy --max-steps 10 --batch-size 2 --n-layer 1 --n-head 1 --n-embd 32 --block-size 32 --device cpu
python scripts/generate.py --checkpoint runs/dummy/ckpt_step_10.pt --prefix ACGT --max-new-tokens 32 --device cpu
python scripts/eval_markov.py --train-path data/processed_dummy/train.txt --test-path data/processed_dummy/test.txt --checkpoint runs/dummy/ckpt_step_10.pt --device cpu
```

## 5M-class config

`configs/train_5m_1024.yaml` uses `n_layer=6`, `n_head=4`, `n_embd=256`, `block_size=1024`.

```bash
python scripts/prepare_windows.py data/raw/moss/genome.fa.gz --out-dir data/processed --block-size 1024 --stride 512 --max-n-rate 0.2
python scripts/train.py --config configs/train_5m_1024.yaml --max-steps 1000
```

Lower DNA-GPT loss than Markov baselines only means the model captured sequence regularities under this split. It does not prove function understanding, SNP effect prediction, adaptation prediction, or cross-species generalization.

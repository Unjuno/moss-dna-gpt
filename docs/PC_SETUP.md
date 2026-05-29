# PC setup

This document targets local clone and dummy training smoke tests.

## Windows PowerShell

```powershell
git clone https://github.com/Unjuno/moss-dna-gpt.git
cd moss-dna-gpt
git switch mvp-initial-dna-gpt
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

## Dummy smoke

```powershell
python scripts/inspect_fasta.py tests/fixtures/dummy.fa
python scripts/prepare_windows.py tests/fixtures/dummy.fa --out-dir data/processed_dummy --block-size 32 --stride 16 --max-n-rate 1.0
python scripts/train.py --train-path data/processed_dummy/train.txt --val-path data/processed_dummy/val.txt --run-dir runs/dummy --max-steps 10 --batch-size 2 --n-layer 1 --n-head 1 --n-embd 32 --block-size 32 --device cpu
python scripts/generate.py --checkpoint runs/dummy/ckpt_step_10.pt --prefix ACGT --max-new-tokens 32 --device cpu
python scripts/eval_markov.py --train-path data/processed_dummy/train.txt --test-path data/processed_dummy/test.txt --checkpoint runs/dummy/ckpt_step_10.pt --device cpu
```

## Real FASTA placement

Place real FASTA under `data/raw/...`. Do not commit it. `data/` is ignored.

## Current non-goals

- Real SNP analysis.
- Environmental adaptation prediction.
- Multi-species training.
- Storing checkpoints or processed datasets in Git.

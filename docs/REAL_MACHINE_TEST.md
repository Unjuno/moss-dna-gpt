# Real machine audit checklist

This checklist is for the first local PC audit after CI success.

## Scope

The target is not to prove biological usefulness. The target is to verify that a user can clone the repository, fetch or use one real FASTA, start training, create a checkpoint, evaluate against Markov baselines, and open the DNA completion UI without committing data or checkpoints.

## Non-goals

- Real SNP analysis.
- Environmental adaptation prediction.
- Multi-species training.
- Claiming biological function understanding.
- Committing real FASTA, processed windows, or checkpoints to Git.

## Phase 0: environment capture

Run these first and save the output.

```powershell
python --version
python -m pip --version
git --version
```

After installation, run:

```powershell
python scripts/check_device.py
```

### PASS

- Python is 3.10 or newer.
- `python scripts/check_device.py` runs without exception.
- If a CUDA GPU is expected, `cuda_available` should be `true`.
- If `cuda_available` is `false`, training can still run on CPU, but GPU audit is not passed.

## Phase 1: clean clone and install

```powershell
git clone https://github.com/Unjuno/moss-dna-gpt.git
cd moss-dna-gpt
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

### PASS

- `pip install -e ".[dev]"` succeeds.
- `pytest` succeeds.

### FAIL

- Editable install fails.
- Import errors occur.
- Any test fails.

## Phase 2: dummy smoke

```powershell
python scripts/inspect_fasta.py tests/fixtures/dummy.fa
python scripts/prepare_windows.py tests/fixtures/dummy.fa --out-dir data/processed_dummy --block-size 32 --stride 16 --max-n-rate 1.0 --max-windows 4
python scripts/train.py --train-path data/processed_dummy/train.txt --val-path data/processed_dummy/val.txt --run-dir runs/dummy --max-steps 10 --batch-size 2 --n-layer 1 --n-head 1 --n-embd 32 --block-size 32 --device cpu
python scripts/generate.py --checkpoint runs/dummy/ckpt_step_10.pt --prefix ACGT --max-new-tokens 32 --device cpu
python scripts/eval_markov.py --train-path data/processed_dummy/train.txt --test-path data/processed_dummy/test.txt --checkpoint runs/dummy/ckpt_step_10.pt --device cpu
```

### PASS

- `runs/dummy/ckpt_step_10.pt` exists.
- Generated sequence contains only A/C/G/T/N.
- `eval_markov.py` outputs Markov and DNA-GPT values.
- `git status --short` does not show tracked data/checkpoint candidates.

## Phase 3: real FASTA quickstart

This path fetches the default moss target into `data/raw/...` and starts a short real-data run.

```powershell
python scripts/run_real_quickstart.py --profile quick --device auto
```

### PASS

- FASTA is stored under `data/raw/...`.
- processed windows are stored under `data/processed/...`.
- checkpoint is stored under `runs/...`.
- the command prints `next_commands`.
- `git status --short` does not show real FASTA, processed windows, or checkpoint files.

### FAIL

- FASTA is created outside `data/`.
- checkpoint is created outside `runs/`.
- `git status --short` shows `.fna`, `.fa`, `.fa.gz`, `.pt`, or processed dataset files as untracked candidates.

## Phase 4: 5M sharded streaming smoke

Do not start with a long run. First test 100 steps.

```powershell
python scripts/run_real_quickstart.py --profile 5m --device auto --max-steps 100
```

### PASS

- `resolved_streaming` is `true` in the run config.
- sharded windows are created under split directories such as `train/`, `val/`, and `test/`.
- `ckpt_step_100.pt` is created.
- `loss.csv` is created.
- no OOM occurs.

### UNCERTAIN

- CPU-only run is slow but does not fail.
- GPU is present but `cuda_available` is `false`.
- checkpoint is produced, but evaluation is too slow.

### FAIL

- OOM.
- checkpoint is not generated.
- `resolved_streaming` is false for a 5M run.
- checkpoint or processed data appears as a Git tracking candidate.

## Phase 5: Markov comparison

Use the exact paths printed by `run_real_quickstart.py`. Example:

```powershell
python scripts/eval_markov.py --train-path data/processed/physcomitrium_patens_5m_1024/train --test-path data/processed/physcomitrium_patens_5m_1024/test --checkpoint runs/physcomitrium_patens_5m_1024/ckpt_step_100.pt --device auto --streaming
```

### PASS

- 0th, 1st, and 5th-order Markov values are printed.
- DNA-GPT `bits_per_base` is printed.
- output can be saved as an evaluation report.

### Scientific caution

A short 100-step smoke run is not expected to beat Markov baselines. It only verifies that the evaluation path runs.

## Phase 6: Streamlit UI

```powershell
pip install -e ".[app]"
streamlit run apps/dna_chat.py -- --checkpoint runs/physcomitrium_patens_5m_1024/ckpt_step_100.pt --device auto
```

### PASS

- UI opens.
- checkpoint path is reflected in the sidebar.
- prefix input accepts A/C/G/T/N.
- output is DNA continuation.
- UI does not present itself as a SNP, gene-function, or adaptation predictor.

## Required audit report

Copy these into the next report.

```text
OS:
Python:
PyTorch:
CUDA available:
Device selected by check_device.py:
GPU name, if any:
Phase 1 pytest: PASS/FAIL
Phase 2 dummy smoke: PASS/FAIL
Phase 3 quickstart: PASS/FAIL
Phase 4 5M 100-step smoke: PASS/FAIL/UNCERTAIN
Phase 5 eval_markov: PASS/FAIL/UNCERTAIN
Phase 6 Streamlit UI: PASS/FAIL/UNCERTAIN
Peak GPU memory, if observed:
Runtime for 5M 100 steps:
First failure command, if any:
Error message, if any:
git status --short after run:
```

## Decision rule

```text
PASS:
  - dummy smoke passes
  - real quickstart passes
  - 5M 100-step smoke creates checkpoint
  - eval_markov runs
  - data and runs remain ignored by Git

UNCERTAIN:
  - CPU-only run is slow but correct
  - GPU not detected despite hardware being present
  - NCBI fetch fails but local FASTA path works

FAIL:
  - install/test fails
  - OOM in quick or 5M smoke
  - checkpoint missing
  - streaming false in 5M run
  - real FASTA/checkpoint appears in Git tracking candidates
```

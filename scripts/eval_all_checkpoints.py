#!/usr/bin/env python
"""
Evaluate all checkpoints and compare with Markov baselines.

Usage:
    python scripts/eval_all_checkpoints.py

Output:
    - Prints eval curve to stdout (JSON)
    - Saves to repo/moss-dna-gpt-20m-patens/eval_curve.json

Paths are relative to the repo root.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ── repo root (this file lives at scripts/eval_all_checkpoints.py) ──────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from moss_dna_gpt.dataset import StreamingDnaWindowDataset
from moss_dna_gpt.metrics import nats_to_bits
from moss_dna_gpt.tokenizer import DnaTokenizer
from moss_dna_gpt.trainer import evaluate, load_checkpoint, resolve_device

# ── config ──────────────────────────────────────────────────────────────
RUN_DIR = REPO_ROOT / "runs" / "physcomitrium_patens_20m_1024_sequence_2weeks"
TEST_PATH = REPO_ROOT / "data" / "processed" / "physcomitrium_patens_5m_1024_sequence" / "test"
OUTPUT_PATH = REPO_ROOT / "repo" / "moss-dna-gpt-20m-patens" / "eval_curve.json"
BATCH_SIZE = 4
EVAL_BATCHES = 200

# Markov baselines (precomputed on the same test split)
MARKOV_BASELINES = {
    "markov_0": {"bits_per_base": 1.9250405735172231, "order": 0},
    "markov_1": {"bits_per_base": 1.9090154216846290, "order": 1},
    "markov_5": {"bits_per_base": 1.8861445601390048, "order": 5},
}

# steps to evaluate (every 500K)
STEPS = list(range(250000, 8_000_001, 500_000))


def main() -> None:
    device = resolve_device("auto")
    tokenizer = DnaTokenizer()

    print(f"Test path: {TEST_PATH}")
    print(f"Loading test dataset (streaming)...", flush=True)
    ds = StreamingDnaWindowDataset(str(TEST_PATH), tokenizer=tokenizer, shuffle_files=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  DataLoader ready (batch_size={BATCH_SIZE}, batches={EVAL_BATCHES})", flush=True)

    curve = []
    for step in STEPS:
        ckpt_path = RUN_DIR / f"ckpt_step_{step}.pt"
        if not ckpt_path.exists():
            print(f"  SKIP {step}: checkpoint not found", flush=True)
            continue

        t0 = time.time()
        model, ckpt_data = load_checkpoint(str(ckpt_path), map_location=device)
        model.to(device)
        loss = evaluate(model, loader, device, EVAL_BATCHES)
        bits = nats_to_bits(loss)
        elapsed = time.time() - t0
        curve.append({"step": step, "bits_per_base": round(bits, 6)})
        print(f"  step={step:>8}  bits/base={bits:.6f}  ({elapsed:.1f}s)", flush=True)

    output = {
        "test_path": str(TEST_PATH.relative_to(REPO_ROOT)),
        "batch_size": BATCH_SIZE,
        "eval_batches": EVAL_BATCHES,
        "markov_baselines": MARKOV_BASELINES,
        "dna_gpt_curve": curve,
    }

    print(f"\n--- Results ---")
    for pt in curve:
        print(f"  step {pt['step']:>8}: {pt['bits_per_base']:.6f} bits/base")
    print(f"\nMarkov baselines:")
    for name, v in MARKOV_BASELINES.items():
        print(f"  {name}: {v['bits_per_base']:.6f} bits/base")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as fp:
        json.dump(output, fp, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

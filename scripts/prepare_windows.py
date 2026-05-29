#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.dataset import prepare_windows_from_fasta


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare fixed-length DNA windows from FASTA.')
    parser.add_argument('fasta')
    parser.add_argument('--out-dir', default='data/processed')
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--max-n-rate', type=float, default=1.0)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-shuffle', action='store_true')
    parser.add_argument('--invalid-policy', choices=['skip', 'replace_n', 'error'], default='skip')
    args = parser.parse_args()

    manifest = prepare_windows_from_fasta(
        args.fasta,
        args.out_dir,
        block_size=args.block_size,
        stride=args.stride,
        max_n_rate=args.max_n_rate,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        invalid_policy=args.invalid_policy,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

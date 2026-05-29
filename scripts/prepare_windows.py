#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.dataset import prepare_window_shards_from_fasta, prepare_windows_from_fasta


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare fixed-length DNA windows from FASTA.')
    parser.add_argument('fasta')
    parser.add_argument('--out-dir', default='data/processed')
    parser.add_argument('--block-size', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--max-n-rate', type=float, default=1.0)
    parser.add_argument('--max-windows', type=int, default=None, help='Cap prepared windows. Omit or use 0 for unlimited.')
    parser.add_argument('--shard', action='store_true', help='Write sharded split directories for streaming training.')
    parser.add_argument('--shard-size', type=int, default=100000)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-shuffle', action='store_true', help='Flat mode only. Sharded mode uses per-window random split assignment.')
    parser.add_argument('--invalid-policy', choices=['skip', 'replace_n', 'error'], default='skip')
    args = parser.parse_args()

    max_windows = None if args.max_windows is None or args.max_windows <= 0 else args.max_windows
    if args.shard:
        manifest = prepare_window_shards_from_fasta(
            args.fasta,
            args.out_dir,
            block_size=args.block_size,
            stride=args.stride,
            max_n_rate=args.max_n_rate,
            max_windows=max_windows,
            shard_size=args.shard_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            invalid_policy=args.invalid_policy,
        )
    else:
        manifest = prepare_windows_from_fasta(
            args.fasta,
            args.out_dir,
            block_size=args.block_size,
            stride=args.stride,
            max_n_rate=args.max_n_rate,
            max_windows=max_windows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            shuffle=not args.no_shuffle,
            invalid_policy=args.invalid_policy,
        )
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

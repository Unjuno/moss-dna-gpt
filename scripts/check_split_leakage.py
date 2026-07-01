#!/usr/bin/env python
from __future__ import annotations

"""Check for potential train/test leakage in window-split manifests.

For sharded datasets with sequence-level split, verifies that each FASTA
record is assigned to exactly one split and reports overlap statistics.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Check train/test split integrity.')
    parser.add_argument('--manifest', default='data/processed/manifest.json', help='Path to manifest.json')
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f'Manifest not found: {manifest_path}')
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    split_policy = manifest.get('split_policy', 'unknown')
    seq_assignments = manifest.get('sequence_assignments', {})
    sequences = manifest.get('sequences', [])

    print(f'Manifest: {manifest_path}')
    print(f'  total_windows: {manifest.get("total_windows", "N/A")}')
    print(f'  split_policy: {split_policy}')
    print(f'  train_ratio: {manifest.get("train_ratio", "N/A")}')
    print(f'  val_ratio: {manifest.get("val_ratio", "N/A")}')
    print(f'  seed: {manifest.get("seed", "N/A")}')
    print(f'  format: {manifest.get("format", "N/A")}')

    if split_policy == 'sequence':
        print(f'\nSequence-level split check:')
        all_splits = set(seq_assignments.values())
        print(f'  unique sequences: {len(seq_assignments)}')
        print(f'  splits found: {sorted(all_splits)}')
        for split_name in all_splits:
            count = sum(1 for v in seq_assignments.values() if v == split_name)
            print(f'    {split_name}: {count} sequences')
        is_valid = all_splits.issubset({'train', 'val', 'test'})
        print(f'  valid splits: {"PASS" if is_valid else "FAIL"}')

        if sequences:
            seq_splits = {s.get('name'): s.get('split') for s in sequences if 'split' in s}
            overlap = set(seq_assignments.keys()) & set(seq_splits.keys())
            print(f'  manifest.sequences vs sequence_assignments overlap: {len(overlap)}')
            mismatch = sum(1 for k in overlap if seq_assignments.get(k) != seq_splits.get(k))
            print(f'  assignment consistency: {"PASS" if mismatch == 0 else f"FAIL ({mismatch} mismatches)"}')

    elif split_policy == 'window':
        print(f'\nWindow-level split (random assignment):')
        print(f'  WARNING: Windows may overlap between splits.')
        print(f'  With stride={manifest.get("stride", "?")} and '
              f'block_size={manifest.get("block_size", "?")}, '
              f'overlapping windows can be assigned to different splits.')
        print(f'  PASS if this is intentional, FAIL if sequence-level holdout was intended.')

    else:
        print(f'\nNo split_policy field — likely flat text (legacy) format.')
        print(f'  PASS if this is a legacy dataset, FAIL if sharded was expected.')


if __name__ == '__main__':
    main()

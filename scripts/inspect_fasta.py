#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.fasta import summarize_fasta


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect FASTA/FASTA.gz base composition.')
    parser.add_argument('fasta')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    summary = summarize_fasta(args.fasta)
    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(f"path: {summary['path']}")
    print(f"sequence_count: {summary['sequence_count']}")
    print(f"total_bp: {summary['total_bp']}")
    print(f"N_rate: {summary['n_rate']:.6f}")
    for base in 'ACGTN':
        print(f"{base}: {summary['counts'][base]} ({summary['ratios'][base]:.6f})")
    for record in summary['sequences']:
        print(f"{record['name']}\tlength={record['length']}\tinvalid={record['invalid_counts']}")


if __name__ == '__main__':
    main()

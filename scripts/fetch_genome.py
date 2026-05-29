#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.genome_fetch import DEFAULT_TAXID, fetch_genome


def main() -> None:
    parser = argparse.ArgumentParser(description='Fetch one moss genome FASTA into data/raw without committing it to Git.')
    parser.add_argument('--taxid', default=DEFAULT_TAXID, help='NCBI taxonomy id. Default: Physcomitrium/Physcomitrella patens taxid 3218.')
    parser.add_argument('--accession', help='Optional exact NCBI assembly accession, e.g. GCA_...')
    parser.add_argument('--out-dir', default='data/raw/physcomitrium_patens')
    parser.add_argument('--force', action='store_true', help='Re-download assembly summary and FASTA even if local files exist.')
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()

    manifest = fetch_genome(
        out_dir=args.out_dir,
        taxid=args.taxid,
        accession=args.accession,
        force=args.force,
        timeout=args.timeout,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from .fasta import ALLOWED, iter_fasta


def load_annotation_db(gff_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    regions: dict[str, list[dict[str, Any]]] = {}
    with gzip.open(gff_path, 'rt') if str(gff_path).endswith('.gz') else open(gff_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            chrom = parts[0]
            feat_type = parts[2]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attrs = parts[8]
            biotype = ''
            for a in attrs.split(';'):
                if a.startswith('biotype='):
                    biotype = a[8:]
            if chrom not in regions:
                regions[chrom] = []
            regions[chrom].append({
                'type': feat_type,
                'start': start,
                'end': end,
                'strand': strand,
                'biotype': biotype,
                'attrs': attrs,
            })
    return regions


def classify_window(
    chrom: str,
    start: int,
    end: int,
    annotation_db: dict[str, list[dict[str, Any]]],
) -> str:
    if chrom not in annotation_db:
        return 'intergenic'
    for feat in annotation_db[chrom]:
        if feat['type'] == 'gene':
            g_start = feat['start']
            g_end = feat['end']
            if start >= g_start and end <= g_end:
                if feat['biotype'] in ('protein_coding',):
                    return 'coding'
                elif feat['biotype']:
                    return feat['biotype']
                return 'genic'
            if abs(start - g_end) < 1000 or abs(end - g_start) < 1000 or (start <= g_end and end >= g_start):
                return 'genic'
    return 'intergenic'


def extract_promoter_regions(
    annotation_db: dict[str, list[dict[str, Any]]],
    upstream_bp: int = 500,
) -> list[dict[str, Any]]:
    promoters = []
    for chrom, feats in annotation_db.items():
        for feat in feats:
            if feat['type'] != 'gene':
                continue
            if feat['biotype'] != 'protein_coding':
                continue
            if feat['strand'] == '+':
                p_start = max(1, feat['start'] - upstream_bp)
                p_end = feat['start']
            else:
                p_start = feat['end']
                p_end = feat['end'] + upstream_bp
            promoters.append({
                'chrom': chrom,
                'start': p_start,
                'end': p_end,
                'strand': feat['strand'],
                'gene_id': feat['attrs'],
            })
    return promoters


def annotate_sequences_from_fasta(
    fasta_path: str | Path,
    annotation_db: dict[str, list[dict[str, Any]]],
    num_windows: int = 200,
    window_size: int = 256,
) -> list[dict[str, Any]]:
    result = []
    for record in iter_fasta(fasta_path):
        seq = record.sequence.upper()
        chrom = record.name.split()[0]
        if len(seq) < window_size:
            continue
        step = max(1, (len(seq) - window_size) // max(num_windows, 1))
        for i in range(0, len(seq) - window_size + 1, step):
            if len(result) >= num_windows:
                break
            w = seq[i:i+window_size]
            if all(c in ALLOWED for c in w):
                region = classify_window(chrom, i + 1, i + window_size, annotation_db)
                gc = (w.count('G') + w.count('C')) / max(len(w), 1)
                result.append({
                    'seq': w,
                    'chrom': chrom,
                    'start': i + 1,
                    'end': i + window_size,
                    'gc': gc,
                    'region': region,
                })
        if len(result) >= num_windows:
            break
    return result

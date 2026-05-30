from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen
import gzip
import hashlib
import json
import re
import time

ASSEMBLY_SUMMARY_GENBANK = 'https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_genbank.txt'
DEFAULT_TAXID = '3218'
DEFAULT_DATASET_ID = 'physcomitrium_patens'


@dataclass(frozen=True)
class AssemblyCandidate:
    row: dict[str, str]
    score: int

    @property
    def accession(self) -> str:
        return self.row['assembly_accession']

    @property
    def organism_name(self) -> str:
        return self.row.get('organism_name', '')

    @property
    def ftp_path(self) -> str:
        return self.row['ftp_path']

    @property
    def fasta_url(self) -> str:
        base = self.ftp_path.rstrip('/')
        name = base.split('/')[-1]
        return f'{base}/{name}_genomic.fna.gz'


def _download_text(url: str, timeout: int = 60) -> str:
    request = Request(url, headers={'User-Agent': 'moss-dna-gpt/0.1'})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode('utf-8')


def _download_binary(url: str, path: Path, timeout: int = 60) -> str:
    request = Request(url, headers={'User-Agent': 'moss-dna-gpt/0.1'})
    h = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with urlopen(request, timeout=timeout) as response, tmp.open('wb') as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            out.write(chunk)
    tmp.replace(path)
    return h.hexdigest()


def verify_gzip_readable(path: str | Path) -> dict:
    p = Path(path)
    try:
        with gzip.open(p, 'rb') as fp:
            sample = fp.read(1)
        return {'verified_readable': True, 'sample_bytes': len(sample), 'error': None}
    except OSError as exc:
        return {'verified_readable': False, 'sample_bytes': 0, 'error': str(exc)}


def parse_assembly_summary(text: str) -> list[dict[str, str]]:
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith('#'):
            if line.startswith('# assembly_accession'):
                header = line[2:].split('\t')
            continue
        if header is None:
            continue
        values = line.split('\t')
        if len(values) < len(header):
            values += [''] * (len(header) - len(values))
        rows.append(dict(zip(header, values)))
    return rows


def score_assembly(row: dict[str, str]) -> int:
    score = 0
    if row.get('version_status') == 'latest':
        score += 1000
    if row.get('genome_rep') == 'Full':
        score += 100
    level = row.get('assembly_level', '')
    score += {'Complete Genome': 80, 'Chromosome': 60, 'Scaffold': 40, 'Contig': 20}.get(level, 0)
    if row.get('release_type') == 'Major':
        score += 10
    if row.get('ftp_path') and row.get('ftp_path') != 'na':
        score += 5
    return score


def select_assembly(rows: list[dict[str, str]], taxid: str = DEFAULT_TAXID, accession: str | None = None) -> AssemblyCandidate:
    if accession:
        matches = [r for r in rows if r.get('assembly_accession') == accession]
    else:
        matches = [r for r in rows if r.get('taxid') == str(taxid) or r.get('species_taxid') == str(taxid)]
    matches = [r for r in matches if r.get('ftp_path') and r.get('ftp_path') != 'na']
    if not matches:
        target = accession or f'taxid={taxid}'
        raise RuntimeError(f'No downloadable NCBI GenBank assembly found for {target}.')
    candidates = [AssemblyCandidate(row=r, score=score_assembly(r)) for r in matches]
    return sorted(candidates, key=lambda c: (c.score, c.row.get('seq_rel_date', ''), c.accession), reverse=True)[0]


def safe_name(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return re.sub(r'_+', '_', text).strip('_') or DEFAULT_DATASET_ID


def fetch_genome(
    out_dir: str | Path = 'data/raw/physcomitrium_patens',
    taxid: str = DEFAULT_TAXID,
    accession: str | None = None,
    summary_url: str = ASSEMBLY_SUMMARY_GENBANK,
    force: bool = False,
    timeout: int = 60,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'assembly_summary_genbank.txt'
    if force or not cache.exists():
        cache.write_text(_download_text(summary_url, timeout=timeout), encoding='utf-8')
    rows = parse_assembly_summary(cache.read_text(encoding='utf-8'))
    selected = select_assembly(rows, taxid=taxid, accession=accession)
    fasta_path = out / f'{selected.accession}_genomic.fna.gz'
    if fasta_path.exists() and not force:
        sha256 = hashlib.sha256(fasta_path.read_bytes()).hexdigest()
    else:
        sha256 = _download_binary(selected.fasta_url, fasta_path, timeout=timeout)
    gzip_verification = verify_gzip_readable(fasta_path)
    if not gzip_verification['verified_readable']:
        raise RuntimeError(f'Downloaded FASTA is not a readable gzip file: {fasta_path}: {gzip_verification["error"]}')
    manifest = {
        'dataset_id': DEFAULT_DATASET_ID,
        'created_at_unix': int(time.time()),
        'summary_url': summary_url,
        'taxid': str(taxid),
        'requested_accession': accession,
        'selected': selected.row,
        'selected_score': selected.score,
        'fasta_url': selected.fasta_url,
        'fasta_path': str(fasta_path),
        'sha256': sha256,
        'gzip_verification': gzip_verification,
        'policy': {
            'git_tracked': False,
            'reason': 'Real genome FASTA is local research data and must stay under data/.'
        },
    }
    (out / 'provenance.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest

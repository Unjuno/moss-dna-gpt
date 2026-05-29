from pathlib import Path

from moss_dna_gpt.fasta import iter_fasta, summarize_fasta


def test_fasta_parser():
    records = list(iter_fasta(Path('tests/fixtures/dummy.fa')))
    assert len(records) == 2
    assert records[0].name == 'contig_1'
    assert records[0].sequence.startswith('ACGT')
    assert records[1].length > 0


def test_fasta_summary_counts():
    summary = summarize_fasta('tests/fixtures/dummy.fa', warn=False)
    assert summary['sequence_count'] == 2
    assert summary['total_bp'] == sum(seq['length'] for seq in summary['sequences'])
    assert set(summary['counts']) == set('ACGTN')

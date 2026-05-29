from moss_dna_gpt.dataset import prepare_windows_from_fasta, windows


def test_windows_max_windows_caps_output():
    seq = 'ACGT' * 20
    out, stats = windows(seq, block_size=8, stride=4, max_windows=3)
    assert len(out) == 3
    assert stats['truncated'] is True


def test_prepare_windows_manifest_records_cap(tmp_path):
    out_dir = tmp_path / 'processed'
    manifest = prepare_windows_from_fasta(
        'tests/fixtures/dummy.fa',
        out_dir,
        block_size=16,
        stride=8,
        max_n_rate=1.0,
        max_windows=4,
    )
    assert manifest['max_windows'] == 4
    assert manifest['total_windows'] == 4
    assert manifest['truncated'] is True
    assert (out_dir / 'train.txt').exists()

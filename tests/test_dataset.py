import pytest

from moss_dna_gpt.dataset import prepare_windows_from_fasta, prepare_window_shards_from_fasta, read_windows, windows


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


def test_read_windows_rejects_directory(tmp_path):
    shard_dir = tmp_path / 'train'
    shard_dir.mkdir()
    (shard_dir / 'shard_00000.txt').write_text('ACGT\n', encoding='utf-8')
    with pytest.raises(IsADirectoryError):
        read_windows(shard_dir)


def test_prepare_window_shards_sequence_split_assigns_each_contig_to_one_split(tmp_path):
    out_dir = tmp_path / 'sharded_seq'
    manifest = prepare_window_shards_from_fasta(
        'tests/fixtures/dummy.fa',
        out_dir,
        block_size=16,
        stride=8,
        max_n_rate=1.0,
        max_windows=20,
        shard_size=5,
        seed=42,
        split_policy='sequence',
    )
    for seq in manifest['sequences']:
        assert 'split' in seq
        assert seq['split'] in ('train', 'val', 'test')
    assert (out_dir / 'train').exists()
    assert (out_dir / 'manifest.json').exists()


def test_prepare_window_shards_sequence_manifest_records_assignments(tmp_path):
    out_dir = tmp_path / 'sharded_seq_man'
    manifest = prepare_window_shards_from_fasta(
        'tests/fixtures/dummy.fa',
        out_dir,
        block_size=16,
        stride=8,
        max_n_rate=1.0,
        max_windows=20,
        shard_size=5,
        seed=42,
        split_policy='sequence',
    )
    assert 'sequence_assignments' in manifest
    assert len(manifest['sequence_assignments']) > 0
    for name, split_name in manifest['sequence_assignments'].items():
        assert split_name in ('train', 'val', 'test')
    assert manifest['split_policy'] == 'sequence'
    assert manifest['train_ratio'] == 0.8
    assert manifest['val_ratio'] == 0.1
    assert manifest['seed'] == 42


def test_prepare_window_shards_window_split_still_works(tmp_path):
    out_dir = tmp_path / 'sharded_win'
    manifest = prepare_window_shards_from_fasta(
        'tests/fixtures/dummy.fa',
        out_dir,
        block_size=16,
        stride=8,
        max_n_rate=1.0,
        max_windows=8,
        shard_size=5,
        seed=42,
        invalid_policy='replace_n',
        split_policy='window',
    )
    assert manifest['format'] == 'sharded_text'
    assert manifest['total_windows'] == 8
    assert manifest['split_policy'] == 'window'
    assert manifest['train_ratio'] == 0.8
    assert manifest['val_ratio'] == 0.1
    assert (out_dir / 'train').exists()

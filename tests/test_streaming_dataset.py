from pathlib import Path

from torch.utils.data import DataLoader

from moss_dna_gpt.dataset import StreamingDnaWindowDataset, prepare_window_shards_from_fasta


def test_prepare_window_shards_and_streaming_dataset(tmp_path):
    out_dir = tmp_path / 'sharded'
    manifest = prepare_window_shards_from_fasta(
        'tests/fixtures/dummy.fa',
        out_dir,
        block_size=16,
        stride=8,
        max_n_rate=1.0,
        max_windows=8,
        shard_size=2,
        seed=1,
        invalid_policy='replace_n',
    )
    assert manifest['format'] == 'sharded_text'
    assert manifest['total_windows'] == 8
    assert manifest['shard_size'] == 2
    assert (out_dir / 'manifest.json').exists()
    assert any((out_dir / 'train').glob('*.txt'))

    ds = StreamingDnaWindowDataset(out_dir / 'train', block_size=16, shuffle_files=False)
    loader = DataLoader(ds, batch_size=2)
    x, y = next(iter(loader))
    assert x.shape[1] == 15
    assert y.shape[1] == 15

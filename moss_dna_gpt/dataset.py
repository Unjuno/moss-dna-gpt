from __future__ import annotations

from pathlib import Path
import json
import random
from collections.abc import Iterator

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .fasta import ALLOWED, iter_fasta
from .tokenizer import DnaTokenizer

SPLITS = ('train', 'val', 'test')


def _normalize_window(w: str, invalid_policy: str) -> tuple[str | None, int]:
    bad = set(w) - ALLOWED
    if not bad:
        return w, 0
    if invalid_policy == 'error':
        raise ValueError(f'invalid bases: {sorted(bad)}')
    if invalid_policy == 'replace_n':
        return ''.join(c if c in ALLOWED else 'N' for c in w), len(bad)
    return None, len(bad)


def iter_sequence_windows(
    seq: str,
    block_size: int = 1024,
    stride: int = 512,
    max_n_rate: float = 1.0,
    invalid_policy: str = 'skip',
) -> Iterator[str]:
    if block_size <= 0 or stride <= 0:
        raise ValueError('block_size and stride must be positive')
    seq = seq.upper()
    if len(seq) < block_size:
        return
    for i in range(0, len(seq) - block_size + 1, stride):
        window = seq[i:i + block_size]
        normalized, _ = _normalize_window(window, invalid_policy)
        if normalized is None:
            continue
        if normalized.count('N') / block_size > max_n_rate:
            continue
        yield normalized


def windows(
    seq: str,
    block_size: int = 1024,
    stride: int = 512,
    max_n_rate: float = 1.0,
    invalid_policy: str = 'skip',
    max_windows: int | None = None,
):
    if max_windows is not None and max_windows < 0:
        raise ValueError('max_windows must be non-negative or None')
    out = []
    stats = {'candidates': 0, 'dropped_n_rate': 0, 'dropped_invalid': 0, 'dropped_short': 0, 'truncated': False}
    seq = seq.upper()
    if len(seq) < block_size:
        stats['dropped_short'] = 1
        return out, stats
    for i in range(0, len(seq) - block_size + 1, stride):
        if max_windows is not None and len(out) >= max_windows:
            stats['truncated'] = True
            break
        stats['candidates'] += 1
        w = seq[i:i + block_size]
        normalized, bad_count = _normalize_window(w, invalid_policy)
        if normalized is None:
            stats['dropped_invalid'] += 1
            continue
        if normalized.count('N') / block_size > max_n_rate:
            stats['dropped_n_rate'] += 1
            continue
        out.append(normalized)
    return out, stats


def split(xs, train_ratio=0.8, val_ratio=0.1, seed=42, shuffle=True):
    xs = list(xs)
    rnd = random.Random(seed)
    if shuffle:
        rnd.shuffle(xs)
    n = len(xs)
    a = int(n * train_ratio)
    b = a + int(n * val_ratio)
    return {'train': xs[:a], 'val': xs[a:b], 'test': xs[b:]}


def prepare_windows_from_fasta(
    fasta,
    out_dir,
    block_size=1024,
    stride=512,
    max_n_rate=1.0,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    shuffle=True,
    invalid_policy='skip',
    max_windows: int | None = None,
):
    if max_windows is not None and max_windows < 0:
        raise ValueError('max_windows must be non-negative or None')
    allw = []
    per = []
    truncated = False
    remaining = max_windows
    for record in iter_fasta(fasta):
        if remaining is not None and remaining <= 0:
            truncated = True
            break
        ws, st = windows(record.sequence, block_size, stride, max_n_rate, invalid_policy, remaining)
        allw += ws
        per.append({'name': record.name, 'length': record.length, 'kept': len(ws), **st})
        if st.get('truncated'):
            truncated = True
        if remaining is not None:
            remaining -= len(ws)
    parts = split(allw, train_ratio, val_ratio, seed, shuffle)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for key, values in parts.items():
        (out_path / f'{key}.txt').write_text('\n'.join(values) + ('\n' if values else ''), encoding='utf-8')
    manifest = {
        'format': 'flat_text',
        'fasta': str(fasta),
        'block_size': block_size,
        'stride': stride,
        'max_n_rate': max_n_rate,
        'max_windows': max_windows,
        'truncated': truncated,
        'total_windows': len(allw),
        'splits': {key: len(value) for key, value in parts.items()},
        'sequences': per,
    }
    (out_path / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


class _ShardWriter:
    def __init__(self, root: Path, split_name: str, shard_size: int):
        self.root = root / split_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.split_name = split_name
        self.shard_size = shard_size
        self.shard_index = 0
        self.count_in_shard = 0
        self.total = 0
        self.files: list[dict] = []
        self.handle = None

    def _open_next(self) -> None:
        if self.handle is not None:
            self.handle.close()
        path = self.root / f'shard_{self.shard_index:05d}.txt'
        self.handle = path.open('w', encoding='utf-8')
        self.files.append({'path': str(path), 'windows': 0})
        self.count_in_shard = 0
        self.shard_index += 1

    def write(self, window: str) -> None:
        if self.handle is None or self.count_in_shard >= self.shard_size:
            self._open_next()
        assert self.handle is not None
        self.handle.write(window + '\n')
        self.count_in_shard += 1
        self.total += 1
        self.files[-1]['windows'] += 1

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def prepare_window_shards_from_fasta(
    fasta,
    out_dir,
    block_size=1024,
    stride=512,
    max_n_rate=1.0,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    invalid_policy='skip',
    max_windows: int | None = None,
    shard_size: int = 100000,
) -> dict:
    if block_size <= 0 or stride <= 0:
        raise ValueError('block_size and stride must be positive')
    if shard_size <= 0:
        raise ValueError('shard_size must be positive')
    if max_windows is not None and max_windows < 0:
        raise ValueError('max_windows must be non-negative or None')
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError('invalid train/val ratios')

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)
    writers = {split_name: _ShardWriter(out_path, split_name, shard_size) for split_name in SPLITS}
    per_sequence = []
    total = 0
    truncated = False

    try:
        for record in iter_fasta(fasta):
            seq_stats = {'name': record.name, 'length': record.length, 'kept': 0, 'candidates': 0, 'dropped_n_rate': 0, 'dropped_invalid': 0, 'dropped_short': 0}
            if record.length < block_size:
                seq_stats['dropped_short'] = 1
                per_sequence.append(seq_stats)
                continue
            sequence = record.sequence.upper()
            for i in range(0, len(sequence) - block_size + 1, stride):
                if max_windows is not None and total >= max_windows:
                    truncated = True
                    break
                seq_stats['candidates'] += 1
                window = sequence[i:i + block_size]
                normalized, _ = _normalize_window(window, invalid_policy)
                if normalized is None:
                    seq_stats['dropped_invalid'] += 1
                    continue
                if normalized.count('N') / block_size > max_n_rate:
                    seq_stats['dropped_n_rate'] += 1
                    continue
                r = rnd.random()
                if r < train_ratio:
                    split_name = 'train'
                elif r < train_ratio + val_ratio:
                    split_name = 'val'
                else:
                    split_name = 'test'
                writers[split_name].write(normalized)
                seq_stats['kept'] += 1
                total += 1
            per_sequence.append(seq_stats)
            if truncated:
                break
    finally:
        for writer in writers.values():
            writer.close()

    manifest = {
        'format': 'sharded_text',
        'fasta': str(fasta),
        'block_size': block_size,
        'stride': stride,
        'max_n_rate': max_n_rate,
        'max_windows': max_windows,
        'truncated': truncated,
        'shard_size': shard_size,
        'seed': seed,
        'split_policy': 'per-window random assignment using train_ratio and val_ratio',
        'total_windows': total,
        'splits': {name: writers[name].total for name in SPLITS},
        'shards': {name: writers[name].files for name in SPLITS},
        'sequences': per_sequence,
    }
    (out_path / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def read_windows(path):
    p = Path(path)
    if p.is_dir():
        values = []
        for shard in sorted(p.glob('*.txt')):
            values.extend(line.strip().upper() for line in shard.read_text(encoding='utf-8').splitlines() if line.strip())
        return values
    return [line.strip().upper() for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]


def resolve_window_files(path: str | Path, split_name: str | None = None) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if split_name is not None and (p / split_name).is_dir():
        p = p / split_name
    if p.is_dir():
        files = sorted(p.glob('*.txt'))
        if not files:
            raise FileNotFoundError(f'No .txt shards found in {p}')
        return files
    raise FileNotFoundError(str(path))


class DnaWindowDataset(Dataset):
    def __init__(self, path, tokenizer=None, block_size=1024):
        self.windows = read_windows(path)
        self.tok = tokenizer or DnaTokenizer()
        self.block_size = block_size

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        ids = self.tok.encode(self.windows[i], unknown='n')[:self.block_size]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


class StreamingDnaWindowDataset(IterableDataset):
    def __init__(self, path, tokenizer=None, block_size=1024, split_name: str | None = None, shuffle_files: bool = True, seed: int = 42):
        self.files = resolve_window_files(path, split_name=split_name)
        self.tok = tokenizer or DnaTokenizer()
        self.block_size = block_size
        self.shuffle_files = shuffle_files
        self.seed = seed

    def _example(self, sequence: str):
        ids = self.tok.encode(sequence, unknown='n')[:self.block_size]
        if len(ids) < 2:
            return None
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y

    def __iter__(self):
        files = list(self.files)
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        if self.shuffle_files:
            rnd = random.Random(self.seed + worker_id)
            rnd.shuffle(files)
        files = files[worker_id::num_workers]
        for file_path in files:
            with file_path.open('r', encoding='utf-8') as fp:
                for line in fp:
                    seq = line.strip().upper()
                    if not seq:
                        continue
                    item = self._example(seq)
                    if item is not None:
                        yield item

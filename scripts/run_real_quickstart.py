#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.dataset import prepare_window_shards_from_fasta, prepare_windows_from_fasta
from moss_dna_gpt.fasta import summarize_fasta
from moss_dna_gpt.genome_fetch import DEFAULT_TAXID, fetch_genome
from moss_dna_gpt.trainer import TrainConfig, train, resolve_device


def profile_model(profile: str, block_size: int) -> dict:
    if profile == 'quick':
        return {
            'vocab_size': 9,
            'block_size': block_size,
            'n_layer': 2,
            'n_head': 2,
            'n_embd': 128,
            'dropout': 0.1,
            'bias': True,
        }
    if profile == '5m':
        return {
            'vocab_size': 9,
            'block_size': block_size,
            'n_layer': 6,
            'n_head': 4,
            'n_embd': 256,
            'dropout': 0.1,
            'bias': True,
        }
    raise ValueError(f'unknown profile: {profile}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='One-command real FASTA quickstart: fetch Physcomitrium patens, prepare windows, and start training.'
    )
    parser.add_argument('--profile', choices=['quick', '5m'], default='quick', help='quick is for first PC run; 5m matches the MVP target model size.')
    parser.add_argument('--taxid', default=DEFAULT_TAXID)
    parser.add_argument('--accession', help='Optional exact NCBI assembly accession.')
    parser.add_argument('--raw-dir', default='data/raw/physcomitrium_patens')
    parser.add_argument('--processed-dir', default=None)
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--fasta-path', help='Use an existing local FASTA/FASTA.gz instead of fetching from NCBI.')
    parser.add_argument('--skip-fetch', action='store_true', help='Do not fetch; requires --fasta-path or an existing provenance.json.')
    parser.add_argument('--force-fetch', action='store_true')
    parser.add_argument('--block-size', type=int, default=None)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--max-n-rate', type=float, default=0.2)
    parser.add_argument('--max-windows', type=int, default=None, help='Cap prepared windows. Default: 20000 for quick, unlimited for 5m. Use 0 for unlimited.')
    parser.add_argument('--shard', action='store_true', help='Force sharded preparation and streaming training. Default: enabled for 5m.')
    parser.add_argument('--flat', action='store_true', help='Force legacy flat text preparation. Not recommended for full 5m runs.')
    parser.add_argument('--shard-size', type=int, default=100000)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()

    device = resolve_device(args.device)
    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    block_size = args.block_size or (256 if args.profile == 'quick' else 1024)
    stride = args.stride or (128 if args.profile == 'quick' else 512)
    max_steps = args.max_steps or (50 if args.profile == 'quick' else 1000)
    batch_size = args.batch_size or (8 if args.profile == 'quick' else 4)
    if args.max_windows is None:
        max_windows = 20000 if args.profile == 'quick' else None
    elif args.max_windows <= 0:
        max_windows = None
    else:
        max_windows = args.max_windows
    use_shards = args.shard or (args.profile == '5m' and not args.flat)
    processed_dir = Path(args.processed_dir or f'data/processed/physcomitrium_patens_{args.profile}_{block_size}')
    run_dir = Path(args.run_dir or f'runs/physcomitrium_patens_{args.profile}_{block_size}')

    provenance = None
    if args.fasta_path:
        fasta_path = Path(args.fasta_path)
    elif args.skip_fetch:
        provenance_path = Path(args.raw_dir) / 'provenance.json'
        if not provenance_path.exists():
            raise SystemExit('skip-fetch requires --fasta-path or an existing data/raw/.../provenance.json')
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        fasta_path = Path(provenance['fasta_path'])
    else:
        provenance = fetch_genome(out_dir=args.raw_dir, taxid=args.taxid, accession=args.accession, force=args.force_fetch)
        fasta_path = Path(provenance['fasta_path'])

    if not fasta_path.exists():
        raise SystemExit(f'FASTA not found: {fasta_path}')

    fasta_summary = summarize_fasta(fasta_path)
    if use_shards:
        manifest = prepare_window_shards_from_fasta(
            fasta_path,
            processed_dir,
            block_size=block_size,
            stride=stride,
            max_n_rate=args.max_n_rate,
            seed=42,
            invalid_policy='replace_n',
            max_windows=max_windows,
            shard_size=args.shard_size,
        )
        train_path = processed_dir / 'train'
        val_path = processed_dir / 'val'
    else:
        manifest = prepare_windows_from_fasta(
            fasta_path,
            processed_dir,
            block_size=block_size,
            stride=stride,
            max_n_rate=args.max_n_rate,
            seed=42,
            shuffle=True,
            invalid_policy='replace_n',
            max_windows=max_windows,
        )
        train_path = processed_dir / 'train.txt'
        val_path = processed_dir / 'val.txt'

    result: dict = {
        'profile': args.profile,
        'device': device,
        'cuda_available': torch.cuda.is_available(),
        'fasta_path': str(fasta_path),
        'fasta_total_bp': fasta_summary['total_bp'],
        'fasta_n_rate': fasta_summary['n_rate'],
        'processed_dir': str(processed_dir),
        'max_windows': max_windows,
        'sharded': use_shards,
        'window_manifest': manifest,
        'provenance': provenance,
    }

    if args.prepare_only:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    train_result = train(TrainConfig(
        train_path=str(train_path),
        val_path=str(val_path),
        run_dir=str(run_dir),
        device=args.device,
        batch_size=batch_size,
        max_steps=max_steps,
        eval_interval=min(100, max_steps),
        eval_batches=10,
        log_interval=10,
        num_threads=args.num_threads,
        num_workers=args.num_workers,
        streaming=use_shards,
        model=profile_model(args.profile, block_size),
    ))
    result['train'] = train_result
    result['next_commands'] = {
        'generate': f'python scripts/generate.py --checkpoint {run_dir / ("ckpt_step_" + str(max_steps) + ".pt")} --prefix ACGT --max-new-tokens 128 --device auto',
        'eval_markov': f'python scripts/eval_markov.py --train-path {train_path} --test-path {processed_dir / "test" if use_shards else processed_dir / "test.txt"} --checkpoint {run_dir / ("ckpt_step_" + str(max_steps) + ".pt")} --device auto',
        'ui': f'streamlit run apps/dna_chat.py -- --checkpoint {run_dir / ("ckpt_step_" + str(max_steps) + ".pt")}',
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

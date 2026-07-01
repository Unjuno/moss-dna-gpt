#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.dataset import DnaWindowDataset, StreamingDnaWindowDataset, resolve_window_files
from moss_dna_gpt.markov import (
    MarkovModel,
    InterpolatedMarkovModel,
    evaluate_markov_orders,
    filter_low_complexity,
    low_complexity_fraction,
    shuffle_sequence,
)
from moss_dna_gpt.trainer import evaluate, load_checkpoint, resolve_device
from moss_dna_gpt.metrics import nats_to_bits
from moss_dna_gpt.tokenizer import DnaTokenizer


def iter_window_strings(path: str | Path):
    for file_path in resolve_window_files(path):
        with file_path.open('r', encoding='utf-8') as fp:
            for line in fp:
                seq = line.strip().upper()
                if seq:
                    yield seq


def evaluate_markov_baselines(
    train_seqs: list[str],
    test_seqs: list[str],
    orders: list[int],
    alpha: float,
    include_imm: bool = False,
    include_shuffled: bool = False,
    lc_threshold: float | None = None,
    bootstrap_n: int = 0,
) -> dict:
    """Run Markov baselines and optional bootstrap CI."""
    result = {}

    if lc_threshold is not None and lc_threshold < 1.0:
        lc_frac = low_complexity_fraction
        train_filtered = filter_low_complexity(train_seqs, lc_threshold)
        test_filtered = filter_low_complexity(test_seqs, lc_threshold)
        orig_test = len(test_seqs)
        result['lc_filter'] = {
            'threshold': lc_threshold,
            'train_seqs_before': len(train_seqs),
            'train_seqs_after': len(train_filtered),
            'test_seqs_before': len(test_seqs),
            'test_seqs_after': len(test_filtered),
            'test_frac_removed': 1.0 - len(test_filtered) / max(orig_test, 1),
        }
        train_seqs = train_filtered
        test_seqs = test_filtered

    markov_results = evaluate_markov_orders(
        train_seqs, test_seqs, orders=orders, alpha=alpha,
        include_imm=include_imm, include_shuffled=include_shuffled,
    )

    if bootstrap_n > 1:
        rng = random.Random(42)
        n_test = len(test_seqs)
        for key, m_result in markov_results.items():
            if key == 'imm':
                imm = InterpolatedMarkovModel(max(orders), alpha).fit(train_seqs)
                estimates = _bootstrap_ce(imm, test_seqs, n_resamples=bootstrap_n, rng=rng)
                markov_results[key]['bootstrap_ci'] = _format_ci(estimates, bootstrap_n)
            elif 'shuffled' in key:
                k = 2 if 'k2' in key else 3
                shuffled_train = [shuffle_sequence(s, k=k) for s in train_seqs]
                m = MarkovModel(max(orders), alpha).fit(shuffled_train)
                estimates = _bootstrap_ce(m, test_seqs, n_resamples=bootstrap_n, rng=rng)
                markov_results[key]['bootstrap_ci'] = _format_ci(estimates, bootstrap_n)
            else:
                order = m_result.get('order', 0)
                if isinstance(order, int):
                    m = MarkovModel(order, alpha).fit(train_seqs)
                    estimates = _bootstrap_ce(m, test_seqs, n_resamples=bootstrap_n, rng=rng)
                    markov_results[key]['bootstrap_ci'] = _format_ci(estimates, bootstrap_n)

    result['markov'] = markov_results
    return result


def _bootstrap_ce(model, test_seqs, n_resamples, rng):
    n = len(test_seqs)
    estimates = []
    for _ in range(n_resamples):
        sample = [test_seqs[rng.randint(0, n - 1)] for _ in range(n)]
        loss, _tokens = model.cross_entropy(sample)
        estimates.append(loss)
    return estimates


def _format_ci(estimates, n_resamples):
    estimates.sort()
    mean = sum(estimates) / len(estimates)
    std = math.sqrt(sum((x - mean) ** 2 for x in estimates) / len(estimates))
    return {
        'n_resamples': n_resamples,
        'mean': mean,
        'std': std,
        'ci_95': [estimates[int(n_resamples * 0.025)], estimates[int(n_resamples * 0.975)]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate Markov baselines and optional DNA-GPT checkpoint.')
    parser.add_argument('--train-path', default='data/processed/train.txt')
    parser.add_argument('--test-path', default='data/processed/test.txt')
    parser.add_argument('--orders', default='0,1,2,3,5,8,10')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--imm', action='store_true', help='Include interpolated Markov model (IMM) baseline')
    parser.add_argument('--shuffled', action='store_true', help='Include dinucleotide/k3-shuffled controls')
    parser.add_argument('--filter-lc', type=float, default=None, metavar='THRESH',
                        help='Filter out low-complexity windows (default: 0.3, set to 0 to skip)')
    parser.add_argument('--checkpoint')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-batches', type=int, default=100)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--streaming', action='store_true', help='Read sharded/directory datasets lazily.')
    parser.add_argument('--bootstrap', type=int, default=0, help='Number of bootstrap resamples for CI (0=skip)')
    args = parser.parse_args()

    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    orders = [int(x) for x in args.orders.split(',') if x.strip()]
    is_streaming = args.streaming or Path(args.train_path).is_dir() or Path(args.test_path).is_dir()

    result: dict = {
        'train_path': str(args.train_path),
        'test_path': str(args.test_path),
        'streaming': is_streaming,
        'settings': {
            'orders': orders,
            'alpha': args.alpha,
            'imm': args.imm,
            'shuffled': args.shuffled,
            'filter_lc': args.filter_lc,
            'bootstrap': args.bootstrap,
        },
    }

    train_seqs = list(iter_window_strings(args.train_path))
    test_seqs = list(iter_window_strings(args.test_path))
    baselines = evaluate_markov_baselines(
        train_seqs, test_seqs, orders=orders, alpha=args.alpha,
        include_imm=args.imm, include_shuffled=args.shuffled,
        lc_threshold=args.filter_lc,
        bootstrap_n=args.bootstrap,
    )
    result.update(baselines)

    if args.checkpoint:
        device = resolve_device(args.device)
        model, ckpt = load_checkpoint(args.checkpoint, map_location=device)
        model.to(device)
        tokenizer = DnaTokenizer()
        if args.streaming or Path(args.test_path).is_dir():
            ds = StreamingDnaWindowDataset(args.test_path, tokenizer=tokenizer, shuffle_files=False)
        else:
            ds = DnaWindowDataset(args.test_path, tokenizer=tokenizer)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        loss = evaluate(model, loader, device, args.eval_batches)
        result['dna_gpt'] = {
            'checkpoint': args.checkpoint,
            'step': ckpt.get('step'),
            'nats_per_base': loss,
            'bits_per_base': nats_to_bits(loss),
            'param_count': model.num_parameters(),
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

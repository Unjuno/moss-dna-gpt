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
from moss_dna_gpt.markov import MarkovModel, InterpolatedMarkovModel, evaluate_markov_orders, shuffle_sequence
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


def evaluate_markov_streaming(train_path: str | Path, test_path: str | Path,
                               orders: list[int], alpha: float,
                               include_imm: bool = False,
                               include_shuffled: bool = False) -> dict:
    train_seqs = list(iter_window_strings(train_path))
    test_seqs = list(iter_window_strings(test_path))
    return evaluate_markov_orders(
        train_seqs, test_seqs, orders=orders, alpha=alpha,
        include_imm=include_imm, include_shuffled=include_shuffled,
    )


def bootstrap_ci(seqs: list[str], model_fn, metric_fn, n_resamples: int = 1000,
                 alpha: float = 0.05, seed: int = 42) -> dict:
    """Bootstrap confidence interval for a model's cross-entropy."""
    rng = random.Random(seed)
    n = len(seqs)
    estimates = []
    for _ in range(n_resamples):
        sample = [seqs[rng.randint(0, n - 1)] for _ in range(n)]
        loss, tokens = metric_fn(sample, model_fn)
        estimates.append(loss)
    estimates.sort()
    ci = {
        'n_resamples': n_resamples,
        'mean': sum(estimates) / len(estimates),
        'std': math.sqrt(sum((x - sum(estimates) / len(estimates)) ** 2 for x in estimates) / len(estimates)),
        f'ci_{int(100*(1-alpha))}': [estimates[int(n_resamples * alpha / 2)], estimates[int(n_resamples * (1 - alpha / 2))]],
    }
    return ci


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate Markov baselines and optional DNA-GPT checkpoint.')
    parser.add_argument('--train-path', default='data/processed/train.txt')
    parser.add_argument('--test-path', default='data/processed/test.txt')
    parser.add_argument('--orders', default='0,1,2,3,5,8,10')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--imm', action='store_true', help='Include interpolated Markov model (IMM) baseline')
    parser.add_argument('--shuffled', action='store_true', help='Include dinucleotide/k3-shuffled controls')
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
            'bootstrap': args.bootstrap,
        },
        'markov': {},
    }

    if not is_streaming:
        result['markov'] = evaluate_markov_streaming(
            args.train_path, args.test_path, orders=orders, alpha=args.alpha,
            include_imm=args.imm, include_shuffled=args.shuffled,
        )

        if args.bootstrap > 1:
            test_seqs = list(iter_window_strings(args.test_path))
            for key, m_result in result['markov'].items():
                order = m_result.get('order', 0)
                if isinstance(order, int):
                    model = MarkovModel(order, args.alpha).fit(iter_window_strings(args.train_path))
                    metric_fn = lambda seqs, m: m.cross_entropy(seqs)
                    ci = bootstrap_ci(test_seqs, model, lambda seqs, m: m.cross_entropy(seqs),
                                      n_resamples=args.bootstrap, seed=42)
                    result['markov'][key]['bootstrap_ci'] = ci
    else:
        train_seqs = list(iter_window_strings(args.train_path))
        test_seqs = list(iter_window_strings(args.test_path))
        result['markov'] = evaluate_markov_orders(
            train_seqs, test_seqs, orders=orders, alpha=args.alpha,
            include_imm=args.imm, include_shuffled=args.shuffled,
        )

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

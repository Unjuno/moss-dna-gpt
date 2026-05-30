#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.dataset import DnaWindowDataset, StreamingDnaWindowDataset, resolve_window_files
from moss_dna_gpt.markov import MarkovModel
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


def evaluate_markov_streaming(train_path: str | Path, test_path: str | Path, orders: list[int], alpha: float) -> dict:
    result = {}
    for order in orders:
        model = MarkovModel(order, alpha=alpha).fit(iter_window_strings(train_path))
        nats, tokens = model.cross_entropy(iter_window_strings(test_path))
        result[order] = {
            'order': order,
            'nats_per_base': nats,
            'bits_per_base': nats_to_bits(nats),
            'tokens': tokens,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate 0/1/5-order Markov baselines and optional DNA-GPT checkpoint.')
    parser.add_argument('--train-path', default='data/processed/train.txt')
    parser.add_argument('--test-path', default='data/processed/test.txt')
    parser.add_argument('--orders', default='0,1,5')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--checkpoint')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-batches', type=int, default=100)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--streaming', action='store_true', help='Read sharded/directory datasets lazily.')
    args = parser.parse_args()

    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    orders = [int(x) for x in args.orders.split(',') if x.strip()]
    result: dict = {
        'train_path': args.train_path,
        'test_path': args.test_path,
        'streaming': args.streaming or Path(args.train_path).is_dir() or Path(args.test_path).is_dir(),
        'markov': evaluate_markov_streaming(args.train_path, args.test_path, orders=orders, alpha=args.alpha),
    }

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

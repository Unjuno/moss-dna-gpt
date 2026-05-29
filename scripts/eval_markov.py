#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.dataset import DnaWindowDataset, read_windows
from moss_dna_gpt.markov import evaluate_markov_orders
from moss_dna_gpt.trainer import evaluate, load_checkpoint, resolve_device
from moss_dna_gpt.metrics import nats_to_bits
from moss_dna_gpt.tokenizer import DnaTokenizer


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
    args = parser.parse_args()

    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    orders = [int(x) for x in args.orders.split(',') if x.strip()]
    train_windows = read_windows(args.train_path)
    test_windows = read_windows(args.test_path)
    result: dict = {
        'train_path': args.train_path,
        'test_path': args.test_path,
        'markov': evaluate_markov_orders(train_windows, test_windows, orders=orders, alpha=args.alpha),
    }

    if args.checkpoint:
        device = resolve_device(args.device)
        model, ckpt = load_checkpoint(args.checkpoint, map_location=device)
        model.to(device)
        ds = DnaWindowDataset(args.test_path, tokenizer=DnaTokenizer())
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

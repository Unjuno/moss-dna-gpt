#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.model import GPTConfig
from moss_dna_gpt.trainer import load_train_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a small DNA-GPT on prepared windows.')
    parser.add_argument('--config', default='configs/train_5m_1024.yaml')
    parser.add_argument('--train-path')
    parser.add_argument('--val-path')
    parser.add_argument('--run-dir')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--device')
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--num-threads', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--streaming', action='store_true', help='Read window files lazily. Use this for sharded datasets.')
    parser.add_argument('--resume-from', help='Path to checkpoint .pt file to resume training from')
    parser.add_argument('--n-layer', type=int)
    parser.add_argument('--n-head', type=int)
    parser.add_argument('--n-embd', type=int)
    parser.add_argument('--block-size', type=int)
    parser.add_argument('--dropout', type=float)
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    for name in ['train_path', 'val_path', 'run_dir', 'max_steps', 'batch_size', 'device', 'learning_rate', 'num_threads', 'num_workers']:
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.streaming:
        cfg.streaming = True
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    model = dict(cfg.model or GPTConfig().to_dict())
    for key in ['n_layer', 'n_head', 'n_embd', 'block_size', 'dropout']:
        value = getattr(args, key)
        if value is not None:
            model[key] = value
    cfg.model = model

    print(json.dumps(train(cfg), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

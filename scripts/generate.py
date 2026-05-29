#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.sampling import sample
from moss_dna_gpt.tokenizer import DnaTokenizer
from moss_dna_gpt.trainer import load_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate DNA continuation from a trained DNA-GPT checkpoint.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prefix', default='ACGT')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-threads', type=int, default=1)
    args = parser.parse_args()

    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    tokenizer = DnaTokenizer()
    prefix_ids = tokenizer.encode(args.prefix, unknown='n')
    idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    allowed = [tokenizer.stoi[b] for b in tokenizer.dna_tokens]
    out = sample(model, idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, allowed_token_ids=allowed)
    print(tokenizer.decode(out[0].tolist(), skip_special=True))


if __name__ == '__main__':
    main()

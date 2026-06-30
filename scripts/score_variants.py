#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moss_dna_gpt.scoring import score_variant
from moss_dna_gpt.tokenizer import DnaTokenizer
from moss_dna_gpt.trainer import load_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score variant effects using a DNA-GPT checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=1)

    sub = parser.add_mutually_exclusive_group(required=True)
    sub.add_argument("--seq", help="Reference sequence (single variant mode)")
    sub.add_argument("--variants", help="CSV file with cols: seq, pos, ref, alt")
    sub.add_argument("--json", type=Path, help="JSON file with list of variant dicts")

    parser.add_argument("--pos", type=int, default=None)
    parser.add_argument("--ref", default=None)
    parser.add_argument("--alt", default=None)

    parser.add_argument("--output", type=Path, help="Write JSON output to file")

    args = parser.parse_args()

    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    tokenizer = DnaTokenizer()

    results: list[dict] = []

    if args.seq:
        if args.pos is None or args.alt is None:
            parser.error("--seq mode requires --pos and --alt")
        result = score_variant(model, tokenizer, args.seq, args.pos, args.alt, device=device)
        results.append(result)

    elif args.variants:
        with open(args.variants) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                result = score_variant(
                    model, tokenizer,
                    row["seq"], int(row["pos"]), row["alt"],
                    device=device,
                )
                result.update(row)
                result.pop("seq", None)
                results.append(result)

    elif args.json:
        with open(args.json) as fp:
            variants = json.load(fp)
        for v in variants:
            result = score_variant(
                model, tokenizer,
                v["seq"], int(v["pos"]), v["alt"],
                device=device,
            )
            result.update(v)
            result.pop("seq", None)
            results.append(result)

    output = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert PyTorch checkpoint to safetensors format.')
    parser.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    parser.add_argument('--out-dir', default=None, help='Output directory (default: same dir as checkpoint)')
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading checkpoint: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    model_state = checkpoint['model']
    model_config = checkpoint['model_config']
    step = checkpoint.get('step', 'unknown')

    safetensors_path = out_dir / 'model.safetensors'
    save_file(model_state, str(safetensors_path))
    total_params = sum(t.numel() for t in model_state.values())
    print(f'Saved safetensors: {safetensors_path} ({len(model_state)} tensors, {total_params:,} params)')

    config_path = out_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f'Saved config: {config_path}')

    metadata_path = out_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump({
            'step': step,
            'model_config': model_config,
            'param_count': total_params,
        }, f, indent=2)
    print(f'Saved metadata: {metadata_path}')

    print(f'\nFiles ready for HuggingFace upload:')
    for p in [safetensors_path, config_path, metadata_path]:
        print(f'  {p}')


if __name__ == '__main__':
    main()

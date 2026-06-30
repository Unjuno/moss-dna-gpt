#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


def main() -> None:
    parser = argparse.ArgumentParser(description='Publish model artifacts to HuggingFace Hub.')
    parser.add_argument('--repo-id', required=True, help='HF repo ID (e.g., your-org/moss-dna-gpt-20m)')
    parser.add_argument('--local-dir', default='repo/moss-dna-gpt-20m-patens', help='Local directory with model artifacts')
    parser.add_argument('--token', default=None, help='HF API token (default: HF_TOKEN env var)')
    parser.add_argument('--private', action='store_true', help='Create private repository')
    parser.add_argument('--dry-run', action='store_true', help='Print what would happen without uploading')
    args = parser.parse_args()

    token = args.token or os.environ.get('HF_TOKEN')
    if not token and not args.dry_run:
        parser.error('HF_TOKEN required (set env var or pass --token)')

    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        parser.error(f'Local directory not found: {local_dir}')

    if args.dry_run:
        print(f'[DRY RUN] Would publish {local_dir} to HuggingFace Hub as {args.repo_id}')
        for p in sorted(local_dir.iterdir()):
            size = p.stat().st_size
            print(f'  {p.name} ({size:,} bytes)')
        print(f'\nFiles to upload: {len(list(local_dir.iterdir()))}')
        return

    api = HfApi(token=token)

    print(f'Creating/verifying repo: {args.repo_id}')
    create_repo(
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        repo_type='model',
        exist_ok=True,
    )

    # Convert safetensors if .pt exists but no .safetensors
    pt_files = list(local_dir.glob('*.pt'))
    st_files = list(local_dir.glob('*.safetensors'))
    if pt_files and not st_files:
        print('Converting checkpoint to safetensors...')
        import torch
        from safetensors.torch import save_file
        for pt in pt_files:
            ckpt = torch.load(pt, map_location='cpu', weights_only=True)
            st_path = local_dir / 'model.safetensors'
            save_file(ckpt['model'], str(st_path))
            print(f'  Created {st_path}')

    print(f'Uploading {local_dir} to {args.repo_id}...')
    upload_folder(
        repo_id=args.repo_id,
        folder_path=str(local_dir),
        token=token,
        repo_type='model',
        ignore_patterns=['*.pt', '*.pth', '__pycache__', '.gitignore'],
    )

    url = f'https://huggingface.co/{args.repo_id}'
    print(f'\nPublished successfully!')
    print(f'  Repo: {url}')
    print(f'  Files: {len(list(local_dir.iterdir()))} uploaded')


if __name__ == '__main__':
    main()

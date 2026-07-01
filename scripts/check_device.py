#!/usr/bin/env python
from __future__ import annotations

import json
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moss_dna_gpt.trainer import resolve_device


def main() -> None:
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'selected_device_for_auto': resolve_device('auto'),
        'cuda_version': torch.version.cuda,
        'device_count': torch.cuda.device_count(),
        'devices': [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            info['devices'].append({
                'index': i,
                'name': prop.name,
                'total_memory_bytes': prop.total_memory,
                'compute_capability': f'{prop.major}.{prop.minor}',
            })
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

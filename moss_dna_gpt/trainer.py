from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import random

import torch
from torch.utils.data import DataLoader
import yaml

from .dataset import DnaWindowDataset
from .model import GPT, GPTConfig
from .tokenizer import DnaTokenizer


@dataclass
class TrainConfig:
    train_path: str = 'data/processed/train.txt'
    val_path: str = 'data/processed/val.txt'
    run_dir: str = 'runs/5m_1024'
    seed: int = 42
    device: str = 'auto'
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 1000
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    num_workers: int = 0
    num_threads: int = 1
    grad_clip: float = 1.0
    model: dict | None = None


def load_train_config(path: str | Path) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    model_cfg = data.pop('model', None)
    return TrainConfig(**data, model=model_cfg)


def resolve_device(device: str) -> str:
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: GPT, loader: DataLoader, device: str, batches: int) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= batches:
                break
            _, loss = model(x.to(device), y.to(device))
            values.append(float(loss))
    model.train()
    return sum(values) / max(len(values), 1)


def train(cfg: TrainConfig) -> dict:
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    tokenizer = DnaTokenizer()
    model_cfg = GPTConfig(**(cfg.model or {}))
    model_cfg.vocab_size = tokenizer.vocab_size

    train_ds = DnaWindowDataset(cfg.train_path, tokenizer, model_cfg.block_size)
    val_ds = DnaWindowDataset(cfg.val_path, tokenizer, model_cfg.block_size)
    if len(train_ds) == 0:
        raise ValueError('empty training dataset')

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps({**cfg.__dict__, 'model': model_cfg.to_dict()}, indent=2), encoding='utf-8')

    rows: list[dict] = []
    data_iter = iter(train_loader)
    last_loss = None
    for step in range(1, cfg.max_steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        _, loss = model(x.to(device), y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        last_loss = float(loss.detach())

        if step % cfg.log_interval == 0 or step == 1:
            rows.append({'step': step, 'train_loss': last_loss, 'val_loss': ''})
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            val_loss = evaluate(model, val_loader, device, cfg.eval_batches) if len(val_ds) else float('nan')
            rows.append({'step': step, 'train_loss': last_loss, 'val_loss': val_loss})
            torch.save({'model': model.state_dict(), 'model_config': model_cfg.to_dict(), 'step': step}, run_dir / f'ckpt_step_{step}.pt')

    with (run_dir / 'loss.csv').open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=['step', 'train_loss', 'val_loss'])
        writer.writeheader()
        writer.writerows(rows)

    return {'run_dir': str(run_dir), 'final_step': cfg.max_steps, 'param_count': model.num_parameters(), 'last_train_loss': last_loss}


def load_checkpoint(path: str | Path, map_location: str = 'cpu') -> tuple[GPT, dict]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    model = GPT(GPTConfig(**checkpoint['model_config']))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint

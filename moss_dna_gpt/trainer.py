from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import random

import torch
from torch.utils.data import DataLoader
import yaml

from .dataset import DnaWindowDataset, StreamingDnaWindowDataset
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
    streaming: bool = False
    model: dict | None = None
    resume_from: str | None = None


def load_train_config(path: str | Path) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    model_cfg = data.pop('model', None)
    return TrainConfig(**data, model=model_cfg)


def resolve_device(device: str) -> str:
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def evaluate(model: GPT, loader: DataLoader, device: str, batches: int | None = None) -> float:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if batches is not None and i >= batches:
                break
            _, loss = model(x.to(device), y.to(device))
            values.append(float(loss))
    model.train()
    return sum(values) / max(len(values), 1)


def _has_data(path: str | Path, streaming: bool) -> bool:
    p = Path(path)
    if streaming or p.is_dir():
        if p.is_file():
            return p.stat().st_size > 0
        return any(child.is_file() and child.suffix == '.txt' and child.stat().st_size > 0 for child in p.glob('**/*.txt'))
    try:
        return len(DnaWindowDataset(p)) > 0
    except FileNotFoundError:
        return False


def _make_dataset(path: str | Path, tokenizer: DnaTokenizer, block_size: int, streaming: bool, split_name: str | None = None):
    p = Path(path)
    if streaming or p.is_dir():
        return StreamingDnaWindowDataset(p, tokenizer=tokenizer, block_size=block_size, split_name=split_name)
    return DnaWindowDataset(p, tokenizer=tokenizer, block_size=block_size)


def train(cfg: TrainConfig) -> dict:
    if cfg.num_threads:
        torch.set_num_threads(cfg.num_threads)
    device = resolve_device(cfg.device)

    tokenizer = DnaTokenizer()

    start_step = 0
    rows: list[dict] = []
    if cfg.resume_from is not None:
        resume_path = Path(cfg.resume_from)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        model_cfg = GPTConfig(**checkpoint['model_config'])
        model_cfg.vocab_size = tokenizer.vocab_size
        model = GPT(model_cfg).to(device)
        model.load_state_dict(checkpoint['model'])
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'rng' in checkpoint:
            rng = checkpoint['rng']
            random.setstate(rng['random'])
            torch.set_rng_state(rng['torch'].cpu())
            if rng['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])
        start_step = checkpoint['step']
        loss_csv = resume_path.parent / 'loss.csv'
        if loss_csv.exists():
            with loss_csv.open('r', encoding='utf-8') as fp:
                for row in csv.DictReader(fp):
                    rows.append(row)
    else:
        set_seed(cfg.seed)
        model_cfg = GPTConfig(**(cfg.model or {}))
        model_cfg.vocab_size = tokenizer.vocab_size
        model = GPT(model_cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    train_streaming = cfg.streaming or Path(cfg.train_path).is_dir()
    val_streaming = cfg.streaming or Path(cfg.val_path).is_dir()
    if not _has_data(cfg.train_path, train_streaming):
        raise ValueError('empty training dataset')

    train_ds = _make_dataset(cfg.train_path, tokenizer, model_cfg.block_size, train_streaming)
    val_ds = _make_dataset(cfg.val_path, tokenizer, model_cfg.block_size, val_streaming) if _has_data(cfg.val_path, val_streaming) else None

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=(not train_streaming), num_workers=cfg.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers) if val_ds is not None else None

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps({**cfg.__dict__, 'model': model_cfg.to_dict(), 'resolved_streaming': train_streaming}, indent=2), encoding='utf-8')

    data_iter = iter(train_loader)
    last_loss = None
    for step in range(start_step + 1, cfg.max_steps + 1):
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
            val_loss = evaluate(model, val_loader, device, cfg.eval_batches) if val_loader is not None else float('nan')
            rows.append({'step': step, 'train_loss': last_loss, 'val_loss': val_loss})
            rng = {
                'torch': torch.get_rng_state(),
                'cuda': [s.cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
                'random': random.getstate(),
            }
            torch.save({
                'model': model.state_dict(),
                'model_config': model_cfg.to_dict(),
                'optimizer': optimizer.state_dict(),
                'rng': rng,
                'step': step,
            }, run_dir / f'ckpt_step_{step}.pt')

    with (run_dir / 'loss.csv').open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=['step', 'train_loss', 'val_loss'])
        writer.writeheader()
        writer.writerows(rows)

    return {'run_dir': str(run_dir), 'final_step': cfg.max_steps, 'param_count': model.num_parameters(), 'last_train_loss': last_loss, 'streaming': train_streaming}


def load_checkpoint(path: str | Path, map_location: str = 'cpu') -> tuple[GPT, dict]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    model = GPT(GPTConfig(**checkpoint['model_config']))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint

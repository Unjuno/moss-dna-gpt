from __future__ import annotations

import torch

from moss_dna_gpt.trainer import TrainConfig, train


def _make_data(tmp_path, block_size=16):
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    seq = 'A' * (block_size + 10)
    lines = [seq] * 20
    (data_dir / 'train.txt').write_text('\n'.join(lines), encoding='utf-8')
    (data_dir / 'val.txt').write_text('\n'.join(lines[:5]), encoding='utf-8')
    return data_dir


def _model_dict():
    return {'n_layer': 1, 'n_head': 1, 'n_embd': 16, 'block_size': 16, 'dropout': 0.0, 'vocab_size': 9}


def test_train_checkpoint_contains_optimizer(tmp_path):
    data_dir = _make_data(tmp_path)
    run_dir = tmp_path / 'run1'
    cfg = TrainConfig(
        train_path=str(data_dir / 'train.txt'),
        val_path=str(data_dir / 'val.txt'),
        run_dir=str(run_dir),
        batch_size=2, max_steps=5, eval_interval=5, log_interval=5, num_workers=0,
        model=_model_dict(),
    )
    train(cfg)
    ckpt = torch.load(run_dir / 'ckpt_step_5.pt', weights_only=True)
    assert 'optimizer' in ckpt
    assert 'rng' in ckpt
    assert 'model' in ckpt
    assert 'model_config' in ckpt
    assert ckpt['step'] == 5


def test_resume_from_checkpoint_continues_training(tmp_path):
    data_dir = _make_data(tmp_path)
    run_dir = tmp_path / 'run2'

    cfg1 = TrainConfig(
        train_path=str(data_dir / 'train.txt'),
        val_path=str(data_dir / 'val.txt'),
        run_dir=str(run_dir),
        batch_size=2, max_steps=5, eval_interval=5, log_interval=5, num_workers=0,
        model=_model_dict(),
    )
    r1 = train(cfg1)
    assert r1['final_step'] == 5

    ckpt_path = run_dir / 'ckpt_step_5.pt'
    assert ckpt_path.exists()

    cfg2 = TrainConfig(
        train_path=str(data_dir / 'train.txt'),
        val_path=str(data_dir / 'val.txt'),
        run_dir=str(run_dir),
        batch_size=2, max_steps=10, eval_interval=5, log_interval=5, num_workers=0,
        model=_model_dict(),
        resume_from=str(ckpt_path),
    )
    r2 = train(cfg2)
    assert r2['final_step'] == 10

    ckpt10 = torch.load(run_dir / 'ckpt_step_10.pt', weights_only=True)
    assert ckpt10['step'] == 10
    assert ckpt10['optimizer'] is not None
    assert ckpt10['rng'] is not None


def test_resume_loss_csv_accumulates(tmp_path):
    data_dir = _make_data(tmp_path)
    run_dir = tmp_path / 'run3'

    cfg1 = TrainConfig(
        train_path=str(data_dir / 'train.txt'),
        val_path=str(data_dir / 'val.txt'),
        run_dir=str(run_dir),
        batch_size=2, max_steps=5, eval_interval=5, log_interval=5, num_workers=0,
        model=_model_dict(),
    )
    train(cfg1)
    loss_csv = run_dir / 'loss.csv'
    assert loss_csv.exists()

    lines_before = len(loss_csv.read_text(encoding='utf-8').strip().splitlines())

    cfg2 = TrainConfig(
        train_path=str(data_dir / 'train.txt'),
        val_path=str(data_dir / 'val.txt'),
        run_dir=str(run_dir),
        batch_size=2, max_steps=10, eval_interval=5, log_interval=5, num_workers=0,
        model=_model_dict(),
        resume_from=str(run_dir / 'ckpt_step_5.pt'),
    )
    train(cfg2)
    lines_after = len(loss_csv.read_text(encoding='utf-8').strip().splitlines())
    assert lines_after > lines_before


def test_resume_backward_compat_no_optimizer(tmp_path):
    data_dir = _make_data(tmp_path)
    run_dir = tmp_path / 'run4'

    cfg1 = TrainConfig(
        train_path=str(data_dir / 'train.txt'),
        val_path=str(data_dir / 'val.txt'),
        run_dir=str(run_dir),
        batch_size=2, max_steps=5, eval_interval=5, log_interval=5, num_workers=0,
        model=_model_dict(),
    )
    train(cfg1)

    ckpt_path = run_dir / 'ckpt_step_5.pt'
    ckpt = torch.load(ckpt_path, weights_only=True)
    del ckpt['optimizer']
    del ckpt['rng']
    torch.save(ckpt, ckpt_path)

    cfg2 = TrainConfig(
        train_path=str(data_dir / 'train.txt'),
        val_path=str(data_dir / 'val.txt'),
        run_dir=str(run_dir),
        batch_size=2, max_steps=10, eval_interval=5, log_interval=5, num_workers=0,
        model=_model_dict(),
        resume_from=str(ckpt_path),
    )
    r2 = train(cfg2)
    assert r2['final_step'] == 10
    ckpt10 = torch.load(run_dir / 'ckpt_step_10.pt', weights_only=True)
    assert ckpt10['step'] == 10
    assert 'optimizer' in ckpt10

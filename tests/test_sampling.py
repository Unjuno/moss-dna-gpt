import torch

from moss_dna_gpt.model import GPT, GPTConfig
from moss_dna_gpt.sampling import sample
from moss_dna_gpt.tokenizer import DnaTokenizer


def test_sample_basic():
    cfg = GPTConfig(vocab_size=9, block_size=16, n_layer=1, n_head=1, n_embd=16, dropout=0.0)
    model = GPT(cfg)
    model.eval()
    tokenizer = DnaTokenizer()
    prefix_ids = tokenizer.encode('ACGT', unknown='n')
    idx = torch.tensor([prefix_ids], dtype=torch.long)
    allowed = [tokenizer.stoi[b] for b in tokenizer.dna_tokens]
    out = sample(model, idx, max_new_tokens=8, temperature=1.0, top_k=None, allowed_token_ids=allowed)
    assert out.shape == (1, 12)
    decoded = tokenizer.decode(out[0].tolist(), skip_special=True)
    assert all(c in 'ACGTN' for c in decoded)


def test_sample_with_temperature():
    cfg = GPTConfig(vocab_size=9, block_size=16, n_layer=1, n_head=1, n_embd=16, dropout=0.0)
    model = GPT(cfg)
    model.eval()
    tokenizer = DnaTokenizer()
    prefix_ids = tokenizer.encode('ACGT', unknown='n')
    idx = torch.tensor([prefix_ids], dtype=torch.long)
    allowed = [tokenizer.stoi[b] for b in tokenizer.dna_tokens]
    out_hot = sample(model, idx, max_new_tokens=4, temperature=0.1, top_k=1, allowed_token_ids=allowed)
    out_cold = sample(model, idx, max_new_tokens=4, temperature=2.0, top_k=None, allowed_token_ids=allowed)
    assert out_hot.shape == (1, 8)
    assert out_cold.shape == (1, 8)


def test_sample_allowed_tokens():
    cfg = GPTConfig(vocab_size=9, block_size=16, n_layer=1, n_head=1, n_embd=16, dropout=0.0)
    model = GPT(cfg)
    model.eval()
    tokenizer = DnaTokenizer()
    prefix_ids = tokenizer.encode('ACGT', unknown='n')
    idx = torch.tensor([prefix_ids], dtype=torch.long)
    out = sample(model, idx, max_new_tokens=4, temperature=0.5, top_k=3, allowed_token_ids=None)
    assert out.shape == (1, 8)

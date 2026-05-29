import torch

from moss_dna_gpt.model import GPT, GPTConfig


def test_model_forward_shape():
    cfg = GPTConfig(vocab_size=9, block_size=16, n_layer=1, n_head=1, n_embd=16, dropout=0.0)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None
    assert model.num_parameters() > 0

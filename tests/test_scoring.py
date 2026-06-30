import torch

from moss_dna_gpt.model import GPT, GPTConfig
from moss_dna_gpt.tokenizer import DnaTokenizer
from moss_dna_gpt.scoring import score_variant


def _make_model():
    cfg = GPTConfig(vocab_size=9, block_size=32, n_layer=1, n_head=1, n_embd=16, dropout=0.0)
    return GPT(cfg).eval()


def test_score_variant_returns_expected_keys():
    tokenizer = DnaTokenizer()
    model = _make_model()
    result = score_variant(model, tokenizer, "ACGTACGT", pos=2, alt="T", device="cpu")
    expected_keys = {
        "llr", "llr_bits", "loss_ref", "loss_ref_bits",
        "loss_mut", "loss_mut_bits", "delta_loss", "delta_loss_bits",
        "ref_base", "alt_base", "position", "seq_length",
    }
    assert set(result.keys()) == expected_keys


def test_score_variant_identity_mutation():
    tokenizer = DnaTokenizer()
    model = _make_model()
    result = score_variant(model, tokenizer, "ACGTACGT", pos=2, alt="G", device="cpu")
    assert result["llr"] == 0.0
    assert result["delta_loss"] == 0.0


def test_score_variant_pos_zero_llr_is_none():
    tokenizer = DnaTokenizer()
    model = _make_model()
    result = score_variant(model, tokenizer, "ACGTACGT", pos=0, alt="T", device="cpu")
    assert result["position"] == 0
    assert result["llr"] is None
    assert result["llr_bits"] is None


def test_score_variant_pos_out_of_range_error():
    tokenizer = DnaTokenizer()
    model = _make_model()
    try:
        score_variant(model, tokenizer, "ACGT", pos=10, alt="T", device="cpu")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_score_variant_alt_not_valid_error():
    tokenizer = DnaTokenizer()
    model = _make_model()
    try:
        score_variant(model, tokenizer, "ACGT", pos=1, alt="X", device="cpu")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_score_variant_loss_ref_is_finite():
    tokenizer = DnaTokenizer()
    model = _make_model()
    result = score_variant(model, tokenizer, "ACGTACGT" * 4, pos=5, alt="T", device="cpu")
    assert torch.isfinite(torch.tensor(result["loss_ref"]))
    assert torch.isfinite(torch.tensor(result["loss_mut"]))
    assert torch.isfinite(torch.tensor(result["delta_loss"]))


def test_score_variant_non_identity_has_nonzero_llr():
    tokenizer = DnaTokenizer()
    model = _make_model()
    result = score_variant(model, tokenizer, "ACGTACGT", pos=2, alt="A", device="cpu")
    assert result["llr"] != 0.0


def test_score_variant_delta_loss_matches_llr_at_last_position():
    tokenizer = DnaTokenizer()
    model = _make_model()
    seq = "ACGTACGT"
    last_pos = len(seq) - 1
    result = score_variant(model, tokenizer, seq, pos=last_pos, alt="T", device="cpu")
    assert result["llr"] is not None
    assert abs(result["delta_loss"] - result["llr"]) < 1e-6

from moss_dna_gpt.tokenizer import DnaTokenizer


def test_tokenizer_roundtrip():
    tok = DnaTokenizer()
    seq = 'ACGTNACGTN'
    ids = tok.encode(seq)
    assert tok.decode(ids) == seq


def test_tokenizer_special_tokens():
    tok = DnaTokenizer()
    ids = tok.encode('AC', add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == 'AC'

from moss_dna_gpt.markov import MarkovModel, evaluate_markov_orders


def test_markov_baseline_runs():
    train = ['ACGTACGTACGT', 'AAAACCCCGGGGTTTT']
    test = ['ACGTACGT', 'CCCCGGGG']
    results = evaluate_markov_orders(train, test, orders=(0, 1, 5), alpha=0.5)
    assert set(results) == {0, 1, 5}
    for item in results.values():
        assert item['nats_per_base'] > 0
        assert item['bits_per_base'] > 0
        assert item['tokens'] > 0


def test_markov_probability_is_smoothed():
    model = MarkovModel(order=1, alpha=0.5).fit(['AAAA'])
    p = model.prob('C', 'G')
    assert 0 < p < 1

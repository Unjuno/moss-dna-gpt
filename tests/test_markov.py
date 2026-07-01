from moss_dna_gpt.markov import MarkovModel, InterpolatedMarkovModel, evaluate_markov_orders, shuffle_sequence


def test_markov_baseline_runs():
    train = ['ACGTACGTACGT', 'AAAACCCCGGGGTTTT']
    test = ['ACGTACGT', 'CCCCGGGG']
    results = evaluate_markov_orders(train, test, orders=(0, 1, 5), alpha=0.5)
    assert set(results) == {'order_0', 'order_1', 'order_5'}
    for item in results.values():
        assert item['nats_per_base'] > 0
        assert item['bits_per_base'] > 0
        assert item['tokens'] > 0


def test_markov_probability_is_smoothed():
    model = MarkovModel(order=1, alpha=0.5).fit(['AAAA'])
    p = model.prob('C', 'G')
    assert 0 < p < 1


def test_interpolated_markov_improves_over_fixed_order():
    train = ['ACGTACGTACGTACGTACGT'] * 10
    test = ['ACGTACGTACGT']
    fixed = evaluate_markov_orders(train, test, orders=(0,), alpha=0.5)
    imm_results = evaluate_markov_orders(train, test, orders=(0, 1, 2, 3), alpha=0.5, include_imm=True)
    assert 'imm' in imm_results
    # IMM should be at least as good as order-0 alone
    assert imm_results['imm']['bits_per_base'] <= fixed['order_0']['bits_per_base'] + 1e-9


def test_shuffled_control():
    train = ['ACGTACGTACGTACGT', 'ACGTACGTACGTACGT']
    test = ['ACGTACGT']
    results = evaluate_markov_orders(train, test, orders=(0, 1, 5), alpha=0.5, include_shuffled=True)
    assert 'shuffled_k2' in results
    assert 'shuffled_k3' in results


def test_shuffle_sequence_preserves_length():
    seq = 'ACGTACGTACGT'
    shuffled = shuffle_sequence(seq, k=2)
    assert len(shuffled) == len(seq)
    assert sorted(shuffled) == sorted(seq)


def test_interpolated_markov_imm_runs():
    train = ['ACGTACGTACGT', 'AAAACCCCGGGGTTTT']
    test = ['ACGTACGT']
    imm = InterpolatedMarkovModel(max_order=3, alpha=0.5)
    imm.fit(train)
    loss, n = imm.cross_entropy(test)
    assert loss > 0
    assert n > 0


def test_low_complexity_fraction_high_entropy():
    from moss_dna_gpt.markov import low_complexity_fraction
    seq = 'ACGTACGTACGTACGTACGTACGTACGTACGT'
    lcf = low_complexity_fraction(seq, window=8, stride=8)
    assert 0 <= lcf <= 0.1


def test_low_complexity_fraction_low_entropy():
    from moss_dna_gpt.markov import low_complexity_fraction
    seq = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    lcf = low_complexity_fraction(seq, window=8, stride=8)
    assert lcf > 0.9


def test_filter_low_complexity():
    from moss_dna_gpt.markov import filter_low_complexity
    seqs = ['ACGTACGTACGTACGTACGTACGTACGTACGT', 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA']
    filtered = filter_low_complexity(seqs, threshold=0.5)
    assert len(filtered) == 1
    assert filtered[0] == seqs[0]

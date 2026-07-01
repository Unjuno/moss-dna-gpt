from __future__ import annotations

from collections import Counter, defaultdict
import math
import random


DNA = 'ACGTN'


class MarkovModel:
    def __init__(self, order: int, alpha: float = 0.5):
        self.order = order
        self.alpha = alpha
        self.counts = defaultdict(Counter)
        self.totals = Counter()

    def fit(self, seqs):
        for s in seqs:
            s = ''.join(c if c in DNA else 'N' for c in s.upper())
            for i, ch in enumerate(s):
                ctx = s[max(0, i - self.order):i] if self.order else ''
                self.counts[ctx][ch] += 1
                self.totals[ctx] += 1
        return self

    def prob(self, ch, ctx):
        ctx = ctx[-self.order:] if self.order else ''
        c = self.counts.get(ctx, Counter())
        t = self.totals.get(ctx, 0)
        V = len(DNA)
        return (c.get(ch, 0) + self.alpha) / (t + self.alpha * V)

    def log_prob(self, ch, ctx):
        return math.log(self.prob(ch, ctx))

    def cross_entropy(self, seqs):
        n = 0
        loss = 0.0
        for s in seqs:
            s = ''.join(c if c in DNA else 'N' for c in s.upper())
            for i, ch in enumerate(s):
                loss -= self.log_prob(ch, s[max(0, i - self.order):i])
                n += 1
        return loss / max(n, 1), n


class InterpolatedMarkovModel:
    """Interpolated Markov Model (IMM) — blends probabilities from order 0 up to max_order.

    Weights are proportional to the total counts observed for each order's context,
    so higher orders contribute more when they have sufficient data.
    """

    def __init__(self, max_order: int, alpha: float = 0.5):
        self.max_order = max_order
        self.alpha = alpha
        self.models: list[MarkovModel] = []

    def fit(self, seqs):
        self.models = [MarkovModel(o, self.alpha).fit(seqs) for o in range(self.max_order + 1)]
        return self

    def prob(self, ch, ctx):
        total_weight = 0.0
        weighted = 0.0
        for order in range(self.max_order + 1):
            m = self.models[order]
            ctx_o = ctx[-order:] if order else ''
            weight = m.totals.get(ctx_o, 0) + 1.0
            total_weight += weight
            weighted += weight * m.prob(ch, ctx)
        return weighted / max(total_weight, 1e-12)

    def log_prob(self, ch, ctx):
        return math.log(self.prob(ch, ctx))

    def cross_entropy(self, seqs):
        n = 0
        loss = 0.0
        for s in seqs:
            s = ''.join(c if c in DNA else 'N' for c in s.upper())
            for i, ch in enumerate(s):
                loss -= self.log_prob(ch, s[max(0, i - self.max_order):i])
                n += 1
        return loss / max(n, 1), n


def shuffle_sequence(seq: str, k: int = 2) -> str:
    """Dinucleotide (k=2) or k-mer shuffle preserving k-mer composition."""
    if len(seq) < k + 1:
        return seq
    kmers = [seq[i:i + k] for i in range(0, len(seq) - k + 1, k)]
    overlap = len(seq) % k
    tail = seq[-overlap:] if overlap else ''
    random.shuffle(kmers)
    return ''.join(kmers) + tail


def evaluate_markov_orders(train, test, orders=(0, 1, 5), alpha=0.5,
                           include_imm: bool = False, include_shuffled: bool = False):
    out = {}
    for o in orders:
        m = MarkovModel(o, alpha).fit(train)
        ce, n = m.cross_entropy(test)
        out[f'order_{o}'] = {
            'order': o,
            'nats_per_base': ce,
            'bits_per_base': ce / math.log(2),
            'tokens': n,
        }
    if include_imm:
        imm = InterpolatedMarkovModel(max(orders), alpha).fit(train)
        ce, n = imm.cross_entropy(test)
        out['imm'] = {
            'order': f'interpolated_0..{max(orders)}',
            'nats_per_base': ce,
            'bits_per_base': ce / math.log(2),
            'tokens': n,
        }
    if include_shuffled:
        for k in (2, 3):
            shuffled_train = [shuffle_sequence(s, k=k) for s in train]
            m_k = MarkovModel(max(orders), alpha).fit(shuffled_train)
            ce, n = m_k.cross_entropy(test)
            out[f'shuffled_k{k}'] = {
                'order': max(orders),
                'shuffle': f'dinucleotide' if k == 2 else f'k{k}',
                'nats_per_base': ce,
                'bits_per_base': ce / math.log(2),
                'tokens': n,
            }
    return out

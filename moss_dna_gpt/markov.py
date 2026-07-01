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
    """Interpolated Markov Model (IMM) — backs off from high to low order.

    For each context, if the observed count at order k exceeds *threshold*,
    the order-k MLE probability is used directly.  Otherwise the probability
    is interpolated with order-(k-1) using weight = count / threshold.
    The recursion bottoms out at order 0 (additive-smoothed marginal).
    """

    def __init__(self, max_order: int, alpha: float = 0.5, threshold: int = 5):
        self.max_order = max_order
        self.alpha = alpha
        self.threshold = threshold
        self.models: list[MarkovModel] = []

    def fit(self, seqs):
        self.models = [MarkovModel(o, self.alpha).fit(seqs) for o in range(self.max_order + 1)]
        return self

    def prob(self, ch, ctx):
        return self._backoff_prob(ch, ctx[-self.max_order:], self.max_order)

    def _backoff_prob(self, ch, ctx, order):
        ctx_o = ctx[-order:] if order else ''
        m = self.models[order]
        t = m.totals.get(ctx_o, 0)
        if order == 0:
            return m.prob(ch, '')
        if t >= self.threshold:
            return m.prob(ch, ctx)
        weight = t / self.threshold
        return weight * m.prob(ch, ctx) + (1 - weight) * self._backoff_prob(ch, ctx, order - 1)

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


def low_complexity_fraction(seq: str, window: int = 80, stride: int = 20) -> float:
    """Fraction of sliding windows with entropy < 1.0 (low-complexity)."""
    if len(seq) < window:
        window = max(len(seq), 8)
        stride = window
    low = 0
    n = 0
    for start in range(0, len(seq) - window + 1, stride):
        w = seq[start:start + window]
        counts = Counter(w)
        ent = -sum((c / window) * math.log2(c / window) for c in counts.values())
        if ent < 1.0:
            low += 1
        n += 1
    return low / max(n, 1)


def filter_low_complexity(seqs: list[str], threshold: float = 0.3) -> list[str]:
    """Keep only sequences with low-complexity fraction below threshold."""
    return [s for s in seqs if low_complexity_fraction(s) < threshold]


def _build_models(counts, totals, max_o, alpha):
    """Build MarkovModel instances from pre-computed counts/totals."""
    models = []
    for o in range(max_o + 1):
        m = MarkovModel(o, alpha)
        m.counts = counts[o]
        m.totals = totals[o]
        models.append(m)
    return models


def evaluate_markov_orders(train, test, orders=(0, 1, 5), alpha=0.5,
                           include_imm: bool = False, include_shuffled: bool = False):
    """Single-pass evaluation of all Markov orders and optional IMM/shuffled baselines."""
    max_o = max(orders) if orders else 0
    V = len(DNA)

    # ---- Train: one pass to fit all orders simultaneously ----
    counts: list[defaultdict[Counter]] = [defaultdict(Counter) for _ in range(max_o + 1)]
    totals: list[Counter] = [Counter() for _ in range(max_o + 1)]

    for s in train:
        s_up = ''.join(c if c in DNA else 'N' for c in s.upper())
        for i, ch in enumerate(s_up):
            for o in range(max_o + 1):
                ctx = s_up[max(0, i - o):i] if o else ''
                counts[o][ctx][ch] += 1
                totals[o][ctx] += 1

    # ---- Evaluate: one pass to compute CE for all orders ----
    out = {}
    test_results: list[float] = [0.0 for _ in range(max_o + 1)]
    test_counts: list[int] = [0 for _ in range(max_o + 1)]

    for s in test:
        s_up = ''.join(c if c in DNA else 'N' for c in s.upper())
        for i, ch in enumerate(s_up):
            for o in range(max_o + 1):
                ctx = s_up[max(0, i - o):i] if o else ''
                c = counts[o].get(ctx, Counter())
                t = totals[o].get(ctx, 0)
                p = (c.get(ch, 0) + alpha) / (t + alpha * V)
                test_results[o] -= math.log(p)
                test_counts[o] += 1

    for o in orders:
        out[f'order_{o}'] = {
            'order': o,
            'nats_per_base': test_results[o] / max(test_counts[o], 1),
            'bits_per_base': test_results[o] / max(test_counts[o], 1) / math.log(2),
            'tokens': test_counts[o],
        }

    # ---- IMM (uses pre-built models via backoff) ----
    if include_imm:
        models = _build_models(counts, totals, max_o, alpha)
        imm = InterpolatedMarkovModel(max_o, alpha)
        imm.models = models
        imm_loss, imm_tokens = imm.cross_entropy(test)
        out['imm'] = {
            'order': f'interpolated_0..{max_o}',
            'nats_per_base': imm_loss,
            'bits_per_base': imm_loss / math.log(2),
            'tokens': imm_tokens,
        }

    # ---- Shuffled controls: separate pass (different training data) ----
    if include_shuffled:
        for k in (2, 3):
            shuffled_train = [shuffle_sequence(s, k=k) for s in train]
            # one-pass fit
            sc: list[defaultdict[Counter]] = [defaultdict(Counter) for _ in range(max_o + 1)]
            stot: list[Counter] = [Counter() for _ in range(max_o + 1)]
            for s in shuffled_train:
                s_up = ''.join(c if c in DNA else 'N' for c in s.upper())
                for i, ch in enumerate(s_up):
                    for o in range(max_o + 1):
                        ctx = s_up[max(0, i - o):i] if o else ''
                        sc[o][ctx][ch] += 1
                        stot[o][ctx] += 1
            # eval using highest order only (consistent with previous behavior)
            o = max_o
            ce_loss = 0.0
            ce_n = 0
            for s in test:
                s_up = ''.join(c if c in DNA else 'N' for c in s.upper())
                for i, ch in enumerate(s_up):
                    ctx = s_up[max(0, i - o):i] if o else ''
                    c = sc[o].get(ctx, Counter())
                    t = stot[o].get(ctx, 0)
                    p = (c.get(ch, 0) + alpha) / (t + alpha * V)
                    ce_loss -= math.log(p)
                    ce_n += 1
            out[f'shuffled_k{k}'] = {
                'order': max_o,
                'shuffle': f'dinucleotide' if k == 2 else f'k{k}',
                'nats_per_base': ce_loss / max(ce_n, 1),
                'bits_per_base': ce_loss / max(ce_n, 1) / math.log(2),
                'tokens': ce_n,
            }

    return out

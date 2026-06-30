from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

from .fasta import ALLOWED, iter_fasta, FastaRecord
from .tokenizer import DnaTokenizer
from .trainer import load_checkpoint


def gc_content(seq: str) -> float:
    s = seq.upper()
    gc = s.count('G') + s.count('C')
    return gc / max(len(s), 1)


def gc_distribution(sequences: list[str]) -> dict:
    gcs = [gc_content(s) for s in sequences]
    return {
        'mean': float(np.mean(gcs)),
        'std': float(np.std(gcs)),
        'min': float(np.min(gcs)),
        'max': float(np.max(gcs)),
        'values': gcs,
    }


def kmer_frequencies(sequences: list[str], k: int = 4) -> dict[str, float]:
    counts: Counter[str] = Counter()
    total = 0
    for s in sequences:
        s = s.upper()
        for i in range(len(s) - k + 1):
            kmer = s[i:i+k]
            if all(c in ALLOWED for c in kmer):
                counts[kmer] += 1
                total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def kmer_js_divergence(
    real_seqs: list[str],
    gen_seqs: list[str],
    k: int = 4,
) -> float:
    real_freq = kmer_frequencies(real_seqs, k)
    gen_freq = kmer_frequencies(gen_seqs, k)
    all_kmers = sorted(set(real_freq) | set(gen_freq))
    p = np.array([real_freq.get(kmer, 0.0) for kmer in all_kmers], dtype=np.float64)
    q = np.array([gen_freq.get(kmer, 0.0) for kmer in all_kmers], dtype=np.float64)
    p += 1e-10
    q += 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q))


def cpg_obs_exp(seq: str) -> float:
    s = seq.upper()
    c = s.count('C')
    g = s.count('G')
    cpg = s.count('CG')
    expected = (c * g) / max(len(s), 1) if len(s) > 0 else 1
    return cpg / max(expected, 1e-10)


def sequence_complexity(seq: str) -> float:
    s = seq.upper()[:1000]
    seen: set[str] = set()
    n = 0
    i = 0
    while i < len(s):
        for j in range(i + 1, len(s) + 1):
            sub = s[i:j]
            if sub in seen:
                continue
            seen.add(sub)
            n += 1
            i = j
            break
    return n / max(len(s), 1)


def shannon_entropy(seq: str, k: int = 2) -> float:
    s = seq.upper()
    counts: Counter[str] = Counter()
    for i in range(len(s) - k + 1):
        counts[s[i:i+k]] += 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        p = c / total
        H -= p * math.log2(p)
    return H


def pyrimidine_dimer_score(
    model: torch.nn.Module,
    tokenizer: DnaTokenizer,
    sequences: list[str],
    device: str = 'cpu',
) -> dict:
    model.eval()
    dimers = ['TT', 'TC', 'CT', 'CC']
    dimer_losses: dict[str, list[float]] = {d: [] for d in dimers}
    other_losses: list[float] = []
    for seq in sequences:
        if len(seq) < 2:
            continue
        ids = tokenizer.encode(seq.upper(), unknown='n')
        if len(ids) < 2:
            continue
        for i in range(len(ids) - 1):
            ctx = ids[:i+1]
            target = ids[i+1]
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(x)
            probs = torch.softmax(logits[0, -1], dim=-1)
            loss = -math.log(max(float(probs[target]), 1e-10))
            pair = seq[i:i+2].upper()
            if pair in dimers:
                dimer_losses[pair].append(loss)
            else:
                other_losses.append(loss)
    result = {
        'dimer_mean': {d: float(np.mean(v)) if v else 0.0 for d, v in dimer_losses.items()},
        'other_mean': float(np.mean(other_losses)) if other_losses else 0.0,
        'dimer_count': {d: len(v) for d, v in dimer_losses.items()},
    }
    return result


def evaluate_biological_quality(
    real_seqs: list[str],
    gen_seqs: list[str],
    model: torch.nn.Module | None = None,
    tokenizer: DnaTokenizer | None = None,
    device: str = 'cpu',
) -> dict:
    real_gc = [gc_content(s) for s in real_seqs]
    gen_gc = [gc_content(s) for s in gen_seqs]

    result: dict = {
        'gc': {
            'real': {'mean': float(np.mean(real_gc)), 'std': float(np.std(real_gc))},
            'generated': {'mean': float(np.mean(gen_gc)), 'std': float(np.std(gen_gc))},
        },
        'kmer_js_3': kmer_js_divergence(real_seqs, gen_seqs, k=3),
        'kmer_js_4': kmer_js_divergence(real_seqs, gen_seqs, k=4),
        'kmer_js_5': kmer_js_divergence(real_seqs, gen_seqs, k=5),
        'cpg': {
            'real': float(np.mean([cpg_obs_exp(s) for s in real_seqs])),
            'generated': float(np.mean([cpg_obs_exp(s) for s in gen_seqs])),
        },
        'entropy': {
            'real': float(np.mean([shannon_entropy(s, k=2) for s in real_seqs])),
            'generated': float(np.mean([shannon_entropy(s, k=2) for s in gen_seqs])),
        },
    }

    if model is not None and tokenizer is not None:
        result['uv_dimer'] = pyrimidine_dimer_score(model, tokenizer, gen_seqs, device=device)

    return result


def sample_windows_from_fasta(
    fasta_path: str | Path,
    num_windows: int = 200,
    window_size: int = 256,
) -> list[str]:
    all_windows: list[str] = []
    for record in iter_fasta(fasta_path):
        seq = record.sequence.upper()
        if len(seq) < window_size:
            continue
        step = max(1, (len(seq) - window_size) // max(num_windows, 1))
        for i in range(0, len(seq) - window_size + 1, step):
            if len(all_windows) >= num_windows:
                break
            w = seq[i:i+window_size]
            if all(c in ALLOWED for c in w):
                all_windows.append(w)
        if len(all_windows) >= num_windows:
            break
    return all_windows[:num_windows]

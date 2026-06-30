from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from .tokenizer import DnaTokenizer


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    sequences: list[str],
    tokenizer: DnaTokenizer,
    device: str = 'cpu',
    pool: str = 'mean',
) -> np.ndarray:
    model.eval()
    all_embs: list[np.ndarray] = []
    for seq in sequences:
        ids = tokenizer.encode(seq.upper(), unknown='n')
        if len(ids) < 2:
            all_embs.append(np.zeros(model.config.n_embd))
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        hidden = model.drop(model.tok_emb(x) + model.pos_emb(torch.arange(x.size(1), device=device)))
        for block in model.blocks:
            hidden = block(hidden)
        hidden = model.ln_f(hidden)
        if pool == 'mean':
            emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()
        elif pool == 'last':
            emb = hidden[0, -1].cpu().numpy()
        else:
            emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()
        all_embs.append(emb)
    return np.array(all_embs)


def build_similarity_graph(
    embeddings: np.ndarray,
    sequences: list[str],
    metadata: list[dict] | None = None,
    threshold: float = 0.85,
    max_nodes: int = 300,
) -> nx.Graph:
    n = min(len(embeddings), max_nodes)
    emb = embeddings[:n]
    seqs = sequences[:n]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_norm = emb / norms
    sim = emb_norm @ emb_norm.T

    G = nx.Graph()
    for i in range(n):
        gc_pct = (seqs[i].upper().count('G') + seqs[i].upper().count('C')) / max(len(seqs[i]), 1)
        attrs: dict[str, Any] = {
            'label': f'Window {i}',
            'title': f'{seqs[i][:60]}...',
            'size': max(5, gc_pct * 30),
            'gc': round(gc_pct, 4),
            'seq': seqs[i],
            'index': i,
        }
        if metadata and i < len(metadata):
            attrs.update(metadata[i])
        G.add_node(i, **attrs)

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > threshold:
                G.add_edge(i, j, weight=round(float(sim[i, j]), 4))

    return G


def build_debruijn_graph(seq: str, k: int = 5) -> nx.DiGraph:
    seq = seq.upper()
    G = nx.DiGraph()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        left = kmer[:-1]
        right = kmer[1:]
        if G.has_edge(left, right):
            G[left][right]['weight'] = G[left][right].get('weight', 1) + 1
        else:
            G.add_edge(left, right, weight=1, label=kmer)
    return G


def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
) -> np.ndarray:
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    return reducer.fit_transform(emb_scaled)


def compute_baseline_stats(sequences: list[str]) -> dict:
    from .evaluate import gc_content, kmer_frequencies, cpg_obs_exp, shannon_entropy
    return {
        'gc_mean': float(np.mean([gc_content(s) for s in sequences])),
        'cpg_mean': float(np.mean([cpg_obs_exp(s) for s in sequences])),
        'entropy_mean': float(np.mean([shannon_entropy(s, k=2) for s in sequences])),
        'kmer_4_top': dict(sorted(kmer_frequencies(sequences, k=4).items(), key=lambda x: -x[1])[:20]),
    }

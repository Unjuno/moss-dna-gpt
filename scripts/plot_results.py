#!/usr/bin/env python3
"""
Plot bits/base comparison bar chart from eval_markov JSON.

Usage:
    python scripts/plot_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def plot_bits_per_base(data: dict, out_path: str | Path) -> None:
    labels = []
    values = []
    colors = []

    for key, entry in data.get("results", {}).items():
        labels.append(entry.get("label", key))
        values.append(entry["bits_per_base"])
        if "dna_gpt" in key.lower():
            colors.append("#2e86ab")
        else:
            colors.append("#a0a0a0")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(values)), values, color=colors, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Bits per base", fontsize=12)
    ax.set_title("DNA next-base prediction: Markov baselines vs DNA-GPT 20M", fontsize=13)
    ax.set_ylim(0, max(values) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    json_path = repo_root / "results" / "eval_markov_20m_step8000000.json"
    out_path = repo_root / "results" / "figures" / "bits_per_base_20m_step8000000.png"

    data = load_results(json_path)

    # Add display label for each entry
    label_map = {
        "markov_order_0": "Markov order 0",
        "markov_order_1": "Markov order 1",
        "markov_order_5": "Markov order 5",
        "dna_gpt_20m": "DNA-GPT 20M",
    }
    for key, entry in data.get("results", {}).items():
        entry["label"] = label_map.get(key, key)

    plot_bits_per_base(data, out_path)


if __name__ == "__main__":
    main()

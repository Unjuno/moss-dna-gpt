#!/usr/bin/env python
"""
Generate a publication-style learning curve figure.

Usage:
    python scripts/plot_learning_curve.py

Output:
    repo/moss-dna-gpt-20m-patens/learning_curve.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EVAL_PATH = REPO_ROOT / "repo" / "moss-dna-gpt-20m-patens" / "eval_curve.json"
OUTPUT_PATH = REPO_ROOT / "repo" / "moss-dna-gpt-20m-patens" / "learning_curve.png"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def main() -> None:
    if not EVAL_PATH.exists():
        print(f"Run scripts/eval_all_checkpoints.py first — {EVAL_PATH} not found.")
        sys.exit(1)

    with open(EVAL_PATH) as fp:
        data = json.load(fp)

    curve = data["dna_gpt_curve"]
    markov = data["markov_baselines"]
    steps = np.array([pt["step"] for pt in curve])
    bits = np.array([pt["bits_per_base"] for pt in curve])

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # DNA-GPT learning curve
    ax.plot(steps / 1e6, bits, "o-", color="#1f77b4", linewidth=1.5, markersize=4,
            label="DNA-GPT (20M)")

    # Markov baselines
    m_styles = [
        ("markov_0", "Markov-0", "#d62728", "--"),
        ("markov_1", "Markov-1", "#ff7f0e", "--"),
        ("markov_5", "Markov-5", "#2ca02c", "--"),
    ]
    for key, label, color, ls in m_styles:
        val = markov[key]["bits_per_base"]
        ax.axhline(val, color=color, linestyle=ls, linewidth=1.2, label=label)
        ax.annotate(f"{val:.4f}", xy=(steps[-1] / 1e6 + 0.05, val),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=9, color=color, va="center")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Test bits / base")
    ax.set_title("DNA-GPT 20M on *Physcomitrium patens* (sequence-level split)")
    ax.set_xlim(0, 9)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Inset: zoom on later steps
    ax_inset = fig.add_axes([0.55, 0.18, 0.35, 0.30])
    ax_inset.plot(steps[3:] / 1e6, bits[3:], "o-", color="#1f77b4", linewidth=1.5, markersize=3)
    for key, label, color, ls in m_styles:
        val = markov[key]["bits_per_base"]
        ax_inset.axhline(val, color=color, linestyle=ls, linewidth=1)
    ax_inset.set_xlim(1.5, 8.5)
    ax_inset.set_ylim(1.25, 1.45)
    ax_inset.set_title("Zoom (≥1.5M)", fontsize=9)
    ax_inset.grid(True, alpha=0.3)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()

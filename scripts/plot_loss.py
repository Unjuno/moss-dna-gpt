#!/usr/bin/env python3
"""
Plot training/validation loss from a loss.csv file.

Usage:
    python scripts/plot_loss.py --loss-csv path/to/loss.csv --out path/to/output.png

The CSV must have a header row with at least a 'step' column.
If 'val_loss' column is present, it is plotted alongside 'train_loss'.
Blank/empty val_loss values are skipped safely.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training loss from CSV")
    parser.add_argument("--loss-csv", required=True, help="Path to loss.csv")
    parser.add_argument("--out", required=True, help="Output PNG path")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.loss_csv)
    out_path = Path(args.out)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required = {"step", "train_loss"}
    if not required.issubset(df.columns):
        print(
            f"Error: CSV must contain columns {required}, got {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df["step"], df["train_loss"], label="Train loss", alpha=0.7, linewidth=0.8)

    if "val_loss" in df.columns:
        val = df[["step", "val_loss"]].dropna()
        if not val.empty:
            ax.plot(
                val["step"], val["val_loss"],
                label="Validation loss", alpha=0.7, linewidth=0.8,
            )

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training loss", fontsize=13)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

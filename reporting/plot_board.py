#!/usr/bin/env python3
"""Plot researcher README figures from results/eval-board.csv.

Usage: python reporting/plot_board.py   (run from the repo root)

Writes:
  results/fig-g1-trained-pairs.png  base model vs EOS-tuned adapter
  results/fig-g2-de-fr.png          de-fr direct vs pivot, base and tuned
  results/fig-g3-loops.png          worst untuned loop rate vs tuned, per pair
  results/figure-data.csv           the numbers behind the figures
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "results"
CSV_PATH = ROOT / "eval-board.csv"
DATA_PATH = ROOT / "figure-data.csv"

LOOP_RE = re.compile(r"(\d+\.\d+)% looped")


def load():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row(rows, pair, setup):
    matches = [r for r in rows if r["pair"] == pair and r["setup"] == setup]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {pair}/{setup}, found {len(matches)}")
    return matches[0]


def loop_rate(record):
    match = LOOP_RE.search(record["notes"])
    if match is None:
        raise ValueError(f"missing repetition rate for {record['pair']}/{record['setup']}")
    rate = float(match.group(1))
    if not 0 <= rate <= 100:
        raise ValueError(f"invalid repetition rate: {rate}")
    return rate


def style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main(output_dir=ROOT):
    rows = load()
    data = []
    pairs = ["en-ro", "ro-en", "de-en"]
    before, after = [], []
    for pair in pairs:
        untuned = [r for r in rows if r["pair"] == pair and r["n"] == "1012"
                   and r["setup"] not in ("base", "eos-tune")]
        if not untuned:
            raise ValueError(f"no full untuned run for {pair}")
        before.append(max(loop_rate(r) for r in untuned))
        after.append(loop_rate(row(rows, pair, "eos-tune")))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # G1: evaluated training directions, base model vs EOS-tuned adapter.
    base = [float(row(rows, p, "base")["chrf"]) for p in pairs]
    tuned = [float(row(rows, p, "eos-tune")["chrf"]) for p in pairs]
    for p, b, t in zip(pairs, base, tuned):
        data.append(("g1", p, "base", f"{b:.2f}"))
        data.append(("g1", p, "eos-tune", f"{t:.2f}"))
        data.append(("g1", p, "delta", f"{t - b:+.2f}"))

    x = range(len(pairs))
    w = 0.35
    fig, ax = plt.subplots()
    ax.bar([i - w / 2 for i in x], base, w, label="Base model")
    bars = ax.bar([i + w / 2 for i in x], tuned, w, label="EOS-tuned adapter")
    for bar, b, t in zip(bars, base, tuned):
        ax.text(bar.get_x() + w / 2, t + 0.4,
                f"{t - b:+.2f}", ha="center", fontsize=9)
    ax.set_xticks(list(x), pairs)
    style(ax, "chrF")
    ax.set_title("Evaluated training directions: base vs EOS-tuned adapter")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig-g1-trained-pairs.png", dpi=150)
    plt.close(fig)

    # G2: de-fr boundary, direct vs pivot.
    labels = ["base direct", "tuned direct", "base pivot", "tuned pivot"]
    vals = [
        float(row(rows, "de-fr", "base")["chrf"]),
        float(row(rows, "de-fr", "eos-tune")["chrf"]),
        float(row(rows, "de-fr", "base-pivot-via-en")["chrf"]),
        float(row(rows, "de-fr", "eos-tune-pivot-via-en")["chrf"]),
    ]
    for lab, v in zip(labels, vals):
        data.append(("g2", "de-fr", lab, f"{v:.2f}"))
    fig, ax = plt.subplots()
    bars = ax.bar(labels, vals)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                f"{v:.2f}", ha="center", fontsize=9)
    style(ax, "chrF")
    ax.set_title("Unseen de-fr: direct vs pivot through English")
    fig.tight_layout()
    fig.savefig(output_dir / "fig-g2-de-fr.png", dpi=150)
    plt.close(fig)

    # G3: worst untuned loop rate vs tuned loop rate per trained pair.
    # Untuned = any full (n=1012) non-eos-tune row; smoke n=10 excluded.
    for p, worst, tuned_rate in zip(pairs, before, after):
        data.append(("g3", p, "worst untuned loop %", f"{worst:.1f}"))
        data.append(("g3", p, "tuned loop %", f"{tuned_rate:.1f}"))
    x = range(len(pairs))
    fig, ax = plt.subplots()
    ax.bar([i - w / 2 for i in x], before, w, label="Worst untuned loop %")
    ax.bar([i + w / 2 for i in x], after, w, label="Tuned loop %")
    ax.set_xticks(list(x), pairs)
    style(ax, "% outputs looped")
    ax.set_title("Repetition: worst untuned configuration vs EOS-tuned")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig-g3-loops.png", dpi=150)
    plt.close(fig)

    data_path = output_dir / DATA_PATH.name
    with open(data_path, "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["figure", "pair", "series", "value"])
        w_.writerows(data)
    print(f"Wrote figures + {data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=ROOT)
    main(parser.parse_args().output_dir)

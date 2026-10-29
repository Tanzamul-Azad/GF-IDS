"""
Figures for the exploratory runs (one seed, CICIoT2023, accuracy on the
development test set), drawn in the same style as the protocol-v2
figures. Reads runs/<tag>_history.pkl.

Usage: python code/make_figs_exploratory.py --out "<manuscript dir>"
"""
import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL1, COL2 = 3.5, 7.16
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.6,
    "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": "#bfbfbf", "grid.linewidth": 0.5,
    "grid.linestyle": "--", "grid.alpha": 0.6, "lines.linewidth": 1.4,
    "legend.frameon": True, "legend.edgecolor": "black",
    "legend.framealpha": 1.0, "figure.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def hist(tag):
    h = pickle.load(open(f"runs/{tag}_history.pkl", "rb"))
    return np.array([x["accuracy"] * 100 for x in h])


def save(fig, out, name):
    p = os.path.join(out, name + ".pdf")
    fig.savefig(p, facecolor="white")
    fig.savefig(p.replace(".pdf", ".png"), facecolor="white", dpi=200)
    plt.close(fig)
    print("wrote", p)


def ablation(out):
    runs = [("BNN-MATCHED_seed42", "GF-IDS (matched)", "#D55E00", "-", "o", 15746),
            ("BNN_seed42", "GF-IDS, extra layer", "#E69F00", "--", "s", 32514),
            ("BNN-FULL_seed42", "Fully binary, larger", "#555555", ":", "^", 32514)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.35),
                               gridspec_kw={"width_ratios": [2.1, 1]})
    for tag, lab, col, ls, mk, _ in runs:
        acc = hist(tag)
        ours = tag == "BNN-MATCHED_seed42"   # drawn heaviest and on top
        a.plot(np.arange(1, len(acc) + 1), acc, color=col, linestyle=ls,
               marker=mk, markevery=6, markersize=3.6, markeredgecolor="white",
               markeredgewidth=0.6, linewidth=1.9 if ours else 1.1,
               zorder=5 if ours else 3, label=lab)
    a.set_xlabel("Communication round")
    a.set_ylabel("Development test accuracy (%)")
    a.set_xlim(1, 45)
    a.set_title("(a) Accuracy per round")
    a.legend(loc="lower right")
    x = np.arange(len(runs))
    best = [hist(t).max() for t, *_ in runs]
    b.bar(x, best, color=[r[2] for r in runs], width=0.62)
    for i, v in enumerate(best):
        b.text(i, v + 1.5, f"{v:.2f}", ha="center", fontsize=6.4)
    b.set_xticks(x)
    b.set_xticklabels([f"{name}\n{r[5]:,} params" for name, r in
                       zip(("Matched", "Extra layer", "Fully binary"), runs)],
                      fontsize=6.0)
    b.set_ylim(0, 108)
    b.set_ylabel("Best accuracy (%)")
    b.set_title("(b) Best accuracy and size")
    fig.tight_layout(w_pad=1.2)
    save(fig, out, "fig_ablation_expl")


def bipru(out):
    models = [("BNN-MATCHED_seed42", "GF-IDS", "#D55E00", 23.51),
              ("BiPruneFL-Repro_seed42", "BiPruneFL\n(adapted)", "#E69F00", 63.76),
              ("MLP_seed42", "MLP-FL", "#0072B2", 63.76)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL1, 2.25),
                               gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(models))
    w = 0.38
    best = [hist(t).max() for t, *_ in models]
    last = [hist(t)[-1] for t, *_ in models]
    cols = [m[2] for m in models]
    a.bar(x - w / 2 - 0.01, best, w, color=cols, label="Best round")
    a.bar(x + w / 2 + 0.01, last, w, color="white", edgecolor=cols,
          hatch="////", linewidth=0.8, label="Last round")
    for i in range(len(models)):
        a.text(x[i] - w / 2, best[i] + 1.5, f"{best[i]:.1f}", ha="center", fontsize=5.6)
        a.text(x[i] + w / 2, last[i] + 1.5, f"{last[i]:.1f}", ha="center", fontsize=5.6)
    a.set_xticks(x)
    a.set_xticklabels([m[1].replace("\n", " ") for m in models], fontsize=6.0,
                      rotation=30, ha="right")
    a.set_ylim(0, 112)
    a.set_ylabel("Development test accuracy (%)")
    a.set_title("(a) Accuracy")
    a.legend(loc="lower left", fontsize=5.8, handlelength=1.4)
    down = [m[3] for m in models]
    b.bar(x, down, color=cols, width=0.62)
    for i, v in enumerate(down):
        b.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.0)
    b.set_xticks(x)
    b.set_xticklabels([m[1].replace("\n", " ") for m in models], fontsize=6.0,
                      rotation=30, ha="right")
    b.set_ylim(0, 75)
    b.set_ylabel("Downlink per round (KiB)")
    b.set_title("(b) Downlink")
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig_bipru_expl")


def agg(out):
    runs = [("BNN-MATCHED_seed42", "FedAvg + projection (ours)", "#D55E00", "-", "o"),
            ("BNN-MATCHED_seed42_fedprox", r"FedProx ($\mu=0.01$)", "#0072B2", "--", "s"),
            ("BNN-MATCHED_seed42_signsgdlat5", "Sign vote, step 0.05", "#009E73", ":", "^"),
            ("BNN-MATCHED_seed42_signsgdlat", "Sign vote, step 0.01", "#CC79A7", "-.", "v"),
            ("BNN-MATCHED_seed42_signsgdlat1", "Sign vote, step 0.001", "#56B4E9",
             (0, (5, 1.5, 1, 1.5)), "D")]
    fig, ax = plt.subplots(figsize=(COL1, 2.55))
    for tag, lab, col, ls, mk in runs:
        acc = hist(tag)
        ours = tag == "BNN-MATCHED_seed42"   # drawn heaviest and on top
        ax.plot(np.arange(1, len(acc) + 1), acc, color=col, linestyle=ls,
                marker=mk, markevery=6, markersize=3.4, markeredgecolor="white",
                markeredgewidth=0.6, linewidth=1.9 if ours else 1.1,
                zorder=5 if ours else 3, label=lab)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Development test accuracy (%)")
    ax.set_xlim(1, 45)
    ax.set_ylim(0, 102)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              fontsize=6.2)
    save(fig, out, "fig_agg_expl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ablation(a.out)
    bipru(a.out)
    agg(a.out)

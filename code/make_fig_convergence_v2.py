"""
Convergence and stability figure for protocol v2.

Pooled client-validation accuracy per round (no test or holdout data),
mean over three seeds with the min-max band. One panel per dataset.
Writes fig_convergence_v2.pdf into the manuscript folder.

Usage: python code/make_fig_convergence_v2.py --out "<manuscript dir>"
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL2 = 7.16
# Fixed categorical order from the paper's validated palette
# (adjacent CVD dE >= 9.6, normal-vision dE >= 18.7 on white).
MODELS = [  # key, label, colour, dash, marker
    ("BNN-MATCHED", "GF-IDS (ours)", "#D55E00", "-", "o"),
    ("MLP-TWIN", "FP32-Twin", "#009E73", (0, (4, 1.5)), "D"),
    ("MLP", "MLP-FL", "#0072B2", "--", "s"),
    ("MLP-INT8", "MLP-INT8", "#CC79A7", (0, (1, 1)), "v"),
]
DATASETS = [("cic", "CICIoT2023"), ("nfton", "NF-ToN-IoT-v2")]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
    "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": "#bfbfbf", "grid.linewidth": 0.5,
    "grid.linestyle": "--", "grid.alpha": 0.6, "lines.linewidth": 1.4,
    "legend.frameon": True, "legend.edgecolor": "black",
    "legend.framealpha": 1.0, "legend.handlelength": 2.6,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def curves(ds, model):
    out = []
    for rd in sorted(glob.glob(f"runs_v2/{ds}/{model}_iid_s*_*/")):
        try:
            c = json.load(open(rd + "config.json"))["config"]
            h = json.load(open(rd + "history.json"))
        except FileNotFoundError:
            continue
        if c["aggregator"] != "fedavg" or c["class_weight"] != "none":
            continue
        out.append([x["val_accuracy"] * 100 for x in h])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.35), sharex=True)
    for ax, (ds, title) in zip(axes, DATASETS):
        for key, label, col, dash, mk in MODELS:
            c = curves(ds, key)
            if len(c) == 0:
                continue
            r = np.arange(1, c.shape[1] + 1)
            ax.fill_between(r, c.min(0), c.max(0), color=col, alpha=0.15,
                            linewidth=0)
            ours = key == "BNN-MATCHED"   # drawn heaviest and on top
            ax.plot(r, c.mean(0), color=col, linestyle=dash, marker=mk,
                    markevery=6, markersize=3.6, markeredgecolor="white",
                    markeredgewidth=0.6, linewidth=1.9 if ours else 1.1,
                    zorder=5 if ours else 3, label=f"{label} ({len(c)} seeds)")
        ax.set_title(title)
        ax.set_xlabel("Communication round")
        ax.set_xlim(1, 45)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].set_ylim(50, 100)
    axes[1].set_ylim(60, 95)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.2)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "fig_convergence_v2.pdf")
    fig.savefig(path, facecolor="white")
    fig.savefig(path.replace(".pdf", ".png"), facecolor="white", dpi=200)
    print("wrote", path)


if __name__ == "__main__":
    main()

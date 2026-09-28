"""Regenerate every figure used in the GF-IDS paper in one consistent style.

Style rules (from the data-visualization checklist used for this paper):
* one fixed colour per model, the same in every figure, taken in order from
  a colour-vision-deficiency-checked categorical palette;
* every line also has its own marker and dash pattern, so the figures still
  read in greyscale print;
* thin marks, a 2 px white gap between bars, solid hairline grids behind the
  data, text in neutral ink (never in the series colour);
* every label stays inside the figure, and every number shown is read from
  the result files, never typed in.

Inputs (all produced by federated_train.py / evaluate.py):
  runs/results_best*.csv, runs/*_history.pkl, runs_nfton/*.csv,
  runs/test_predictions_seed42.npz (from code/cache_predictions.py).
Run from the repository root:  python code/make_paper_figures.py
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(__file__))
from binary_ops import BinaryLinear  # noqa: E402
from models import MODEL_REGISTRY  # noqa: E402
from quant_ops import QuantLinear  # noqa: E402

OUT = "figures"
COL1, COL2 = 3.5, 7.16

INK, INK2, MUTED, GRID = "#000000", "#000000", "#777777", "#bfbfbf"

# Categorical palette, slots 1-8 in fixed order (validated: adjacent CVD
# dE >= 9.1, normal-vision dE >= 19.6 on the light surface).
SLOT = ["#D55E00", "#0072B2", "#009E73", "#E69F00",
        "#56B4E9", "#CC79A7", "#F0E442", "#56B4E9"]

MODELS = {  # key: (label, colour, marker, dash)
    "BNN-MATCHED": ("BNN-FL (proposed)", SLOT[0], "o", "-"),
    "MLP": ("MLP-FL", SLOT[1], "s", "--"),
    "LSTM": ("LSTM-FL", SLOT[2], "D", ":"),
    "CNN": ("CNN-FL", SLOT[3], "^", "-."),
    "BNN-INT8IO": ("BNN-INT8IO", SLOT[4], "P", (0, (5, 1.5, 1, 1.5))),
    "MLP-INT8": ("MLP-INT8", SLOT[5], "v", (0, (1, 1))),
    "BiPruneFL-Repro": ("BiPruneFL-Repro", SLOT[6], "h", "-"),
    "BNN-INT8IO-MATCHED": ("BNN-INT8IO (matched)", SLOT[7], "X", "-"),
    # ablation variants: darker and grey versions of the BNN-FL colour
    "BNN": ("BNN-FL (extra layer)", "#8B4000", "s", "--"),
    "BNN-FULL": ("BNN-FL (fully binarized)", "#999999", "^", ":"),
}
ORDER = ["BNN-MATCHED", "MLP", "LSTM", "CNN", "BNN-INT8IO", "MLP-INT8"]


def short(key):
    return MODELS[key][0].replace(" (proposed)", "")


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.6,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "black", "axes.linewidth": 0.8,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "--", "grid.alpha": 0.6,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": True, "legend.edgecolor": "black", "legend.framealpha": 1.0, "legend.fancybox": True, "legend.handlelength": 2.4,
    "lines.linewidth": 1.3, "lines.markersize": 3.6,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), facecolor="white")
    plt.close(fig)
    print("  wrote", name)


def line(ax, x, y, key, label=None, dash=None, marker=None, every=4, **kw):
    lab, c, m, ls = MODELS[key]
    mk = marker or m
    ring = c if mk in ("x", "+") else "black"   # open markers need their own colour
    ax.plot(x, y, color=c, linestyle=dash or ls, marker=mk,
            markevery=every, markeredgecolor=ring,
            markeredgewidth=0.9 if mk in ("x", "+") else 0.4,
            label=label or lab, **kw)


def bars(ax, xs, heights, colors, **kw):
    return ax.bar(xs, heights, color=colors, edgecolor="black",
                  linewidth=0.45, width=kw.pop("width", 0.72), **kw)


def label_bars(ax, rects, fmt, pad_frac=0.012):
    top = ax.get_ylim()[1]
    for r in rects:
        h = r.get_y() + r.get_height()
        ax.text(r.get_x() + r.get_width() / 2, h + top * pad_frac,
                fmt(h), ha="center", va="bottom", fontsize=6.2, color=INK2)


def hist(tag):
    with open(os.path.join("runs", f"{tag}_history.pkl"), "rb") as f:
        return [h["accuracy"] * 100 for h in pickle.load(f)], None


def hist_full(tag):
    with open(os.path.join("runs", f"{tag}_history.pkl"), "rb") as f:
        return pickle.load(f)


R = pd.read_csv("runs/results_best.csv").set_index("Model")
ACC = R["Accuracy(%)"]
DOWN = R["PackedPayload(KB)"]
UP = R["Float32Payload(KB)"]
PARAMS = R["Parameters"]


# ---------------------------------------------------------------- energy
def energy_nj(key):
    """Energy per inference, same pricing as Table VIII (Horowitz, 45 nm)."""
    from thop import profile
    from evaluate import count_binary_linear, count_quant_linear
    m = MODEL_REGISTRY[key](31, 34).eval()
    tot, _ = profile(m, inputs=(torch.randn(1, 31),), verbose=False,
                     custom_ops={BinaryLinear: count_binary_linear,
                                 QuantLinear: count_quant_linear})
    bins = [b for b in m.modules() if isinstance(b, BinaryLinear)]
    first = bins[0].in_features * bins[0].out_features if bins else 0
    rest = sum(b.in_features * b.out_features for b in bins[1:])
    i8 = sum(q.in_features * q.out_features
             for q in m.modules() if isinstance(q, QuantLinear))
    fl = tot - first - rest - i8
    add = 0.03 if "INT8" in key else 0.9
    return (fl * 4.6 + i8 * 0.23 + first * add + rest * 0.03) / 1000


# ------------------------------------------------------------ figures
def fig_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.55))
    for key in ORDER:
        h = hist_full(f"{key}_seed42")
        r = np.arange(1, len(h) + 1)
        line(axes[0], r, [x["accuracy"] * 100 for x in h], key)
        line(axes[1], r, [x["loss"] for x in h], key)
    axes[0].set_ylabel("Global test accuracy (%)")
    axes[0].set_ylim(0, 102)
    axes[1].set_ylabel("Global test loss (log scale)")
    axes[1].set_yscale("log")
    for ax, t in zip(axes, ("(a) Accuracy", "(b) Loss")):
        ax.set_xlabel("Communication round")
        ax.set_title(t)
        ax.set_xlim(0, 46)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig_convergence")


def _short_class(name):
    return (name.replace("_Fragmentation", "-Frag").replace("_Flood", "")
            .replace("_Malware", "").replace("Recon-", "Rec-")
            .replace("_Spoofing", "-Spoof").replace("_Attack", "")
            .replace("DictionaryBruteForce", "DictBruteForce")
            .replace("VulnerabilityScan", "VulnScan")
            .replace("HostDiscovery", "HostDisc"))


def fig_perclass(P, names):
    y = P["labels"]
    f1m = f1_score(y, P["pred_mlp"], average=None, labels=range(len(names)))
    f1b = f1_score(y, P["pred_bnn"], average=None, labels=range(len(names)))
    fig, ax = plt.subplots(figsize=(COL2, 2.9))
    x = np.arange(len(names))
    w = 0.4
    ax.bar(x - w / 2, f1m, w, color=MODELS["MLP"][1], edgecolor="black",
           linewidth=0.3, label="MLP-FL")
    ax.bar(x + w / 2, f1b, w, color=MODELS["BNN-MATCHED"][1],
           edgecolor="black", linewidth=0.3, hatch="///",
           label="BNN-FL (proposed)")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_class(n) for n in names], rotation=55,
                       ha="right", fontsize=6, rotation_mode="anchor")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.7, len(names) - 0.3)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, "fig_perclass_f1")


def fig_roc(P, k):
    yb = label_binarize(P["labels"], classes=np.arange(k))
    fig, ax = plt.subplots(figsize=(COL1, 2.75))
    curves = {}
    for key, pk in (("MLP", "prob_mlp"), ("BNN-MATCHED", "prob_bnn")):
        fpr, tpr, _ = roc_curve(yb.ravel(), P[pk].ravel())
        curves[key] = (fpr, tpr)
        lab, c, _, ls = MODELS[key]
        ax.plot(fpr, tpr, color=c, linestyle=ls,
                label=f"{lab.replace(' (proposed)', '')} (AUC {auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], color="black", alpha=0.4, linewidth=0.8, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ins = ax.inset_axes([0.34, 0.30, 0.46, 0.42])
    for key, (fpr, tpr) in curves.items():
        ins.plot(fpr, tpr, color=MODELS[key][1], linestyle=MODELS[key][3],
                 linewidth=1.1)
    ins.set_xlim(0, 0.05)
    ins.set_ylim(0.95, 1.001)
    ins.set_xticks([0, 0.025, 0.05])
    ins.set_yticks([0.95, 0.975, 1.0])
    ins.tick_params(labelsize=5.6)
    ins.set_title("zoom: top-left corner", fontsize=6, pad=2, color=INK2)
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_color("black")
    fig.tight_layout()
    save(fig, "fig_roc")


def fig_misclass(P, names, top=10):
    y, p = P["labels"], P["pred_bnn"]
    cm = confusion_matrix(y, p, labels=np.arange(len(names)))
    rate = cm / cm.sum(1, keepdims=True).clip(min=1)
    pairs = [(i, j, rate[i, j], cm[i].sum()) for i in range(len(names))
             for j in range(len(names)) if i != j and rate[i, j] >= 0.10]
    pairs = sorted(pairs, key=lambda t: -t[2])[:top]
    fig, ax = plt.subplots(figsize=(COL1, 2.85))
    yy = np.arange(len(pairs))
    ax.barh(yy, [t[2] for t in pairs], color=MODELS["BNN-MATCHED"][1],
            edgecolor="black", linewidth=0.45, height=0.68)
    for k, t in enumerate(pairs):
        ax.text(t[2] + 0.015, k, f"{t[2] * 100:.1f}%", va="center",
                fontsize=6.2, color=INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{names[t[0]]} (n={t[3]:,})\n→ {names[t[1]]}"
                        for t in pairs], fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("Predicted as benign (% of true class)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_misclassification")


def fig_ablation():
    keys = ["BNN-MATCHED", "BNN", "BNN-FULL"]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.45),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    for key in keys:
        a = [x["accuracy"] * 100 for x in hist_full(f"{key}_seed42")]
        line(axes[0], np.arange(1, len(a) + 1), a, key)
    axes[0].set_xlabel("Communication round")
    axes[0].set_ylabel("Global test accuracy (%)")
    axes[0].set_ylim(40, 101)
    axes[0].set_xlim(0, 46)
    axes[0].set_title("(a) Accuracy during training")
    axes[0].legend(loc="lower right")
    x = np.arange(3)
    r = bars(axes[1], x, [ACC[k] for k in keys], [MODELS[k][1] for k in keys])
    axes[1].set_ylim(0, 110)
    label_bars(axes[1], r, lambda v: f"{v:.2f}%")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Proposed\n" f"{PARAMS[keys[0]]:,} params",
                             "Extra layer\n" f"{PARAMS[keys[1]]:,} params",
                             "Fully binarized\n" f"{PARAMS[keys[2]]:,} params"],
                            fontsize=6.3)
    axes[1].set_ylabel("Best accuracy (%)")
    axes[1].grid(axis="x", visible=False)
    axes[1].set_title("(b) Best accuracy and model size")
    fig.tight_layout()
    save(fig, "fig_ablation")


def fig_efficiency():
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.45))
    x = np.arange(len(ORDER))
    cols = [MODELS[k][1] for k in ORDER]
    names = [short(k) for k in ORDER]
    e = [energy_nj(k) for k in ORDER]
    r = bars(axes[0], x, e, cols)
    axes[0].set_yscale("log")
    axes[0].set_ylim(3, 3e4)
    for rect, v in zip(r, e):
        axes[0].text(rect.get_x() + rect.get_width() / 2, v * 1.25,
                     f"{v:.1f}" if v < 100 else f"{v:,.0f}", ha="center",
                     va="bottom", fontsize=5.8, color=INK2)
    axes[0].set_ylabel("Energy per inference (nJ, log scale)")
    axes[0].set_title("(a) Estimated energy")
    d = np.array([DOWN[k] for k in ORDER])
    u = np.array([UP[k] for k in ORDER])
    bars(axes[1], x, d, cols)
    bars(axes[1], x, u, cols, bottom=d, alpha=0.45, hatch="///")
    axes[1].set_ylim(0, 760)
    for k in range(len(x)):
        axes[1].text(x[k], d[k] + u[k] + 12, f"{d[k] + u[k]:.0f}",
                     ha="center", va="bottom", fontsize=5.8, color=INK2)
    axes[1].set_ylabel("Traffic per client per round (KB)")
    axes[1].set_title("(b) Round trip")
    axes[1].legend(handles=[Patch(facecolor="#777777", edgecolor="black", label="Downlink"),
                            Patch(facecolor="#dddddd", edgecolor="black", hatch="///", label="Uplink")],
                   loc="upper left")
    p = [PARAMS[k] for k in ORDER]
    bars(axes[2], x, np.array(p) / 1000, cols)
    axes[2].set_ylabel("Trainable parameters (thousands)")
    axes[2].set_title("(c) Model size")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=6.2,
                           rotation_mode="anchor")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_efficiency")


def fig_pareto():
    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    offs = {"BNN-MATCHED": (8, -3, "left"), "MLP": (7, -1, "left"),
            "LSTM": (-7, 4, "right"), "CNN": (-7, -9, "right"),
            "BNN-INT8IO": (0, 7, "center"), "MLP-INT8": (7, -7, "left")}
    for key in ORDER:
        lab, c, m, _ = MODELS[key]
        ax.scatter(DOWN[key], ACC[key], s=46, color=c, marker=m,
                   edgecolor="black", linewidth=0.45, zorder=3)
        dx, dy, ha = offs[key]
        ax.annotate(short(key), (DOWN[key], ACC[key]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, fontsize=6.4, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(8, 700)
    ax.set_ylim(80, 100)
    ax.set_xlabel("Downlink per client per round (KB, log scale)")
    ax.set_ylabel("Best accuracy (%)")
    ax.text(8.8, 99.2, "Upper left is better: more accurate, less traffic",
            fontsize=6.0, color=INK2, va="center")
    ax.set_yticks([80, 85, 90, 95, 100])
    fig.tight_layout()
    save(fig, "fig_pareto")


def fig_precision_tiers():
    keys = ["MLP", "MLP-INT8", "BNN-MATCHED", "BNN-INT8IO", "BNN-INT8IO-MATCHED"]
    ticks = ["MLP-FL", "MLP-INT8", "BNN-FL", "BNN-INT8IO\n(old)",
             "BNN-INT8IO\n(matched)"]
    cols = [MODELS[k][1] for k in keys]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4))
    r = bars(axes[0], x, [ACC[k] for k in keys], cols)
    axes[0].set_ylim(0, 110)
    r[-1].set_hatch("...")   # matched variant: same colour family, dotted fill
    label_bars(axes[0], r, lambda v: f"{v:.2f}")
    axes[0].set_ylabel("Best accuracy (%)")
    axes[0].set_title("(a) Detection accuracy")
    d = np.array([DOWN[k] for k in keys])
    u = np.array([UP[k] for k in keys])
    rd = bars(axes[1], x, d, cols)
    rd[-1].set_hatch("...")
    bars(axes[1], x, u, cols, bottom=d, alpha=0.45, hatch="///")
    axes[1].set_ylim(0, 185)
    for k in range(len(x)):
        axes[1].text(x[k], d[k] + u[k] + 2.5, f"{d[k] + u[k]:.1f}",
                     ha="center", va="bottom", fontsize=6.2, color=INK2)
    axes[1].set_ylabel("Traffic per client per round (KB)")
    axes[1].set_title("(b) Round trip")
    axes[1].legend(handles=[Patch(facecolor="#777777", edgecolor="black", label="Downlink"),
                            Patch(facecolor="#dddddd", edgecolor="black", hatch="///", label="Uplink")],
                   loc="upper left", ncol=2)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=6.3)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_precision_tiers")


def fig_bipru():
    keys = ["BNN-MATCHED", "BiPruneFL-Repro", "MLP"]
    ticks = ["BNN-FL", "BiPruneFL\n(reproduced)", "MLP-FL"]
    cols = [MODELS[k][1] for k in keys]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 2.35))
    r = bars(axes[0], x, [ACC[k] for k in keys], cols)
    axes[0].set_ylim(0, 112)
    label_bars(axes[0], r, lambda v: f"{v:.2f}")
    axes[0].set_ylabel("Best accuracy (%)")
    axes[0].set_title("(a) Accuracy")
    r = bars(axes[1], x, [DOWN[k] for k in keys], cols)
    axes[1].set_ylim(0, 75)
    label_bars(axes[1], r, lambda v: f"{v:.1f}")
    axes[1].set_ylabel("Downlink per round (KB)")
    axes[1].set_title("(b) Downlink")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=5.8)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_bipru")


def fig_noniid():
    d = {s: pd.read_csv(f"runs/results_best{s}.csv").set_index("Model")
         for s in ("", "_dir0.5", "_dir0.1")}
    s43 = {s: pd.read_csv(f"runs/results_best_seed43_{s}.csv").set_index("Model")
           for s in ("iid", "dir0.5", "dir0.1")}
    fig, ax = plt.subplots(figsize=(COL1, 3.05))
    x = np.arange(3)
    for key in ORDER:
        line(ax, x, [d[s].loc[key, "Accuracy(%)"] for s in d], key,
             label=short(key) + " (42)", every=1)
    for key in ("BNN-MATCHED", "BNN-INT8IO"):
        line(ax, x, [s43[s].loc[key, "Accuracy(%)"] for s in s43], key,
             label=short(key) + " (43)", dash=(0, (1, 1.2)), marker="x",
             every=1, linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(["IID", "α = 0.5\n(moderate)", "α = 0.1\n(severe)"])
    ax.set_xlim(-0.12, 2.12)
    ax.set_ylim(62, 100)
    ax.set_ylabel("Best accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              fontsize=6.1, handlelength=2.2, columnspacing=1.0)
    fig.tight_layout()
    save(fig, "fig_noniid")


def fig_agg():
    runs = [("", "FedAvg + re-binarize (ours)", SLOT[0], "-", "o"),
            ("_fedprox", "FedProx (μ = 0.01)", SLOT[1], "--", "s"),
            ("_signsgdlat", "SignSGD vote, step 0.01", SLOT[2], ":", "D"),
            ("_signsgdlat5", "SignSGD vote, step 0.05", SLOT[3], "-.", "^"),
            ("_signsgdlat1", "SignSGD vote, step 0.001", SLOT[4],
             (0, (5, 1.5, 1, 1.5)), "P")]
    fig, ax = plt.subplots(figsize=(COL1, 3.0))
    for tag, lab, c, ls, m in runs:
        a = [h["accuracy"] * 100 for h in hist_full(f"BNN-MATCHED_seed42{tag}")]
        ax.plot(np.arange(1, len(a) + 1), a, color=c, linestyle=ls, marker=m,
                markevery=4, markeredgecolor="black", markeredgewidth=0.4,
                label=lab)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Global test accuracy (%)")
    ax.set_ylim(0, 102)
    ax.set_xlim(0, 46)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              fontsize=6.1, columnspacing=1.0)
    fig.tight_layout()
    save(fig, "fig_agg")


if __name__ == "__main__":
    names = [str(n) for n in np.load("data/class_names.npy", allow_pickle=True)]
    P = np.load("runs/test_predictions_seed42.npz")
    fig_convergence()
    fig_perclass(P, names)
    fig_roc(P, len(names))
    fig_misclass(P, names)
    fig_ablation()
    fig_efficiency()
    fig_pareto()
    fig_precision_tiers()
    fig_bipru()
    fig_noniid()
    fig_agg()

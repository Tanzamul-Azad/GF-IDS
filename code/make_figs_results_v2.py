"""
Result figures for protocol v2, all from v2 run outputs.

fig_perclass_v2.pdf  per-class holdout F1 of BNN-FL and FP32-Twin on
                     CICIoT2023, mean over three seeds, classes sorted
                     by holdout support
fig_flips_v2.pdf     share of hidden weights whose sign changes in each
                     round, BNN-FL, both datasets, three seeds

Usage: python code/make_figs_results_v2.py --out "<manuscript dir>"
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL1, COL2 = 3.5, 7.16
BNN, TWIN = "#D55E00", "#009E73"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
    "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": "#bfbfbf", "grid.linewidth": 0.5,
    "grid.linestyle": "--", "grid.alpha": 0.6, "legend.frameon": True,
    "legend.edgecolor": "black", "legend.framealpha": 1.0,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def runs(ds, model):
    out = []
    for rd in sorted(glob.glob(f"runs_v2/{ds}/{model}_iid_s*_*/")):
        try:
            r = json.load(open(rd + "results.json"))
            h = json.load(open(rd + "history.json"))
        except FileNotFoundError:
            continue
        c = r["config"]
        if c["aggregator"] == "fedavg" and c["class_weight"] == "none":
            out.append((r, h))
    return out


def perclass(out):
    b = [r["best/holdout"]["per_class"] for r, _ in runs("cic", "BNN-MATCHED")]
    t = [r["best/holdout"]["per_class"] for r, _ in runs("cic", "MLP-TWIN")]
    classes = sorted(b[0], key=lambda c: -b[0][c]["support"])
    fb = np.array([[x[c]["f1"] for c in classes] for x in b])
    ft = np.array([[x[c]["f1"] for c in classes] for x in t])
    x = np.arange(len(classes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(COL2, 2.6))
    ax.bar(x - w / 2 - 0.01, fb.mean(0), w, color=BNN, label="BNN-FL (ours)",
           yerr=fb.std(0, ddof=1), error_kw={"elinewidth": 0.5, "capsize": 0})
    ax.bar(x + w / 2 + 0.01, ft.mean(0), w, color=TWIN, label="FP32-Twin",
           hatch="////", edgecolor="white", linewidth=0,
           yerr=ft.std(0, ddof=1), error_kw={"elinewidth": 0.5, "capsize": 0})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c} ({b[0][c]['support']:,})" for c in classes],
                       rotation=65, ha="right")
    ax.set_ylabel("Holdout F1")
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(-0.6, len(classes) - 0.4)
    # Support groups used in the text: >10,000, 1,000-10,000, <1,000 flows.
    sup = np.array([b[0][c]["support"] for c in classes])
    groups = [(sup > 10000, "large, >10k flows"),
              ((sup >= 1000) & (sup <= 10000), "mid-sized, 1k-10k"),
              (sup < 1000, "small, <1k")]
    for i, (g, name) in enumerate(groups):
        idx = np.where(g)[0]
        lo, hi = idx.min() - 0.5, idx.max() + 0.5
        if i == 1:
            ax.axvspan(lo, hi, color="#f0f0f0", zorder=0, linewidth=0)
        if i > 0:
            ax.axvline(lo, color="#555555", linewidth=0.6, linestyle=":")
        ax.text((lo + hi) / 2, 1.105,
                f"{len(idx)} {name}\nmean F1 {fb.mean(0)[g].mean():.2f} vs "
                f"{ft.mean(0)[g].mean():.2f}",
                ha="center", va="center", fontsize=6.2, linespacing=1.15)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    path = os.path.join(out, "fig_perclass_v2.pdf")
    fig.savefig(path, facecolor="white")
    fig.savefig(path.replace(".pdf", ".png"), facecolor="white", dpi=200)
    print("wrote", path)


def flips(out):
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    n_hidden = 128 * 64 + 64 * 32
    first = {}
    for ds, title, dash in (("cic", "CICIoT2023", "-"),
                            ("nfton", "NF-ToN-IoT-v2", "--")):
        fl = np.array([[h_["sign_flip_rate"] * 100 for h_ in h]
                       for _, h in runs(ds, "BNN-MATCHED")])
        r = np.arange(1, fl.shape[1] + 1)
        col = BNN if ds == "cic" else "#0072B2"
        ax.plot(r, fl.mean(0), linestyle=dash, color=col, linewidth=1.4)
        first[ds] = fl[:, 0].mean()
        later = fl[:, 1:].mean() / 100 * n_hidden
        # Direct labels instead of a legend.
        if ds == "cic":
            ax.text(16, 0.45, f"{title}: about {later:.0f} of {n_hidden:,}\n"
                    "weights change per round", color=col, fontsize=6.2,
                    ha="left", va="bottom")
        else:
            ax.text(16, 0.0025, f"{title}: no change after round 1",
                    color=col, fontsize=6.2, ha="left", va="bottom")
    ax.annotate(f"Round 1: {first['cic']:.0f}% (CICIoT2023),\n"
                f"{first['nfton']:.0f}% (NF-ToN-IoT-v2)",
                xy=(1.15, first["cic"]), xytext=(4, 4), fontsize=6.2,
                va="center", arrowprops={"arrowstyle": "->", "linewidth": 0.6})
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Hidden signs changed (%)")
    ax.set_xlim(1, 45)
    path = os.path.join(out, "fig_flips_v2.pdf")
    fig.savefig(path, facecolor="white")
    fig.savefig(path.replace(".pdf", ".png"), facecolor="white", dpi=200)
    print("wrote", path)


LABEL = {"BNN-MATCHED": "BNN-FL (ours)", "MLP-TWIN": "FP32-Twin",
         "MLP": "MLP-FL", "MLP-INT8": "MLP-INT8",
         "BNN-INT8IO-MATCHED": "BNN-INT8IO", "CNN": "CNN-FL",
         "LSTM": "LSTM-FL", "BNN-FULL-MATCHED": "BNN-Full"}
STYLE = {"BNN-MATCHED": ("#D55E00", "o"), "MLP-TWIN": ("#009E73", "D"),
         "MLP": ("#0072B2", "s"), "MLP-INT8": ("#CC79A7", "v"),
         "BNN-INT8IO-MATCHED": ("#E69F00", "P"), "CNN": ("#56B4E9", "^"),
         "LSTM": ("#777777", "X"), "BNN-FULL-MATCHED": ("#000000", "*")}


def save(fig, out, name):
    path = os.path.join(out, name + ".pdf")
    fig.savefig(path, facecolor="white")
    fig.savefig(path.replace(".pdf", ".png"), facecolor="white", dpi=200)
    plt.close(fig)
    print("wrote", path)


def pareto(out):
    """Holdout accuracy against measured downlink per round, both datasets,
    with the seed range as error bars. Labels stay inside the axes."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.55))
    for ax, (ds, title) in zip(axes, (("cic", "CICIoT2023"),
                                      ("nfton", "NF-ToN-IoT-v2"))):
        pts = []
        for m in LABEL:
            rs = runs(ds, m)
            if not rs:
                continue
            acc = np.array([r["best/holdout"]["accuracy"] * 100 for r, _ in rs])
            down = np.mean([np.mean([x["down_bytes"] for x in h[1:]])
                            for _, h in rs]) / 1024
            pts.append((m, down, acc))
        xs = [d for _, d, _ in pts]
        ys = np.concatenate([a for _, _, a in pts])
        ax.set_xscale("log")
        ax.set_xlim(min(xs) / 1.5, max(xs) * 1.5)
        pad = 0.12 * (ys.max() - ys.min())
        ax.set_ylim(ys.min() - pad, ys.max() + 1.8 * pad)
        for m, down, acc in pts:
            col, mk = STYLE[m]
            ax.errorbar(down, acc.mean(), yerr=[[acc.mean() - acc.min()],
                                                [acc.max() - acc.mean()]],
                        fmt=mk, color=col, markersize=6, capsize=2,
                        markeredgecolor="white", markeredgewidth=0.7,
                        elinewidth=0.8)
            right = down > max(xs) / 3          # near the right edge
            off = {("nfton", "MLP-TWIN"): (0, 9, "center"),
                   ("nfton", "MLP"): (0, -12, "center"),
                   ("nfton", "MLP-INT8"): (0, 8, "center"),
                   ("nfton", "BNN-MATCHED"): (6, -9, "left"),
                   ("cic", "MLP-TWIN"): (-7, 3, "right"),
                   ("cic", "MLP"): (-7, -9, "right"),
                   }.get((ds, m), ((-7, -2, "right") if right else (7, -2, "left")))
            ax.annotate(LABEL[m].replace(" (ours)", ""), (down, acc.mean()),
                        textcoords="offset points", xytext=off[:2],
                        ha=off[2], fontsize=6.2)
        lo_x, hi_x = ax.get_xlim()
        ticks = [t for t in (5, 10, 20, 30, 50, 100, 200, 400) if lo_x <= t <= hi_x]
        ax.set_xticks(ticks)
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel("Downlink per client and round (KiB, log scale)")
        ax.set_title(title)
        ax.annotate("better", xy=(0.03, 0.98), xytext=(0.12, 0.92),
                    xycoords="axes fraction", textcoords="axes fraction",
                    fontsize=6.2, color="#555555", ha="left", va="top",
                    arrowprops={"arrowstyle": "->", "color": "#555555",
                                "linewidth": 0.7})
    axes[0].set_ylabel("Holdout accuracy (%)")
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "fig_pareto_v2")


def efficiency(out):
    """Energy proxy, per-round traffic and parameter count per model, all
    on CICIoT2023 shapes."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_tables_v2 import cost_breakdown, message_kib
    models = [m for m in LABEL if runs("cic", m) or runs("nfton", m)]
    e, p, dn, up = [], [], [], []
    for m in models:
        b = cost_breakdown(m, 31, 34)
        d_, u_ = message_kib(m, 31, 34)
        e.append(b["energy_nJ"]); p.append(b["params"])
        dn.append(d_); up.append(u_)
    x = np.arange(len(models))
    cols = [STYLE[m][0] for m in models]
    names = [LABEL[m].replace(" (ours)", "") for m in models]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))
    axes[0].bar(x, e, color=cols, width=0.7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Energy proxy (nJ, log)")
    axes[0].set_title("(a) Energy per inference")
    w = 0.38
    axes[1].bar(x - w / 2 - 0.01, dn, w, color=cols, label="Downlink")
    axes[1].bar(x + w / 2 + 0.01, up, w, color="white", edgecolor=cols,
                hatch="////", linewidth=0.8, label="Uplink")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("KiB per client and round (log)")
    axes[1].set_title("(b) Traffic per round")
    axes[1].legend(loc="upper left", handlelength=1.4)
    axes[2].bar(x, np.array(p) / 1000, color=cols, width=0.7)
    axes[2].set_ylabel("Trainable parameters (thousands)")
    axes[2].set_title("(c) Model size")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=6.2)
        ax.set_xlim(-0.6, len(models) - 0.4)
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "fig_efficiency_v2")


def roc_and_confusion(out):
    """Macro one-vs-rest ROC of BNN-FL and FP32-Twin (seed 42, CICIoT2023
    holdout) and the ten classes BNN-FL most often labels benign."""
    import sys
    import torch
    from sklearn.metrics import roc_curve
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import MODEL_REGISTRY
    X = np.load("data_v2/cic/X_holdout.npy").astype(np.float32)
    y = np.load("data_v2/cic/y_holdout.npy")
    names = list(np.load("data/class_names.npy", allow_pickle=True))
    C = len(names)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = np.linspace(0, 1, 2001)
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    preds = {}
    for m in ("BNN-MATCHED", "MLP-TWIN"):
        rd = glob.glob(f"runs_v2/cic/{m}_iid_s42_*/")[0]
        net = MODEL_REGISTRY[m](X.shape[1], C).to(dev).eval()
        net.load_state_dict(torch.load(rd + "best.pt", map_location=dev))
        probs = []
        with torch.no_grad():
            for s in range(0, len(X), 16384):
                probs.append(torch.softmax(net(torch.from_numpy(X[s:s + 16384])
                                               .to(dev)), 1).cpu().numpy())
        pr = np.concatenate(probs)
        preds[m] = pr.argmax(1)
        tpr = np.zeros_like(grid)
        for c in range(C):
            f, t, _ = roc_curve(y == c, pr[:, c])
            tpr += np.interp(grid, f, t)
        tpr /= C
        auc = np.trapz(tpr, grid)
        col, _ = STYLE[m]
        ours = m == "BNN-MATCHED"   # drawn heaviest and on top
        ax.plot(grid, tpr, color=col, linestyle="-" if ours else "--",
                linewidth=1.9 if ours else 1.1, zorder=5 if ours else 3,
                label=f"{LABEL[m]} (macro AUC {auc:.4f})")
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.6, linestyle=":")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("False positive rate (symlog)")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    save(fig, out, "fig_roc_v2")

    b = names.index("BenignTraffic")
    p = preds["BNN-MATCHED"]
    share = []
    for c in range(C):
        if c == b:
            continue
        mask = y == c
        share.append((float((p[mask] == b).mean()), names[c], int(mask.sum())))
    share = sorted(share, reverse=True)[:10][::-1]
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    ax.barh(range(len(share)), [s[0] * 100 for s in share], color=STYLE["BNN-MATCHED"][0])
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels([f"{s[1]} ({s[2]:,})" for s in share], fontsize=6.3)
    for i, s in enumerate(share):
        ax.text(s[0] * 100 + 1, i, f"{s[0] * 100:.1f}%", va="center", fontsize=6.2)
    ax.set_xlabel("Share predicted as benign (%)")
    ax.set_xlim(0, 100)
    save(fig, out, "fig_misclass_v2")


def noniid(out):
    """Holdout accuracy and attack miss rate on NF-ToN-IoT-v2 for IID,
    moderate and severe label skew. Mean over seeds with the range as error bars."""
    settings = [("iid", None, "IID"), ("dirichlet", 0.5, r"$\alpha=0.5$"),
                ("dirichlet", 0.1, r"$\alpha=0.1$")]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3))
    for m, dash, dx in (("BNN-MATCHED", "-", -0.07),
                        ("MLP-TWIN", (0, (4, 1.5)), 0.0),
                        ("MLP-INT8", (0, (1, 1)), 0.07)):
        acc, miss = [], []
        for part, alpha, _ in settings:
            vals = []
            for rd in glob.glob(f"runs_v2/nfton/{m}_*_s*_*/"):
                try:
                    r = json.load(open(rd + "results.json"))
                except FileNotFoundError:
                    continue
                c = r["config"]
                if (c["partition"] == part and c["alpha"] == alpha
                        and c["aggregator"] == "fedavg"
                        and c["class_weight"] == "none"):
                    vals.append((r["best/holdout"]["accuracy"] * 100,
                                 r["best/holdout"]["attack_miss_rate"] * 100))
            acc.append(np.array([v[0] for v in vals]))
            miss.append(np.array([v[1] for v in vals]))
        x = np.arange(len(settings)) + dx
        col, mk = STYLE[m]
        for ax, data in zip(axes, (acc, miss)):
            mean = np.array([d.mean() for d in data])
            lo = mean - np.array([d.min() for d in data])
            hi = np.array([d.max() for d in data]) - mean
            ours = m == "BNN-MATCHED"   # drawn heaviest and on top
            ax.errorbar(x, mean, yerr=[lo, hi], color=col, linestyle=dash,
                        marker=mk, markersize=5, capsize=2,
                        linewidth=1.9 if ours else 1.1, zorder=5 if ours else 3,
                        markeredgecolor="white", markeredgewidth=0.6,
                        label=LABEL[m])
    for ax, lab in zip(axes, ("Holdout accuracy (%)",
                              "Attacks labeled benign (%)")):
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels([s[2] for s in settings])
        ax.set_xlabel("Client partition")
        ax.set_ylabel(lab)
    axes[0].legend(loc="lower left")
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "fig_noniid_v2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    perclass(a.out)
    flips(a.out)
    pareto(a.out)
    efficiency(a.out)
    roc_and_confusion(a.out)
    noniid(a.out)

"""
Build the manuscript's result tables from the v2 run outputs.

Every number in these tables is read from runs_v2/<dataset>/*/results.json
and history.json, which eval_v2.py and fl_v2.py write. Nothing is typed
by hand. The script also writes numbers_v2.json with the values quoted
in the text, so the prose can be checked against the runs.

Usage: python code/make_tables_v2.py --out "<manuscript>/tables"
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

from models import MODEL_REGISTRY

NAMES = {"BNN-MATCHED": "GF-IDS (ours)", "MLP-TWIN": "FP32-Twin",
         "MLP": "MLP-FL", "MLP-INT8": "MLP-INT8",
         "BNN-INT8IO-MATCHED": "BNN-INT8IO", "CNN": "CNN-FL",
         "LSTM": "LSTM-FL", "BNN-FULL-MATCHED": "BNN-Full"}
ORDER = list(NAMES)
READY = {"nfton": (0.90, 0.60), "cic": (0.95, 0.55)}
# Single-seed exploratory runs scored at their last round on the v2 holdout
# (docs/revision_v2/v1_final_on_holdout.py). Used only where a model has no
# protocol-v2 run on a dataset, and marked with a dagger in the tables.
_V1 = "./docs/revision_v2/v1_final_on_holdout.json"
V1_LAST = json.load(open(_V1)) if os.path.exists(_V1) else {}
DATASETS = {"cic": "CICIoT2023", "nfton": "NF-ToN-IoT-v2"}


def load(ds):
    runs = []
    for rd in glob.glob(f"./runs_v2/{ds}/*/"):
        p = os.path.join(rd, "results.json")
        if os.path.exists(p):
            r = json.load(open(p))
            r["history"] = json.load(open(os.path.join(rd, "history.json")))
            runs.append(r)
    return runs


def variant(c):
    """Short label for everything except model and seed."""
    tags = []
    if c["partition"] != "iid":
        tags.append(f"dir{c['alpha']}")
    if c["aggregator"] != "fedavg":
        tags.append(c["aggregator"])
    if c["class_weight"] != "none":
        tags.append(f"cw-{c['class_weight']}")
    if c.get("init_broadcast", "float") != "float":
        tags.append(f"init-{c['init_broadcast']}")
    return "+".join(tags) or "base"


def group(runs):
    g = defaultdict(list)
    for r in runs:
        c = r["config"]
        g[(c["model"], variant(c))].append(r)
    for k in g:
        g[k].sort(key=lambda r: r["config"]["seed"])
    return g


def stat(rs, key, metric, scale=100.0):
    v = np.array([r[key][metric] for r in rs], float) * scale
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan")), v


def fmt(m, s, nd=2):
    if np.isnan(s):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f}$\\pm${s:.{nd}f}"


def readiness(r, ds):
    acc_t, f1_t = READY[ds]
    total = 0.0
    for h in r["history"]:
        total += h["down_bytes"] + float(np.mean(h["up_bytes_per_client"]))
        if h["val_accuracy"] >= acc_t and h["val_macro_f1"] >= f1_t:
            return h["round"], total / 2 ** 20
    return None, None


def cost_breakdown(model_name, d, C):
    """Operation counts split by precision and the arithmetic energy proxy
    (45 nm, Horowitz primitives plus explicit modelling assumptions): a
    Float32 operation counted by thop is priced as a 4.6 pJ MAC equivalent,
    a real-fed signed addition as a 0.9 pJ Float32 addition, and an 8-bit
    MAC at 0.23 pJ. Binary terms use the 0.03 pJ *8-bit addition* price as
    a proxy; Horowitz does not report that as a one-bit XNOR-popcount cost.
    INT8 estimates assume an integer datapath not executed by this code.
    These assumptions preserve the archived estimates; they are not hardware
    measurements or a packed-kernel cost model."""
    from thop import profile
    from binary_ops import BinaryLinear
    from quant_ops import QuantLinear
    from evaluate import count_binary_linear, count_quant_linear
    m = MODEL_REGISTRY[model_name](d, C).eval()
    ops, _ = profile(m, inputs=(torch.randn(1, d),), verbose=False,
                     custom_ops={BinaryLinear: count_binary_linear,
                                 QuantLinear: count_quant_linear})
    bins = [mod for mod in m.modules() if isinstance(mod, BinaryLinear)]
    quants = [mod for mod in m.modules() if isinstance(mod, QuantLinear)]
    b_terms = [mod.in_features * mod.out_features for mod in bins]
    q_terms = sum(mod.in_features * mod.out_features for mod in quants)
    float_ops = int(ops - sum(b_terms) - q_terms)
    real_fed = {"BNN-MATCHED": 1, "BNN-FULL-MATCHED": 2}.get(model_name, 0)
    signed = sum(t for i, t in enumerate(b_terms) if i < real_fed)
    binary = sum(t for i, t in enumerate(b_terms) if i >= real_fed)
    e = float_ops * 4.6 + q_terms * 0.23 + signed * 0.9 + binary * 0.03
    return {"float_ops": float_ops, "int8_macs": int(q_terms),
            "signed_adds": int(signed), "binary_terms": int(binary),
            "energy_nJ": e / 1000.0,
            "params": sum(p.numel() for p in m.parameters())}


def energy_nj(model_name, d, C):
    b = cost_breakdown(model_name, d, C)
    return b["energy_nJ"], b["params"]


def message_kib(model_name, d, C):
    """Downlink and uplink payload from the codec, for the model shape."""
    from codec import encode
    from edgepop_ops import edgepop_frozen_keys
    m = MODEL_REGISTRY[model_name](d, C)
    ex = edgepop_frozen_keys(m)
    return (len(encode(m, m, "down", ex)) / 1024,
            len(encode(m, m, "up", ex)) / 1024)


def tex_detection(groups_by_ds, out):
    ds_list = [ds for ds in ("cic", "nfton") if ds in groups_by_ds]
    cols = "l" + "rrrr" * len(ds_list)
    head1 = " & ".join(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{DATASETS[ds]}}}}}"
                       for ds in ds_list)
    rules = "".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}"
                    for i in range(len(ds_list)))
    head2 = " & ".join(["\\textbf{Acc.}", "\\textbf{Macro F1}",
                        "\\textbf{FAR}", "\\textbf{Miss}"] * len(ds_list))
    lines = [f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", "\\toprule",
             f"& {head1} \\\\", rules, f"\\textbf{{Model}} & {head2} \\\\",
             "\\midrule"]
    for m in ORDER:
        if m == "BNN-FULL-MATCHED":
            continue
        cells = []
        for ds in ds_list:
            rs = groups_by_ds[ds].get((m, "base"))
            if not rs:
                v1 = V1_LAST.get(f"{ds}/{m}_seed42")
                if v1:
                    h = v1["holdout"]
                    cells += [f"{h['accuracy'] * 100:.2f}$^\\dagger$",
                              f"{h['macro_f1']:.3f}$^\\dagger$",
                              f"{h['benign_far'] * 100:.2f}$^\\dagger$",
                              f"{h['attack_miss_rate'] * 100:.2f}$^\\dagger$"]
                else:
                    cells += ["--"] * 4
                continue
            a = stat(rs, "best/holdout", "accuracy")
            f = stat(rs, "best/holdout", "macro_f1", 1.0)
            far = stat(rs, "best/holdout", "benign_far")
            mi = stat(rs, "best/holdout", "attack_miss_rate")
            cells += [fmt(a[0], a[1]), fmt(f[0], f[1], 3), fmt(far[0], far[1]),
                      fmt(mi[0], mi[1])]
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "tab_detection.tex"), "w").write("\n".join(lines))


def tex_nonoverlap(groups, out, ds="nfton"):
    lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
             "& \\multicolumn{2}{c}{\\textbf{All holdout rows}} & "
             "\\multicolumn{2}{c}{\\textbf{Unseen rows only}} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "\\textbf{Model} & \\textbf{Acc.} & \\textbf{Macro F1} & "
             "\\textbf{Acc.} & \\textbf{Macro F1} \\\\", "\\midrule"]
    for m in ORDER:
        rs = groups.get((m, "base"))
        if not rs or m == "BNN-FULL-MATCHED":
            continue
        a = stat(rs, "best/holdout", "accuracy")
        f = stat(rs, "best/holdout", "macro_f1", 1.0)
        a2 = stat(rs, "best/holdout_nonoverlap", "accuracy")
        f2 = stat(rs, "best/holdout_nonoverlap", "macro_f1", 1.0)
        lines.append(f"{NAMES[m]} & {fmt(*a[:2])} & {fmt(*f[:2], 3)} & "
                     f"{fmt(*a2[:2])} & {fmt(*f2[:2], 3)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, f"tab_nonoverlap_{ds}.tex"), "w").write("\n".join(lines))


def tex_cost(groups_by_ds, out, dims):
    """Operations by precision, energy proxy and message sizes, all on
    CICIoT2023 shapes (d = 31, C = 34) so every model is comparable."""
    d, C = 31, 34
    lines = ["\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
             "& & \\multicolumn{4}{c}{\\textbf{Operations per inference}} & & "
             "\\multicolumn{3}{c}{\\textbf{KiB per client and round}} \\\\",
             "\\cmidrule(lr){3-6}\\cmidrule(lr){8-10}",
             "\\textbf{Model} & \\textbf{Params} & \\textbf{Float32} & "
             "\\textbf{INT8 MAC} & \\textbf{$\\pm1$ add} & \\textbf{Binary} & "
             "\\textbf{nJ} & \\textbf{Down} & \\textbf{Up} & \\textbf{Total} \\\\",
             "\\midrule"]
    numbers = {}
    for m in ORDER:
        if not any(groups_by_ds[ds].get((m, "base")) for ds in groups_by_ds):
            continue
        b = cost_breakdown(m, d, C)
        down, up = message_kib(m, d, C)
        numbers[f"cost/{m}"] = {**b, "down_KiB": down, "up_KiB": up}
        fi = lambda v: f"{v:,}" if v else "--"
        lines.append(f"{NAMES[m]} & {b['params']:,} & {fi(b['float_ops'])} & "
                     f"{fi(b['int8_macs'])} & {fi(b['signed_adds'])} & "
                     f"{fi(b['binary_terms'])} & {b['energy_nJ']:.2f} & "
                     f"{down:.2f} & {up:.2f} & {down + up:.2f} \\\\")
    for ds in groups_by_ds:
        for m in ORDER:
            rs = groups_by_ds[ds].get((m, "base"), [])
            if rs:
                numbers[f"ready/{ds}/{m}"] = [readiness(r, ds) for r in rs]
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "tab_cost.tex"), "w").write("\n".join(lines))
    return numbers


def tex_controls(groups, out, ds):
    rows = [("BNN-MATCHED", "base", "GF-IDS (ours)"),
            ("MLP-TWIN", "base", "FP32-Twin"),
            ("BNN-MATCHED", "latent_broadcast", "GF-IDS, no projection"),
            ("BNN-INT8IO-MATCHED", "base", "BNN-INT8IO"),
            ("BNN-FULL-MATCHED", "base", "BNN-Full")]
    lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
             "\\textbf{Variant} & \\textbf{Down KiB} & \\textbf{Acc.} & "
             "\\textbf{Macro F1} & \\textbf{FAR} \\\\", "\\midrule"]
    for m, v, label in rows:
        rs = groups.get((m, v))
        if not rs:
            continue
        down = np.mean([np.mean([h["down_bytes"] for h in r["history"][1:]])
                        for r in rs]) / 1024
        a = stat(rs, "best/holdout", "accuracy")
        f = stat(rs, "best/holdout", "macro_f1", 1.0)
        far = stat(rs, "best/holdout", "benign_far")
        lines.append(f"{label} & {down:.2f} & {fmt(*a[:2])} & {fmt(*f[:2], 3)} "
                     f"& {fmt(*far[:2])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, f"tab_controls_{ds}.tex"), "w").write("\n".join(lines))


DEFAULT_METRICS = (("macro_f1", "Macro F1", 1.0, 3), ("benign_far", "FAR", 100.0, 2))


def tex_variants(groups, out, ds, variants, fname, header,
                 metrics=DEFAULT_METRICS):
    k = len(metrics)
    lines = ["\\begin{tabular}{@{}l" + "r" * (k * len(variants)) + "@{}}",
             "\\toprule",
             "& " + " & ".join(f"\\multicolumn{{{k}}}{{c}}{{\\textbf{{{h}}}}}"
                               for h in header) + " \\\\",
             "".join(f"\\cmidrule(lr){{{2 + k * i}-{1 + k * (i + 1)}}}"
                     for i in range(len(variants))),
             "\\textbf{Model} & " + " & ".join(
                 [f"\\textbf{{{m[1]}}}" for m in metrics] * len(variants)) + " \\\\",
             "\\midrule"]
    for m in ("BNN-MATCHED", "MLP-TWIN", "MLP-INT8"):
        cells = []
        for v in variants:
            rs = groups.get((m, v))
            if not rs:
                cells += ["--"] * k
                continue
            for key, _, scale, nd in metrics:
                s = stat(rs, "best/holdout", key, scale)
                cells.append(fmt(*s[:2], nd))
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, fname), "w").write("\n".join(lines))


def tex_readiness(groups_by_ds, out):
    """Rounds and cumulative traffic until the validation targets are met,
    per client and for the five-client federation, mean over seeds."""
    lines = ["\\begin{tabular}{@{}llrrrc@{}}", "\\toprule",
             "\\textbf{Data} & \\textbf{Model} & \\textbf{Rounds} & "
             "\\textbf{MiB/client} & \\textbf{MiB, $K{=}5$} & "
             "\\textbf{Reached} \\\\", "\\midrule"]
    for ds in ("cic", "nfton"):
        g = groups_by_ds.get(ds)
        if not g:
            continue
        for m in ("BNN-MATCHED", "MLP-TWIN", "MLP", "MLP-INT8"):
            rs = g.get((m, "base"))
            if not rs:
                continue
            vals = [readiness(r, ds) for r in rs]
            ok = [v for v in vals if v[0] is not None]
            if ok:
                rd = np.mean([v[0] for v in ok])
                mib = np.mean([v[1] for v in ok])
                cells = f"{rd:.1f} & {mib:.2f} & {5 * mib:.2f}"
            else:
                cells = "\\multicolumn{3}{c}{not reached in 45 rounds}"
            lines.append(f"{DATASETS[ds].split('-')[0]} & {NAMES[m]} & {cells} & "
                         f"{len(ok)}/{len(vals)} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(os.path.join(out, "tab_readiness.tex"), "w").write("\n".join(lines))


def tex_stability(groups_by_ds, out):
    """Best versus last-round holdout accuracy and late-round swings of
    the validation curve, mean over seeds."""
    lines = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
             "\\textbf{Data} & \\textbf{Model} & \\textbf{Best} & "
             "\\textbf{Last} & \\textbf{Worst last} & \\textbf{SD, last 10} \\\\",
             "\\midrule"]
    for ds in ("cic", "nfton"):
        g = groups_by_ds.get(ds)
        if not g:
            continue
        for m in ("BNN-MATCHED", "MLP-TWIN", "MLP", "MLP-INT8"):
            rs = g.get((m, "base"))
            if not rs:
                continue
            best = np.mean([r["best/holdout"]["accuracy"] for r in rs]) * 100
            last = [r["final/holdout"]["accuracy"] * 100 for r in rs]
            sd = np.mean([np.std([h["val_accuracy"] * 100
                                  for h in r["history"][-10:]]) for r in rs])
            lines.append(f"{DATASETS[ds].split('-')[0]} & {NAMES[m]} & {best:.2f} & "
                         f"{np.mean(last):.2f} & {min(last):.2f} & {sd:.2f} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(os.path.join(out, "tab_stability.tex"), "w").write("\n".join(lines))


def tex_int8(groups_by_ds, out):
    """Binary versus 8-bit compression: downlink and detection, both datasets."""
    lines = ["\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
             "\\textbf{Data} & \\textbf{Model} & \\textbf{Down KiB} & "
             "\\textbf{Acc.} & \\textbf{Macro F1} & \\textbf{FAR} \\\\",
             "\\midrule"]
    for ds in ("cic", "nfton"):
        g = groups_by_ds.get(ds)
        if not g:
            continue
        for m in ("BNN-MATCHED", "MLP-INT8", "BNN-INT8IO-MATCHED"):
            rs = g.get((m, "base"))
            if not rs:
                continue
            down = np.mean([np.mean([h["down_bytes"] for h in r["history"][1:]])
                            for r in rs]) / 1024
            a = stat(rs, "best/holdout", "accuracy")
            f = stat(rs, "best/holdout", "macro_f1", 1.0)
            far = stat(rs, "best/holdout", "benign_far")
            lines.append(f"{'CIC' if ds == 'cic' else 'NF'} & {NAMES[m]} & {down:.2f} & "
                         f"{fmt(*a[:2])} & {fmt(*f[:2], 3)} & {fmt(*far[:2])} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    open(os.path.join(out, "tab_int8.tex"), "w").write("\n".join(lines))


def tex_partition(groups_by_ds, out, ds="nfton", seed=42):
    """Client training sizes and classes present under each partition
    (one seed), read from the run configs."""
    g = groups_by_ds.get(ds)
    if not g:
        return
    cols = []
    for v in ("base", "dir0.5", "dir0.1"):
        rs = [r for r in g.get(("BNN-MATCHED", v), [])
              if r["config"]["seed"] == seed]
        if not rs:
            return
        cfg = json.load(open(glob.glob(
            f"./runs_v2/{ds}/{rs[0]['run_id']}/config.json")[0]))
        cols.append(list(zip(cfg["client_train_sizes"],
                             cfg["client_classes_present"])))
    C = len(np.load("./data_nfton/class_names.npy", allow_pickle=True)) \
        if ds == "nfton" else 34
    lines = ["\\begin{tabular}{@{}lrrr@{}}", "\\toprule",
             f"& \\multicolumn{{3}}{{c}}{{\\textbf{{Training flows (classes present of {C})}}}} \\\\",
             "\\cmidrule(lr){2-4}",
             "\\textbf{Client} & \\textbf{IID} & \\textbf{$\\alpha=0.5$} & "
             "\\textbf{$\\alpha=0.1$} \\\\", "\\midrule"]
    for k in range(len(cols[0])):
        lines.append(f"{k + 1} & " + " & ".join(
            f"{c[k][0]:,} ({c[k][1]})" for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, f"tab_partition_{ds}.tex"), "w").write("\n".join(lines))


def tex_perseed(groups_by_ds, out):
    lines = ["\\begin{tabular}{@{}lllrrrrrrr@{}}", "\\toprule",
             "\\textbf{Data} & \\textbf{Model} & \\textbf{Setting} & "
             "\\textbf{Seed} & \\textbf{Round} & \\textbf{Acc.} & "
             "\\textbf{Macro F1} & \\textbf{FAR} & \\textbf{Miss} & "
             "\\textbf{F1, unseen} \\\\", "\\midrule"]
    for ds in ("cic", "nfton"):
        g = groups_by_ds.get(ds)
        if not g:
            continue
        for (m, v) in sorted(g, key=lambda k: (ORDER.index(k[0])
                                               if k[0] in ORDER else 99, k[1])):
            for r in g[(m, v)]:
                h, u = r["best/holdout"], r["best/holdout_nonoverlap"]
                lines.append(
                    f"{DATASETS[ds].split('-')[0]} & {NAMES.get(m, m)} & "
                    f"{v.replace('_', ' ')} & {r['config']['seed']} & "
                    f"{r['done']['best']['round']} & {h['accuracy']*100:.2f} & "
                    f"{h['macro_f1']:.4f} & {h['benign_far']*100:.2f} & "
                    f"{h['attack_miss_rate']*100:.2f} & {u['macro_f1']:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out, "supp_perseed.tex"), "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    groups_by_ds, dims, numbers = {}, {}, {}
    for ds in ("cic", "nfton"):
        runs = load(ds)
        if not runs:
            continue
        g = group(runs)
        groups_by_ds[ds] = g
        X = np.load(("./data/" if ds == "cic" else "./data_nfton/")
                    + "X_test.npy", mmap_mode="r")
        C = len(np.load(("./data/" if ds == "cic" else "./data_nfton/")
                        + "class_names.npy", allow_pickle=True))
        dims[ds] = (X.shape[1], C)
        for (m, v), rs in g.items():
            for split in ("holdout", "holdout_nonoverlap", "dev_test"):
                for met in ("accuracy", "macro_f1", "mcc", "benign_far",
                            "attack_miss_rate", "macro_fpr", "roc_auc_macro_ovr"):
                    vals = [r[f"best/{split}"].get(met) for r in rs]
                    if None not in vals:
                        numbers[f"{ds}/{m}/{v}/{split}/{met}"] = {
                            "values": vals, "mean": float(np.mean(vals)),
                            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                            "seeds": [r["config"]["seed"] for r in rs]}
            numbers[f"{ds}/{m}/{v}/best_rounds"] = [r["done"]["best"]["round"] for r in rs]
            numbers[f"{ds}/{m}/{v}/zero_recall"] = [r["best/holdout"]["zero_recall_classes"] for r in rs]
            flips = [[h["sign_flip_rate"] for h in r["history"]] for r in rs]
            if flips and flips[0][0] is not None:
                numbers[f"{ds}/{m}/{v}/sign_flips"] = flips
        tex_controls(g, a.out, ds)
        tex_variants(g, a.out, ds, ["base", "dir0.5", "dir0.1"],
                     f"tab_noniid_{ds}.tex", ["IID", "$\\alpha=0.5$", "$\\alpha=0.1$"],
                     metrics=(("accuracy", "Acc.", 100.0, 2),
                              ("macro_f1", "Macro F1", 1.0, 3),
                              ("attack_miss_rate", "Miss", 100.0, 2)))
        tex_variants(g, a.out, ds, ["base", "cw-sqrt"],
                     f"tab_classweight_{ds}.tex", ["Unweighted", "Weighted"])
        tex_nonoverlap(g, a.out, ds)
    tex_detection(groups_by_ds, a.out)
    tex_perseed(groups_by_ds, a.out)
    tex_readiness(groups_by_ds, a.out)
    tex_stability(groups_by_ds, a.out)
    tex_int8(groups_by_ds, a.out)
    tex_partition(groups_by_ds, a.out)
    numbers.update(tex_cost(groups_by_ds, a.out, dims))
    json.dump(numbers, open(os.path.join(a.out, "numbers_v2.json"), "w"),
              indent=1, default=float)
    print("tables written to", a.out)


if __name__ == "__main__":
    main()

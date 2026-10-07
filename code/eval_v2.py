"""
Final evaluation for protocol v2. Run only after training has ended.

For each finished run in runs_v2/<dataset>/ this scores the
validation-selected checkpoint (best.pt) and the last-round checkpoint
(final.pt, no selection) on

  holdout              data_v2/<dataset>/X_holdout.npy, never used in
                       development
  holdout_nonoverlap   holdout rows whose processed features and label
                       do not occur in the training pool
  dev_test             the v1 test file (exploratory: v1 picked
                       checkpoints and the round budget on it)
  dev_test_nonoverlap  the same, without rows that occur in training

and writes results.json inside the run directory. A summary across
seeds (mean, sample SD and the per-seed values) is then written to
runs_v2/<dataset>/summary.json and summary.csv.

Usage: python code/eval_v2.py --dataset nfton
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from codec import encode
from edgepop_ops import edgepop_frozen_keys
from fl_v2 import DATA_DIRS, OUT_ROOT, device
from metrics_v2 import headline, security_report
from models import MODEL_REGISTRY


def row_hash(x, y):
    h = np.empty(len(x), dtype=np.uint64)
    for s in range(0, len(x), 200_000):
        f = pd.DataFrame(np.asarray(x[s:s + 200_000]))
        f["label"] = np.asarray(y[s:s + 200_000])
        h[s:s + len(f)] = pd.util.hash_pandas_object(f, index=False).to_numpy()
    return h


def eval_sets(ds):
    """Evaluation arrays plus their training-overlap masks (cached)."""
    ddir, vdir = DATA_DIRS[ds], f"./data_v2/{ds}/"
    cache = os.path.join(vdir, "overlap_masks.npz")
    Xtr = np.load(ddir + "X_train.npy", mmap_mode="r")
    sets = {"holdout": (np.load(vdir + "X_holdout.npy"),
                        np.load(vdir + "y_holdout.npy")),
            "dev_test": (np.load(ddir + "X_test.npy"),
                         np.load(ddir + "y_test.npy"))}
    if os.path.exists(cache):
        masks = dict(np.load(cache))
    else:
        htr = row_hash(Xtr, np.load(ddir + "y_train.npy", mmap_mode="r"))
        masks = {}
        for k, (X, y) in sets.items():
            assert X.dtype == Xtr.dtype, (k, X.dtype, Xtr.dtype)
            masks[k] = np.isin(row_hash(X, y), htr)
        np.savez_compressed(cache, **masks)
    return sets, masks


@torch.no_grad()
def predict(model, X, bs=16384):
    model.eval()
    preds, probs = [], []
    for s in range(0, len(X), bs):
        z = model(torch.from_numpy(np.asarray(X[s:s + bs], np.float32))
                  .to(device))
        p = torch.softmax(z, 1)
        preds.append(p.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(preds), np.concatenate(probs)


def evaluate_run(rdir, sets, masks, names):
    cfg = json.load(open(os.path.join(rdir, "config.json")))
    c = cfg["config"]
    Model = MODEL_REGISTRY[c["model"]]
    d = sets["holdout"][0].shape[1]
    out = {"run_id": cfg["run_id"], "config": c,
           "done": json.load(open(os.path.join(rdir, "done.json")))}
    for ck in ("best", "final"):
        m = Model(d, len(names)).to(device)
        m.load_state_dict(torch.load(os.path.join(rdir, f"{ck}.pt"),
                                     map_location=device))
        excl = edgepop_frozen_keys(m)
        out[f"{ck}_down_bytes"] = len(encode(m, m, "down", excl))
        out[f"{ck}_up_bytes"] = len(encode(m, m, "up", excl))
        for k, (X, y) in sets.items():
            p, pr = predict(m, X)
            out[f"{ck}/{k}"] = security_report(y, p, names, pr)
            keep = ~masks[k]
            out[f"{ck}/{k}_nonoverlap"] = security_report(
                y[keep], p[keep], names, pr[keep])
    json.dump(out, open(os.path.join(rdir, "results.json"), "w"), indent=1)
    return out


GROUP_KEYS = ["model", "partition", "alpha", "aggregator", "mu",
              "class_weight", "init_broadcast", "lr", "lr_decay", "rounds",
              "epochs", "batch_size", "clients"]


def summarize(ds, results):
    groups = defaultdict(list)
    for r in results:
        c = r["config"]
        groups[tuple(c.get(k) for k in GROUP_KEYS)].append(r)
    rows, full = [], []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rs = sorted(rs, key=lambda r: r["config"]["seed"])
        g = dict(zip(GROUP_KEYS, key))
        row = {**g, "seeds": [r["config"]["seed"] for r in rs],
               "best_rounds": [r["done"]["best"]["round"] for r in rs],
               "down_KiB": rs[0]["best_down_bytes"] / 1024,
               "up_KiB": rs[0]["best_up_bytes"] / 1024}
        for ck in ("best", "final"):
            for split in ("holdout", "holdout_nonoverlap", "dev_test",
                          "dev_test_nonoverlap"):
                for met in ("accuracy", "macro_f1", "mcc", "benign_far",
                            "attack_miss_rate", "macro_fpr",
                            "roc_auc_macro_ovr"):
                    vals = [r[f"{ck}/{split}"].get(met) for r in rs]
                    if any(v is None for v in vals):
                        continue
                    a = np.array(vals, float)
                    pre = f"{ck}/{split}/{met}"
                    row[pre + "/mean"] = float(a.mean())
                    row[pre + "/sd"] = float(a.std(ddof=1)) if len(a) > 1 else None
                    row[pre + "/values"] = [float(v) for v in a]
        rows.append(row)
        full.append({**row, "per_class_best_holdout": {
            r["config"]["seed"]: r["best/holdout"]["per_class"] for r in rs}})
    base = os.path.join(OUT_ROOT, ds)
    json.dump(full, open(os.path.join(base, "summary.json"), "w"), indent=1)
    flat = [{k: (json.dumps(v) if isinstance(v, list) else v)
             for k, v in r.items()} for r in rows]
    pd.DataFrame(flat).to_csv(os.path.join(base, "summary.csv"), index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATA_DIRS))
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    names = list(np.load(DATA_DIRS[a.dataset] + "class_names.npy",
                         allow_pickle=True))
    sets, masks = eval_sets(a.dataset)
    for k in sets:
        print(f"{k}: {len(sets[k][1]):,} rows, "
              f"{masks[k].mean()*100:.2f}% also in training")
    results = []
    for rdir in sorted(glob.glob(os.path.join(OUT_ROOT, a.dataset, "*", ""))):
        if not os.path.exists(os.path.join(rdir, "done.json")):
            continue
        res_path = os.path.join(rdir, "results.json")
        if os.path.exists(res_path) and not a.redo:
            r = json.load(open(res_path))
        else:
            r = evaluate_run(rdir, sets, masks, names)
        h = headline(r["best/holdout"])
        print(f"{r['run_id']}: holdout acc={h['accuracy']*100:.2f} "
              f"F1={h['macro_f1']:.4f} FAR={h['benign_far']*100:.2f} "
              f"miss={h['attack_miss_rate']*100:.2f}", flush=True)
        results.append(r)
    if results:
        summarize(a.dataset, results)
        print("summary written")


if __name__ == "__main__":
    main()

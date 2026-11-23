"""
Re-check of the CICIoT2023 training-overlap masks (17 Nov 2026).

eval_v2.py marks a holdout row as "seen" when an exact bit hash of its
processed features and label occurs in the training pool. The v1
training rows were scaled with sklearn MinMaxScaler (x * scale_ + min_),
while prepare_v2.py scales the holdout as (x - lo) / span. The two
formulas agree to about 1e-16 but not bit for bit, so exact hashing can
miss true duplicates. Here the features are rounded before hashing.

Writes data_v2/<ds>/overlap_masks_rounded.npz (the original cache is
left untouched) and prints the overlap share for each rounding.

Usage (from the repository root): python code/overlap_recheck_v2.py --dataset cic
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

DATA_DIRS = {"cic": "./data/", "nfton": "./data_nfton/"}


def row_hash(x, y, decimals=None):
    h = np.empty(len(x), dtype=np.uint64)
    for s in range(0, len(x), 200_000):
        a = np.asarray(x[s:s + 200_000], dtype=np.float64)
        if decimals is not None:
            a = np.round(a, decimals) + 0.0   # + 0.0 folds -0.0 into 0.0
        f = pd.DataFrame(a)
        f["label"] = np.asarray(y[s:s + 200_000])
        h[s:s + len(f)] = pd.util.hash_pandas_object(f, index=False).to_numpy()
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cic", choices=["cic", "nfton"])
    ap.add_argument("--decimals", type=int, nargs="+", default=[12, 10, 8, 6])
    a = ap.parse_args()
    ddir, vdir = DATA_DIRS[a.dataset], f"./data_v2/{a.dataset}/"
    Xtr = np.load(ddir + "X_train.npy", mmap_mode="r")
    ytr = np.load(ddir + "y_train.npy", mmap_mode="r")
    sets = {"holdout": (np.load(vdir + "X_holdout.npy", mmap_mode="r"),
                        np.load(vdir + "y_holdout.npy")),
            "dev_test": (np.load(ddir + "X_test.npy", mmap_mode="r"),
                         np.load(ddir + "y_test.npy"))}
    old = dict(np.load(vdir + "overlap_masks.npz"))
    report = {"old_exact_cache": {k: float(v.mean()) for k, v in old.items()}}
    masks = {}
    for d in [None] + a.decimals:
        htr = np.unique(row_hash(Xtr, ytr, d))
        res = {}
        for k, (X, y) in sets.items():
            m = np.isin(row_hash(X, y, d), htr)
            res[k] = {"overlap": float(m.mean()), "rows": int(len(m)),
                      "seen_rows": int(m.sum())}
            masks[f"{k}_dec{d}"] = m
        report[f"decimals={d}"] = res
        print(d, json.dumps(res))
    np.savez_compressed(os.path.join(vdir, "overlap_masks_rounded.npz"), **masks)
    with open(os.path.join(vdir, "overlap_recheck.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

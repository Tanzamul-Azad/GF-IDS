"""
Exact CICIoT2023 training-overlap masks computed on the raw CSV values
(17 Nov 2026). See overlap_recheck_v2.py for why the cached masks in
data_v2/cic/overlap_masks.npz are too low: they hash scaled features,
and the holdout was scaled with a formula that differs from the v1
MinMaxScaler in the last bits.

Here a row is "seen" when its 31 kept raw feature values and its label
occur in the cleaned training file (NaN/inf rows removed, as in v1).
The holdout and the development test file are cleaned exactly as in
prepare_v2.py and preprocessing.py, so the masks align row by row with
X_holdout.npy and X_test.npy; the script checks that alignment.

Writes data_v2/cic/overlap_masks_raw.npz and overlap_raw.json. The old
cache is left untouched.

Usage (from the repository root):
    python code/overlap_raw_cic_v2.py --raw ../CICIOT23
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from prepare_v2 import CIC_FEATURES


def clean(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def key_hash(df):
    f = df[CIC_FEATURES + ["label"]].reset_index(drop=True)
    return pd.util.hash_pandas_object(f, index=False).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../CICIOT23")
    a = ap.parse_args()
    t = json.load(open("./data_v2/cic/transform.json"))
    lo, hi = np.array(t["min"]), np.array(t["max"])
    span = np.where(hi - lo == 0, 1.0, hi - lo)

    # Training pool: hashes of every cleaned training row, plus a count of
    # the rows that survive full-row deduplication (should equal v1).
    tr_keys, full = [], []
    for ch in pd.read_csv(os.path.join(a.raw, "train", "train.csv"),
                          chunksize=500_000):
        ch = clean(ch)
        tr_keys.append(key_hash(ch))
        full.append(pd.util.hash_pandas_object(ch, index=False).to_numpy())
    tr_keys = np.unique(np.concatenate(tr_keys))
    n_train_dedup = int(len(np.unique(np.concatenate(full))))
    v1_train = np.load("./data/X_train.npy", mmap_mode="r").shape[0]
    print(f"train rows after full dedup {n_train_dedup:,} (v1 {v1_train:,})")

    out, report = {}, {"train_rows_after_dedup": n_train_dedup,
                       "v1_train_rows": int(v1_train),
                       "unique_train_keys": int(len(tr_keys))}
    for name, csv, xpath in [
            ("holdout", os.path.join(a.raw, "validation", "validation.csv"),
             "./data_v2/cic/X_holdout.npy"),
            ("dev_test", os.path.join(a.raw, "test", "test.csv"),
             "./data/X_test.npy")]:
        df = clean(pd.read_csv(csv)).drop_duplicates()
        X = np.load(xpath, mmap_mode="r")
        if len(df) != len(X):
            raise SystemExit(f"{name}: {len(df):,} cleaned rows but "
                             f"{len(X):,} in {xpath}")
        idx = np.linspace(0, len(X) - 1, 20_000).astype(int)
        err = float(np.abs((df[CIC_FEATURES].values[idx] - lo) / span
                           - X[idx]).max())
        if err > 1e-9:
            raise SystemExit(f"{name}: rows do not align (max diff {err})")
        m = np.isin(key_hash(df), tr_keys)
        out[name] = m
        report[name] = {"rows": int(len(m)), "seen_rows": int(m.sum()),
                        "overlap": float(m.mean()), "align_max_diff": err}
        print(name, report[name])
    np.savez_compressed("./data_v2/cic/overlap_masks_raw.npz", **out)
    json.dump(report, open("./data_v2/cic/overlap_raw.json", "w"), indent=1)


if __name__ == "__main__":
    main()

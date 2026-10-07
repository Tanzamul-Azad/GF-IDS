"""
Holdout preparation for the corrected (v2) protocol.

v1 used one test file both to pick checkpoints and to report results.
v2 keeps the v1 training arrays unchanged and adds a final holdout
that played no part in any development decision:

  CICIoT2023     the release's validation.csv, which no v1 script read
  NF-ToN-IoT-v2  a fresh 12% sample of the test CSV rows that the v1
                 sampler did not draw

The v1 preprocessing never saved its fitted transform, so it is
rebuilt here from the raw training file and checked against the v1
arrays before it is applied. The rebuilt transform (feature order,
min/max bounds, label order) is written to transform.json next to the
holdout arrays, so every later run applies the same mapping.

Usage (from the repository root):
    python code/prepare_v2.py --dataset cic
    python code/prepare_v2.py --dataset nfton
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

RAW = {
    "cic": r"F:\UIU\11th\green\CICIOT23",
    "nfton": r"F:\UIU\11th\green\datasets\NF-ToN-IoT-v2",
}
V1_DIR = {"cic": "./data/", "nfton": "./data_nfton/"}
OUT_DIR = {"cic": "./data_v2/cic/", "nfton": "./data_v2/nfton/"}

# The 31 CICIoT2023 features kept by v1 (Table II of the manuscript),
# in CSV column order.
CIC_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate",
    "Drate", "fin_flag_number", "syn_flag_number", "psh_flag_number",
    "ece_flag_number", "cwr_flag_number", "urg_count", "rst_count", "HTTP",
    "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP",
    "ARP", "ICMP", "IPv", "Tot sum", "Min", "IAT", "Covariance", "Variance"]

NF_DROP = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "Label"]


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def save(out, name, arr):
    np.save(os.path.join(out, name), arr)


# ─────────────────────────────────────────────────────────────
# CICIoT2023
# ─────────────────────────────────────────────────────────────
def prepare_cic():
    raw, out = RAW["cic"], OUT_DIR["cic"]
    os.makedirs(out, exist_ok=True)
    train_csv = os.path.join(raw, "train", "train.csv")
    hold_csv = os.path.join(raw, "validation", "validation.csv")
    classes = list(np.load(V1_DIR["cic"] + "class_names.npy",
                           allow_pickle=True))

    # 1. Min/max of the kept features over the rows v1 kept. Dropping
    # duplicates cannot change a min or max, so only the NaN/inf filter
    # matters here.
    lo = hi = None
    n_valid = 0
    for ch in pd.read_csv(train_csv, chunksize=500_000):
        ch = ch.replace([np.inf, -np.inf], np.nan).dropna()
        n_valid += len(ch)
        f = ch[CIC_FEATURES]
        lo = f.min() if lo is None else np.minimum(lo, f.min())
        hi = f.max() if hi is None else np.maximum(hi, f.max())
    lo, hi = lo.values.astype(np.float64), hi.values.astype(np.float64)
    span = np.where(hi - lo == 0, 1.0, hi - lo)   # MinMaxScaler convention

    # 2. Check against the v1 arrays on the first 300k raw rows. v1
    # dropped NaN/inf rows, then exact duplicates (keeping the first),
    # so the v1 rows are the surviving raw rows in file order.
    head = pd.read_csv(train_csv, nrows=300_000)
    head = head.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
    Xh = (head[CIC_FEATURES].values - lo) / span
    v1 = np.load(V1_DIR["cic"] + "X_train.npy", mmap_mode="r")
    n = min(len(Xh), 250_000)   # stay clear of duplicates resolved later
    err = float(np.abs(Xh[:n] - v1[:n]).max())
    print(f"rebuilt transform vs v1 X_train, first {n:,} rows: "
          f"max abs diff {err:.3e}")
    if err > 1e-9:
        raise SystemExit("transform does not reproduce v1; stopping")

    # 3. Holdout: same cleaning as the v1 test file (NaN/inf rows and
    # exact duplicates removed), same features, same bounds, no clip.
    df = pd.read_csv(hold_csv)
    rows_raw = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    rows_valid = len(df)
    df = df.drop_duplicates()
    unknown = sorted(set(df["label"]) - set(classes))
    if unknown:
        raise SystemExit(f"holdout has unseen labels: {unknown}")
    X = (df[CIC_FEATURES].values.astype(np.float64) - lo) / span
    y = np.array([classes.index(c) for c in df["label"]], dtype=np.int64)
    save(out, "X_holdout.npy", X)
    save(out, "y_holdout.npy", y)

    meta = {
        "dataset": "CICIoT2023",
        "train_source": train_csv, "train_sha256": sha256(train_csv),
        "train_valid_rows": int(n_valid),
        "holdout_source": hold_csv, "holdout_sha256": sha256(hold_csv),
        "holdout_rows_raw": int(rows_raw),
        "holdout_rows_after_nan_inf": int(rows_valid),
        "holdout_rows_after_dedup": int(len(df)),
        "features": CIC_FEATURES, "min": lo.tolist(), "max": hi.tolist(),
        "clip_to_unit_interval": False,
        "classes": classes,
        "rebuild_check_max_abs_diff": err,
        "holdout_class_support": np.bincount(y, minlength=len(classes)).tolist(),
    }
    with open(os.path.join(out, "transform.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items()
                      if k not in ("min", "max", "features")}, indent=1))


# ─────────────────────────────────────────────────────────────
# NF-ToN-IoT-v2
# ─────────────────────────────────────────────────────────────
def v1_sample_index(path, seed, frac=0.12):
    """Row labels drawn by prepare_nfton.sample(), replayed exactly."""
    rng = np.random.RandomState(seed)
    picked = []
    for ch in pd.read_csv(path, chunksize=500_000, usecols=["Attack"]):
        picked.append(ch.sample(frac=frac,
                                random_state=rng.randint(1e9)).index.values)
    return np.concatenate(picked)


def prepare_nfton():
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    raw, out = RAW["nfton"], OUT_DIR["nfton"]
    os.makedirs(out, exist_ok=True)
    tr_csv = os.path.join(raw, "NF-ToN-IoT-v2-train.csv")
    te_csv = os.path.join(raw, "NF-ToN-IoT-v2-test.csv")

    # 1. Replay the v1 training sample and fit the same transform.
    rng = np.random.RandomState(42)
    parts = []
    for ch in pd.read_csv(tr_csv, chunksize=500_000):
        parts.append(ch.sample(frac=0.12, random_state=rng.randint(1e9)))
    tr = pd.concat(parts).drop(columns=NF_DROP).drop_duplicates()
    tr = tr.sample(n=min(1_000_000, len(tr)), random_state=42)
    enc = LabelEncoder().fit(tr["Attack"])
    classes = list(enc.classes_)
    Xtr = tr.drop(columns=["Attack"]).astype(np.float64)
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan).fillna(0)
    sc = MinMaxScaler().fit(Xtr)
    A = pd.DataFrame(sc.transform(Xtr), columns=Xtr.columns)
    corr = A.sample(min(len(A), 200_000), random_state=0).corr().abs() \
        .fillna(0).values
    keep = []
    for i in range(corr.shape[0]):
        if all(corr[i, j] <= 0.80 for j in keep):
            keep.append(i)
    cols = [A.columns[i] for i in keep]
    v1 = np.load(V1_DIR["nfton"] + "X_train.npy")
    rebuilt = A[cols].values.astype(np.float32)
    same = rebuilt.shape == v1.shape and np.array_equal(rebuilt, v1)
    print(f"rebuilt NF train equals v1 X_train: {same}")
    if not same:
        raise SystemExit("NF transform does not reproduce v1; stopping")

    # 2. Holdout from test-file rows the v1 sampler never drew.
    used = set(v1_sample_index(te_csv, 43).tolist())
    rng = np.random.RandomState(2026)
    parts, scanned = [], 0
    for ch in pd.read_csv(te_csv, chunksize=500_000):
        scanned += len(ch)
        ch = ch[~ch.index.isin(used)]
        parts.append(ch.sample(frac=0.12, random_state=rng.randint(1e9)))
    te = pd.concat(parts).drop(columns=NF_DROP).drop_duplicates()
    te = te[te["Attack"].isin(classes)]
    Xte = te.drop(columns=["Attack"]).astype(np.float64)
    Xte = Xte.replace([np.inf, -np.inf], np.nan).fillna(0)
    B = pd.DataFrame(np.clip(sc.transform(Xte), 0, 1), columns=Xte.columns)
    save(out, "X_holdout.npy", B[cols].values.astype(np.float32))
    y = enc.transform(te["Attack"]).astype(np.int64)
    save(out, "y_holdout.npy", y)

    meta = {
        "dataset": "NF-ToN-IoT-v2",
        "train_source": tr_csv, "train_sha256": sha256(tr_csv),
        "holdout_source": te_csv, "holdout_sha256": sha256(te_csv),
        "test_rows_scanned": int(scanned),
        "v1_test_rows_excluded": int(len(used)),
        "holdout_sampling": "12% of each 500k-row chunk of unused test rows, "
                            "RandomState(2026), then exact-duplicate removal",
        "holdout_rows": int(len(y)),
        "features": cols, "min": sc.data_min_[keep].tolist(),
        "max": sc.data_max_[keep].tolist(), "clip_to_unit_interval": True,
        "classes": classes,
        "holdout_class_support": np.bincount(y, minlength=len(classes)).tolist(),
    }
    with open(os.path.join(out, "transform.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: v for k, v in meta.items()
                      if k not in ("min", "max", "features")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cic", "nfton"])
    {"cic": prepare_cic, "nfton": prepare_nfton}[ap.parse_args().dataset]()

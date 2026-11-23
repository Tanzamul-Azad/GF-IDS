"""
Re-score the CICIoT2023 "unseen rows" subset with the exact raw-value
overlap mask from overlap_raw_cic_v2.py (17 Nov 2026).

For every finished run in runs_v2/cic/ the validation-selected (best)
and last-round (final) checkpoints are scored on the holdout. The
all-rows score is compared with the stored results.json as a check, and
the unseen-rows score is written to results_unseen_raw.json in the run
directory. results.json is not modified.

Usage (from the repository root): python code/rescore_unseen_cic_v2.py
"""

import glob
import json
import os

import numpy as np
import torch

from eval_v2 import predict
from fl_v2 import device
from metrics_v2 import security_report
from models import MODEL_REGISTRY


def main():
    X = np.load("./data_v2/cic/X_holdout.npy")
    y = np.load("./data_v2/cic/y_holdout.npy")
    seen = np.load("./data_v2/cic/overlap_masks_raw.npz")["holdout"]
    keep = ~seen
    names = list(np.load("./data/class_names.npy", allow_pickle=True))
    print(f"holdout {len(y):,} rows, unseen {keep.sum():,}")
    rows = []
    for rdir in sorted(glob.glob("./runs_v2/cic/*/")):
        if not os.path.exists(os.path.join(rdir, "results.json")):
            continue
        old = json.load(open(os.path.join(rdir, "results.json")))
        c = old["config"]
        out = {"run_id": old["run_id"], "mask": "overlap_masks_raw.npz",
               "unseen_rows": int(keep.sum())}
        for ck in ("best", "final"):
            m = MODEL_REGISTRY[c["model"]](X.shape[1], len(names)).to(device)
            m.load_state_dict(torch.load(os.path.join(rdir, f"{ck}.pt"),
                                         map_location=device))
            p, pr = predict(m, X)
            acc_all = float((p == y).mean())
            d = abs(acc_all - old[f"{ck}/holdout"]["accuracy"])
            if d > 1e-6:
                raise SystemExit(f"{rdir} {ck}: all-rows accuracy differs by {d}")
            out[f"{ck}/holdout_nonoverlap"] = security_report(
                y[keep], p[keep], names, pr[keep])
            out[f"{ck}/holdout_nonoverlap_old"] = {
                k: old[f"{ck}/holdout_nonoverlap"].get(k)
                for k in ("accuracy", "macro_f1", "benign_far",
                          "attack_miss_rate")}
        json.dump(out, open(os.path.join(rdir, "results_unseen_raw.json"), "w"),
                  indent=1, default=float)
        n, o = out["best/holdout_nonoverlap"], out["best/holdout_nonoverlap_old"]
        rows.append((old["run_id"], o["accuracy"], n["accuracy"],
                     o["macro_f1"], n["macro_f1"]))
        print(f"{old['run_id']:45s} acc {o['accuracy']*100:6.2f} -> "
              f"{n['accuracy']*100:6.2f}  F1 {o['macro_f1']:.4f} -> "
              f"{n['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()

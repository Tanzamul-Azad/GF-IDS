"""Cache test-set predictions of MLP-FL and BNN-FL (seed 42, best round).

The per-class F1, ROC and misclassification figures all need the same
predictions on the 1,170,467-sample global test set. Computing them once
and saving them lets the figure script run in seconds.
Run from the repository root:  python code/cache_predictions.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import figures as F  # noqa: E402

X = np.load("data/X_test.npy")
y = np.load("data/y_test.npy")
out = {"labels": y}
for name, key in (("MLP", "mlp"), ("BNN-MATCHED", "bnn")):
    F.RUN_DIR = "./runs/"
    model = F.load_model(name, X.shape[1], int(y.max()) + 1)
    preds, labels, probs = F.predict(model, X, y, return_probs=True)
    assert (labels == y).all()
    out[f"pred_{key}"] = preds.astype(np.int16)
    out[f"prob_{key}"] = probs.astype(np.float32)
    print(name, "accuracy", round(float((preds == y).mean()) * 100, 2))
np.savez_compressed("runs/test_predictions_seed42.npz", **out)
print("saved runs/test_predictions_seed42.npz")

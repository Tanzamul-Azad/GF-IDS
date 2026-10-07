"""
Detection metrics for the corrected (v2) evaluation protocol.

The v1 code reported one "FPR", the mean of the C one-vs-rest false
positive rates. That is a multiclass macro FPR. It is not the false
alarm rate an IDS operator sees, which counts benign flows flagged as
any attack. Both are reported here under separate names:

  macro_fpr         mean over classes of FP_c / (FP_c + TN_c)
  benign_far        benign rows predicted as any attack / benign rows
  attack_miss_rate  attack rows predicted as benign / attack rows
  attack_recall     1 - attack_miss_rate (attack-vs-benign detection)

ROC AUC is given with its averaging named. "macro" averages the
per-class one-vs-rest AUCs. "micro" flattens the indicator matrix,
which is what the v1 figure script did, and is dominated by the
large classes.
"""

import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)


def benign_index(class_names):
    """Index of the benign class, found by name so the label order of
    each dataset does not have to be remembered."""
    hits = [i for i, n in enumerate(class_names)
            if str(n).lower().startswith("benign")]
    if len(hits) != 1:
        raise ValueError(f"expected one benign class, found {hits}")
    return hits[0]


def security_report(labels, preds, class_names, probs=None):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    C = len(class_names)
    b = benign_index(class_names)
    cm = confusion_matrix(labels, preds, labels=np.arange(C))

    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    with np.errstate(invalid="ignore", divide="ignore"):
        fpr_c = np.where(FP + TN > 0, FP / (FP + TN), 0.0)

    is_benign = labels == b
    pred_benign = preds == b
    n_benign = int(is_benign.sum())
    n_attack = int((~is_benign).sum())
    benign_far = float((is_benign & ~pred_benign).sum() / max(n_benign, 1))
    attack_miss = float((~is_benign & pred_benign).sum() / max(n_attack, 1))

    prec, rec, f1, sup = precision_recall_fscore_support(
        labels, preds, labels=np.arange(C), zero_division=0)

    out = {
        "n": int(len(labels)),
        "accuracy": float((labels == preds).mean()),
        "macro_f1": float(f1_score(labels, preds, labels=np.arange(C),
                                   average="macro", zero_division=0)),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "macro_fpr": float(fpr_c.mean()),
        "benign_far": benign_far,
        "attack_miss_rate": attack_miss,
        "attack_recall": 1.0 - attack_miss,
        "n_benign": n_benign,
        "n_attack": n_attack,
        "zero_recall_classes": [str(class_names[i]) for i in range(C)
                                if sup[i] > 0 and rec[i] == 0],
        "per_class": {str(class_names[i]): {
            "support": int(sup[i]), "precision": float(prec[i]),
            "recall": float(rec[i]), "f1": float(f1[i]),
            "predicted_benign_share": float(cm[i, b] / sup[i]) if sup[i] else 0.0}
            for i in range(C)},
    }

    if probs is not None:
        present = np.unique(labels)
        if len(present) == C:
            out["roc_auc_macro_ovr"] = float(roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"))
        else:
            # sklearn needs every class present for multiclass macro AUC.
            # Average only over the classes that appear, and say so.
            aucs = []
            for c in present:
                aucs.append(roc_auc_score(labels == c, probs[:, c]))
            out["roc_auc_macro_ovr_present_classes"] = float(np.mean(aucs))
        onehot = np.eye(C, dtype=np.uint8)[labels]
        out["roc_auc_micro"] = float(roc_auc_score(onehot.ravel(),
                                                   probs.ravel()))
    return out


def headline(r):
    """The flat numbers used in tables, without the per-class detail."""
    keys = ["n", "accuracy", "macro_f1", "macro_precision", "macro_recall",
            "mcc", "macro_fpr", "benign_far", "attack_miss_rate",
            "attack_recall", "roc_auc_macro_ovr", "roc_auc_micro"]
    return {k: r[k] for k in keys if k in r}

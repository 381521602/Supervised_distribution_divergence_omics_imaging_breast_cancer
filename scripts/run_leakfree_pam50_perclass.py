# -*- coding: utf-8 -*-
"""Leak-free mRNA PAM50 per-class metrics and confusion matrices.

Reuses the exact fold-internal pipeline of run_leakfree_single_omics.py and
saves fold-level predictions so per-class precision/recall/F1 and the
aggregated confusion matrix can be computed.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_leakfree_single_omics import (  # noqa: E402
    RANDOM_STATE, K_FOLDS, PAM50_K, PAM50_SIZE,
    build_adaptive_pipeline, jsd_scores, make_images, load_aligned,
    ml_models, train_torch, MLP, FullSizeCNN, Dense,
)

OUT_DIR = ROOT / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    omics = "mRNA"
    X, lab = load_aligned(omics)
    m = lab["pam50_4class"].isin(["Luminal A", "Luminal B", "Basal-like", "HER2-enriched"])
    X = X[m.to_numpy()]
    lab = lab.loc[m].reset_index(drop=True)
    enc = LabelEncoder()
    y = enc.fit_transform(lab["pam50_4class"].values)
    class_names = list(enc.classes_)
    k = PAM50_K[omics]
    size = PAM50_SIZE[omics]
    model_names = ["LogisticRegression", "RandomForest", "GradientBoosting", "SVC", "KNN",
                   "MLP", "FullSizeCNN", "Dense"]
    ytrue = {n: [] for n in model_names}
    ypred = {n: [] for n in model_names}
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        pipe = build_adaptive_pipeline(omics, "PAM50_4class", k=k).fit(X[tr], y[tr])
        Xtr = pipe.transform(X[tr])
        Xte = pipe.transform(X[te])
        order = np.argsort(jsd_scores(Xtr, y[tr]))[::-1]
        imgs_tr = make_images(Xtr[:, order], np.arange(Xtr.shape[1]), size)
        imgs_te = make_images(Xte[:, order], np.arange(Xte.shape[1]), size)
        for name, clf in ml_models().items():
            pred = clf.fit(Xtr, y[tr]).predict(Xte)
            ytrue[name].extend(y[te].tolist())
            ypred[name].extend(pred.tolist())
        p = train_torch(MLP(Xtr.shape[1], 4), Xtr, y[tr], Xte, binary=False).argmax(1)
        ytrue["MLP"].extend(y[te].tolist()); ypred["MLP"].extend(p.tolist())
        p = train_torch(FullSizeCNN(size, 4), imgs_tr, y[tr], imgs_te, binary=False).argmax(1)
        ytrue["FullSizeCNN"].extend(y[te].tolist()); ypred["FullSizeCNN"].extend(p.tolist())
        p = train_torch(Dense(Xtr.shape[1], 4), Xtr, y[tr], Xte, binary=False).argmax(1)
        ytrue["Dense"].extend(y[te].tolist()); ypred["Dense"].extend(p.tolist())
        print(f"fold {fold + 1}/{K_FOLDS} done (k={Xtr.shape[1]})", flush=True)

    metric_rows = []
    cm_rows = []
    for name in model_names:
        yt = np.asarray(ytrue[name])
        yp = np.asarray(ypred[name])
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, labels=np.arange(len(class_names)), zero_division=0)
        for ci, cls in enumerate(class_names):
            metric_rows.append({"model": name, "class": cls,
                                "precision": round(float(p[ci]), 4),
                                "recall": round(float(r[ci]), 4),
                                "f1": round(float(f1[ci]), 4)})
        macro = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        weighted = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)
        metric_rows.append({"model": name, "class": "__macro__",
                            "precision": round(float(macro[0]), 4),
                            "recall": round(float(macro[1]), 4),
                            "f1": round(float(macro[2]), 4)})
        metric_rows.append({"model": name, "class": "__weighted__",
                            "precision": round(float(weighted[0]), 4),
                            "recall": round(float(weighted[1]), 4),
                            "f1": round(float(weighted[2]), 4)})
        cm = confusion_matrix(yt, yp, labels=np.arange(len(class_names)))
        for ri, cls in enumerate(class_names):
            cm_rows.append({"model": name, "true_class": cls, **{c: int(cm[ri, ci]) for ci, c in enumerate(class_names)}})

    with (OUT_DIR / "pam50_mrna_perclass_metrics.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "class", "precision", "recall", "f1"], delimiter="\t")
        w.writeheader(); w.writerows(metric_rows)
    with (OUT_DIR / "pam50_mrna_confusion_matrices.tsv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "true_class"] + class_names, delimiter="\t")
        w.writeheader(); w.writerows(cm_rows)

    print("\nPer-class F1 (rows = model, cols = class):")
    for name in model_names:
        f1s = [r["f1"] for r in metric_rows if r["model"] == name and not r["class"].startswith("__")]
        print(f"{name:20s} " + "  ".join(f"{v:.3f}" for v in f1s))
    print("\nDone ->", OUT_DIR)


if __name__ == "__main__":
    main()

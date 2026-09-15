#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 2: PAM50 MLP grid."""

import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_batch1_pam50_grid import align, make_selector, FEATURE_METHODS, BASE_K, MULTIPLIERS, OMICS, RANDOM_STATE, K_FOLDS

OUT = ROOT / "data" / "batch2_pam50_mlp_results.tsv"
EPOCHS = 15
BATCH_SIZE = 64

class MLP(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,dout))
    def forward(self,x): return self.net(x)

def main():
    labels = pd.read_csv(ROOT/"data/brca_labels_modeling_ready.tsv", sep="\t", dtype={"case_id":str})
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=["omics","multiplier","k","feature_method","model","metric","mean","std"], delimiter="\t"); w.writeheader()
    for omics in OMICS:
        X, y_raw = align(omics)
        enc = LabelEncoder(); y = enc.fit_transform(y_raw)
        for mult in MULTIPLIERS:
            k = BASE_K[omics]*mult
            for fs in FEATURE_METHODS:
                cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                accs, f1s = [], []
                for tr, te in cv.split(np.zeros(len(y)), y):
                    sel = make_selector(fs,k); sel.fit(X[tr],y[tr])
                    Xtr = StandardScaler().fit_transform(sel.transform(X[tr]))
                    Xte = StandardScaler().fit(sel.transform(X[tr])).transform(sel.transform(X[te]))
                    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
                    model=MLP(Xtr.shape[1], len(enc.classes_))
                    opt=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); crit=nn.CrossEntropyLoss()
                    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(y[tr],dtype=torch.long)
                    n=Xt.shape[0]
                    for _ in range(EPOCHS):
                        perm=torch.randperm(n)
                        for i in range(0,n,BATCH_SIZE):
                            idx=perm[i:i+BATCH_SIZE]
                            if idx.shape[0]<2: continue
                            opt.zero_grad(); loss=crit(model(Xt[idx]),yt[idx]); loss.backward(); opt.step()
                    model.eval()
                    with torch.no_grad(): pred=model(torch.tensor(Xte,dtype=torch.float32)).argmax(dim=1).numpy()
                    accs.append(accuracy_score(y[te],pred)); f1s.append(f1_score(y[te],pred,average="macro"))
                with OUT.open("a", encoding="utf-8", newline="") as f:
                    w=csv.DictWriter(f, fieldnames=["omics","multiplier","k","feature_method","model","metric","mean","std"], delimiter="\t")
                    w.writerow({"omics":omics,"multiplier":mult,"k":k,"feature_method":fs,"model":"MLP","metric":"accuracy","mean":round(float(np.mean(accs)),4),"std":round(float(np.std(accs)),4)})
                    w.writerow({"omics":omics,"multiplier":mult,"k":k,"feature_method":fs,"model":"MLP","metric":"macro_f1","mean":round(float(np.mean(f1s)),4),"std":round(float(np.std(f1s)),4)})
                print(f"{omics} mult={mult} {fs} MLP done")
    print(f"Done -> {OUT}")

if __name__=="__main__":
    main()

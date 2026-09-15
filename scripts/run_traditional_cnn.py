#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train traditional CNN on NSRE-ordered images."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT=Path(__file__).resolve().parent.parent
PAM_DIR=ROOT/"data/final_datasets/PAM50"
SUR_DIR=ROOT/"data/final_datasets/Survival"
IMG=ROOT/"data/images"
OUT=ROOT/"data/traditional_cnn_results.tsv"
RANDOM_STATE=42
K_FOLDS=5
EPOCHS=30
BATCH_SIZE=64
OMICS=["mRNA","CNV","miRNA"]


class CNN(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
        )
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,out_dim))
    def forward(self,x): return self.head(self.features(x))


def train(Xtr,ytr,Xte,yte,dout,binary=False):
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    model=CNN(dout); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    if binary:
        crit=nn.BCEWithLogitsLoss(); yt=torch.tensor(ytr,dtype=torch.float32).view(-1,1)
    else:
        crit=nn.CrossEntropyLoss(); yt=torch.tensor(ytr,dtype=torch.long)
    Xt=torch.tensor(Xtr,dtype=torch.float32)
    n=Xt.shape[0]
    for _ in range(EPOCHS):
        perm=torch.randperm(n)
        for i in range(0,n,BATCH_SIZE):
            idx=perm[i:i+BATCH_SIZE]
            if idx.shape[0]<2: continue
            opt.zero_grad(); loss=crit(model(Xt[idx]),yt[idx]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out=model(torch.tensor(Xte,dtype=torch.float32))
    if binary:
        return torch.sigmoid(out).numpy().ravel()
    return out.argmax(dim=1).numpy()


def eval_pam50():
    rows=[]
    for omics in OMICS:
        imgs=np.load(IMG/f"PAM50/{omics}/images.npy")
        df=pd.read_csv(PAM_DIR/f"{omics}_PAM50_final.tsv",sep="\t")
        y_raw=df["pam50"].values; enc=LabelEncoder(); y=enc.fit_transform(y_raw)
        cv=StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=RANDOM_STATE)
        accs,f1s=[],[]
        for tr,te in cv.split(np.zeros(len(y)),y):
            pred=train(imgs[tr],y[tr],imgs[te],y[te],len(enc.classes_))
            accs.append(accuracy_score(y[te],pred)); f1s.append(f1_score(y[te],pred,average="macro"))
        rows.append({"task":"PAM50","omics":omics,"model":"CNN","metric":"accuracy","mean":round(float(np.mean(accs)),4),"std":round(float(np.std(accs)),4)})
        rows.append({"task":"PAM50","omics":omics,"model":"CNN","metric":"macro_f1","mean":round(float(np.mean(f1s)),4),"std":round(float(np.std(f1s)),4)})
    return rows


def eval_survival():
    rows=[]
    for omics in OMICS:
        imgs=np.load(IMG/f"Survival/{omics}/images.npy")
        df=pd.read_csv(SUR_DIR/f"{omics}_Survival_final.tsv",sep="\t")
        y=df["os_event"].astype(int).values
        cv=StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=RANDOM_STATE)
        aucs=[]
        for tr,te in cv.split(np.zeros(len(y)),y):
            prob=train(imgs[tr],y[tr],imgs[te],y[te],1,binary=True)
            aucs.append(roc_auc_score(y[te],prob))
        rows.append({"task":"Survival","omics":omics,"model":"CNN","metric":"roc_auc","mean":round(float(np.mean(aucs)),4),"std":round(float(np.std(aucs)),4)})
    return rows


def main():
    rows=eval_pam50()+eval_survival()
    with OUT.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["task","omics","model","metric","mean","std"],delimiter="\t"); w.writeheader(); w.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__=="__main__":
    main()

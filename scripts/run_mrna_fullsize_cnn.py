#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-size CNN for mRNA single-omics images."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT=Path(__file__).resolve().parent.parent
PAM_DIR=ROOT/"data/final_datasets/PAM50"
SUR_DIR=ROOT/"data/final_datasets/Survival"
IMG=ROOT/"data/images"
OUT=ROOT/"data/mrna_fullsize_cnn_results.tsv"
RANDOM_STATE=42; K_FOLDS=5; EPOCHS=30; BATCH_SIZE=64


class FullSizeCNN(nn.Module):
    def __init__(self,size,out_dim):
        super().__init__()
        self.conv=nn.Conv2d(1,32,kernel_size=(size,size))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(32,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,out_dim))
    def forward(self,x):
        x=self.conv(x); x=F.relu(x); return self.head(x)


def train(Xtr,ytr,Xte,yte,size,dout,binary=False):
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    model=FullSizeCNN(size,dout); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    if binary:
        crit=nn.BCEWithLogitsLoss(); yt=torch.tensor(ytr,dtype=torch.float32).view(-1,1)
    else:
        crit=nn.CrossEntropyLoss(); yt=torch.tensor(ytr,dtype=torch.long)
    Xt=torch.tensor(Xtr,dtype=torch.float32); n=Xt.shape[0]
    for _ in range(EPOCHS):
        perm=torch.randperm(n)
        for i in range(0,n,BATCH_SIZE):
            idx=perm[i:i+BATCH_SIZE]
            if idx.shape[0]<2: continue
            opt.zero_grad(); loss=crit(model(Xt[idx]),yt[idx]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): out=model(torch.tensor(Xte,dtype=torch.float32))
    return torch.sigmoid(out).numpy().ravel() if binary else out.argmax(dim=1).numpy()


def eval_pam50():
    imgs=np.load(IMG/"PAM50/mRNA/images.npy"); df=pd.read_csv(PAM_DIR/"mRNA_PAM50_final.tsv",sep="\t")
    y_raw=df["pam50"].values; enc=LabelEncoder(); y=enc.fit_transform(y_raw)
    cv=StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=RANDOM_STATE)
    accs,f1s=[],[]
    for tr,te in cv.split(np.zeros(len(y)),y):
        pred=train(imgs[tr],y[tr],imgs[te],y[te],20,len(enc.classes_))
        accs.append(accuracy_score(y[te],pred)); f1s.append(f1_score(y[te],pred,average="macro"))
    return [
        {"task":"PAM50","model":"FullSizeCNN","metric":"accuracy","mean":round(float(np.mean(accs)),4),"std":round(float(np.std(accs)),4)},
        {"task":"PAM50","model":"FullSizeCNN","metric":"macro_f1","mean":round(float(np.mean(f1s)),4),"std":round(float(np.std(f1s)),4)},
    ]


def eval_survival():
    imgs=np.load(IMG/"Survival/mRNA/images.npy"); df=pd.read_csv(SUR_DIR/"mRNA_Survival_final.tsv",sep="\t")
    y_event=df["os_event"].astype(int).values; y_time=df["os_time_days"].astype(float).values
    cv=StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=RANDOM_STATE)
    aucs,cis=[],[]
    for tr,te in cv.split(np.zeros(len(y_event)),y_event):
        prob=train(imgs[tr],y_event[tr],imgs[te],y_event[te],15,1,binary=True)
        aucs.append(roc_auc_score(y_event[te],prob)); cis.append(concordance_index(y_time[te],-prob,y_event[te]))
    return [
        {"task":"Survival","model":"FullSizeCNN","metric":"roc_auc","mean":round(float(np.mean(aucs)),4),"std":round(float(np.std(aucs)),4)},
        {"task":"Survival","model":"FullSizeCNN","metric":"c_index","mean":round(float(np.mean(cis)),4),"std":round(float(np.std(cis)),4)},
    ]


def main():
    rows=eval_pam50()+eval_survival()
    with OUT.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["task","model","metric","mean","std"],delimiter="\t"); w.writeheader(); w.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__=="__main__":
    main()

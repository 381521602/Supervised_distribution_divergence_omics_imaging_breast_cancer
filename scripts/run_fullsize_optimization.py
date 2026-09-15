#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test FullSizeCNN optimization variants."""

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
OUT=ROOT/"data/fullsize_optimization_results.tsv"
RANDOM_STATE=42; K_FOLDS=5; EPOCHS=25; BATCH_SIZE=64


class VariantNet(nn.Module):
    def __init__(self,size,out_dim,variant,raw_dim=None):
        super().__init__(); self.variant=variant; self.channels=64 if variant in ["3_channels64","combined"] else 32
        self.conv=nn.Conv2d(1,self.channels,kernel_size=(size,size))
        self.gn=nn.GroupNorm(min(8,self.channels),self.channels)
        self.bn=nn.BatchNorm2d(self.channels)
        feat=self.channels
        if variant in ["1_raw_skip","6_residual"]:
            feat=self.channels + (raw_dim if variant=="1_raw_skip" else self.channels)
        self.head=nn.Sequential(nn.Linear(feat,64),nn.Mish() if variant in ["2_mish"] else nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,out_dim))
    def forward(self,x,raw=None):
        z=self.conv(x)
        z=self.gn(z) if self.variant in ["4_groupnorm","combined"] else self.bn(z)
        z=F.mish(z) if self.variant in ["2_mish","combined"] else F.relu(z)
        z=F.adaptive_avg_pool2d(z,(1,1)).flatten(1)
        if self.variant=="1_raw_skip":
            z=torch.cat([z,raw],dim=1)
        elif self.variant=="6_residual":
            z=torch.cat([z,raw],dim=1)
        return self.head(z)


def train(variant,Xtr,ytr,Xte,yte,size,dout,raw_tr=None,raw_te=None,binary=False):
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    model=VariantNet(size,dout,variant,None if raw_tr is None else raw_tr.shape[1])
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    if binary:
        crit=nn.BCEWithLogitsLoss(); yt=torch.tensor(ytr,dtype=torch.float32).view(-1,1)
    else:
        crit=nn.CrossEntropyLoss(); yt=torch.tensor(ytr,dtype=torch.long)
    Xt=torch.tensor(Xtr); n=Xt.shape[0]
    Rt=torch.tensor(raw_tr) if raw_tr is not None else None
    for _ in range(EPOCHS):
        perm=torch.randperm(n)
        for i in range(0,n,BATCH_SIZE):
            idx=perm[i:i+BATCH_SIZE]
            if idx.shape[0]<2: continue
            opt.zero_grad()
            raw_batch=Rt[idx] if Rt is not None else None
            loss=crit(model(Xt[idx],raw_batch),yt[idx]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out=model(torch.tensor(Xte), None if raw_te is None else torch.tensor(raw_te))
    return torch.sigmoid(out).numpy().ravel() if binary else out.argmax(dim=1).numpy()


def eval_variant(variant,task):
    size=20 if task=="PAM50" else 15
    imgs=np.load(IMG/f"{task}/mRNA/images.npy")
    if task=="PAM50":
        df=pd.read_csv(PAM_DIR/"mRNA_PAM50_final.tsv",sep="\t"); raw=df.drop(columns=["pam50"]).to_numpy(dtype=np.float32); y_raw=df["pam50"].values; enc=LabelEncoder(); y=enc.fit_transform(y_raw)
    else:
        df=pd.read_csv(SUR_DIR/"mRNA_Survival_final.tsv",sep="\t"); raw=df.drop(columns=["case_id","os_event","os_time_days"]).to_numpy(dtype=np.float32); y=df["os_event"].astype(int).values
    cv=StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=RANDOM_STATE)
    if task=="PAM50":
        accs,f1s=[],[]
        for tr,te in cv.split(np.zeros(len(y)),y):
            pred=train(variant,imgs[tr],y[tr],imgs[te],y[te],size,len(enc.classes_),raw[tr] if variant in ["1_raw_skip","6_residual"] else None,raw[te] if variant in ["1_raw_skip","6_residual"] else None)
            accs.append(accuracy_score(y[te],pred)); f1s.append(f1_score(y[te],pred,average="macro"))
        return [{"variant":variant,"task":"PAM50","metric":"accuracy","mean":round(float(np.mean(accs)),4),"std":round(float(np.std(accs)),4)},{"variant":variant,"task":"PAM50","metric":"macro_f1","mean":round(float(np.mean(f1s)),4),"std":round(float(np.std(f1s)),4)}]
    else:
        y_time=df["os_time_days"].astype(float).values; aucs,cis=[],[]
        for tr,te in cv.split(np.zeros(len(y)),y):
            prob=train(variant,imgs[tr],y[tr],imgs[te],y[te],size,1,raw[tr] if variant in ["1_raw_skip","6_residual"] else None,raw[te] if variant in ["1_raw_skip","6_residual"] else None,binary=True)
            aucs.append(roc_auc_score(y[te],prob)); cis.append(concordance_index(y_time[te],-prob,y[te]))
        return [{"variant":variant,"task":"Survival","metric":"roc_auc","mean":round(float(np.mean(aucs)),4),"std":round(float(np.std(aucs)),4)},{"variant":variant,"task":"Survival","metric":"c_index","mean":round(float(np.mean(cis)),4),"std":round(float(np.std(cis)),4)}]


def main():
    variants=["combined"]
    rows=[]
    for v in variants:
        rows+=eval_variant(v,"PAM50"); rows+=eval_variant(v,"Survival"); print("done",v)
    with OUT.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["variant","task","metric","mean","std"],delimiter="\t"); w.writeheader(); w.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__=="__main__":
    main()

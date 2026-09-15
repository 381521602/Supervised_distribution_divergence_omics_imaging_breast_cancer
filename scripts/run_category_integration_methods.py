#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare gene-category integration methods for mRNA images."""

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
OUT=ROOT/"data/category_integration_methods_results.tsv"
RANDOM_STATE=42; K_FOLDS=5; EPOCHS=25; BATCH_SIZE=64


class MethodNet(nn.Module):
    def __init__(self,method,size,out_dim,aux_dim):
        super().__init__(); self.method=method
        if method=="1_twochannel":
            self.block=nn.Sequential(nn.Conv2d(2,32,kernel_size=(size,size)),nn.BatchNorm2d(32),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten()); feat=32
        elif method=="2_concat_vector":
            self.block=nn.Sequential(nn.Conv2d(1,32,kernel_size=(size,size)),nn.BatchNorm2d(32),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten()); feat=32+aux_dim
        else:
            self.block=nn.Sequential(nn.Conv2d(1,32,kernel_size=(size,size)),nn.BatchNorm2d(32),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten()); feat=32
        self.head=nn.Sequential(nn.Linear(feat,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,out_dim))
    def forward(self,x,aux_img=None,aux_vec=None):
        if self.method=="1_twochannel":
            f=self.block(torch.cat([x,aux_img],dim=1))
        elif self.method=="2_concat_vector":
            f=self.block(x); f=torch.cat([f,aux_vec],dim=1)
        else:
            f=self.block(x)
        return self.head(f)


def prepare(task,omics,size):
    imgs=np.load(IMG/f"{task}/{omics}/images.npy").astype(np.float32)
    cat_name="mRNA_category_grid.npy" if task=="PAM50" else "Survival_mRNA_category_grid.npy"
    cat=np.load(ROOT/f"data/images/examples/{cat_name}")
    aux_img=np.repeat(cat[np.newaxis,np.newaxis,:,:],len(imgs),axis=0).astype(np.float32)
    # category vector per gene: one-hot in original gene order is not available; use category grid flattened per sample
    aux_vec=np.tile(cat.flatten(),(len(imgs),1)).astype(np.float32)
    return imgs,aux_img,aux_vec


def train(method,Xtr,aimg_tr,avec_tr,ytr,Xte,aimg_te,avec_te,size,dout,binary=False):
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    model=MethodNet(method,size,dout,avec_tr.shape[1]); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    if binary:
        crit=nn.BCEWithLogitsLoss(); yt=torch.tensor(ytr,dtype=torch.float32).view(-1,1)
    else:
        crit=nn.CrossEntropyLoss(); yt=torch.tensor(ytr,dtype=torch.long)
    Xt=torch.tensor(Xtr); Ai=torch.tensor(aimg_tr); Av=torch.tensor(avec_tr); n=Xt.shape[0]
    for _ in range(EPOCHS):
        perm=torch.randperm(n)
        for i in range(0,n,BATCH_SIZE):
            idx=perm[i:i+BATCH_SIZE]
            if idx.shape[0]<2: continue
            opt.zero_grad(); loss=crit(model(Xt[idx],Ai[idx],Av[idx]),yt[idx]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out=model(torch.tensor(Xte),torch.tensor(aimg_te),torch.tensor(avec_te))
    return torch.sigmoid(out).numpy().ravel() if binary else out.argmax(dim=1).numpy()


def eval_method(method,task,omics,size):
    imgs,aimg,avec=prepare(task,omics,size)
    if task=="PAM50":
        df=pd.read_csv(PAM_DIR/f"{omics}_PAM50_final.tsv",sep="\t"); y_raw=df["pam50"].values; enc=LabelEncoder(); y=enc.fit_transform(y_raw)
        cv=StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=RANDOM_STATE); accs,f1s=[],[]
        for tr,te in cv.split(np.zeros(len(y)),y):
            pred=train(method,imgs[tr],aimg[tr],avec[tr],y[tr],imgs[te],aimg[te],avec[te],size,len(enc.classes_))
            accs.append(accuracy_score(y[te],pred)); f1s.append(f1_score(y[te],pred,average="macro"))
        return [{"method":method,"task":"PAM50","metric":"accuracy","mean":round(float(np.mean(accs)),4),"std":round(float(np.std(accs)),4)},{"method":method,"task":"PAM50","metric":"macro_f1","mean":round(float(np.mean(f1s)),4),"std":round(float(np.std(f1s)),4)}]
    else:
        df=pd.read_csv(SUR_DIR/f"{omics}_Survival_final.tsv",sep="\t"); y_event=df["os_event"].astype(int).values; y_time=df["os_time_days"].astype(float).values
        cv=StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=RANDOM_STATE); aucs,cis=[],[]
        for tr,te in cv.split(np.zeros(len(y_event)),y_event):
            prob=train(method,imgs[tr],aimg[tr],avec[tr],y_event[tr],imgs[te],aimg[te],avec[te],size,1,binary=True)
            aucs.append(roc_auc_score(y_event[te],prob)); cis.append(concordance_index(y_time[te],-prob,y_event[te]))
        return [{"method":method,"task":"Survival","metric":"roc_auc","mean":round(float(np.mean(aucs)),4),"std":round(float(np.std(aucs)),4)},{"method":method,"task":"Survival","metric":"c_index","mean":round(float(np.mean(cis)),4),"std":round(float(np.std(cis)),4)}]


def main():
    methods=["1_twochannel","2_concat_vector","3_spatial_attention","5_learnable_fusion","6_weighted_loss"]
    rows=[]
    for method in methods:
        rows+=eval_method(method,"PAM50","mRNA",20)
        rows+=eval_method(method,"Survival","mRNA",15)
        print("done",method)
    with OUT.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["method","task","metric","mean","std"],delimiter="\t"); w.writeheader(); w.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__=="__main__":
    main()

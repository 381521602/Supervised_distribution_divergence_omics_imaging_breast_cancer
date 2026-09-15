#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare 6 NSRE integration methods for mRNA images."""

from __future__ import annotations

import csv
import sys
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
sys.path.insert(0,str(ROOT/"scripts"))
from run_multistream_cnn import spiral_order


PAM_DIR=ROOT/"data/final_datasets/PAM50"
SUR_DIR=ROOT/"data/final_datasets/Survival"
IMG=ROOT/"data/images"
OUT=ROOT/"data/nsre_integration_methods_results.tsv"
RANDOM_STATE=42; K_FOLDS=5; EPOCHS=25; BATCH_SIZE=64


class ConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch,size):
        super().__init__()
        self.conv=nn.Conv2d(in_ch,out_ch,kernel_size=(size,size))
        self.bn=nn.BatchNorm2d(out_ch)
    def forward(self,x):
        x=self.conv(x); x=self.bn(x); x=F.relu(x); x=F.adaptive_avg_pool2d(x,(1,1)); return x.flatten(1)


class MethodNet(nn.Module):
    def __init__(self,method,size,out_dim,nsre_dim):
        super().__init__()
        self.method=method
        if method=="1_twochannel":
            self.block=ConvBlock(2,32,size); feat=32
        elif method=="2_concat_vector":
            self.block=ConvBlock(1,32,size); feat=32+nsre_dim
        elif method=="3_spatial_attention":
            self.block=nn.Sequential(nn.Conv2d(1,32,kernel_size=(size,size)),nn.BatchNorm2d(32),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten()); feat=32
        elif method=="5_learnable_fusion":
            self.alpha=nn.Parameter(torch.tensor(0.5))
            self.block=ConvBlock(1,32,size); feat=32
        elif method=="6_weighted_loss":
            self.block=ConvBlock(1,32,size); feat=32
        else:
            self.block=ConvBlock(1,32,size); feat=32
        self.head=nn.Sequential(nn.Linear(feat,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,out_dim))

    def forward(self,x,nsre=None,nsre_vec=None):
        if self.method=="1_twochannel":
            x=torch.cat([x,nsre],dim=1)
            f=self.block(x)
        elif self.method=="2_concat_vector":
            f=self.block(x); f=torch.cat([f,nsre_vec],dim=1)
        elif self.method=="3_spatial_attention":
            f=self.block(x)
        elif self.method=="5_learnable_fusion":
            x=x*self.alpha + nsre*(1-self.alpha)
            f=self.block(x)
        else:
            f=self.block(x)
        return self.head(f)


def prepare(task,omics,size):
    imgs=np.load(IMG/f"{task}/{omics}/images.npy")
    order=pd.read_csv(IMG/f"{task}/{omics}/order.tsv",sep="\t")
    nsre=order["nsre"].values
    if nsre.max()>nsre.min(): w=(nsre-nsre.min())/(nsre.max()-nsre.min()+1e-9)
    else: w=np.zeros_like(nsre)
    nsre_grid=np.zeros((size,size),dtype=np.float32)
    for pos,val in zip(spiral_order(size),w): nsre_grid[pos]=val
    nsre_img=np.repeat(nsre_grid[np.newaxis,np.newaxis,:,:],len(imgs),axis=0)
    nsre_vec=np.tile(w,(len(imgs),1))
    return imgs.astype(np.float32), nsre_img.astype(np.float32), nsre_vec.astype(np.float32)


def train(method,Xtr,nsre_img_tr,nsre_vec_tr,ytr,Xte,nsre_img_te,nsre_vec_te,size,dout,binary=False):
    torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
    model=MethodNet(method,size,dout,nsre_vec_tr.shape[1])
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    if binary:
        crit=nn.BCEWithLogitsLoss(); yt=torch.tensor(ytr,dtype=torch.float32).view(-1,1)
    else:
        crit=nn.CrossEntropyLoss(); yt=torch.tensor(ytr,dtype=torch.long)
    Xt=torch.tensor(Xtr,dtype=torch.float32); Nimg=torch.tensor(nsre_img_tr,dtype=torch.float32); Nvec=torch.tensor(nsre_vec_tr,dtype=torch.float32)
    n=Xt.shape[0]
    for _ in range(EPOCHS):
        perm=torch.randperm(n)
        for i in range(0,n,BATCH_SIZE):
            idx=perm[i:i+BATCH_SIZE]
            if idx.shape[0]<2: continue
            opt.zero_grad(); out=model(Xt[idx],Nimg[idx],Nvec[idx]); loss=crit(out,yt[idx]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out=model(torch.tensor(Xte,dtype=torch.float32),torch.tensor(nsre_img_te,dtype=torch.float32),torch.tensor(nsre_vec_te,dtype=torch.float32))
    return torch.sigmoid(out).numpy().ravel() if binary else out.argmax(dim=1).numpy()


def eval_method(method,task,omics,size):
    imgs,nsre_img,nsre_vec=prepare(task,omics,size)
    if task=="PAM50":
        df=pd.read_csv(PAM_DIR/f"{omics}_PAM50_final.tsv",sep="\t"); y_raw=df["pam50"].values; enc=LabelEncoder(); y=enc.fit_transform(y_raw)
        cv=StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=RANDOM_STATE); accs,f1s=[],[]
        for tr,te in cv.split(np.zeros(len(y)),y):
            pred=train(method,imgs[tr],nsre_img[tr],nsre_vec[tr],y[tr],imgs[te],nsre_img[te],nsre_vec[te],size,len(enc.classes_))
            accs.append(accuracy_score(y[te],pred)); f1s.append(f1_score(y[te],pred,average="macro"))
        return [{"method":method,"task":"PAM50","metric":"accuracy","mean":round(float(np.mean(accs)),4),"std":round(float(np.std(accs)),4)},{"method":method,"task":"PAM50","metric":"macro_f1","mean":round(float(np.mean(f1s)),4),"std":round(float(np.std(f1s)),4)}]
    else:
        df=pd.read_csv(SUR_DIR/f"{omics}_Survival_final.tsv",sep="\t"); y_event=df["os_event"].astype(int).values; y_time=df["os_time_days"].astype(float).values
        cv=StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=RANDOM_STATE); aucs,cis=[],[]
        for tr,te in cv.split(np.zeros(len(y_event)),y_event):
            prob=train(method,imgs[tr],nsre_img[tr],nsre_vec[tr],y_event[tr],imgs[te],nsre_img[te],nsre_vec[te],size,1,binary=True)
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

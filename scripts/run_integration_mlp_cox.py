#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplement integration with MLP (PAM50) and CoxPH (survival)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
PAM_DIR = ROOT / "data" / "final_datasets" / "PAM50"
SUR_DIR = ROOT / "data" / "final_datasets" / "Survival"
OUT = ROOT / "data" / "integration_mlp_cox_results.tsv"
RANDOM_STATE = 42
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 64
COMBOS = [("mRNA","CNV"),("mRNA","miRNA"),("CNV","miRNA"),("mRNA","CNV","miRNA")]


class MLP(nn.Module):
    def __init__(self,din,dout):
        super().__init__(); self.net=nn.Sequential(nn.Linear(din,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,dout))
    def forward(self,x): return self.net(x)


def build_pam50(combo):
    dfs=[pd.read_csv(PAM_DIR/f"{o}_PAM50_final.tsv",sep="\t") for o in combo]
    common=dfs[0].index
    for d in dfs[1:]: common=common.intersection(d.index)
    common=list(common)
    X=np.hstack([d.loc[common].drop(columns=["pam50"]).to_numpy(dtype=float) for d in dfs])
    y_raw=dfs[0].loc[common,"pam50"].values
    return X,y_raw


def build_survival(combo):
    dfs=[pd.read_csv(SUR_DIR/f"{o}_Survival_final.tsv",sep="\t") for o in combo]
    common=dfs[0].index
    for d in dfs[1:]: common=common.intersection(d.index)
    common=list(common)
    X=np.hstack([d.loc[common].drop(columns=["case_id","os_event","os_time_days"]).to_numpy(dtype=float) for d in dfs])
    y_event=dfs[0].loc[common,"os_event"].astype(int).values
    y_time=dfs[0].loc[common,"os_time_days"].astype(float).values
    return X,y_event,y_time


def eval_pam50_mlp(combo):
    X,y_raw=build_pam50(combo); enc=LabelEncoder(); y=enc.fit_transform(y_raw)
    cv=StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=RANDOM_STATE)
    accs,f1s=[],[]
    for tr,te in cv.split(np.zeros(len(y)),y):
        scaler=StandardScaler().fit(X[tr]); Xtr=scaler.transform(X[tr]); Xte=scaler.transform(X[te])
        torch.manual_seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)
        model=MLP(Xtr.shape[1],len(enc.classes_)); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4); crit=nn.CrossEntropyLoss()
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
    return [
        {"task":"PAM50","combo":"+".join(combo),"model":"MLP","metric":"accuracy","mean":round(float(np.mean(accs)),4),"std":round(float(np.std(accs)),4)},
        {"task":"PAM50","combo":"+".join(combo),"model":"MLP","metric":"macro_f1","mean":round(float(np.mean(f1s)),4),"std":round(float(np.std(f1s)),4)},
    ]


def eval_survival_cox(combo):
    X,y_event,y_time=build_survival(combo)
    cv=StratifiedKFold(n_splits=K_FOLDS,shuffle=True,random_state=RANDOM_STATE)
    cis=[]; aucs=[]
    for tr,te in cv.split(np.zeros(len(y_event)),y_event):
        scaler=StandardScaler().fit(X[tr]); Xtr=scaler.transform(X[tr]); Xte=scaler.transform(X[te])
        train_df=pd.DataFrame(Xtr); train_df["time"]=y_time[tr]; train_df["event"]=y_event[tr]
        test_df=pd.DataFrame(Xte)
        cph=CoxPHFitter(penalizer=0.1); cph.fit(train_df,duration_col="time",event_col="event")
        hazard=cph.predict_partial_hazard(test_df)
        cis.append(concordance_index(y_time[te],-hazard,y_event[te])); aucs.append(roc_auc_score(y_event[te],hazard))
    return [
        {"task":"Survival","combo":"+".join(combo),"model":"CoxPH","metric":"c_index","mean":round(float(np.mean(cis)),4),"std":round(float(np.std(cis)),4)},
        {"task":"Survival","combo":"+".join(combo),"model":"CoxPH","metric":"roc_auc","mean":round(float(np.mean(aucs)),4),"std":round(float(np.std(aucs)),4)},
    ]


def main():
    rows=[]
    for combo in COMBOS:
        rows+=eval_pam50_mlp(combo)
        rows+=eval_survival_cox(combo)
        print("done",combo)
    with OUT.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["task","combo","model","metric","mean","std"],delimiter="\t"); w.writeheader(); w.writerows(rows)
    print(f"Wrote -> {OUT}")


if __name__=="__main__":
    main()

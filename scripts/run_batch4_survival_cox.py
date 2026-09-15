#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 4: survival CoxPH grid."""

import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_batch1_pam50_grid import make_selector, FEATURE_METHODS, BASE_K, MULTIPLIERS, OMICS, RANDOM_STATE, K_FOLDS, load_omics

OUT = ROOT / "data" / "batch4_survival_cox_results.tsv"

def align_os_time(omics):
    matrix = load_omics(omics)
    labels = pd.read_csv(ROOT/"data/brca_labels_modeling_ready.tsv", sep="\t", dtype={"case_id":str})
    cases = labels.loc[labels["os_event"].isin([0,1]) & labels["os_time_days"].notna(), "case_id"].tolist()
    case_cols = {}
    for col in matrix.columns:
        case_cols.setdefault(col[:12], []).append(col)
    best_cols = {c: ([x for x in cols if x.endswith("-01")] or cols)[0] for c, cols in case_cols.items()}
    aligned = [c for c in cases if c in best_cols]
    X = matrix[[best_cols[c] for c in aligned]].T.to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    lab = labels.set_index("case_id").loc[aligned]
    y_event = lab["os_event"].astype(int).values
    y_time = lab["os_time_days"].astype(float).values
    return X, y_event, y_time

def main():
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=["omics","multiplier","k","feature_method","model","metric","mean","std"], delimiter="\t"); w.writeheader()
    for omics in OMICS:
        X, y_event, y_time = align_os_time(omics)
        for mult in MULTIPLIERS:
            k = BASE_K[omics]*mult
            for fs in FEATURE_METHODS:
                cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                cis=[]
                for tr,te in cv.split(np.zeros(len(y_event)), y_event):
                    sel=make_selector(fs,k); sel.fit(X[tr], y_event[tr])
                    Xtr=StandardScaler().fit_transform(sel.transform(X[tr]))
                    Xte=StandardScaler().fit(sel.transform(X[tr])).transform(sel.transform(X[te]))
                    if Xtr.shape[1] > 100:
                        var = np.var(Xtr, axis=0)
                        top = np.argsort(var)[::-1][:100]
                        Xtr = Xtr[:, top]
                        Xte = Xte[:, top]
                    train_df=pd.DataFrame(Xtr); train_df["time"]=y_time[tr]; train_df["event"]=y_event[tr]
                    test_df=pd.DataFrame(Xte)
                    cph=CoxPHFitter(penalizer=0.1); cph.fit(train_df, duration_col="time", event_col="event")
                    hazard=cph.predict_partial_hazard(test_df)
                    cis.append(concordance_index(y_time[te], -hazard, y_event[te]))
                with OUT.open("a", encoding="utf-8", newline="") as f:
                    w=csv.DictWriter(f, fieldnames=["omics","multiplier","k","feature_method","model","metric","mean","std"], delimiter="\t")
                    w.writerow({"omics":omics,"multiplier":mult,"k":k,"feature_method":fs,"model":"CoxPH","metric":"c_index","mean":round(float(np.mean(cis)),4),"std":round(float(np.std(cis)),4)})
                print(f"{omics} mult={mult} {fs} Cox done")
    print(f"Done -> {OUT}")

if __name__=="__main__":
    main()

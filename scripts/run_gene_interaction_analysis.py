from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parent.parent
sys_path = ROOT / "scripts"
import sys
sys.path.insert(0, str(sys_path))
from run_multistream_cnn import spiral_order

PAM = ROOT / "data/final_datasets/PAM50/mRNA_PAM50_final.tsv"
ORDER = ROOT / "data/images/PAM50/mRNA/order.tsv"
CONSENSUS = ROOT / "data/interpretability/mrna_pam50_consensus_key_factors.tsv"
OUT_DIR = ROOT / "data/gene_interaction"
OUT_DIR.mkdir(parents=True, exist_ok=True)
COMM_IMG_DIR = ROOT / "data/images/PAM50/mRNA_community"
COMM_IMG_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


class FullSizeCNN(nn.Module):
    def __init__(self, size, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(size, size))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.head(F.relu(self.conv(x)))


def train_fold(Xtr, ytr, Xte, size, out_dim, epochs=20):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    model = FullSizeCNN(size, out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    n = Xt.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            if idx.shape[0] < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte, dtype=torch.float32))
    return logits.argmax(dim=1).numpy()


def build_images(X, order, size=20):
    positions = spiral_order(size)
    imgs = np.zeros((X.shape[0], 1, size, size), dtype=np.float32)
    for i in range(X.shape[0]):
        for (r, c), j in zip(positions, order):
            if j < X.shape[1]:
                imgs[i, 0, r, c] = X[i, j]
    return imgs


def evaluate(images, y, enc):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    for tr, te in cv.split(np.zeros(len(y)), y):
        pred = train_fold(images[tr], y[tr], images[te], 20, len(enc.classes_))
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return round(float(np.mean(accs)), 4), round(float(np.std(accs)), 4), round(float(np.mean(f1s)), 4), round(float(np.std(f1s)), 4)


def main():
    df = pd.read_csv(PAM, sep="\t")
    y_raw = df["pam50"].values
    X = df.drop(columns=["pam50"]).to_numpy(dtype=float)
    genes = df.drop(columns=["pam50"]).columns.tolist()
    jsd_df = pd.read_csv(ORDER, sep="\t")
    jsd = {row.feature: float(row.jsd) for row in jsd_df.itertuples()}

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    G = nx.Graph()
    G.add_nodes_from(range(len(genes)))
    threshold = 0.75
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            if abs(corr[i, j]) >= threshold:
                G.add_edge(i, j, weight=abs(float(corr[i, j])))

    communities = sorted(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"), key=len, reverse=True)
    community_of = {}
    community_rows = []
    edge_rows = []
    for ci, comm in enumerate(communities):
        for node in comm:
            community_of[node] = ci
            community_rows.append({"gene": genes[node], "community": ci})
    for node in range(len(genes)):
        if node not in community_of:
            community_of[node] = len(communities)
            community_rows.append({"gene": genes[node], "community": len(communities)})
    for i, j in G.edges():
        edge_rows.append({"gene1": genes[i], "gene2": genes[j], "correlation": round(float(G[i][j]["weight"]), 4)})
    pd.DataFrame(edge_rows).to_csv(OUT_DIR / "mrna_coexpression_edges.tsv", sep="\t", index=False)
    pd.DataFrame(community_rows).to_csv(OUT_DIR / "mrna_coexpression_communities.tsv", sep="\t", index=False)

    order = sorted(
        range(len(genes)),
        key=lambda j: (community_of[j], -jsd.get(genes[j], 0.0), genes[j]),
    )
    pd.DataFrame({"feature": [genes[j] for j in order], "community": [community_of[j] for j in order], "jsd": [jsd.get(genes[j], 0.0) for j in order]}).to_csv(COMM_IMG_DIR / "order.tsv", sep="\t", index=False)
    images = build_images(X, order, 20)
    np.save(COMM_IMG_DIR / "images.npy", images)

    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    acc, acc_std, f1, f1_std = evaluate(images, y, enc)
    pd.DataFrame([{"variant": "community_reorder", "task": "PAM50", "metric": "accuracy", "mean": acc, "std": acc_std}, {"variant": "community_reorder", "task": "PAM50", "metric": "macro_f1", "mean": f1, "std": f1_std}]).to_csv(OUT_DIR / "community_reorder_fullsize_cnn_results.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    top_genes = [genes[j] for j in order[:60]]
    top_idx = [genes.index(g) for g in top_genes]
    sub = corr[np.ix_(top_idx, top_idx)]
    im = ax.imshow(sub, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top_genes)), top_genes, rotation=90, fontsize=5)
    ax.set_yticks(range(len(top_genes)), top_genes, fontsize=5)
    ax.set_title("mRNA co-expression communities (top 60 network-ordered genes)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_mrna_coexpression_communities.png", dpi=200)
    print("community count", len(communities), "edges", G.number_of_edges())
    print("FullSizeCNN community order acc", acc, "f1", f1)
    print("saved to", OUT_DIR)


if __name__ == "__main__":
    main()

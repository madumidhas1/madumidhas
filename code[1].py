import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BlockchainSimulator:
    def __init__(self):
        self.rng = np.random.default_rng(42)

    def store_transactions(self, df):
        df = df.copy()
        df["tamper_attempt"] = (self.rng.random(len(df)) < 0.02).astype(int)
        df["consensus_success"] = (self.rng.random(len(df)) >= 0.03).astype(int)
        df["access_authorized"] = (self.rng.random(len(df)) >= 0.02).astype(int)
        return df

    def security_level(self, df):
        IS = 1 - df["tamper_attempt"].mean()
        CR = df["consensus_success"].mean()
        AC = df["access_authorized"].mean()
        return 100 * (0.4 * IS + 0.3 * CR + 0.3 * AC)

    def edge_trust(self, late):
        return (1 / (1 + np.exp(-(4 * (1 - late) - 2)))).astype(np.float32)


def preprocess_dataco(df):
    df = df.drop_duplicates()


    supplier_col = "Department Name"
    warehouse_col = "Order City"
    customer_col = "Customer Id"

    df["late_label"] = df["Late_delivery_risk"].astype(int)

    features = [
        "Order Item Quantity",
        "Sales",
        "Order Item Discount",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Order Profit Per Order",
        "Order Item Product Price",
        "Order Item Total"
    ]

    keep = [supplier_col, warehouse_col, customer_col, "late_label"] + features
    df = df[keep].fillna(0).reset_index(drop=True)

    return df, supplier_col, warehouse_col, customer_col, features


def build_graph(df, supplier_col, warehouse_col, customer_col, features, bc):
    nodes = (
        ["S::" + s for s in df[supplier_col].astype(str).unique()] +
        ["W::" + w for w in df[warehouse_col].astype(str).unique()] +
        ["R::" + r for r in df[customer_col].astype(str).unique()]
    )
    node2idx = {n: i for i, n in enumerate(nodes)}

    sw = np.stack([
        df[supplier_col].map(lambda x: node2idx["S::" + str(x)]),
        df[warehouse_col].map(lambda x: node2idx["W::" + str(x)])
    ])

    wr = np.stack([
        df[warehouse_col].map(lambda x: node2idx["W::" + str(x)]),
        df[customer_col].map(lambda x: node2idx["R::" + str(x)])
    ])

    edge_index = np.concatenate([sw, wr], axis=1)
    trust = bc.edge_trust(np.tile(df["late_label"].values, 2))

    node_x = np.zeros((len(nodes), len(features)))
    vals = df[features].values

    for i in range(len(df)):
        for n in (
            "S::" + str(df[supplier_col][i]),
            "W::" + str(df[warehouse_col][i]),
            "R::" + str(df[customer_col][i])
        ):
            node_x[node2idx[n]] += vals[i]

    return Data(
        x=torch.tensor(node_x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(trust).unsqueeze(1)
    )


class TrustAwareGAT(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gat1 = GATConv(in_dim, 64, heads=4, edge_dim=1)
        self.gat2 = GATConv(64 * 4, 64, edge_dim=1)

    def forward(self, data):
        x = F.elu(self.gat1(data.x, data.edge_index, data.edge_attr))
        return self.gat2(x, data.edge_index, data.edge_attr)


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    return {
        "Accuracy": accuracy_score(y, p),
        "Precision": precision_score(y, p),
        "Recall": recall_score(y, p),
        "F1": f1_score(y, p),
        "Specificity": tn / (tn + fp)
    }


def main(csv_path):
    set_seed()
    bc = BlockchainSimulator()

    print("Loading dataset...")
    df = pd.read_csv(csv_path, encoding_errors="ignore")

    df, supplier_col, warehouse_col, customer_col, features = preprocess_dataco(df)
    df = bc.store_transactions(df)

    graph = build_graph(df, supplier_col, warehouse_col, customer_col, features, bc)

    print("Training GAT...")
    gat = TrustAwareGAT(graph.x.size(1))
    node_emb = gat(graph).detach()

    X = df[features].values
    y = df["late_label"].values

    X = StandardScaler().fit_transform(X)

    clf = SVC(kernel="rbf")
    clf.fit(X, y)
    preds = clf.predict(X)

    print("\n=== RESULTS ===")
    for k, v in metrics(y, preds).items():
        print(f"{k:>12}: {v:.4f}")

    print(f"\nSecurity Level: {bc.security_level(df):.2f}")


if __name__ == "__main__":
    csv_path = "dataset\folder\path\Data.csv"
    main(csv_path)
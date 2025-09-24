import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler
import torch
import yaml
from dataclasses import dataclass

@dataclass
class GraphData:
    A: torch.Tensor             # [N, N] normalized adjacency
    X: torch.Tensor             # [T, N, F] features
    Y: torch.Tensor             # [T, N] target usage_kw
    nodes: pd.DataFrame
    edges: pd.DataFrame
    times: np.ndarray           # array of timestamps

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_adjacency(edges: pd.DataFrame, nodes: pd.DataFrame) -> np.ndarray:
    id2idx = {nid: i for i, nid in enumerate(nodes["node_id"].tolist())}
    N = len(nodes)
    A = np.zeros((N, N), dtype=np.float32)
    for _, e in edges.iterrows():
        if e["u"] in id2idx and e["v"] in id2idx:
            i, j = id2idx[e["u"]], id2idx[e["v"]]
            cost = float(e.get("impedance", e.get("travel_time_min", 1.0)))
            w = 1.0 / (1.0 + cost)
            A[i, j] = max(A[i, j], w)
            A[j, i] = max(A[j, i], w)
    # add self loops
    for i in range(N):
        A[i, i] = 1.0
    # normalize A_hat = D^{-1/2} A D^{-1/2}
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-6)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm.astype(np.float32)

def load_graph_timeseries(cfg: dict) -> GraphData:
    nodes = pd.read_csv(cfg["paths"]["nodes"])
    edges = pd.read_csv(cfg["paths"]["edges"])
    ts = pd.read_csv(cfg["paths"]["timeseries"], parse_dates=["timestamp"])

    # Align to single node set and pivot to [T, N, F]
    features_cols = ["traffic_index","temp_c","precip_mm","income_k","poi_density","holiday","hour_sin","hour_cos"]
    # engineer cyclical hour of day
    ts["hour"] = ts["timestamp"].dt.hour
    ts["hour_sin"] = np.sin(2*np.pi*ts["hour"]/24.0)
    ts["hour_cos"] = np.cos(2*np.pi*ts["hour"]/24.0)
    ts["holiday"] = ts["holiday"].astype(int)

    # Targets and features
    y_pivot = ts.pivot(index="timestamp", columns="node_id", values="usage_kw").sort_index()
    X_list = []
    for col in features_cols:
        X_list.append(ts.pivot(index="timestamp", columns="node_id", values=col).sort_index())
    # ensure same index/columns
    X_list = [x.reindex_like(y_pivot).fillna(method="ffill").fillna(0.0) for x in X_list]
    Y = y_pivot.fillna(0.0).to_numpy(dtype=np.float32)  # [T, N]
    X = np.stack([x.to_numpy(dtype=np.float32) for x in X_list], axis=-1)  # [T, N, F]

    # scale features (not targets)
    T, N, F = X.shape
    scaler = StandardScaler()
    X = X.reshape(T*N, F)
    X = scaler.fit_transform(X)
    X = X.reshape(T, N, F).astype(np.float32)

    # adjacency
    A = build_adjacency(edges, nodes)

    return GraphData(
        A=torch.tensor(A, dtype=torch.float32),
        X=torch.tensor(X, dtype=torch.float32),
        Y=torch.tensor(Y, dtype=torch.float32),
        nodes=nodes,
        edges=edges,
        times=y_pivot.index.to_numpy()
    )

def make_sequences(gd: GraphData, time_window: int, horizon: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X, Y = gd.X, gd.Y
    T = X.shape[0]
    xs, ys = [], []
    for t in range(time_window, T - horizon):
        xwin = X[t-time_window:t]   # [Tw, N, F]
        ytar = Y[t:t+horizon]       # [H, N]
        xs.append(xwin.numpy())
        ys.append(ytar.numpy())
    X_seq = torch.tensor(np.stack(xs), dtype=torch.float32)       # [B, Tw, N, F]
    Y_seq = torch.tensor(np.stack(ys), dtype=torch.float32)       # [B, H, N]
    return X_seq, Y_seq, torch.arange(time_window, T - horizon)

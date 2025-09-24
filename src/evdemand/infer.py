import os
import numpy as np
import pandas as pd
import torch
from .data import load_config, load_graph_timeseries
from .model import STGCN
from datetime import timedelta

def infer(config_path: str, horizon: int, out_path: str):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(cfg["paths"]["artifacts_dir"], "model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ckpt["config"]["model"]
    model = STGCN(
        in_features=model_cfg["in_features"],
        gcn_hidden=model_cfg["gcn_hidden"],
        gcn_layers=model_cfg["gcn_layers"],
        temporal_hidden=model_cfg["temporal_hidden"],
        dropout=model_cfg["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    gd = load_graph_timeseries(cfg)
    A = gd.A.to(device)
    Tw = model_cfg["time_window"]
    X = gd.X[-Tw:].unsqueeze(0).to(device)  # [1, Tw, N, F]

    means = []
    stds = []
    # Iterative one-step ahead
    x_roll = gd.X.clone()
    x_window = x_roll[-Tw:].unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(horizon):
            mean, logvar = model(x_window, A)
            std = torch.exp(0.5 * logvar)
            means.append(mean.squeeze(0).cpu().numpy())
            stds.append(std.squeeze(0).cpu().numpy())
            # naive roll by appending predicted mean to features as proxy for usage_kw if needed
            # (features do not include target; so we simply advance window without change)
            x_window = x_window  # no change

    means = np.stack(means, axis=0)  # [H, N]
    stds = np.stack(stds, axis=0)    # [H, N]

    z_p90 = 1.2815515655446004
    p50 = means
    p90 = means + z_p90 * stds

    # queue risk if demand exceeds capacity
    cap = gd.nodes["capacity_kw"].to_numpy(dtype=float)
    cap = np.where(cap>0, cap, np.inf)
    risk = (p90 > cap[None, :]).astype(float)

    # write csv
    start_ts = pd.to_datetime(gd.times[-1]) + pd.Timedelta(hours=1)
    timestamps = [start_ts + pd.Timedelta(hours=i) for i in range(horizon)]
    rows = []
    node_ids = gd.nodes["node_id"].tolist()
    for h_idx, ts in enumerate(timestamps):
        for j, nid in enumerate(node_ids):
            rows.append({
                "timestamp": ts.isoformat(),
                "node_id": nid,
                "p50_kw": float(p50[h_idx, j]),
                "p90_kw": float(p90[h_idx, j]),
                "queue_risk": float(risk[h_idx, j])
            })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--out", default="artifacts/forecasts.csv")
    args = ap.parse_args()
    infer(args.config, args.horizon, args.out)

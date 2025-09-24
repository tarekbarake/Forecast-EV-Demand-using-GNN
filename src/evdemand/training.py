import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from .data import load_config, load_graph_timeseries, make_sequences
from .model import STGCN, gaussian_nll

def train(config_path: str):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["paths"]["artifacts_dir"], exist_ok=True)

    gd = load_graph_timeseries(cfg)
    A = gd.A.to(device)
    time_window = int(cfg["model"]["time_window"])
    horizon = int(cfg["forecast"]["horizon"])
    X_seq, Y_seq, _ = make_sequences(gd, time_window, horizon=1)  # next-hour target for training
    # Collapse horizon=1
    Y_seq = Y_seq[:, 0, :]  # [B, N]

    ds = TensorDataset(X_seq, Y_seq)
    val_frac = float(cfg["train"]["val_split"])
    val_len = int(len(ds) * val_frac)
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(cfg["train"]["seed"]))

    dl_tr = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    dl_va = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = STGCN(
        in_features=cfg["model"]["in_features"],
        gcn_hidden=cfg["model"]["gcn_hidden"],
        gcn_layers=cfg["model"]["gcn_layers"],
        temporal_hidden=cfg["model"]["temporal_hidden"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    best_val = math.inf
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        tr_loss = 0.0
        for xb, yb in tqdm(dl_tr, desc=f"epoch {epoch+1}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            mean, logvar = model(xb, A)
            loss = gaussian_nll(mean, logvar, yb).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                mean, logvar = model(xb, A)
                loss = gaussian_nll(mean, logvar, yb).mean()
                va_loss += loss.item() * xb.size(0)
        va_loss /= max(1, len(val_ds))

        print(f"epoch {epoch+1}: train_nll={tr_loss:.4f} val_nll={va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model_state": model.state_dict(), "config": cfg}, os.path.join(cfg["paths"]["artifacts_dir"], "model.pt"))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    train(args.config)

import os
import argparse
import pandas as pd
from evdemand.accessibility import compute_accessibility
from evdemand.selection import rank_candidates
from evdemand.data import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--out", default="artifacts/ranked_candidates.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    acc_df = compute_accessibility(cfg["paths"]["nodes"], cfg["paths"]["edges"], "artifacts/forecasts.csv")
    os.makedirs("artifacts", exist_ok=True)
    acc_path = "artifacts/accessibility.csv"
    acc_df.to_csv(acc_path, index=False)

    equity_w = float(cfg.get("selection", {}).get("equity_weight", 0.0))
    ranked = rank_candidates(cfg["paths"]["nodes"], acc_path, args.budget, equity_weight=equity_w)
    ranked.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

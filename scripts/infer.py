from evdemand.infer import infer
import argparse
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--out", default="artifacts/forecasts.csv")
    args = ap.parse_args()
    infer(args.config, args.horizon, args.out)

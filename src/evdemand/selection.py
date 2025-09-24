import pandas as pd
import numpy as np

def rank_candidates(nodes_csv: str, accessibility_csv: str, budget: float, equity_weight: float = 0.0) -> pd.DataFrame:
    nodes = pd.read_csv(nodes_csv)
    acc = pd.read_csv(accessibility_csv)
    df = nodes.merge(acc, on="node_id", how="left")
    df["install_cost"] = df["install_cost"].fillna(0.0)
    df["access_score"] = df["access_score"].fillna(0.0)
    df["equity"] = df.get("equity", pd.Series(np.zeros(len(df))))

    # Score: accessibility * demand proxy * (1 + equity_weight * equity)
    df["score"] = df["access_score"] * (1.0 + df["p50_kw"].fillna(0.0)) * (1.0 + equity_weight * df["equity"].fillna(0.0))

    # Greedy by value density subject to budget
    candidates = df[df["is_candidate"] == 1].copy()
    candidates["density"] = candidates["score"] / candidates["install_cost"].replace(0.0, np.inf)
    candidates = candidates.sort_values(by="density", ascending=False)

    chosen = []
    spend = 0.0
    for _, row in candidates.iterrows():
        c = float(row["install_cost"])
        if spend + c <= budget:
            chosen.append(row)
            spend += c

    chosen_df = pd.DataFrame(chosen)
    chosen_df = chosen_df.sort_values(by="score", ascending=False)
    return chosen_df[["node_id","score","install_cost","access_score","p50_kw","capacity_kw"]]

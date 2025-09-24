import pandas as pd
import numpy as np
import networkx as nx

def build_graph(edges_csv: str) -> nx.Graph:
    edges = pd.read_csv(edges_csv)
    G = nx.Graph()
    for _, e in edges.iterrows():
        u, v = e["u"], e["v"]
        w = float(e.get("travel_time_min", e.get("impedance", 1.0)))
        G.add_edge(u, v, weight=w)
    return G

def compute_accessibility(nodes_csv: str, edges_csv: str, forecasts_csv: str) -> pd.DataFrame:
    nodes = pd.read_csv(nodes_csv)
    forecasts = pd.read_csv(forecasts_csv, parse_dates=["timestamp"])
    latest = forecasts["timestamp"].max()
    f = forecasts[forecasts["timestamp"] == latest].copy()

    G = build_graph(edges_csv)
    pop = nodes.set_index("node_id")["pop_weight"].to_dict()
    is_cand = nodes.set_index("node_id")["is_candidate"].to_dict()

    # For each candidate, compute population-weighted accessibility as sum(pop_i / (1 + tt_i->cand))
    # Using travel_time_min weights
    acc_rows = []
    for cand in nodes[nodes["is_candidate"] == 1]["node_id"].tolist():
        tt = {}
        try:
            lengths = nx.single_source_dijkstra_path_length(G, cand, weight="weight")
        except Exception:
            lengths = {}
        score = 0.0
        for nid, p in pop.items():
            t = lengths.get(nid, 1e6)
            score += p / (1.0 + t)
        # demand proxy from forecast p50
        dmean = f[f["node_id"] == cand]["p50_kw"].mean()
        acc_rows.append({"node_id": cand, "access_score": score, "p50_kw": dmean})
    df = pd.DataFrame(acc_rows).fillna(0.0)
    return df

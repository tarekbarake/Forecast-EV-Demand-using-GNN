import os
import argparse
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

def gen_graph(num_nodes: int):
    # make ids
    node_ids = [f"n{i}" for i in range(num_nodes)]
    types = rng.choice(["intersection","station","poi","grid_tie"], size=num_nodes, p=[0.6,0.2,0.15,0.05])
    lat0, lon0 = 45.5, -73.6
    lat = lat0 + rng.normal(0, 0.05, size=num_nodes)
    lon = lon0 + rng.normal(0, 0.08, size=num_nodes)
    pop_weight = (rng.pareto(2.0, size=num_nodes) + 1.0) * 100
    existing = (types == "station").astype(int)
    candidates = (rng.random(num_nodes) < 0.3).astype(int)
    install_cost = (rng.uniform(5000, 100000, size=num_nodes) * candidates).round(0)
    capacity_kw = (existing * rng.uniform(50, 350, size=num_nodes)).round(1)

    nodes = pd.DataFrame({
        "node_id": node_ids,
        "type": types,
        "lat": lat,
        "lon": lon,
        "pop_weight": pop_weight.round(1),
        "is_existing": existing,
        "is_candidate": candidates,
        "install_cost": install_cost,
        "capacity_kw": capacity_kw
    })

    # edges (random geometric-ish)
    edges = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if rng.random() < 0.05:
                dist = np.hypot(lat[i]-lat[j], lon[i]-lon[j]) * 111 * 1.2
                tt = dist / rng.uniform(20, 50) * 60.0
                edges.append({"u": node_ids[i], "v": node_ids[j], "travel_time_min": round(tt,2), "distance_km": round(dist,2), "impedance": round(tt,2)})
    edges = pd.DataFrame(edges)
    return nodes, edges

def gen_timeseries(nodes: pd.DataFrame, hours: int):
    ts_rows = []
    base = pd.Timestamp("2025-01-01 00:00:00Z")
    for h in range(hours):
        ts = base + pd.Timedelta(hours=h)
        hour = h % 24
        temp = 20 - 15*np.cos(2*np.pi*hour/24.0) + rng.normal(0, 2)
        precip = max(0.0, rng.normal(0.5, 0.7))
        traffic = max(0.0, 50 + 30*np.sin(2*np.pi*hour/24.0) + rng.normal(0,5))
        holiday = int((pd.Timestamp(ts).dayofweek >= 5))
        for _, r in nodes.iterrows():
            nid = r["node_id"]
            income_k = rng.uniform(35, 120)
            poi_density = rng.uniform(0.0, 1.0)
            base_demand = 5 + 3*holiday + 2*np.sin(2*np.pi*hour/24.0)
            if r["type"] == "station":
                base_demand += 15
            usage = max(0.0, base_demand + 0.03*traffic + 0.1*poi_density - 0.05*precip + rng.normal(0,1.0))
            ts_rows.append({
                "timestamp": ts.isoformat(),
                "node_id": nid,
                "usage_kw": round(usage, 3),
                "traffic_index": round(traffic, 3),
                "temp_c": round(temp, 3),
                "precip_mm": round(precip, 3),
                "income_k": round(income_k, 3),
                "poi_density": round(poi_density, 3),
                "holiday": holiday
            })
    return pd.DataFrame(ts_rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/synth")
    ap.add_argument("--num-nodes", type=int, default=80)
    ap.add_argument("--hours", type=int, default=24*7)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    nodes, edges = gen_graph(args.num_nodes)
    ts = gen_timeseries(nodes, args.hours)
    nodes.to_csv(os.path.join(args.outdir, "nodes.csv"), index=False)
    edges.to_csv(os.path.join(args.outdir, "edges.csv"), index=False)
    ts.to_csv(os.path.join(args.outdir, "timeseries.csv"), index=False)
    print(f"Wrote synthetic dataset to {args.outdir}")

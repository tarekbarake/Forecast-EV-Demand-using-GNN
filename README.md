# Forecast EV Demand using GNN

End-to-end pipeline to forecast hourly EV charging demand and queue risk, compute accessibility with shortest paths, and select candidate charger sites under a budget.

## Features
- Spatiotemporal GNN (Temporal Conv + GCN) to forecast demand mean/variance.
- P50/P90 estimates from Gaussian outputs.
- Queue risk vs station capacity.
- Accessibility via Dijkstra/A* over a road graph.
- Budget-aware greedy site selection with optional equity weights.
- Synthetic data generator for quick tests.

## Quickstart
```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate
pip install -U pip

# Install package (editable) and CLI deps
pip install -e .

# Generate small synthetic dataset
python scripts/generate_synth.py --outdir data/synth --num-nodes 80 --hours 168

# Train model
python scripts/train.py --config configs/default.yaml

# Forecast next 24h
python scripts/infer.py --config configs/default.yaml --horizon 24 --out artifacts/forecasts.csv

# Rank candidate sites under a budget (e.g., $500k)
python scripts/rank_sites.py --config configs/default.yaml --budget 500000 --out artifacts/ranked_candidates.csv
```

## Data format
- `data/nodes.csv`: node_id,type,lat,lon,pop_weight,is_existing,is_candidate,install_cost,capacity_kw
- `data/edges.csv`: u,v,travel_time_min,distance_km,impedance
- `data/timeseries.csv`: timestamp,node_id,usage_kw,traffic_index,temp_c,precip_mm,income_k,poi_density,holiday

Timestamps are ISO8601 in UTC. Add more features as columns.

## Configuration
See `configs/default.yaml` for file paths and hyperparameters.

## Outputs
- `artifacts/model.pt`: trained weights
- `artifacts/forecasts.csv`: per-node P50/P90 demand and queue risk
- `artifacts/ranked_candidates.csv`: sorted list with accessibility and ROI proxies


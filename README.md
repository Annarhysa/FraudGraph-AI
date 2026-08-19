<<<<<<< HEAD
# FraudGraph AI

Explainable, graph-based fraud detection. Accounts, cards, and merchants
are modeled as a graph; transactions are scored for anomalousness using
graph structure + rule-based signals (baseline), with a GNN + SHAP layer
planned next. Every alert comes with a human-readable "why."

## Data

`data/data_transaction.csv` — IBM TabFormer credit card transactions
(2,000 users, ~24.4M transactions, 1991–2020, ~0.12% fraud rate).
Not committed to git (see `.gitignore`) — put your own copy at that path.

## Pipeline

```
data/data_transaction.csv (raw, 24M rows)
        │  fraudgraph.data_prep
        ▼
data/processed/transactions_full.parquet     (full dataset, columnar)
data/processed/transactions_sample.parquet   (300 users + all fraud users, 2019+)
        │  fraudgraph.graph_build
        ▼
data/processed/graph.pkl        User -> Card -> Merchant graph (networkx)
        │  fraudgraph.anomaly
        ▼
data/processed/alerts.csv       top-200 scored transactions + explanations
        │  app.py (Streamlit)
        ▼
Interactive dashboard: alert feed, "why" panel, card neighborhood graph
```

### Why a sample?

The raw data has only 2,000 users but ~29 years of daily history each,
so 24M rows come from history depth, not breadth. `data_prep.py` carves
out transactions from 2019 onward for ~300 users (plus every user who
ever has a fraud transaction, so positives aren't lost) — enough to
build and iterate on the graph without needing a database.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Run

```
.venv\Scripts\python -m src.fraudgraph.data_prep    # ~2 min, one-time
.venv\Scripts\python -m src.fraudgraph.graph_build
.venv\Scripts\python -m src.fraudgraph.anomaly
.venv\Scripts\streamlit run app.py
```

## Current baseline

Rule/graph-only scoring (velocity, amount z-score vs. user history, new
merchant, merchant rarity, error flags — no label used in scoring) gets
**~4.9x lift over random** at precision@200 on the 2019+ sample. That's
the number the GNN + SHAP stage needs to beat.

## Next steps

- [ ] PyTorch Geometric GNN (GraphSAGE) trained on the sample graph
- [ ] SHAP for tabular feature attribution + GNNExplainer for graph-path
      attribution, feeding the "why" panel
- [ ] Shared-attribute ring detection (connected components on merchants
      shared across flagged accounts)
- [ ] FastAPI backend to decouple scoring from the UI
=======
# FraudGraph-AI
Explainable Real-Time Fraud Detection.

## Problem at hand
AI fraud models generate too many false positives, while sophisticated fraud rings operate across multiple accounts and transactions.

## Solution
FraudGraph AI, a graph-based fraud detection system where accounts, cards, devices and transactions are nodes/edges. Combining Graph Neural Networks/graph algorithms with anomaly detection with added SHAP/explainability, so every flagged transaction gets a human-readable reason. 
>>>>>>> 7ca87c377bd818908ec7e8dfc8bdca692708cd1d

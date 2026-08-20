# FraudGraph AI

Explainable, graph-based financial fraud detection and investigation platform.

> **This is a research/prototype system, not a production banking fraud
> engine.** It's built to demonstrate an end-to-end approach — graph
> modeling, GraphSAGE, explainability, and an analyst investigation
> workflow — on a public dataset, at a scale that runs on a laptop.

## 1. Problem

Fraud models built purely on transaction-level features generate too many
false positives, because they look at each transaction in isolation.
Real fraud — especially coordinated fraud rings — shows up in the
*relationships* between accounts, cards, and merchants: a handful of
cards funneling transactions through the same small set of merchants, or
a burst of new accounts sharing a device or a payment instrument. Rule
engines catch some of this but don't generalize; black-box ML models
catch more of it but can't tell an analyst *why*.

## 2. Motivation

The Bank for International Settlements and other financial-crime
research consistently point to network-based analysis as the tool best
suited to detecting coordinated fraud, because it's the structure of
the connections — not any single transaction — that gives coordinated
fraud away. That's the premise of this project.

## 3. Solution

Model accounts, cards, and merchants as a graph, train a Graph Neural
Network (GraphSAGE) on top of it to score individual transactions, back
that up with a transparent rule-based baseline, and surface every score
with three explicitly separated categories of evidence — rule signals,
network/graph signals, and model-derived signals — so an analyst never
just sees "97% fraud probability" with no explanation.

## 4. Architecture

```
                 ┌──────────────────────┐
                 │   Streamlit UI        │   Fraud Analyst Dashboard
                 │   app.py + ui/*.py     │   11 pages, see §15
                 └──────────┬────────────┘
                            │ imports directly (see note below)
                            ▼
                 ┌──────────────────────┐
                 │   fraudgraph package   │
                 │   (src/fraudgraph/)    │
                 └──────────┬────────────┘
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   Fraud Models        Graph Engine        Database
   anomaly.py           NetworkX /          SQLite
   gnn.py (GraphSAGE)   PyTorch Geometric   (cases.py)
          │
          ▼
   Explainability
   explainability.py — rule / network / GNNExplainer signals

                 ┌──────────────────────┐
                 │   FastAPI backend      │   api/main.py + routes/*.py
                 │   (independent service)│   mirrors the same read/write
                 └──────────────────────┘   operations over HTTP
```

**Note on the API:** the spec calls for the UI to call the API "where
practical." In this prototype, `app.py` imports `fraudgraph` modules
directly rather than calling the FastAPI service over HTTP — this keeps
`streamlit run app.py` working standalone with zero extra moving parts,
which the spec also requires. The FastAPI service (`uvicorn
api.main:app`) is a fully working, independently runnable layer over the
same operations (see §16), ready to be the thing the UI calls once this
moves toward an actual client/server deployment.

## 5. Dataset

[IBM TabFormer](https://github.com/IBM/TabFormer) credit card
transactions (`data/data_transaction.csv`, not committed — see
`.gitignore`):

- 2,000 users, ~24.4M transactions, 1991–2020
- ~0.12% fraud rate overall
- Columns: User, Card, Year/Month/Day/Time, Amount, Use Chip, Merchant
  Name/City/State/Zip, MCC, Errors?, Is Fraud?

## 6. Data pipeline

```
data/data_transaction.csv (raw, 24.4M rows, 2.3GB)
        │  fraudgraph.data_prep   — DuckDB, out-of-core CSV → Parquet
        ▼
data/processed/transactions_full.parquet
data/processed/transactions_sample.parquet   (300 users + every fraud-labeled
                                               user, restricted to 2019+)
        │  fraudgraph.graph_build — networkx User→Card→Merchant graph
        ▼
data/processed/graph.pkl
        │  fraudgraph.anomaly     — rule-based scoring
        │  fraudgraph.gnn         — GraphSAGE training/inference
        │  fraudgraph.scoring     — merges rule + GNN + network signals
        ▼
data/processed/alerts_master.parquet   ← single source of truth for every
                                          dashboard page and API route
        │  fraudgraph.clusters    — Louvain community detection
        ▼
data/processed/clusters.parquet, cluster_members.parquet
        │  fraudgraph.cases       — SQLite (analyst decisions, cases)
        ▼
data/processed/fraudgraph.db
```

**Never loads the full 24M-row CSV into memory at once** — `data_prep.py`
uses DuckDB for the CSV→Parquet conversion, and every downstream stage
works off the 678K-row Parquet sample or smaller.

### Why a sample, not the full 24M rows?

The raw data has only 2,000 users but ~29 years of daily history each —
the 24M rows come from history depth, not breadth. `data_prep.py` keeps
transactions from 2019 onward for 300 users plus every user who ever has
a fraud transaction (so positives aren't lost), producing a 678K-row
working sample. This is what the graph, model, and dashboard operate on.

## 7. Graph construction

Tripartite graph: **Card ↔ Transaction ↔ Merchant** (`fraudgraph.gnn.build_graph`,
also mirrored as a `networkx` graph in `fraudgraph.graph_build` for the
original prototype visualization). Transaction nodes carry the real
per-transaction features (amount, time-of-day/day-of-week, channel,
error flag); card and merchant nodes contribute structure only. Card IDs
are namespaced per-user (`user_id:card_id`) since the raw data only
guarantees card uniqueness within a user.

- **694,300 nodes** (1,502 cards, 677,930 transactions, 14,868 merchants)
- **2,711,720 edges**

## 8. Rule-based baseline (`fraudgraph.anomaly`)

No training required — five hand-written signals a fraud analyst would
recognize, each contributing to a weighted composite score:

- transaction velocity (count in the trailing hour, same card)
- amount z-score vs. that user's own historical spending
- first-time merchant for this user
- merchant rarity (very few distinct users have used it)
- swipe/chip error flag present

## 9. GraphSAGE model (`fraudgraph.gnn`)

2-layer GraphSAGE, hidden dim 32, class-weighted BCE loss (pos_weight
for the ~0.3% fraud base rate in the sample).

### Temporal evaluation

Train on the earliest 50% of transactions, validate on the next 10%,
test on the most recent 40% — no future information leaks into training,
the same constraint a production system would face. (Split points landed
at 50/10/40 rather than a more standard 70/15/15 because fraud labels in
this sample only occur in the first ~70% of the time-sorted rows — a
property of the source dataset, not a modeling choice; see the comment
in `temporal_split()`.)

## 10. Results

Metrics computed live from `alerts_master.parquet` on the temporal test
split (`fraudgraph.metrics`), not hardcoded:

| Model | ROC-AUC | PR-AUC | Precision@200 | Lift@200 |
|---|---|---|---|---|
| Rule-based baseline | 0.739 | 0.0034 | 1.0% | **7.7x** |
| **GraphSAGE** | **0.897** | **0.0142** | **5.0%** | **38.3x** |

(Test split: 271,172 transactions, 0.131% fraud base rate.)

GraphSAGE substantially outperforms the rule baseline on every metric,
while the rule baseline remains useful as a transparent, zero-training
fallback signal and as one of the three "why" evidence categories.

### Score calibration

The raw model sigmoid output is poorly calibrated for this base rate —
heavy class weighting during training pushes a large share of the
population toward high raw scores, even though the *ranking* (what
precision@K and ROC-AUC measure) is good. `fraudgraph.scoring` Platt-scales
the raw score against the held-out validation split into a real
calibrated probability, then min-max-rescales that onto a 0–100 display
score so the risk buckets (LOW/MEDIUM/HIGH/CRITICAL) have a realistic
shape. Both transforms are monotonic, so they do not change the
rank-based metrics above.

GNN and rule-based scores are kept **separate** everywhere in the UI —
no arbitrary combined score is computed or hidden from the analyst.

## 11. Explainability (`fraudgraph.explainability`)

Every flagged transaction's "Why?" panel on the Investigation page shows
three explicitly labeled categories, so an analyst never has to guess
what's a heuristic vs. what's model-derived:

- **Rule signals** — straight from `anomaly.py` (velocity, z-score, new
  merchant, merchant rarity, error flag)
- **Network signals** — graph connectivity computed in `scoring.py`
  (e.g. "merchant is also used by N other high-risk cards", "card shares
  a merchant with M other suspicious accounts")
- **Model signals** — [GNNExplainer](https://arxiv.org/abs/1903.03894)
  (via `torch_geometric.explain`), run on-demand on a bounded 2-hop
  subgraph around the transaction (not the full 694K-node graph, for
  interactivity), returning real top-contributing input features and
  top-contributing edges/neighbors from the trained model — nothing here
  is invented; it's the actual explainer output for that transaction.

## 12. Fraud-ring / suspicious-cluster detection (`fraudgraph.clusters`)

Builds a card-level graph where two cards are linked if they share a
merchant *and* at least one of those shared transactions is high-risk,
then runs Louvain community detection over it. Clusters are reported as
**"potential fraud rings" / "suspicious clusters"** — this is a proxy
signal for coordinated activity, not a claim of confirmed criminal
organization. Each cluster shows user/card/merchant/transaction counts,
known-fraud count, risk score, and first/last activity, with an
interactive graph view.

## 13. Scoring your own data (`fraudgraph.scoring.score_uploaded`)

The **Upload Data** page accepts a CSV/Excel file in the same raw column
schema as the source dataset (User, Card, Year, Month, Day, Time, Amount,
Use Chip, Merchant Name, Merchant City, Merchant State, Zip, MCC —
`Errors?` and `Is Fraud?` optional). It's scored with the **already-trained**
GraphSAGE model and the rule engine — no retraining, since GraphSAGE
aggregates neighbor features rather than memorizing fixed node embeddings,
so it generalizes to a fresh graph built from the upload. The Platt
calibrator and 0-100 display scale fit during `scoring.build_master()`
are persisted (`data/processed/calibrator.pkl`, `score_scale.json`) and
reused here, so an upload's risk scores land on the exact same scale as
the main dashboard's, rather than fitting an unstable calibration on a
handful of rows.

Scope/limits: network/graph signals for an uploaded transaction are
computed only within that upload's own batch, not cross-referenced
against the main 678K-transaction sample. Arbitrary column schemas aren't
supported — the file must match the raw schema above (see `sample/sample.csv`
for a working example, including some real fraud-labeled rows).

## 14. Accounts & persistence (`fraudgraph.auth`, `fraudgraph.batches`)

The dashboard is gated behind a login screen (sign up / log in). Passwords
are salted and PBKDF2-hashed (200,000 iterations) in the same SQLite DB as
cases. Signing in issues a session token carried in the page URL
(`?token=...`) — that's what makes login survive a browser refresh rather
than living only in Streamlit's in-memory `session_state`.

Every file scored on the **Upload Data** page is saved as a batch tied to
the signed-in user: metadata in SQLite, the scored transactions in
Parquet (`data/processed/uploads/`). Upload history persists across
refreshes, logouts, and app restarts, each user only sees their own
batches, and any batch can be downloaded as a CSV report. This is
prototype-grade auth — no rate limiting, no password reset, no email
verification — not something to point at real credentials.

## 15. Dashboard (`app.py` + `src/fraudgraph/ui/`)

Sidebar-navigated Streamlit app, 11 pages (login-gated — see §14):

1. **Overview** — KPIs, fraud-risk-over-time, risk distribution, model
   comparison table, recent high-risk alerts
2. **Fraud Alerts** — analyst work queue over `alerts_master.parquet`,
   filterable by risk/status/split/amount/merchant/card, searchable, sortable
3. **Investigation** — the centerpiece: transaction overview, GraphSAGE
   vs. rule-based score bars, the three-category "why" panel (with
   on-demand GNNExplainer), related-transaction history (same
   card/user/merchant), interactive network view, and analyst decision
   buttons (Confirm/Dismiss/Escalate/Create Case)
4. **Network Explorer** — search any user/card/merchant/transaction ID,
   explore its neighborhood at 1–3 hops, filter by risk
5. **Fraud Rings** — suspicious clusters table + per-cluster graph and
   transaction drill-down
6. **Transactions** — searchable/paginated transaction explorer
7. **Upload Data** — score your own CSV/Excel with the pretrained model
   (see §13), saved per-account with history and CSV report downloads
8. **Model Performance** — real ROC/PR curves and metrics, rule vs.
   GraphSAGE, model architecture info
9. **Model Monitoring** — prediction distribution, feature drift (PSI,
   train vs. test split), missing values, live-measured batch inference
   latency, honest "not implemented" notes for what isn't real (concept
   drift, production request-level latency)
10. **Cases** — SQLite-backed case management: create a case from any
    transaction, track status/priority/notes
11. **Settings / About** — dataset, graph stats, model architecture,
    tech stack, accounts/persistence, and an explicit limitations section

Graph views default to a 1-hop, 100-node neighborhood and cap at 300
nodes even when expanded, so nothing renders an unreadable hairball.
Everything expensive (the graph, the trained model, the master alerts
table, cluster detection) is cached with `st.cache_resource` /
`st.cache_data` so navigating the app doesn't retrain or rebuild on
every click.

## 16. API (`api/`)

FastAPI service mirroring the dashboard's read/write operations —
independently runnable, not required for the Streamlit app (see §4):

```
GET  /health
GET  /alerts                       (filters: risk_level, status, split, limit, offset)
GET  /alerts/{alert_id}
POST /alerts/{alert_id}/decision
GET  /transactions                 GET /transactions/{transaction_id}
GET  /cards/{card_key}
GET  /users/{user_id}
GET  /merchants/{merchant_id}
GET  /network/{entity_id}?depth=1
GET  /clusters                     GET /clusters/{cluster_id}
GET  /model/metrics                GET /model/status
GET  /cases   POST /cases          PATCH /cases/{case_id}
```

## 17. Installation

```
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt
```

(`torch` is installed separately first from the CPU-only index — the
default PyPI wheel bundles CUDA and is much larger than needed here.)

Place the IBM TabFormer CSV at `data/data_transaction.csv`.

## 18. Running locally

Build the pipeline once (each step caches its output, so this only needs
to be redone if you change the data or the code):

```
.venv\Scripts\python -m src.fraudgraph.data_prep      # ~2 min, one-time
.venv\Scripts\python -m src.fraudgraph.graph_build
.venv\Scripts\python -m src.fraudgraph.anomaly
.venv\Scripts\python -m src.fraudgraph.gnn             # ~5-10 min on CPU
.venv\Scripts\python -m src.fraudgraph.scoring          # merges everything, ~1 min
```

Then launch the dashboard:

```
.venv\Scripts\streamlit run app.py
```

First time in, use the **Create account** tab on the login screen — the
dashboard is gated behind login (see §14), and your Upload Data history
is tied to that account.

Or the API (optional, independent of the dashboard):

```
.venv\Scripts\uvicorn api.main:app --reload
```

`app.py` will error with a clear message (not a stack trace) if the
pipeline hasn't been run yet.

## 19. Example investigation workflow

1. Analyst opens the dashboard → **Overview** shows current alert volume
   and risk distribution.
2. Goes to **Fraud Alerts**, filters to CRITICAL, sorts by "Most connected."
3. Opens a transaction → **Investigation** page: sees GraphSAGE score
   97%+, rule score comparatively lower, reads the "why" — a rule signal
   (velocity), a network signal (shares a merchant with 3 other
   high-risk cards), and runs GNNExplainer to see which specific
   neighboring node contributed most.
4. Checks **Related Transaction History** for the same card — spots two
   more recent high-score transactions.
5. Opens the **network view**, expands to 2 hops, sees the card sitting
   in a dense little cluster of merchants shared with other flagged cards.
6. Jumps to **Fraud Rings**, finds the cluster this card belongs to —
   5 users, 4 cards, 2 merchants, 11 of 38 transactions already
   fraud-labeled.
7. Returns to **Investigation**, clicks **Create Case**, adds a note
   ("Card connected to 3 previously flagged merchants via RING0038"),
   sets priority CRITICAL.
8. Confirms the alert as fraud; the case and the transaction status both
   update.

## 20. Limitations

- Research/prototype system — not a production banking fraud engine.
- Working sample is 300 users (+ all fraud-labeled users), 2019+, not
  the full 24.4M-row dataset.
- Fraud labels in the sample only occur in the first ~70% of the
  time-sorted rows — a source-data artifact that constrains where
  positive examples exist for evaluation.
- GNNExplainer runs on a bounded 2-hop subgraph for interactivity, not
  the full graph.
- Model Monitoring reflects one static batch-scoring pass, not a live
  production pipeline — no concept-drift detection or real
  request-level latency tracking.
- The FastAPI service and the Streamlit app both read the same
  parquet/SQLite files directly rather than the UI calling the API over
  HTTP (see §4 for why).
- No combined GNN+rule risk score — shown separately, deliberately, so
  no arbitrary weighting is hidden from the analyst.
- Auth is prototype-grade: PBKDF2-hashed passwords, but no rate limiting,
  password reset, or email verification. Don't reuse a real password.

## 21. Future work

- SHAP for tabular feature attribution alongside GNNExplainer
- Wire the Streamlit UI to call the FastAPI service over HTTP instead of
  importing `fraudgraph` directly, for an actual client/server split
- Concept-drift and production-latency monitoring
- Expand the working sample beyond 300 users / 2019+ as infra allows
  (e.g. a real database instead of parquet + SQLite)
- Community detection alternatives (label propagation) alongside Louvain
  for fraud-ring detection, with a proper held-out evaluation of ring
  quality against labeled coordinated-fraud cases if such labels become
  available

## Repository structure

```
fraudgraph-ai/
├── app.py                     Streamlit entrypoint (router)
├── api/
│   ├── main.py                 FastAPI app
│   ├── services.py              plain-Python cached data/model access
│   └── routes/                  alerts, transactions, network, clusters, cases, model
├── src/fraudgraph/
│   ├── data_prep.py             CSV → Parquet (DuckDB)
│   ├── graph_build.py           networkx graph (original prototype)
│   ├── anomaly.py                rule-based scoring
│   ├── gnn.py                    GraphSAGE model + training
│   ├── scoring.py                 merges rule + GNN + network signals → alerts_master
│   ├── explainability.py          rule/network/GNNExplainer signal extraction
│   ├── clusters.py                Louvain fraud-ring detection
│   ├── cases.py                   SQLite case management
│   ├── auth.py                     SQLite accounts + refresh-surviving sessions
│   ├── batches.py                  per-user persisted upload history
│   ├── metrics.py                 live evaluation metrics (no hardcoded numbers)
│   ├── netviz.py                   shared network-graph rendering
│   ├── store.py                    Streamlit-cached data/model access
│   └── ui/                        one module per dashboard page (incl. upload_data.py)
├── sample/
│   └── sample.csv                  example file for the Upload Data page
├── data/
│   ├── data_transaction.csv       raw (not committed)
│   └── processed/                  parquet, graph, model, db, uploads/ (not committed)
├── requirements.txt
└── README.md
```

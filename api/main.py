"""
FraudGraph AI backend API.

Run:
    uvicorn api.main:app --reload

This is a thin read/write layer over the same fraudgraph modules the
Streamlit app uses (src/fraudgraph/scoring.py, gnn.py, clusters.py,
cases.py) — the app.py dashboard does not currently call this over HTTP
(it imports the modules directly, per the spec's requirement that
`streamlit run app.py` keeps working standalone), but every read/write
operation the dashboard performs is also available here for external
integration or testing.
"""
from fastapi import FastAPI

from .routes import alerts, cases, clusters, model, network, transactions

app = FastAPI(title="FraudGraph AI API", version="1.0")

app.include_router(alerts.router)
app.include_router(transactions.router)
app.include_router(network.router)
app.include_router(clusters.router)
app.include_router(cases.router)
app.include_router(model.router)


@app.get("/health")
def health():
    return {"status": "ok"}

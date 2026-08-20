from fastapi import APIRouter

from .. import services

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metrics")
def model_metrics():
    from fraudgraph import metrics

    df = services.get_alerts_df()
    comp = metrics.model_comparison(df)
    return comp


@router.get("/status")
def model_status():
    data, meta, model = services.get_graph_and_model()
    df = services.get_alerts_df()
    return {
        "model": "GraphSAGE",
        "version": "v1",
        "layers": 2,
        "evaluation": "temporal split (50/10/40)",
        "graph_nodes": data.num_nodes,
        "graph_edges": data.num_edges,
        "n_transactions": len(df),
        "high_risk_rate": round(float(df["risk_level"].isin(["HIGH", "CRITICAL"]).mean()), 4),
    }

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .. import services

router = APIRouter(prefix="/alerts", tags=["alerts"])


class DecisionRequest(BaseModel):
    status: str  # OPEN | UNDER REVIEW | CONFIRMED FRAUD | DISMISSED | ESCALATED


@router.get("")
def list_alerts(
    risk_level: str | None = Query(None, description="LOW, MEDIUM, HIGH, or CRITICAL"),
    status: str | None = None,
    split: str = Query(
        "val,test",
        description="Comma-separated: train, val, test. Defaults to val,test — excludes "
        "transactions whose fraud label was fed directly into the training loss, so alerts "
        "reflect genuine held-out predictions rather than memorized training examples.",
    ),
    limit: int = 50,
    offset: int = 0,
):
    df = services.merged_alerts()
    if risk_level:
        df = df[df["risk_level"] == risk_level.upper()]
    if status:
        df = df[df["status"] == status.upper()]
    allowed_splits = [s.strip() for s in split.split(",") if s.strip()]
    if allowed_splits:
        df = df[df["split"].isin(allowed_splits)]
    df = df.sort_values("gnn_score_pct", ascending=False)
    total = len(df)
    page = df.iloc[offset: offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "alerts": services.df_to_json(page)}


@router.get("/{alert_id}")
def get_alert(alert_id: str):
    df = services.merged_alerts()
    match = df[df["transaction_id"] == alert_id]
    if match.empty:
        raise HTTPException(404, f"No alert with id {alert_id}")
    return services.row_to_json(match.iloc[0])


@router.post("/{alert_id}/decision")
def decide_alert(alert_id: str, body: DecisionRequest):
    from fraudgraph import cases as case_store

    if body.status not in case_store.STATUSES:
        raise HTTPException(400, f"status must be one of {case_store.STATUSES}")
    df = services.get_alerts_df()
    if alert_id not in set(df["transaction_id"]):
        raise HTTPException(404, f"No alert with id {alert_id}")
    case_store.set_alert_status(alert_id, body.status)
    return {"transaction_id": alert_id, "status": body.status}

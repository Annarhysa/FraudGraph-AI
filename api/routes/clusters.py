from fastapi import APIRouter, HTTPException

from .. import services

router = APIRouter(prefix="/clusters", tags=["clusters"])


@router.get("")
def list_clusters():
    clusters_df, _ = services.get_clusters()
    return {"total": len(clusters_df), "clusters": services.df_to_json(clusters_df)}


@router.get("/{cluster_id}")
def get_cluster(cluster_id: str):
    clusters_df, members_df = services.get_clusters()
    match = clusters_df[clusters_df["cluster_id"] == cluster_id]
    if match.empty:
        raise HTTPException(404, f"No cluster with id {cluster_id}")

    card_keys = members_df[members_df["cluster_id"] == cluster_id]["card_key"].tolist()
    df = services.get_alerts_df()
    txns = df[df["card_key"].isin(card_keys)].sort_values("gnn_score_pct", ascending=False).head(100)

    result = services.row_to_json(match.iloc[0])
    result["card_keys"] = card_keys
    result["transactions"] = services.df_to_json(txns)
    return result

from fastapi import APIRouter, HTTPException

from .. import services

router = APIRouter(tags=["network"])


def _resolve_node_id(entity_id: str, df, meta):
    if entity_id.upper().startswith("TXN"):
        match = df[df["transaction_id"].str.upper() == entity_id.upper()]
        if len(match):
            row_idx = int(match.iloc[0]["transaction_id"].replace("TXN", ""))
            return meta["txn_offset"] + row_idx
    if ":" in entity_id:
        keys = list(dict.fromkeys((df["user_id"].astype(str) + ":" + df["card_id"].astype(str)).values))
        if entity_id in keys:
            return keys.index(entity_id)
    try:
        merchant_id = int(entity_id)
        merchants = df["merchant_id"].unique().tolist()
        if merchant_id in merchants:
            return meta["txn_offset"] + meta["n_txn"] + merchants.index(merchant_id)
    except ValueError:
        pass
    return None


@router.get("/network/{entity_id}")
def get_network(entity_id: str, depth: int = 1):
    from fraudgraph import netviz

    df = services.get_alerts_df()
    data, meta, model = services.get_graph_and_model()

    node_id = _resolve_node_id(entity_id, df, meta)
    if node_id is None:
        raise HTTPException(404, f"No entity found for id {entity_id}")

    sub_nodes, sub_edge_index, center_local = netviz.get_ego_subgraph(node_id, data.edge_index, depth)

    nodes = []
    for local_idx, global_id in enumerate(sub_nodes):
        t = netviz.node_type(int(global_id), meta)
        nodes.append({
            "id": int(global_id),
            "type": t,
            "label": netviz.node_label(int(global_id), meta, df),
            "is_center": local_idx == center_local,
        })
    edges = [{"src": int(sub_nodes[a]), "dst": int(sub_nodes[b])} for a, b in sub_edge_index.numpy().T]

    return {"center": entity_id, "depth": depth, "nodes": nodes, "edges": edges}

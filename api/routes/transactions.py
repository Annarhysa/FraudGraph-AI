from fastapi import APIRouter, HTTPException, Query

from .. import services

router = APIRouter(tags=["entities"])


@router.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: str):
    df = services.merged_alerts()
    match = df[df["transaction_id"] == transaction_id]
    if match.empty:
        raise HTTPException(404, f"No transaction with id {transaction_id}")
    return services.row_to_json(match.iloc[0])


@router.get("/transactions")
def list_transactions(limit: int = 50, offset: int = 0, min_amount: float = 0, fraud_only: bool = False):
    df = services.merged_alerts()
    df = df[df["amount"] >= min_amount]
    if fraud_only:
        df = df[df["is_fraud"]]
    df = df.sort_values("txn_ts", ascending=False)
    total = len(df)
    page = df.iloc[offset: offset + limit]
    return {"total": total, "limit": limit, "offset": offset, "transactions": services.df_to_json(page)}


@router.get("/cards/{card_key}")
def get_card(card_key: str):
    df = services.merged_alerts()
    txns = df[df["card_key"] == card_key]
    if txns.empty:
        raise HTTPException(404, f"No card with id {card_key}")
    return {
        "card_key": card_key,
        "n_transactions": len(txns),
        "n_fraud_transactions": int(txns["is_fraud"].sum()),
        "avg_risk_score": round(float(txns["gnn_score_pct"].mean()), 1),
        "max_risk_score": round(float(txns["gnn_score_pct"].max()), 1),
        "transactions": services.df_to_json(txns.sort_values("txn_ts", ascending=False).head(50)),
    }


@router.get("/users/{user_id}")
def get_user(user_id: int):
    df = services.merged_alerts()
    txns = df[df["user_id"] == user_id]
    if txns.empty:
        raise HTTPException(404, f"No user with id {user_id}")
    return {
        "user_id": user_id,
        "n_cards": txns["card_key"].nunique(),
        "n_transactions": len(txns),
        "n_fraud_transactions": int(txns["is_fraud"].sum()),
        "avg_risk_score": round(float(txns["gnn_score_pct"].mean()), 1),
        "transactions": services.df_to_json(txns.sort_values("txn_ts", ascending=False).head(50)),
    }


@router.get("/merchants/{merchant_id}")
def get_merchant(merchant_id: int):
    df = services.merged_alerts()
    txns = df[df["merchant_id"] == merchant_id]
    if txns.empty:
        raise HTTPException(404, f"No merchant with id {merchant_id}")
    return {
        "merchant_id": merchant_id,
        "n_cards": txns["card_key"].nunique(),
        "n_transactions": len(txns),
        "n_fraud_transactions": int(txns["is_fraud"].sum()),
        "avg_risk_score": round(float(txns["gnn_score_pct"].mean()), 1),
        "transactions": services.df_to_json(txns.sort_values("txn_ts", ascending=False).head(50)),
    }

"""
Per-user persisted upload batches — what makes "score my file" survive a
refresh/restart instead of living only in Streamlit's in-memory
session_state. Batch metadata lives in SQLite; the scored transactions
themselves are parquet files (same pattern as the rest of the pipeline).
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/processed/fraudgraph.db")
BATCH_DIR = Path("data/processed/uploads")


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                filename TEXT,
                created_at TEXT NOT NULL,
                n_transactions INTEGER,
                n_high_risk INTEGER,
                n_critical INTEGER,
                has_fraud_labels INTEGER
            )
        """)


def save_batch(user_id: int, filename: str, scored_df: pd.DataFrame) -> str:
    init_db()
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batch_id = f"BATCH{int(datetime.now(timezone.utc).timestamp() * 1000)}"

    # Re-key transaction_id with the batch prefix so it's globally unique —
    # plain "UPLOAD000003" would collide across different users' batches
    # once alert-decision status is tracked by transaction_id.
    scored_df = scored_df.copy()
    scored_df["transaction_id"] = [f"{batch_id}-{i:06d}" for i in range(len(scored_df))]

    scored_df.to_parquet(BATCH_DIR / f"{batch_id}.parquet")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO batches (batch_id, user_id, filename, created_at, n_transactions, "
            "n_high_risk, n_critical, has_fraud_labels) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                batch_id, user_id, filename, datetime.now(timezone.utc).isoformat(),
                len(scored_df),
                int(scored_df["risk_level"].isin(["HIGH", "CRITICAL"]).sum()),
                int((scored_df["risk_level"] == "CRITICAL").sum()),
                int(bool(scored_df["has_fraud_labels"].iloc[0])) if len(scored_df) else 0,
            ),
        )
    return batch_id


def list_batches(user_id: int) -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM batches WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def parse_row_index(transaction_id: str) -> int:
    """Recovers the row position within its batch from a "{batch_id}-{i:06d}"
    transaction_id — that position is also the transaction's node offset
    within the graph built from that batch's dataframe."""
    return int(transaction_id.rsplit("-", 1)[-1])


def load_batch(batch_id: str) -> pd.DataFrame:
    return pd.read_parquet(BATCH_DIR / f"{batch_id}.parquet").reset_index(drop=True)


def delete_batch(batch_id: str, user_id: int) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM batches WHERE batch_id = ? AND user_id = ?", (batch_id, user_id))
    path = BATCH_DIR / f"{batch_id}.parquet"
    if path.exists():
        path.unlink()

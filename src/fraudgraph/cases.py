"""
Case management + alert-decision persistence, backed by SQLite (no extra
infra to run locally — this is the "Database" box in the architecture).
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/processed/fraudgraph.db")

STATUSES = ["OPEN", "UNDER REVIEW", "CONFIRMED FRAUD", "DISMISSED", "ESCALATED"]
PRIORITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


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
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                transaction_id TEXT NOT NULL,
                analyst TEXT,
                status TEXT NOT NULL,
                priority TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_decisions (
                transaction_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_case(transaction_id: str, analyst: str = "analyst", priority: str = "MEDIUM", notes: str = "") -> str:
    init_db()
    with _connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        case_id = f"CASE{n + 1:05d}"
        now = _now()
        conn.execute(
            "INSERT INTO cases (case_id, transaction_id, analyst, status, priority, notes, created_at, updated_at) "
            "VALUES (?, ?, ?, 'OPEN', ?, ?, ?, ?)",
            (case_id, transaction_id, analyst, priority, notes, now, now),
        )
    return case_id


def list_cases() -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM cases ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_case(case_id: str) -> dict | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        return dict(row) if row else None


def update_case(case_id: str, status: str | None = None, notes: str | None = None,
                 priority: str | None = None) -> None:
    init_db()
    with _connect() as conn:
        current = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if current is None:
            raise ValueError(f"No such case: {case_id}")
        conn.execute(
            "UPDATE cases SET status = ?, notes = ?, priority = ?, updated_at = ? WHERE case_id = ?",
            (
                status or current["status"],
                notes if notes is not None else current["notes"],
                priority or current["priority"],
                _now(),
                case_id,
            ),
        )


def set_alert_status(transaction_id: str, status: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO alert_decisions (transaction_id, status, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(transaction_id) DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at",
            (transaction_id, status, _now()),
        )


def get_alert_statuses() -> dict[str, str]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT transaction_id, status FROM alert_decisions").fetchall()
        return {r["transaction_id"]: r["status"] for r in rows}

"""
Lightweight username/password auth, backed by the same SQLite DB as cases.

Not production-grade security (no rate limiting, no password-reset flow,
no HTTPS enforcement — that's the app's job to provide). Good enough for
a prototype that needs real per-user accounts and sessions that survive a
page refresh.

Sessions persist via a token in the page's URL query string (?token=...),
which Streamlit keeps across a browser refresh — that's what makes login
(and therefore a user's saved uploads) survive a refresh instead of living
only in in-memory session_state.
"""
import hashlib
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/processed/fraudgraph.db")

MIN_PASSWORD_LENGTH = 6


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
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000).hex()


def create_user(username: str, password: str) -> int:
    username = username.strip()
    if not username:
        raise ValueError("Username can't be empty")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")

    init_db()
    salt = secrets.token_bytes(16)
    pw_hash = _hash_password(password, salt)
    with _connect() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (username, pw_hash, salt.hex(), datetime.now(timezone.utc).isoformat()),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' is already taken")
        return cur.lastrowid


def verify_user(username: str, password: str) -> dict | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
        if row is None:
            return None
        if _hash_password(password, bytes.fromhex(row["salt"])) == row["password_hash"]:
            return {"user_id": row["user_id"], "username": row["username"]}
        return None


def create_session(user_id: int) -> str:
    init_db()
    token = secrets.token_urlsafe(32)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, datetime.now(timezone.utc).isoformat()),
        )
    return token


def get_user_from_token(token: str) -> dict | None:
    if not token:
        return None
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT sessions.user_id AS user_id, users.username AS username "
            "FROM sessions JOIN users ON users.user_id = sessions.user_id "
            "WHERE token = ?",
            (token,),
        ).fetchone()
        return dict(row) if row else None


def delete_session(token: str) -> None:
    if not token:
        return
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))

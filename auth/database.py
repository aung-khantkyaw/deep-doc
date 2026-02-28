"""SQLite database layer for users, chat history, and system settings."""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = "data/deepdoc.db"

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables and seed default system settings."""
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            role          TEXT    NOT NULL DEFAULT 'user',
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            mode       TEXT    NOT NULL,          -- 'chat' | 'quiz'
            role       TEXT    NOT NULL,          -- 'user' | 'assistant'
            content    TEXT    NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS system_settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)

    # Default system settings (INSERT OR IGNORE — never overwrite admin changes)
    defaults = {
        "model_name":   "llama3.1:8b",
        "top_k":        "4",
        "temperature":  "0.2",
        "chunk_size":   "512",
        "chunk_overlap": "80",
    }
    for key, val in defaults.items():
        cur.execute(
            "INSERT OR IGNORE INTO system_settings (key, value) VALUES (?, ?)",
            (key, val),
        )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_user_by_username(username: str) -> sqlite3.Row | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return row


def get_user_by_id(user_id: int) -> sqlite3.Row | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return row


def create_user(
    username: str,
    email: str,
    password_hash: str,
    role: str = "user",
) -> int:
    """Insert a new user. Returns the new row id."""
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
        (username, email, password_hash, role),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return user_id


def username_exists(username: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return row is not None


def email_exists(email: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM users WHERE email = ?", (email,)
    ).fetchone()
    conn.close()
    return row is not None


def admin_exists() -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM users WHERE role = 'admin' LIMIT 1"
    ).fetchone()
    conn.close()
    return row is not None


# ---------------------------------------------------------------------------
# System settings
# ---------------------------------------------------------------------------

def get_all_settings() -> dict[str, str]:
    conn = get_conn()
    rows = conn.execute("SELECT key, value FROM system_settings").fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


def update_setting(key: str, value: str) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO system_settings (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

def save_message(
    user_id: int,
    mode: str,
    role: str,
    content: str,
) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO chat_history (user_id, mode, role, content) VALUES (?, ?, ?, ?)",
        (user_id, mode, role, content),
    )
    conn.commit()
    conn.close()


def get_user_history(
    user_id: int,
    mode: str = "chat",
    limit: int = 100,
) -> list[sqlite3.Row]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT role, content, created_at
        FROM chat_history
        WHERE user_id = ? AND mode = ?
        ORDER BY created_at ASC
        LIMIT ?
        """,
        (user_id, mode, limit),
    ).fetchall()
    conn.close()
    return rows


def clear_user_history(user_id: int, mode: str = "chat") -> None:
    conn = get_conn()
    conn.execute(
        "DELETE FROM chat_history WHERE user_id = ? AND mode = ?",
        (user_id, mode),
    )
    conn.commit()
    conn.close()

"""SQLite database layer for users, chat history, and system settings."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DB_PATH = "data/deepdoc.db"

DEFAULT_LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = os.getenv("TOP_K", "4")
DEFAULT_TEMPERATURE = os.getenv("TEMPERATURE", "0.2")
DEFAULT_CHUNK_SIZE = os.getenv("CHUNK_SIZE", "512")
DEFAULT_CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", "80")

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

        CREATE TABLE IF NOT EXISTS study_rooms (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            title         TEXT    NOT NULL,
            chroma_dir    TEXT    NOT NULL,
            model_name    TEXT    NOT NULL,
            embedding_model TEXT  NOT NULL DEFAULT 'all-MiniLM-L6-v2',
            top_k         INTEGER NOT NULL,
            chunk_size    INTEGER NOT NULL,
            chunk_overlap INTEGER NOT NULL,
            temperature   REAL    NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS room_files (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id     INTEGER NOT NULL,
            file_name   TEXT    NOT NULL,
            file_path   TEXT    NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (room_id) REFERENCES study_rooms(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS system_settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)

    # Backward-compatible migration for room-based history
    columns = [
        row["name"]
        for row in cur.execute("PRAGMA table_info(chat_history)").fetchall()
    ]
    if "room_id" not in columns:
        cur.execute("ALTER TABLE chat_history ADD COLUMN room_id INTEGER")

    room_columns = [
        row["name"]
        for row in cur.execute("PRAGMA table_info(study_rooms)").fetchall()
    ]
    if "embedding_model" not in room_columns:
        cur.execute(
            "ALTER TABLE study_rooms ADD COLUMN embedding_model TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2'"
        )

    # Default system settings (INSERT OR IGNORE — never overwrite admin changes)
    defaults = {
        "model_name": DEFAULT_LLM_MODEL,
        "top_k": DEFAULT_TOP_K,
        "temperature": DEFAULT_TEMPERATURE,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
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


def update_user_profile(user_id: int, username: str, email: str) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE users SET username = ?, email = ? WHERE id = ?",
        (username, email, user_id),
    )
    conn.commit()
    conn.close()


def update_user_password(user_id: int, password_hash: str) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (password_hash, user_id),
    )
    conn.commit()
    conn.close()


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
    room_id: int | None = None,
) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO chat_history (user_id, mode, role, content, room_id) VALUES (?, ?, ?, ?, ?)",
        (user_id, mode, role, content, room_id),
    )
    conn.commit()
    conn.close()


def get_user_history(
    user_id: int,
    mode: str = "chat",
    limit: int = 100,
    room_id: int | None = None,
) -> list[sqlite3.Row]:
    conn = get_conn()
    if room_id is None:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM chat_history
            WHERE user_id = ? AND mode = ? AND room_id IS NULL
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (user_id, mode, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM chat_history
            WHERE user_id = ? AND mode = ? AND room_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (user_id, mode, room_id, limit),
        ).fetchall()
    conn.close()
    return rows


def clear_user_history(user_id: int, mode: str = "chat", room_id: int | None = None) -> None:
    conn = get_conn()
    if room_id is None:
        conn.execute(
            "DELETE FROM chat_history WHERE user_id = ? AND mode = ? AND room_id IS NULL",
            (user_id, mode),
        )
    else:
        conn.execute(
            "DELETE FROM chat_history WHERE user_id = ? AND mode = ? AND room_id = ?",
            (user_id, mode, room_id),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Study rooms (timeline of uploaded file sets)
# ---------------------------------------------------------------------------

def create_study_room(
    user_id: int,
    title: str,
    chroma_dir: str,
    model_name: str,
    embedding_model: str,
    top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    temperature: float,
) -> int:
    conn = get_conn()
    cur = conn.execute(
        """
        INSERT INTO study_rooms (
            user_id, title, chroma_dir, model_name, embedding_model,
            top_k, chunk_size, chunk_overlap, temperature
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            title,
            chroma_dir,
            model_name,
            embedding_model,
            top_k,
            chunk_size,
            chunk_overlap,
            temperature,
        ),
    )
    conn.commit()
    room_id = cur.lastrowid
    conn.close()
    return room_id


def touch_study_room(room_id: int) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE study_rooms SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (room_id,),
    )
    conn.commit()
    conn.close()


def get_study_room(room_id: int, user_id: int) -> sqlite3.Row | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM study_rooms WHERE id = ? AND user_id = ?",
        (room_id, user_id),
    ).fetchone()
    conn.close()
    return row


def list_user_study_rooms(user_id: int, limit: int = 50) -> list[sqlite3.Row]:
    conn = get_conn()
    rows = conn.execute(
        """
     SELECT r.id, r.title, r.chroma_dir, r.model_name, r.top_k, r.chunk_size, r.chunk_overlap,
         r.embedding_model, r.temperature, r.created_at, r.updated_at,
         COUNT(f.id) AS file_count
     FROM study_rooms r
     LEFT JOIN room_files f ON f.room_id = r.id
     WHERE r.user_id = ?
     GROUP BY r.id
     ORDER BY datetime(r.updated_at) DESC, r.id DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    conn.close()
    return rows


def add_room_file(room_id: int, file_name: str, file_path: str) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO room_files (room_id, file_name, file_path) VALUES (?, ?, ?)",
        (room_id, file_name, file_path),
    )
    conn.commit()
    conn.close()


def get_room_files(room_id: int) -> list[sqlite3.Row]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT file_name, file_path, uploaded_at
        FROM room_files
        WHERE room_id = ?
        ORDER BY datetime(uploaded_at) ASC, id ASC
        """,
        (room_id,),
    ).fetchall()
    conn.close()
    return rows

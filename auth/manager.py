"""High-level auth and settings management."""
from __future__ import annotations

import bcrypt

from auth.database import (
    admin_exists,
    create_user,
    email_exists,
    get_all_settings,
    get_user_by_id,
    get_user_by_username,
    update_user_password,
    update_user_profile,
    update_setting,
    username_exists,
)

ADMIN_DEFAULT_USERNAME = "admin"
ADMIN_DEFAULT_EMAIL    = "admin@deepdoc.local"
ADMIN_DEFAULT_PASSWORD = "admin123"

# ---------------------------------------------------------------------------
# Password
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def create_default_admin() -> None:
    """Create the default admin account on first run if no admin exists."""
    if not admin_exists():
        create_user(
            username=ADMIN_DEFAULT_USERNAME,
            email=ADMIN_DEFAULT_EMAIL,
            password_hash=hash_password(ADMIN_DEFAULT_PASSWORD),
            role="admin",
        )
        print(
            f"[DeepDoc] Default admin created — "
            f"username: {ADMIN_DEFAULT_USERNAME!r}  "
            f"password: {ADMIN_DEFAULT_PASSWORD!r}  "
            f"(change this after first login)"
        )


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def register(
    username: str,
    email: str,
    password: str,
) -> tuple[bool, str]:
    """Register a new user.

    Returns (success: bool, message: str).
    """
    username = username.strip()
    email    = email.strip().lower()

    if not username or not email or not password:
        return False, "All fields are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if username_exists(username):
        return False, "Username is already taken."
    if email_exists(email):
        return False, "Email is already registered."

    create_user(username, email, hash_password(password), role="user")
    return True, "Account created successfully."


def login(
    username: str,
    password: str,
) -> tuple[bool, str, dict | None]:
    """Authenticate a user.

    Returns (success, message, user_dict | None).
    user_dict contains: id, username, email, role.
    """
    row = get_user_by_username(username.strip())
    if row is None:
        return False, "Invalid username or password.", None
    if not verify_password(password, row["password_hash"]):
        return False, "Invalid username or password.", None

    user = {
        "id":       row["id"],
        "username": row["username"],
        "email":    row["email"],
        "role":     row["role"],
    }
    return True, f"Welcome back, {row['username']}!", user


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def get_system_settings() -> dict:
    """Return settings with proper Python types."""
    raw = get_all_settings()
    return {
        "model_name":    raw.get("model_name",    "llama3.1:8b"),
        "embedding_model": raw.get("embedding_model", "all-MiniLM-L6-v2"),
        "top_k":         int(raw.get("top_k",     "4")),
        "temperature":   float(raw.get("temperature", "0.2")),
        "chunk_size":    int(raw.get("chunk_size", "512")),
        "chunk_overlap": int(raw.get("chunk_overlap", "80")),
    }


def save_system_settings(
    model_name: str,
    embedding_model: str,
    top_k: int,
    temperature: float,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    update_setting("model_name",    model_name)
    update_setting("embedding_model", embedding_model)
    update_setting("top_k",         str(top_k))
    update_setting("temperature",   str(temperature))
    update_setting("chunk_size",    str(chunk_size))
    update_setting("chunk_overlap", str(chunk_overlap))


# ---------------------------------------------------------------------------
# User profile
# ---------------------------------------------------------------------------

def update_profile(user_id: int, username: str, email: str) -> tuple[bool, str, dict | None]:
    username = username.strip()
    email = email.strip().lower()

    if not username or not email:
        return False, "Username and email are required.", None
    if len(username) < 3:
        return False, "Username must be at least 3 characters.", None

    current = get_user_by_id(user_id)
    if current is None:
        return False, "User not found.", None

    if username != current["username"] and username_exists(username):
        return False, "Username is already taken.", None
    if email != current["email"] and email_exists(email):
        return False, "Email is already registered.", None

    update_user_profile(user_id, username, email)
    updated = get_user_by_id(user_id)
    if updated is None:
        return False, "Could not load updated profile.", None

    user = {
        "id": updated["id"],
        "username": updated["username"],
        "email": updated["email"],
        "role": updated["role"],
    }
    return True, "Profile updated.", user


def change_password(user_id: int, current_password: str, new_password: str) -> tuple[bool, str]:
    row = get_user_by_id(user_id)
    if row is None:
        return False, "User not found."
    if not verify_password(current_password, row["password_hash"]):
        return False, "Current password is incorrect."
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."

    update_user_password(user_id, hash_password(new_password))
    return True, "Password updated."

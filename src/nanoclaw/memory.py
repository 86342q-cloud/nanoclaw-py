"""Unified SQLite memory for all agent modes.

Single messages table with per-chat history.
Works for both DeepSeek (Anthropic SDK) and MiniMax (httpx).
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Any

from nanoclaw.config import ASSISTANT_NAME, DATA_DIR, WORKSPACE_DIR

DB_PATH = DATA_DIR / "messages.db"
SCHEMA_VERSION = 1

# Rough token estimation: ~3 chars per token for mixed RU/EN
CHARS_PER_TOKEN = 3


def _get_conn() -> sqlite3.Connection:
    """Get SQLite connection (create DB and tables if needed)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            tokens INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_messages_chat_time
        ON messages(chat_id, created_at DESC)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.execute("""
        INSERT OR IGNORE INTO memory_meta (key, value)
        VALUES ('schema_version', ?)
    """, (str(SCHEMA_VERSION),))
    conn.commit()


def estimate_tokens(text: str) -> int:
    """Rough token count. ~3 chars per token for mixed RU/EN text."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def save_message(chat_id: int, user_id: int, role: str, content: str) -> int:
    """Save a message and return its ID."""
    conn = _get_conn()
    tokens = estimate_tokens(content)
    now = time.time()
    cursor = conn.execute(
        "INSERT INTO messages (chat_id, user_id, role, content, tokens, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (chat_id, user_id, role, content, tokens, now),
    )
    conn.commit()
    conn.close()
    return cursor.lastrowid


def get_history(
    chat_id: int,
    max_messages: int = 30,
    max_tokens: int = 8000,
) -> list[dict[str, Any]]:
    """Get recent messages for a chat, trimming to stay within token budget."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, chat_id, user_id, role, content, tokens, created_at "
        "FROM messages "
        "WHERE chat_id = ? "
        "ORDER BY created_at DESC "
        "LIMIT ?",
        (chat_id, max_messages * 2),
    ).fetchall()
    conn.close()

    if not rows:
        return []

    messages = [dict(r) for r in reversed(rows)]

    total_tokens = sum(m["tokens"] for m in messages)
    while total_tokens > max_tokens and len(messages) > 2:
        removed = messages.pop(0)
        total_tokens -= removed["tokens"]

    if len(messages) > max_messages:
        messages = messages[-max_messages:]

    return messages


def clear_history(chat_id: int) -> int:
    """Delete all messages for a chat. Returns count of deleted rows."""
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


def get_chat_stats(chat_id: int) -> dict[str, Any]:
    """Get memory stats for a chat."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) as total, "
        "COALESCE(SUM(tokens), 0) as total_tokens, "
        "COALESCE(MIN(created_at), 0) as first_msg, "
        "COALESCE(MAX(created_at), 0) as last_msg "
        "FROM messages WHERE chat_id = ?",
        (chat_id,),
    ).fetchone()
    conn.close()
    return dict(row)


def format_history(history: list[dict[str, Any]]) -> str:
    """Format message history for inclusion in a prompt."""
    if not history:
        return ""

    lines = ["[Previous conversation]"]
    for msg in history:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role_label}: {msg['content']}")
    lines.append("")
    return "\n".join(lines)


_INITIAL_CLAUDE_MD = f"""# {ASSISTANT_NAME} — Корпоративный AI-ассистент

Ты — {ASSISTANT_NAME}, корпоративный ассистент в Telegram.

## Твои возможности
- Читать, писать, редактировать файлы (Read, Write, Edit)
- Запускать команды и скрипты (Bash)
- Искать в интернете (WebSearch, WebFetch)
- Отправлять сообщения пользователю (mcp__nanoclaw__send_message)
- Выполнять SQL-запросы (MCP postgres — по необходимости)

## Память
- История чата хранится в SQLite (data/messages.db)
- Этот файл (CLAUDE.md) — долговременная память, обновляй его

## Пользователи и домены
- У каждого пользователя свой домен и набор инструментов
- Инструкции домена: domains/<домен>/AGENTS.md

## Важно
- Пожелания «передай Антону» → сохраняй в таблицу suggestions
- Не выдумывай функциональность, которой нет в твоих tools
- Будь вежливым, полезным, с лёгким юмором
"""


def ensure_workspace() -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    claude_md = WORKSPACE_DIR / "CLAUDE.md"
    if not claude_md.exists():
        claude_md.write_text(_INITIAL_CLAUDE_MD)


def format_history_for_minimax(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Format history as OpenAI-compatible messages for MiniMax API."""
    result = []
    for msg in history:
        result.append({
            "role": msg["role"],
            "content": msg["content"],
        })
    return result

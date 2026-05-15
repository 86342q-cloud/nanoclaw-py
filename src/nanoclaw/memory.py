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

CHARS_PER_TOKEN = 3


def _get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            summary TEXT NOT NULL,
            topics TEXT NOT NULL DEFAULT '',
            msg_count INTEGER NOT NULL DEFAULT 0,
            started_at REAL NOT NULL,
            ended_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sessions_chat_time
        ON session_summaries(chat_id, ended_at DESC)
    """)
    conn.commit()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def save_message(chat_id: int, user_id: int, role: str, content: str) -> int:
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
    conn = _get_conn()
    cursor = conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


def get_chat_stats(chat_id: int) -> dict[str, Any]:
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
    result = []
    for msg in history:
        result.append({
            "role": msg["role"],
            "content": msg["content"],
        })
    return result


import os
import httpx
import asyncio

async def _summarize_via_deepseek(messages: list[dict[str, Any]]) -> str:
    conversation = []
    for m in messages:
        role = "Пользователь" if m["role"] == "user" else "Ассистент"
        conversation.append(f"{role}: {m['content']}")

    prompt = (
        "Сделай краткую сводку диалога. Выдели:\n"
        "1. Основные темы (2-3 слова через запятую)\n"
        "2. Что сделано / решено\n"
        "3. Что в работе / не закончено\n"
        "4. Важные факты и договорённости\n\n"
        "Диалог:\n" + "\n".join(conversation[-60:]) + "\n\n"
        "Сводка:"
    )

    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 600,
                    "temperature": 0.2,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        return ""
    except Exception:
        return ""


def end_session(chat_id: int, user_id: int, summary: str = "") -> dict[str, Any]:
    history = get_history(chat_id, max_messages=200, max_tokens=999999)
    msg_count = len(history)

    if msg_count == 0:
        return {"summary": "", "topics": "", "msg_count": 0, "kept_messages": 0}

    first_ts = history[0]["created_at"] if history else time.time()
    last_ts = history[-1]["created_at"] if history else time.time()

    kept_messages = 0

    if not summary:
        if msg_count < 20:
            summary = f"Короткая сессия ({msg_count} сообщ.)"
        else:
            summary = asyncio.run(_summarize_via_deepseek(history))
            if not summary:
                summary = f"Сессия из {msg_count} сообщений"

    if summary and msg_count >= 20:
        if msg_count > 50:
            kept_messages = 15
            conn = _get_conn()
            cutoff_id = history[-15]["id"]
            conn.execute(
                "DELETE FROM messages WHERE chat_id = ? AND id < ?",
                (chat_id, cutoff_id),
            )
            conn.commit()
            conn.close()
        else:
            clear_history(chat_id)

    topics = summary.split("\n")[0] if summary else ""
    if topics.startswith("1."):
        topics = topics[2:].strip()
    if len(topics) > 200:
        topics = topics[:200]

    conn = _get_conn()
    conn.execute(
        "INSERT INTO session_summaries (chat_id, user_id, summary, topics, msg_count, started_at, ended_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (chat_id, user_id, summary, topics, msg_count, first_ts, last_ts),
    )
    conn.commit()
    conn.close()

    return {
        "summary": summary,
        "topics": topics,
        "msg_count": msg_count,
        "kept_messages": kept_messages,
    }


def get_last_session(chat_id: int) -> dict[str, Any] | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM session_summaries WHERE chat_id = ? ORDER BY ended_at DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_session_topics(chat_id: int, limit: int = 5) -> list[str]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT DISTINCT topics FROM session_summaries WHERE chat_id = ? AND topics != '' ORDER BY ended_at DESC LIMIT ?",
        (chat_id, limit),
    ).fetchall()
    conn.close()
    return [r["topics"] for r in rows]

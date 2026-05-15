"""Domain loading and user profile management.

Each domain lives in domains/<name>/ with AGENTS.md and config.yaml.
User profiles are in SQLite (messages.db, users + role_configs tables).
"""

import json
import sqlite3
import yaml
from pathlib import Path
from typing import Any

from nanoclaw.config import DATA_DIR

DOMAINS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "domains"


def _get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATA_DIR / "messages.db"))
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create user and role tables if needed."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            telegram_username TEXT,
            department TEXT NOT NULL,
            role TEXT NOT NULL,
            domain TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS role_configs (
            role_id TEXT PRIMARY KEY,
            greeting TEXT,
            mcp_servers TEXT NOT NULL DEFAULT '[]',
            menu_json TEXT NOT NULL DEFAULT '[]',
            capabilities TEXT NOT NULL DEFAULT '{}'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            domain TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'new',
            created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
        )
    """)
    conn.commit()


def load_domain_config(domain_name: str) -> dict[str, Any] | None:
    """Load config.yaml for a domain. Returns None if domain doesn't exist."""
    config_path = DOMAINS_DIR / domain_name / "config.yaml"
    if not config_path.exists():
        return None
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_domain_agents_md(domain_name: str) -> str | None:
    """Load AGENTS.md for a domain."""
    path = DOMAINS_DIR / domain_name / "AGENTS.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def get_user_profile(user_id: int) -> dict[str, Any] | None:
    """Get user profile from SQLite. Returns None if not found."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE user_id = ? AND is_active = 1",
        (user_id,),
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def add_user(
    user_id: int,
    name: str,
    telegram_username: str,
    department: str,
    role: str,
    domain: str,
) -> None:
    """Add or update a user profile."""
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO users (user_id, name, telegram_username, department, role, domain, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?, 1)",
        (user_id, name, telegram_username, department, role, domain),
    )
    conn.commit()
    conn.close()


def get_greeting(domain_name: str, user_name: str) -> str:
    """Get domain-specific greeting for a user."""
    config = load_domain_config(domain_name)
    if config and config.get("user", {}).get("greeting_template"):
        return config["user"]["greeting_template"].format(name=user_name)
    return f"Привет, {user_name}! 👋"


def get_domain_buttons(domain_name: str) -> list[dict[str, Any]]:
    """Get button configuration for a domain."""
    config = load_domain_config(domain_name)
    if config:
        return config.get("buttons", [])
    return []


def add_suggestion(user_name: str, domain: str, message: str) -> int:
    """Save a 'передай Антону' suggestion. Returns ID."""
    conn = _get_conn()
    cursor = conn.execute(
        "INSERT INTO suggestions (user_name, domain, message) VALUES (?, ?, ?)",
        (user_name, domain, message),
    )
    conn.commit()
    conn.close()
    return cursor.lastrowid


def get_suggestions(domain: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
    """Get suggestions, optionally filtered by domain/status."""
    conn = _get_conn()
    query = "SELECT * FROM suggestions WHERE 1=1"
    params = []
    if domain:
        query += " AND domain = ?"
        params.append(domain)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def domain_has_capability(domain_name: str, capability: str) -> bool:
    """Check if domain has a specific capability enabled."""
    config = load_domain_config(domain_name)
    if not config:
        return False
    caps = config.get("capabilities", {})
    return caps.get(capability, False)


def build_system_prompt(domain_name: str, user_name: str) -> str:
    """Build domain-specific system prompt for the agent."""
    config = load_domain_config(domain_name)
    agents_md = load_domain_agents_md(domain_name)

    assistant = config.get("assistant", {}) if config else {}
    tone = assistant.get("tone", "friendly_professional")
    assistant_name = assistant.get("name", "Ассист")

    tone_map = {
        "confident_casual": "уверенный, деловой когда нужно, с лёгким юмором. На «ты». Коротко, без воды.",
        "friendly_professional": "дружелюбный и профессиональный. На «ты». Понятно и по делу.",
        "formal": "формальный, деловой, на «вы».",
        "strict": "строгий, только по делу, без болтовни.",
    }
    tone_desc = tone_map.get(tone, tone_map["friendly_professional"])

    prompt = f"""Ты — {assistant_name}, персональный AI-ассистент. Твой пользователь: {user_name}. Домен: {domain_name}.

Твой тон: {tone_desc}

ПРАВИЛА ОБЩЕНИЯ:
1. Если пользователь уходит не в тему — после 2-3 сообщений мягко верни в рабочий контекст: «Давай вернёмся к делу — у нас много планов!»
2. Если не уверен что правильно понял — УТОЧНИ СМЫСЛ: «Правильно ли я понимаю, что речь о НОВОЙ логике расчёта или обновлении старой?» Задавай ОТКРЫТЫЕ уточняющие вопросы — не бинарные «да/нет», а смысловые: «О каком именно периоде речь — май целиком или последняя неделя?», «Ты про все авто или только новые?»
3. Если это просьба/задача — зафиксируй: «Понял: нужно X. Зафиксировал.»
4. Если это просто болтовня — отвечай легко, но не затягивай.
5. Если запрос требует данных (трафик, продажи, отчёты) — предложи нажать кнопку или скажи что данные уточняются.
6. В конце длинного обсуждения предложи подвести итог: «Давай зафиксирую что решили?»

Ты НЕ говоришь «я всё умею». Ты НЕ врёшь про возможности. Ты — экспериментальный ассистент, который учится каждый день."""

    if agents_md:
        context_excerpt = agents_md[:2000]
        prompt += f"\n\nКонтекст домена:\n{context_excerpt}"

    return prompt
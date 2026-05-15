import base64
import logging
import httpx
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters

from nanoclaw.agent import run_agent
from nanoclaw.memory import clear_history, get_chat_stats
from nanoclaw.domain import (
    get_user_profile,
    get_greeting,
    get_domain_buttons,
    add_suggestion,
    load_domain_config,
    domain_has_capability,
    build_system_prompt,
)
from nanoclaw.config import (
    ASSISTANT_NAME,
    OWNER_ID,
    TELEGRAM_BOT_TOKEN,
    ANTHROPIC_API_KEY,
    POLZA_API_KEY,
    DATA_DIR,
    get_mode,
    set_mode,
    get_test_proxy,
    set_test_proxy,
    MODE_DEEPSEEK,
    MODE_MINIMAX,
)

logger = logging.getLogger(__name__)

_TELEGRAM_MAX_LENGTH = 4096
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "output"

MODE_LABELS = {
    MODE_DEEPSEEK: "🧠 DeepSeek (умный)",
    MODE_MINIMAX: "⚡ MiniMax (быстрый)",
}

MODE_DESCRIPTIONS = {
    MODE_DEEPSEEK: (
        "🧠 **DeepSeek V4 Flash**\n"
        "Всё через одну модель. Tools, веб-поиск, файлы, память.\n"
        "Медленнее, но умнее. ~$0.09/запрос."
    ),
    MODE_MINIMAX: (
        "⚡ **MiniMax M2.7-highspeed**\n"
        "Быстрые ответы без tools. Для болтовни и простых вопросов.\n"
        "Мгновенно. Самый дешёвый."
    ),
}


def _effective_user_id(update: Update) -> int:
    """Return the effective user ID — test proxy if active, else real user."""
    proxy = get_test_proxy()
    if proxy and update.effective_user.id == OWNER_ID:
        return proxy["user_id"]
    return update.effective_user.id


def _is_owner(update: Update) -> bool:
    """Check if user is owner OR registered domain user."""
    if update.effective_user is None:
        return False
    user_id = update.effective_user.id
    if user_id == OWNER_ID:
        return True
    from nanoclaw.domain import get_user_profile
    profile = get_user_profile(user_id)
    return profile is not None


def _build_mode_keyboard() -> InlineKeyboardMarkup:
    current = get_mode()
    buttons = []
    for mode, label in MODE_LABELS.items():
        prefix = "✅ " if mode == current else ""
        buttons.append([InlineKeyboardButton(f"{prefix}{label}", callback_data=f"mode_{mode}")])
    buttons.append([InlineKeyboardButton("📊 Статус", callback_data="status")])
    return InlineKeyboardMarkup(buttons)


def _build_domain_keyboard(domain_buttons: list[dict]) -> InlineKeyboardMarkup:
    buttons = []
    for btn in domain_buttons:
        buttons.append([InlineKeyboardButton(
            btn["label"],
            callback_data=f"domain_{btn['id']}"
        )])
    return InlineKeyboardMarkup(buttons)


async def _start(update: Update, context) -> None:
    user_id = _effective_user_id(update)
    username = update.effective_user.username or "no_username"
    logger.info(f"/start from user_id={user_id} @{username}")
    if not _is_owner(update):
        return
    profile = get_user_profile(user_id)

    if profile:
        greeting = _test_prefix() + get_greeting(profile["domain"], profile["name"])
        buttons = get_domain_buttons(profile["domain"])
        keyboard = _build_domain_keyboard(buttons)
        await update.message.reply_text(
            greeting,
            reply_markup=keyboard,
            parse_mode="Markdown",
        )
    else:
        current = get_mode()
        await update.message.reply_text(
            f"🤖 **{ASSISTANT_NAME}** — автономный TG-бот\n\n"
            f"Текущий режим: {MODE_LABELS[current]}\n\n"
            "Команды:\n"
            "/mode — выбрать режим\n"
            "/status — состояние бота\n"
            "/clear — сбросить память\n"
            "/test _domain_ — тестовый прокси",
            reply_markup=_build_mode_keyboard(),
            parse_mode="Markdown",
        )
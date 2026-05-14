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
)
from nanoclaw.config import (
    ASSISTANT_NAME,
    OWNER_ID,
    TELEGRAM_BOT_TOKEN,
    POLZA_API_KEY,
    get_mode,
    set_mode,
    MODE_DEEPSEEK,
    MODE_MINIMAX,
)

logger = logging.getLogger(__name__)

_TELEGRAM_MAX_LENGTH = 4096
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "output"

MODE_LABELS = {
    MODE_DEEPSEEK: "DeepSeek (umnyy)",
    MODE_MINIMAX: "MiniMax (bystryy)",
}

MODE_DESCRIPTIONS = {
    MODE_DEEPSEEK: "DeepSeek V4 Flash. Tools, web, pamyat.",
    MODE_MINIMAX: "MiniMax M2.7-highspeed. Bystrye otvety bez tools.",
}


def _is_owner(update: Update) -> bool:
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
        prefix = "V " if mode == current else ""
        buttons.append([InlineKeyboardButton(f"{prefix}{label}", callback_data=f"mode_{mode}")])
    buttons.append([InlineKeyboardButton("Status", callback_data="status")])
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
    if not _is_owner(update):
        return
    user_id = update.effective_user.id
    profile = get_user_profile(user_id)

    if profile:
        greeting = get_greeting(profile["domain"], profile["name"])
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
            f"Bot **{ASSISTANT_NAME}**\n\n"
            f"Mode: {MODE_LABELS[current]}\n\n"
            "/mode /status /clear",
            reply_markup=_build_mode_keyboard(),
            parse_mode="Markdown",
        )


async def _mode_command(update: Update, context) -> None:
    if not _is_owner(update):
        return
    current = get_mode()
    await update.message.reply_text(
        f"Mode: {MODE_LABELS[current]}\n\n{MODE_DESCRIPTIONS[current]}",
        reply_markup=_build_mode_keyboard(),
        parse_mode="Markdown",
    )


async def _status_command(update: Update, context) -> None:
    if not _is_owner(update):
        return
    chat_id = update.effective_chat.id
    current = get_mode()
    stats = get_chat_stats(chat_id)
    msg_count = stats["total"]
    tok_count = stats["total_tokens"]
    await update.message.reply_text(
        f"Status: Active\nMode: {MODE_LABELS[current]}\n"
        f"Memory: SQLite | {msg_count} msgs (~{tok_count} tokens)",
        reply_markup=_build_mode_keyboard(),
        parse_mode="Markdown",
    )


async def _mode_callback(update: Update, context) -> None:
    query = update.callback_query
    if not _is_owner(update):
        await query.answer("No access")
        return

    data = query.data

    if data == "status":
        await query.answer()
        chat_id = update.effective_chat.id
        current = get_mode()
        stats = get_chat_stats(chat_id)
        msg_count = stats["total"]
        await query.edit_message_text(
            f"Status: Active\nMode: {MODE_LABELS[current]}\nMemory: {msg_count} msgs",
            reply_markup=_build_mode_keyboard(),
            parse_mode="Markdown",
        )
        return

    if data.startswith("domain_"):
        await _handle_domain_callback(update, context)
        return

    mode = data.replace("mode_", "")
    if mode not in MODE_LABELS:
        await query.answer("Unknown mode")
        return

    old_mode = get_mode()
    if mode == old_mode:
        await query.answer(f"Already in {MODE_LABELS[mode]}")
        return

    set_mode(mode)
    await query.answer(f"Switched: {MODE_LABELS[mode]}")
    await query.edit_message_text(
        f"Mode: {MODE_LABELS[old_mode]} -> {MODE_LABELS[mode]}",
        reply_markup=_build_mode_keyboard(),
        parse_mode="Markdown",
    )


async def _clear(update: Update, context) -> None:
    if not _is_owner(update):
        return
    chat_id = update.effective_chat.id
    deleted = clear_history(chat_id)
    await update.message.reply_text(f"Memory cleared! {deleted} msgs removed.")


async def _transcribe_voice(file_path: str) -> str:
    if not POLZA_API_KEY:
        return ""
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else "ogg"
        mime_map = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg", "oga": "audio/ogg"}
        mime = mime_map.get(ext, "audio/ogg")
        payload = {
            "model": "openai/gpt-4o-mini-transcribe",
            "file": f"data:{mime};base64,{audio_b64}",
            "language": "ru",
            "response_format": "json",
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {POLZA_API_KEY}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post("https://polza.ai/api/v1/audio/transcriptions", json=payload, headers=headers)
        if r.status_code == 200:
            return r.json().get("text", "")
        return ""
    except Exception as e:
        logger.exception("STT failed: %s", e)
        return ""


async def _handle_voice(update: Update, context) -> None:
    if not _is_owner(update) or not update.message or not update.message.voice:
        return

    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    voice = update.message.voice

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    voice_path = str(OUTPUT_DIR / f"voice_{voice.file_id}.ogg")
    tg_file = await context.bot.get_file(voice.file_id)
    await tg_file.download_to_drive(voice_path)

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    text = await _transcribe_voice(voice_path)
    if not text:
        await update.message.reply_text("Cannot transcribe voice")
        return

    if any(phrase in text.lower() for phrase in [
        "pereday antonu", "peredat antonu", "dlya antona",
        "antonu pereday", "skazhi antonu", "napishi antonu",
    ]):
        profile = get_user_profile(user_id)
        if profile:
            import re
            msg = re.sub(r'(?i)(pereday|peredat)\s+antonu[:\s,]*', '', text).strip()
            if msg:
                add_suggestion(profile["name"], profile["domain"], msg)
                await update.message.reply_text("Saved! Will pass to developer.")
            else:
                await update.message.reply_text("What to pass?")
        return

    profile = get_user_profile(user_id)
    if profile and not domain_has_capability(profile["domain"], "chat"):
        await update.message.reply_text(
            "Voice accepted but chat is disabled. Use buttons.",
            reply_markup=_build_domain_keyboard(get_domain_buttons(profile["domain"])),
        )
        return

    agent_prompt = f"[Voice message]: {text}"
    response = await run_agent(agent_prompt, context.bot, chat_id, user_id)
    for i in range(0, len(response), _TELEGRAM_MAX_LENGTH):
        chunk = response[i : i + _TELEGRAM_MAX_LENGTH]
        await update.message.reply_text(chunk)


async def _handle_message(update: Update, context) -> None:
    if not _is_owner(update) or not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text
    user_id = update.effective_user.id

    if any(phrase in user_text.lower() for phrase in [
        "pereday antonu", "peredat antonu", "dlya antona",
        "antonu pereday", "skazhi antonu", "napishi antonu",
    ]):
        logger.info(f"Suggestion detected: {user_text[:200]}")
        profile = get_user_profile(user_id)
        if profile:
            import re
            msg = re.sub(r'(?i)(pereday|peredat)\s+antonu[:\s,]*', '', user_text).strip()
            if msg:
                add_suggestion(profile["name"], profile["domain"], msg)
                await update.message.reply_text("Saved! Will pass to developer.")
            else:
                await update.message.reply_text("What to pass?")
        return

    if user_text.lower() in ["mode", "mode?", "rezhim"]:
        current = get_mode()
        await update.message.reply_text(
            f"Mode: {MODE_LABELS[current]}",
            reply_markup=_build_mode_keyboard(),
            parse_mode="Markdown",
        )
        return

    profile = get_user_profile(user_id)
    if profile and not domain_has_capability(profile["domain"], "chat"):
        await update.message.reply_text(
            "Chat disabled in test mode. Use buttons.\n\nFor suggestions: start with 'pereday Antonu'",
            reply_markup=_build_domain_keyboard(get_domain_buttons(profile["domain"])),
        )
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    response = await run_agent(user_text, context.bot, chat_id, user_id)

    for i in range(0, len(response), _TELEGRAM_MAX_LENGTH):
        chunk = response[i : i + _TELEGRAM_MAX_LENGTH]
        await update.message.reply_text(chunk)


def setup_bot() -> Application:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", _start))
    app.add_handler(CommandHandler("mode", _mode_command))
    app.add_handler(CommandHandler("status", _status_command))
    app.add_handler(CommandHandler("clear", _clear))
    app.add_handler(CommandHandler("adduser", _add_user_command))
    app.add_handler(CallbackQueryHandler(_mode_callback))
    app.add_handler(MessageHandler(filters.VOICE, _handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
    return app


async def _handle_domain_callback(update: Update, context) -> None:
    query = update.callback_query
    if not _is_owner(update):
        await query.answer("No access")
        return

    data = query.data
    button_id = data.replace("domain_", "")
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    profile = get_user_profile(user_id)

    if not profile:
        await query.answer("Profile not found")
        return

    domain_config = load_domain_config(profile["domain"])
    if not domain_config:
        await query.answer("Domain not found")
        return

    button_config = None
    for btn in domain_config.get("buttons", []):
        if btn["id"] == button_id:
            button_config = btn
            break

    if not button_config:
        await query.answer("Unknown button")
        return

    if button_config.get("action") == "suggest":
        await query.answer()
        await query.edit_message_text(
            "To send suggestions to developer, start message with 'pereday Antonu'",
            reply_markup=_build_domain_keyboard(get_domain_buttons(profile["domain"])),
            parse_mode="Markdown",
        )
        return

    script_path = button_config.get("script")
    if script_path:
        await query.answer(f"Running {button_config['label']}...")
        await _run_domain_script(update, context, script_path, button_config, profile)
        return

    await query.answer("Action not configured")


async def _run_domain_script(update, context, script_path: str, button_config: dict, profile: dict) -> None:
    import subprocess
    chat_id = update.effective_chat.id

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    full_script_path = project_root / script_path

    if not full_script_path.exists():
        await context.bot.send_message(chat_id=chat_id, text=f"Script not found: {script_path}")
        return

    try:
        result = subprocess.run(
            ["python", str(full_script_path)],
            capture_output=True, text=True, timeout=30,
            cwd=str(project_root),
        )
        output = result.stdout.strip()
        logger.info(f"Script {script_path}: {output[:200]}")

        file_path = None
        for line in output.split("\n"):
            if line.startswith("PATH:"):
                file_path = Path(line.replace("PATH:", "").strip())
                break

        if file_path and file_path.exists():
            await context.bot.send_document(
                chat_id=chat_id,
                document=open(str(file_path), "rb"),
                caption=f"{button_config['label']} - ready!",
            )
            if update.callback_query:
                await update.callback_query.edit_message_text(
                    f"{button_config['label']} - file sent!",
                    reply_markup=_build_domain_keyboard(get_domain_buttons(profile["domain"])),
                )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Done!\n\n```\n{output[:1000]}\n```",
                parse_mode="Markdown",
            )
    except subprocess.TimeoutExpired:
        await context.bot.send_message(chat_id=chat_id, text="Script timed out.")
    except Exception as e:
        logger.exception(f"Script {script_path} failed")
        await context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")


async def _add_user_command(update: Update, context) -> None:
    if not _is_owner(update):
        return
    from nanoclaw.domain import add_user
    args = update.message.text.split()[1:] if update.message.text else []
    if len(args) < 6:
        await update.message.reply_text(
            "Usage: /adduser user_id name username department role domain",
            parse_mode="Markdown",
        )
        return
    try:
        user_id = int(args[0])
        name, username, department, role, domain = args[1], args[2], args[3], args[4], args[5]
        add_user(user_id, name, username, department, role, domain)
        await update.message.reply_text(f"User {name} added to {domain}!")
    except (ValueError, IndexError):
        await update.message.reply_text("Invalid format.")

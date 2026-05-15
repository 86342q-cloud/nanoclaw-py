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


async def _mode_command(update: Update, context) -> None:
    if not _is_owner(update):
        return
    current = get_mode()
    await update.message.reply_text(
        f"🎮 **Выбор режима**\n\n"
        f"Текущий: {MODE_LABELS[current]}\n\n"
        f"{MODE_DESCRIPTIONS[current]}\n\n"
        "Нажми кнопку чтобы переключить:",
        reply_markup=_build_mode_keyboard(),
        parse_mode="Markdown",
    )


async def _test_command(update: Update, context) -> None:
    """/test <domain> — owner impersonates domain user for testing."""
    if update.effective_user.id != OWNER_ID:
        return
    args = update.message.text.split()[1:] if update.message.text else []
    if not args:
        await update.message.reply_text(
            "🎭 **Тестовый прокси**\n\n"
            "Использование: `/test marketing`\n"
            "Доступные домены: " + _list_domains() + "\n\n"
            "`/testoff` — выключить",
            parse_mode="Markdown",
        )
        return
    domain_name = args[0]
    config = load_domain_config(domain_name)
    if not config:
        await update.message.reply_text(f"❌ Домен `{domain_name}` не найден")
        return
    import sqlite3
    conn = sqlite3.connect(str(DATA_DIR / "messages.db"))
    conn.row_factory = sqlite3.Row
    user = conn.execute(
        "SELECT * FROM users WHERE domain = ? AND is_active = 1 LIMIT 1",
        (domain_name,),
    ).fetchone()
    conn.close()
    if not user:
        await update.message.reply_text(f"❌ В домене `{domain_name}` нет пользователей. Сначала `/adduser`.")
        return
    set_test_proxy(user["user_id"], user["name"], domain_name)
    await update.message.reply_text(
        f"🎭 **Тестовый прокси ВКЛ**\n\n"
        f"Ты теперь: **{user['name']}** ({domain_name})\n"
        f"Все сообщения и голос — как от неё.\n\n"
        f"`/testoff` — выключить",
        parse_mode="Markdown",
    )


async def _testoff_command(update: Update, context) -> None:
    """/testoff — exit test proxy."""
    if update.effective_user.id != OWNER_ID:
        return
    proxy = get_test_proxy()
    if proxy:
        set_test_proxy(None)
        await update.message.reply_text("🎭 Тестовый прокси выключен. Ты снова Антон 🧠")
    else:
        await update.message.reply_text("🎭 Прокси и так выключен.")


def _list_domains() -> str:
    domains_dir = Path(__file__).resolve().parent.parent.parent.parent / "domains"
    names = []
    for d in sorted(domains_dir.iterdir()):
        if d.is_dir() and (d / "config.yaml").exists():
            names.append(f"`{d.name}`")
    return ", ".join(names) if names else "нет"


def _test_prefix() -> str:
    proxy = get_test_proxy()
    if proxy:
        return "\U0001f3ad [" + proxy["user_name"] + "] "
    return ""


async def _status_command(update: Update, context) -> None:
    if not _is_owner(update):
        return
    chat_id = update.effective_chat.id
    current = get_mode()
    stats = get_chat_stats(chat_id)
    msg_count = stats["total"]
    tok_count = stats["total_tokens"]
    await update.message.reply_text(
        f"📊 **Статус бота**\n\n"
        f"• Состояние: 🟢 **Активен**\n"
        f"• Режим: {MODE_LABELS[current]}\n"
        f"• Модель: {'DeepSeek V4 Flash' if current != MODE_MINIMAX else 'MiniMax M2.7-highspeed'}\n"
        f"• Память: SQLite | {msg_count} сообщений (~{tok_count} токенов)\n"
        f"• Бот: @sm\\_data\\_assistant\\_bot\n"
        f"• Ядро: nanoclaw-py",
        reply_markup=_build_mode_keyboard(),
        parse_mode="Markdown",
    )


async def _voice_confirm_callback(update: Update, context) -> None:
    """Handle voice confirmation: ✅ → run agent, ❌ → ask to repeat."""
    query = update.callback_query
    if not _is_owner(update):
        await query.answer("Нет доступа")
        return

    data = query.data
    chat_id = update.effective_chat.id
    user_id = _effective_user_id(update)
    text = context.user_data.pop("voice_pending_text", None)
    context.user_data.pop("voice_pending_user_id", None)

    if data == "voice_confirm":
        if not text:
            await query.edit_message_text("⚠️ Что-то пошло не так. Попробуй ещё раз 🎤")
            await query.answer()
            return
        display_text = text[:200] + ("…" if len(text) > 200 else "")
        await query.edit_message_text(f"🎤 «{display_text}»\n\n✅ Принято! Сейчас подумаю…")
        await query.answer()

        if any(phrase in text.lower() for phrase in [
            "передай антону", "передать антону", "для антона",
            "антону передай", "антону передать", "скажи антону",
            "напиши антону", "пожелание", "идея для",
        ]):
            profile = get_user_profile(user_id)
            if profile:
                import re
                msg = re.sub(r'(?i)(передай|передать)\s+антону[:\s,]*', '', text).strip()
                if msg:
                    add_suggestion(profile["name"], profile["domain"], msg)
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="✍️ Записал. Скину Антону, он разрулит 👊"
                    )
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="📨 Что передать Антону? Скажи после «передай Антону» — я мигом."
                    )
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="📨 Ок, Антон увидит в логах."
                )
            return

        profile = get_user_profile(user_id)
        if profile and not domain_has_capability(profile["domain"], "chat"):
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"🎤 Голосовые принимаю. Болтовню пока отключили — жми кнопки 👇",
                reply_markup=_build_domain_keyboard(get_domain_buttons(profile["domain"])),
            )
            return

        agent_prompt = f"[Голосовое сообщение]: {text}"
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        profile = get_user_profile(user_id)
        sys_prompt = build_system_prompt(profile["domain"], profile["name"]) if profile else ""
        response = await run_agent(agent_prompt, context.bot, chat_id, user_id, sys_prompt)
        for i in range(0, len(response), _TELEGRAM_MAX_LENGTH):
            chunk = response[i : i + _TELEGRAM_MAX_LENGTH]
            await context.bot.send_message(chat_id=chat_id, text=chunk)

    elif data == "voice_reject":
        await query.edit_message_text("🎤 Понял, давай по новой! Запиши ещё раз — я послушаю 👂")
        await query.answer()


async def _mode_callback(update: Update, context) -> None:
    query = update.callback_query
    if not _is_owner(update):
        await query.answer("Нет доступа")
        return

    data = query.data

    if data in ("voice_confirm", "voice_reject"):
        await _voice_confirm_callback(update, context)
        return

    if data == "status":
        await query.answer()
        chat_id = update.effective_chat.id
        current = get_mode()
        stats = get_chat_stats(chat_id)
        msg_count = stats["total"]
        await query.edit_message_text(
            f"📊 **Статус**\n\n"
            f"• Состояние: 🟢 **Активен**\n"
            f"• Режим: {MODE_LABELS[current]}\n"
            f"• Память: SQLite | {msg_count} сообщений\n"
            f"• Цена: ~$0.09/запрос (DeepSeek)",
            reply_markup=_build_mode_keyboard(),
            parse_mode="Markdown",
        )
        return

    if data.startswith("domain_"):
        await _handle_domain_callback(update, context)
        return

    mode = data.replace("mode_", "")
    if mode not in MODE_LABELS:
        await query.answer("Неизвестный режим")
        return

    old_mode = get_mode()
    if mode == old_mode:
        await query.answer(f"Уже в режиме {MODE_LABELS[mode]}")
        return

    set_mode(mode)

    await query.answer(f"Переключено: {MODE_LABELS[mode]}")
    await query.edit_message_text(
        f"🎮 **Режим переключён**\n\n"
        f"Было: {MODE_LABELS[old_mode]}\n"
        f"Стало: {MODE_LABELS[mode]}\n\n"
        f"{MODE_DESCRIPTIONS[mode]}",
        reply_markup=_build_mode_keyboard(),
        parse_mode="Markdown",
    )


async def _clear(update: Update, context) -> None:
    if not _is_owner(update):
        return
    chat_id = update.effective_chat.id
    deleted = clear_history(chat_id)
    await update.message.reply_text(f"🧹 Память очищена! Удалено {deleted} сообщений. Начинаем с чистого листа!")


async def _transcribe_voice(file_path: str) -> str:
    if not POLZA_API_KEY:
        logger.warning("POLZA_API_KEY not set, cannot transcribe voice")
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
        logger.warning("Polza.ai STT error: %s", r.status_code)
        return ""
    except Exception as e:
        logger.exception("STT failed: %s", e)
        return ""


async def _correct_stt(raw_text: str, domain_name: str = "") -> str:
    """Correct STT transcription errors using DeepSeek API."""
    if not raw_text.strip():
        return raw_text

    try:
        context_hint = ""
        if domain_name:
            domain_config = load_domain_config(domain_name)
            if domain_config:
                domain_display = domain_config.get("domain", {}).get("display_name", domain_name)
                context_hint = (
                    f" Контекст: общение с AI-ассистентом в домене «{domain_display}». "
                    f"Возможная тематика: маркетинг, трафик, продажи, автомобили, отчёты."
                )

        prompt = (
            "Исправь ошибки распознавания речи (ASR) в тексте. "
            "Сохрани смысл, стиль и пунктуацию. Убери слова-паразиты. "
            "Не добавляй ничего от себя. Верни ТОЛЬКО исправленный текст, без пояснений."
            f"{context_hint}\n\n"
            f"Исходный текст: {raw_text}\n\n"
            "Исправленный текст:"
        )

        api_key = ANTHROPIC_API_KEY
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.1,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )

        if r.status_code == 200:
            corrected = r.json()["choices"][0]["message"]["content"].strip()
            if not corrected or len(corrected) < len(raw_text) * 0.3:
                logger.warning("STT correction returned suspicious result, using original")
                return raw_text
            return corrected

        logger.warning(f"STT correction API error: {r.status_code} {r.text[:200]}")
        return raw_text

    except Exception as e:
        logger.warning(f"STT correction failed: {e}")
        return raw_text


async def _handle_voice(update: Update, context) -> None:
    if not _is_owner(update) or not update.message or not update.message.voice:
        return

    chat_id = update.effective_chat.id
    user_id = _effective_user_id(update)
    voice = update.message.voice
    logger.info("Voice received: %s bytes, %ss", voice.file_size, voice.duration)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    voice_path = str(OUTPUT_DIR / f"voice_{voice.file_id}.ogg")
    tg_file = await context.bot.get_file(voice.file_id)
    await tg_file.download_to_drive(voice_path)

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    text = await _transcribe_voice(voice_path)
    if not text:
        await update.message.reply_text("🎤 Не смог распознать голосовое 😕")
        return
    logger.info("STT: %s", text[:200])

    text = text.strip()

    profile = get_user_profile(user_id)
    domain_name = profile["domain"] if profile else ""
    original_text = text
    text = await _correct_stt(text, domain_name)
    if text != original_text:
        logger.info(f"STT corrected: [{original_text[:80]}] → [{text[:80]}]")

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Да, всё верно", callback_data="voice_confirm"),
            InlineKeyboardButton("❌ Нет, не так", callback_data="voice_reject"),
        ]
    ])
    context.user_data["voice_pending_text"] = text
    context.user_data["voice_pending_user_id"] = user_id

    await update.message.reply_text(
        f"{_test_prefix()}🎤 Я услышал:\n\n«{text}»\n\nПравильно понял?",
        reply_markup=keyboard,
    )


async def _handle_message(update: Update, context) -> None:
    if not _is_owner(update) or not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text
    user_id = _effective_user_id(update)

    if any(phrase in user_text.lower() for phrase in [
        "передай антону", "передать антону", "для антона",
        "антону передай", "антону передать", "скажи антону",
        "напиши антону", "пожелание", "идея для",
    ]):
        logger.info(f"Suggestion detected: {user_text[:200]}")
        profile = get_user_profile(user_id)
        if profile:
            import re
            msg = re.sub(r'(?i)(передай|передать)\s+антону[:\s,]*', '', user_text).strip()
            if msg:
                logger.info(f"Saving suggestion: {msg[:200]}")
                add_suggestion(profile["name"], profile["domain"], msg)
                await update.message.reply_text("✍️ Записал. Антон получит 👊")
            else:
                await update.message.reply_text("📨 Что передать разработчику? Напиши после «передай Антону» — я мигом.")
        else:
            await update.message.reply_text("📨 Я пока не знаю твой профиль. Но разработчик увидит это сообщение в логах!")
        return

    if user_text.lower() in ["какой режим?", "какой режим", "режим?", "режим", "mode", "mode?"]:
        current = get_mode()
        await update.message.reply_text(
            f"Сейчас: **{MODE_LABELS[current]}**\n\n{MODE_DESCRIPTIONS[current]}",
            reply_markup=_build_mode_keyboard(),
            parse_mode="Markdown",
        )
        return

    profile = get_user_profile(user_id)
    if profile and not domain_has_capability(profile["domain"], "chat"):
        await update.message.reply_text(
            f"Болтовню пока отключили. Жми кнопки 👇 или пиши «передай Антону» ✍️",
            reply_markup=_build_domain_keyboard(get_domain_buttons(profile["domain"])),
        )
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    profile = get_user_profile(user_id)
    sys_prompt = build_system_prompt(profile["domain"], profile["name"]) if profile else ""
    response = await run_agent(user_text, context.bot, chat_id, user_id, sys_prompt)

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
    app.add_handler(CommandHandler("test", _test_command))
    app.add_handler(CommandHandler("testoff", _testoff_command))
    app.add_handler(CallbackQueryHandler(_mode_callback))
    app.add_handler(MessageHandler(filters.VOICE, _handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
    return app


async def _handle_domain_callback(update: Update, context) -> None:
    query = update.callback_query
    if not _is_owner(update):
        await query.answer("Нет доступа")
        return

    data = query.data
    button_id = data.replace("domain_", "")
    user_id = _effective_user_id(update)
    chat_id = update.effective_chat.id
    profile = get_user_profile(user_id)

    if not profile:
        await query.answer("Профиль не найден")
        return

    domain_config = load_domain_config(profile["domain"])
    if not domain_config:
        await query.answer("Домен не найден")
        return

    button_config = None
    for btn in domain_config.get("buttons", []):
        if btn["id"] == button_id:
            button_config = btn
            break

    if not button_config:
        await query.answer("Неизвестная кнопка")
        return

    if button_config.get("action") == "suggest":
        await query.answer()
        await query.edit_message_text(
            f"📨 Чтобы передать пожелания разработчику, просто напиши сообщение начиная с «передай Антону» и я всё запишу ✍️\n\n"
            f"Например: *передай Антону: хочу отчёт по ROMI за месяц*",
            reply_markup=_build_domain_keyboard(get_domain_buttons(profile["domain"])),
            parse_mode="Markdown",
        )
        return

    script_path = button_config.get("script")
    if script_path:
        await query.answer(f"Запускаю {button_config['label']}...")
        await _run_domain_script(update, context, script_path, button_config, profile)
        return

    await query.answer("Действие не настроено")


async def _run_domain_script(update, context, script_path: str, button_config: dict, profile: dict) -> None:
    import subprocess
    chat_id = update.effective_chat.id

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    full_script_path = project_root / script_path

    if not full_script_path.exists():
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"⚠️ Скрипт не найден: {script_path}"
        )
        return

    try:
        import os as _os
        env = _os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            ["python", str(full_script_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            cwd=str(project_root),
            env=env,
        )
        output = (result.stdout or "").strip()
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
                caption=f"📥 {button_config['label']} — готово!",
            )
            text_output = "\n".join(
                line for line in output.split("\n")
                if not line.strip().startswith("PATH:")
            ).strip()
            if text_output:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"```\n{text_output[:3500]}\n```",
                    parse_mode="Markdown",
                )
            if update.callback_query:
                await update.callback_query.edit_message_text(
                    f"✅ {button_config['label']} — файл отправлен!\n\nЕщё что-то нужно?",
                    reply_markup=_build_domain_keyboard(get_domain_buttons(profile["domain"])),
                )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"✅ {button_config['label']} — выполнено!\n\n```\n{output[:1000]}\n```",
                parse_mode="Markdown",
            )
    except subprocess.TimeoutExpired:
        await context.bot.send_message(
            chat_id=chat_id,
            text="⏰ Скрипт выполняется слишком долго. Попробуй позже."
        )
    except Exception as e:
        logger.exception(f"Script {script_path} failed")
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ Ошибка при выполнении: {e}"
        )


async def _add_user_command(update: Update, context) -> None:
    if not _is_owner(update):
        return
    from nanoclaw.domain import add_user
    args = update.message.text.split()[1:] if update.message.text else []
    if len(args) < 6:
        await update.message.reply_text(
            "Использование: `/adduser user_id name username department role domain`\n"
            "Пример: `/adduser 123456789 Алиса melali_ka marketing marketing_director marketing`",
            parse_mode="Markdown",
        )
        return
    try:
        user_id = int(args[0])
        name = args[1]
        username = args[2]
        department = args[3]
        role = args[4]
        domain = args[5]
        add_user(user_id, name, username, department, role, domain)
        await update.message.reply_text(f"✅ Пользователь {name} (@{username}) добавлен в домен {domain}!")
    except (ValueError, IndexError):
        await update.message.reply_text("❌ Неверный формат.", parse_mode="Markdown")
"""Agent routing: DeepSeek (Anthropic SDK) + MiniMax (httpx) + auto-fallback.

DeepSeek uses Anthropic SDK as a bridge (handles thinking-mode quirks)
but WITHOUT SDK session persistence. Memory is SQLite-based, shared
across both models.
"""

import asyncio
import logging
import re
from typing import Any, AsyncGenerator

import httpx

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

from nanoclaw.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL,
    WORKSPACE_DIR,
    get_mode,
    MODE_DEEPSEEK,
    MODE_MINIMAX,
)
from nanoclaw.memory import (
    save_message,
    get_history,
    clear_history,
    format_history,
    format_history_for_minimax,
)

logger = logging.getLogger(__name__)
_agent_lock = asyncio.Lock()

_fallback_used: set[int] = set()


def _create_tools(bot: Any, chat_id: int) -> list:
    @tool("send_message", "Send a message to the user on Telegram", {"text": str})
    async def send_message(args: dict[str, Any]) -> dict[str, Any]:
        await bot.send_message(chat_id=chat_id, text=args["text"])
        return {"content": [{"type": "text", "text": "Message sent."}]}

    return [send_message]


async def _make_prompt(text: str) -> AsyncGenerator[dict, None]:
    yield {"type": "user", "message": {"role": "user", "content": text}}


async def run_agent(prompt: str, bot: Any, chat_id: int, user_id: int) -> str:
    async with _agent_lock:
        mode = get_mode()

        if mode == MODE_MINIMAX:
            return await _run_minimax(prompt, chat_id, user_id)

        try:
            result = await _run_deepseek(prompt, bot, chat_id, user_id)
            if result.startswith("[MiniMax unavailable") or result.startswith("Sorry"):
                raise Exception("DeepSeek returned error")
            return result
        except Exception as e:
            logger.warning(f"DeepSeek failed, auto-fallback to MiniMax: {e}")
            fallback_key = hash((chat_id, user_id, prompt))
            if fallback_key in _fallback_used:
                _fallback_used.discard(fallback_key)
                return "Both models unavailable. Try later."
            _fallback_used.add(fallback_key)
            try:
                result = await _run_minimax(prompt, chat_id, user_id)
                _fallback_used.discard(fallback_key)
                return f"[DeepSeek unavailable, MiniMax fallback]\n\n{result}"
            except Exception as e2:
                _fallback_used.discard(fallback_key)
                logger.exception("MiniMax fallback also failed")
                return "Both models unavailable. Try later."


async def _run_deepseek(prompt: str, bot: Any, chat_id: int, user_id: int) -> str:
    history = get_history(chat_id)
    history_text = format_history(history)

    if history_text:
        full_prompt = f"{history_text}\n[Current message]\nUser: {prompt}"
    else:
        full_prompt = prompt

    save_message(chat_id, user_id, "user", prompt)

    tools = _create_tools(bot, chat_id)
    mcp_server = create_sdk_mcp_server(name="nanoclaw", tools=tools)

    env = {"ANTHROPIC_API_KEY": ANTHROPIC_API_KEY}
    if ANTHROPIC_BASE_URL:
        env["ANTHROPIC_BASE_URL"] = ANTHROPIC_BASE_URL

    options = ClaudeAgentOptions(
        cwd=str(WORKSPACE_DIR),
        setting_sources=["project"],
        allowed_tools=[
            "Bash", "Read", "Write", "Edit", "Glob", "Grep",
            "WebSearch", "WebFetch", "mcp__nanoclaw__send_message",
        ],
        permission_mode="bypassPermissions",
        mcp_servers={"nanoclaw": mcp_server},
        env=env,
    )

    response_parts: list[str] = []

    try:
        async for message in query(prompt=_make_prompt(full_prompt), options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                pass
    except Exception:
        if not response_parts:
            logger.exception("Agent error")
            return "Sorry, something went wrong."
        logger.debug("Ignoring query cleanup error", exc_info=True)

    response = "".join(response_parts) or "Done."
    save_message(chat_id, 0, "assistant", response)
    return response


async def _run_minimax(prompt: str, chat_id: int, user_id: int) -> str:
    import os

    history = get_history(chat_id)
    messages = format_history_for_minimax(history)
    messages.append({"role": "user", "content": prompt})

    save_message(chat_id, user_id, "user", prompt)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "https://api.minimaxi.chat/v1/chat/completions",
                json={
                    "model": "MiniMax-M2.7-highspeed",
                    "messages": messages,
                    "max_tokens": 1000,
                    "stop": ["<think>"],
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('MINIMAX_API_KEY', '')}",
                    "Content-Type": "application/json",
                },
            )

        if r.status_code != 200:
            raise Exception(f"HTTP {r.status_code}")

        raw = r.json()["choices"][0]["message"]["content"] or "..."
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        if "<think>" in raw:
            raw = raw.split("<think>")[0]
        response = raw.strip() or "..."

    except Exception as e:
        logger.warning(f"MiniMax failed: {e}")
        return f"[MiniMax unavailable: {e}. Try /mode to switch.]"

    save_message(chat_id, 0, "assistant", response)
    return response

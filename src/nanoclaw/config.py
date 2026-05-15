import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Required
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OWNER_ID = int(os.environ["OWNER_ID"])
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# Optional
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Ассист")
SCHEDULER_INTERVAL = int(os.getenv("SCHEDULER_INTERVAL", "60"))
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
POLZA_API_KEY = os.getenv("POLZA_API_KEY", "")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE_DIR = BASE_DIR / "workspace"
STORE_DIR = BASE_DIR / "store"
DATA_DIR = BASE_DIR / "data"
DB_PATH = STORE_DIR / "nanoclaw.db"
STATE_FILE = DATA_DIR / "state.json"
MODE_FILE = DATA_DIR / "mode.json"

# Agent modes
MODE_DEEPSEEK = "deepseek"
MODE_MINIMAX = "minimax"

# Test proxy — owner impersonates a domain user for testing
TEST_PROXY_FILE = DATA_DIR / "test_proxy.json"


def get_mode() -> str:
    """Get current agent mode. Default: deepseek."""
    if MODE_FILE.exists():
        data = json.loads(MODE_FILE.read_text())
        return data.get("mode", MODE_DEEPSEEK)
    return MODE_DEEPSEEK


def set_mode(mode: str) -> None:
    """Set agent mode."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODE_FILE.write_text(json.dumps({"mode": mode}))


def get_test_proxy() -> dict | None:
    """Get current test proxy state. Returns None if not in test mode."""
    if TEST_PROXY_FILE.exists():
        return json.loads(TEST_PROXY_FILE.read_text())
    return None


def set_test_proxy(user_id: int | None, user_name: str = "", domain: str = "") -> None:
    """Set test proxy. user_id=None to disable."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if user_id is None:
        TEST_PROXY_FILE.unlink(missing_ok=True)
    else:
        TEST_PROXY_FILE.write_text(json.dumps({
            "user_id": user_id,
            "user_name": user_name,
            "domain": domain,
        }))


def get_chat_workspace(chat_id: int) -> Path:
    return WORKSPACE_DIR

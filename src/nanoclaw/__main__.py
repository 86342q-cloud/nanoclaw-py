import asyncio
import logging
import sys

from nanoclaw.bot import setup_bot
from nanoclaw.config import ASSISTANT_NAME, DATA_DIR, DB_PATH, STORE_DIR, WORKSPACE_DIR
from nanoclaw.db import init_db
from nanoclaw.memory import ensure_workspace

# Force UTF-8 output (Windows cp1251 fix)
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def _prepare_runtime() -> None:
    for d in (WORKSPACE_DIR, STORE_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
    await init_db(str(DB_PATH))
    logger.info("Database initialized at %s", DB_PATH)
    ensure_workspace()
    logger.info("Workspace ready at %s", WORKSPACE_DIR)


def _run_bot() -> None:
    app = setup_bot()
    logger.info("%s is starting...", ASSISTANT_NAME)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


def main() -> None:
    asyncio.run(_prepare_runtime())
    _run_bot()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down...")

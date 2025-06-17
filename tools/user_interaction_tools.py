"""Utility tools for user interaction and logging simple messages."""
import logging

LOGGER = logging.getLogger("UserInteractionTools")
logging.basicConfig(level=logging.INFO)


def log_message(message: str) -> dict:
    """Log a message for debugging or narrative purposes."""
    LOGGER.info(message)
    return {"status": "success", "message": message}

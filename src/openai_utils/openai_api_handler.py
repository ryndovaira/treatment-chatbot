from openai import OpenAI

from src.env_config import (
    OPENAI_API_KEY,
    OPENAI_PROJECT_ID,
)
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def get_openai_client():
    """Create an OpenAI client using environment variables."""
    if not OPENAI_API_KEY or not OPENAI_PROJECT_ID:
        raise ValueError("OPENAI_API_KEY and OPENAI_PROJECT_ID must be set in the environment.")
    return OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID)

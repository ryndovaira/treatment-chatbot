import json
from pathlib import Path
from typing import Dict, Any

from langchain.schema import Document

from src.logging_config import setup_logger

logger = setup_logger(__name__)

LOG_FILE = Path("query_logs_0.json")


def document_to_serializable(doc: Document) -> Dict[str, Any]:
    """
    Convert a Document object into a JSON-serializable dictionary.

    Args:
        doc (Document): The Document object to serialize.

    Returns:
        Dict[str, Any]: A dictionary representation of the Document.
    """
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    }


def log_query(query: str, results: Dict[str, Any]) -> None:
    """
    Log the query and its results to a JSON file.

    Args:
        query (str): The user query.
        results (Dict[str, Any]): The results including answers and source documents.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        if LOG_FILE.exists():
            with LOG_FILE.open("r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(
            {
                "query": query,
                "public_answer": results["public_answer"],
                "private_answer": results["private_answer"],
                "public_sources": [
                    document_to_serializable(doc) for doc in results["public_sources"]
                ],
                "private_sources": [
                    document_to_serializable(doc) for doc in results["private_sources"]
                ],
            }
        )

        with LOG_FILE.open("w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4)

        logger.info(f"Query logged successfully to {LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to log query: {e}")

import json
from pathlib import Path
from typing import Dict, Any

from src.logging_config import setup_logger

logger = setup_logger(__name__)

# Path for logging queries
LOG_FILE = Path("artifacts/query_logs.json")


def serialize_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a document into a JSON-serializable dictionary.

    Args:
        doc (Dict[str, Any]): The document to serialize.

    Returns:
        Dict[str, Any]: A dictionary representation of the document.
    """
    return {
        "text": doc.get("text", ""),
        "metadata": doc.get("metadata", {}),
    }


def log_query(query: str, results: Dict[str, Any]) -> None:
    """
    Log the query and its results to a JSON file.

    Args:
        query (str): The user query.
        results (Dict[str, Any]): The results including summaries and source documents.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load existing logs if the file exists
        if LOG_FILE.exists():
            with LOG_FILE.open("r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        # Append the new log entry
        logs.append(
            {
                "query": query,
                "public_summary": results.get("public_summary", ""),
                "private_summary": results.get("private_summary", ""),
                "combined_summary": results.get("combined_summary", ""),
                "public_sources": [
                    serialize_document(doc) for doc in results.get("public_sources", [])
                ],
                "private_sources": [
                    serialize_document(doc) for doc in results.get("private_sources", [])
                ],
            }
        )

        # Write logs back to the file
        with LOG_FILE.open("w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4)

        logger.info(f"Query and results logged successfully to {LOG_FILE}")

    except Exception as e:
        logger.error(f"Failed to log query and results: {e}")

import hashlib
import json
from pathlib import Path

from openai import OpenAI

from src.logging_config import setup_logger

logger = setup_logger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash for a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def retrieve_batch_results(batch_id: str) -> str:
    """Retrieve batch results using OpenAI API."""
    logger.info(f"Retrieving results for batch ID: {batch_id}")
    client = OpenAI()
    result_file = client.batches.get_results(batch_id)
    logger.info(f"Results retrieved successfully for batch ID: {batch_id}")
    return result_file.text


def is_batch_already_submitted(file_hash: str, tracking_file: Path) -> bool:
    """Check if a batch is already submitted based on the file hash."""
    if not tracking_file.exists():
        return False
    with open(tracking_file, "r") as f:
        tracking_data = json.load(f)
    return file_hash in tracking_data.get("batch_hashes", {})


def save_batch_hash(file_hash: str, batch_id: str, tracking_file: Path) -> None:
    """Save a hash and its associated batch ID to the tracking file."""
    tracking_data = {"batch_hashes": {}, "batches": {}}
    if tracking_file.exists():
        with open(tracking_file, "r") as f:
            tracking_data = json.load(f)

    # Update tracking data
    tracking_data["batch_hashes"][file_hash] = batch_id
    tracking_data["batches"][batch_id] = {"status": "submitted"}

    with open(tracking_file, "w") as f:
        json.dump(tracking_data, f, indent=4)

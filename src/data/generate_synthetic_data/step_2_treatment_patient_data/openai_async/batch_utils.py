import hashlib
import json
from pathlib import Path

from src.logging_config import setup_logger
from src.openai_utils.openai_api_handler import get_openai_client

logger = setup_logger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash for a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def retrieve_batch_results(batch_id: str, save_path: Path) -> str:
    """
    Retrieve batch results using OpenAI API and save them to a file.

    :param batch_id: The ID of the batch to retrieve results for.
    :param save_path: The path where the results file should be saved.
    :return: True if the results were successfully saved, False otherwise.
    """
    client = get_openai_client()

    # Check the batch status
    batch = client.batches.retrieve(batch_id)
    logger.info(f"Batch ID {batch_id} status: {batch.status}")

    if batch.status != "completed":
        logger.info(f"Batch ID {batch_id} is not yet ready. Current status: {batch.status}")
        return ""  # Indicate that the batch is not ready

    # Retrieve the output file ID
    output_file_id = batch.output_file_id
    if not output_file_id:
        logger.warning(f"No output file found for batch ID {batch_id}. This may indicate an issue.")
        if batch.error_file_id:
            logger.error(
                f"Batch ID {batch_id} encountered an error. Retrieving error file {batch.error_file_id}"
            )
            error_file = client.files.content(batch.error_file_id)
            logger.error(f"Error file content: {error_file.text}")
        return ""  # Indicate that the output file is missing
    # Retrieve results
    logger.info(f"Retrieving results for output file ID: {output_file_id}")
    result_file = client.files.content(output_file_id)

    # Save the results to the specified file
    try:
        logger.info(f"Saving results to: {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        with open(save_path, "w") as f:
            f.write(result_file.text)
        logger.info(f"Results successfully saved to {save_path}")
        return result_file
    except Exception as e:
        logger.error(f"Failed to save results to {save_path}: {e}")
        return ""


def is_batch_already_submitted(file_hash: str, tracking_file: Path) -> bool:
    """Check if a batch is already submitted based on the file hash."""
    if not tracking_file.exists():
        return False
    with open(tracking_file, "r") as f:
        tracking_data = json.load(f)
    return file_hash in tracking_data.get("batch_hashes", {})


def save_batch_hash(file_hash: str, batch_id: str, tracking_file: Path, input_file: Path) -> None:
    """
    Save a hash and its associated batch ID to the tracking file, along with input/output file metadata.

    :param file_hash: Hash of the batch file.
    :param batch_id: ID of the batch job.
    :param tracking_file: Path to the tracking file.
    :param input_file: Path to the input batch file.
    """
    tracking_data = {"batch_hashes": {}, "batches": {}}
    if tracking_file.exists():
        with open(tracking_file, "r") as f:
            tracking_data = json.load(f)

    # Update tracking data
    tracking_data["batch_hashes"][file_hash] = batch_id
    tracking_data["batches"][batch_id] = {
        "status": "submitted",
        "input_file": str(input_file),
        "output_file": None,
    }

    with open(tracking_file, "w") as f:
        json.dump(tracking_data, f, indent=4)

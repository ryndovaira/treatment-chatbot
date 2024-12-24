from pathlib import Path

from batch_utils import compute_file_hash, is_batch_already_submitted, save_batch_hash
from src.data.generate_synthetic_data.step_2_treatment_patient_data.openai_async.config import (
    BATCH_TRACKING_FILE,
    BATCH_INPUT_FILE,
)
from src.logging_config import setup_logger
from src.openai_utils.openai_api_handler import get_openai_client

logger = setup_logger(__name__)


def submit_batch(batch_file_path):
    """
    Submit a batch request to OpenAI's Batch API.

    :param batch_file_path: Path to the .jsonl batch input file.
    """
    client = get_openai_client()
    batch_file_path = Path(batch_file_path)

    if not batch_file_path.exists():
        logger.error(f"Batch file {batch_file_path} does not exist.")
        raise FileNotFoundError(f"Batch file {batch_file_path} does not exist.")

    # Compute the file hash to check for duplicates
    file_hash = compute_file_hash(batch_file_path)
    logger.info(f"Computed hash for batch file: {file_hash}")

    if is_batch_already_submitted(file_hash, BATCH_TRACKING_FILE):
        logger.info(
            f"Batch with hash {file_hash} is already submitted. Check the status using `check_status.py`."
        )
        return

    # Upload the batch file
    logger.info(f"Uploading batch file: {batch_file_path}")
    batch_input_file = client.files.create(file=open(batch_file_path, "rb"), purpose="batch")
    batch_input_file_id = batch_input_file["id"]
    logger.info(f"Batch file uploaded with ID: {batch_input_file_id}")

    # Submit the batch job
    logger.info(f"Submitting batch job for file ID: {batch_input_file_id}")
    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    batch_id = batch_job["id"]
    logger.info(f"Batch submitted successfully. Batch ID: {batch_id}")

    # Save the hash and batch ID to the tracking file
    save_batch_hash(file_hash, batch_id, BATCH_TRACKING_FILE)
    logger.info(f"Batch tracking updated in {BATCH_TRACKING_FILE}")


if __name__ == "__main__":
    submit_batch(BATCH_INPUT_FILE)

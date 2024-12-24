import json

from src.data.generate_synthetic_data.step_2_treatment_patient_data.openai_async.config import (
    BATCH_TRACKING_FILE,
)
from src.logging_config import setup_logger
from src.openai_utils.openai_api_handler import get_openai_client

# Configure logger
logger = setup_logger(__name__)


def load_tracking_data(tracking_file):
    """
    Load the tracking metadata file.

    :param tracking_file: Path to the tracking file.
    :return: Parsed JSON data.
    """
    if not tracking_file.exists():
        logger.error(f"Tracking file {tracking_file} does not exist.")
        return {}
    try:
        with open(tracking_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {tracking_file}.")
        return {}


def check_all_batches(tracking_file):
    """
    Check the status of all tracked batches.

    :param tracking_file: Path to the tracking file.
    """
    data = load_tracking_data(tracking_file)
    if not data or "batches" not in data:
        logger.info("No tracked batches found.")
        return

    for batch_id, batch_info in data["batches"].items():
        logger.info(f"Batch ID: {batch_id}")
        logger.info(f"  Status: {batch_info['status']}")
        logger.info(f"  Input File: {batch_info.get('input_file', 'Not available')}")
        logger.info(f"  Output File: {batch_info.get('output_file', 'Not available')}")


def check_batch_status(batch_id, tracking_file):
    """
    Check the status of a specific batch.

    :param batch_id: The ID of the batch to check.
    :param tracking_file: Path to the tracking file.
    """
    client = get_openai_client()
    try:
        batch_status = client.batches.retrieve(batch_id)
        logger.info(f"Batch ID: {batch_id}")
        logger.info(f"Status: {batch_status['status']}")
        logger.info(f"Completed Requests: {batch_status['request_counts']['completed']}")
        logger.info(f"Failed Requests: {batch_status['request_counts']['failed']}")
        logger.info(f"Total Requests: {batch_status['request_counts']['total']}")

        # Update the status in the tracking file
        data = load_tracking_data(tracking_file)
        if batch_id in data.get("batches", {}):
            data["batches"][batch_id]["status"] = batch_status["status"]
            with open(tracking_file, "w") as f:
                json.dump(data, f, indent=4)
                logger.info(f"Updated status for batch {batch_id} in {tracking_file}")
    except Exception as e:
        logger.error(f"Error checking status for batch {batch_id}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # If no arguments are provided, check all batches
        check_all_batches(BATCH_TRACKING_FILE)
    elif len(sys.argv) == 2:
        # If a batch ID is provided, check its status
        batch_id = sys.argv[1]
        check_batch_status(batch_id, BATCH_TRACKING_FILE)
    else:
        logger.error("Usage: python check_status.py [batch_id]")

import json

from src.data.generate_synthetic_data.step_2_treatment_patient_data.config import (
    BATCH_TRACKING_FILE,
    BATCH_OUTPUT_FILE,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.helpers import (
    save_data_as_json,
    save_data_as_jsonl,
)
from src.logging_config import setup_logger
from src.openai_utils.openai_api_handler import get_openai_client

logger = setup_logger(__name__)


def load_tracking_data(tracking_file) -> dict:
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


def usage_info(parsed_results):
    """
    Log the model and usage information for the parsed results.
    :param parsed_results: Parsed JSON results.
    """
    model = parsed_results[0]["response"]["body"]["model"]
    usage_prompt_tokens = sum(
        [result["response"]["body"]["usage"]["prompt_tokens"] for result in parsed_results]
    )
    usage_completion_tokens = sum(
        [result["response"]["body"]["usage"]["completion_tokens"] for result in parsed_results]
    )
    usage_total_tokens = sum(
        [result["response"]["body"]["usage"]["total_tokens"] for result in parsed_results]
    )

    logger.info(f"Model: {model}")
    logger.info(
        f"Usage: Prompt Tokens: {usage_prompt_tokens}, Completion Tokens: {usage_completion_tokens}, Total Tokens: {usage_total_tokens}"
    )


def save_output_text_as_json(output_text, path, record_index):
    """
    Save the output text as JSON.
    :param output_text: Output text to save.
    :param path: Path to save the output text.
    :param record_index: Index of the record in the batch
    """
    logger.info("Parsing batch results...")
    parsed_results = [json.loads(line) for line in output_text.splitlines()]
    clean_parsed_results = [
        json.loads(result["response"]["body"]["choices"][0]["message"]["content"])
        | {"patient_id": int(result["custom_id"].split("-")[1])}
        for result in parsed_results
    ]

    usage_info(parsed_results)

    save_data_as_json(
        data=clean_parsed_results, data_name="output", path=path, record_index=record_index
    )


def retrieve_save_track_batch_file_content(
    name, batch_id, file_id, tracking_data, record_index: int
) -> str:
    """
    Retrieve the file content for a batch and save it to a file.
    :param name: Name of the file.
    :param batch_id: Batch ID.
    :param file_id: File ID.
    :param tracking_data: Tracking data.
    :param record_index: Index of the record in the batch.
    :return: The file content.
    """
    logger.info(f"Retrieving {name} file for batch {batch_id}...")
    file = client.files.content(file_id)
    file_text = file.text

    file_path_with_ext = save_data_as_jsonl(
        data=file_text, data_name=name, path=BATCH_OUTPUT_FILE, record_index=record_index
    )

    logger.info(f"Updating tracking file with {name} file path for batch {batch_id}")
    update_tracking_file(
        BATCH_TRACKING_FILE, tracking_data, batch_id, f"{name}_file", str(file_path_with_ext)
    )

    return file_text


def retrieve_results(batch, batch_id, tracking_data, record_index: int):
    """
    Retrieve the results for a batch and save them to a file.
    :param batch: Batch object.
    :param batch_id: Batch ID.
    :param tracking_data: Tracking data.
    :param record_index: Index of the record in the batch.
    """
    logger.info(f"Retrieving results for batch ID: {batch_id}")
    output_file_id = batch.output_file_id
    if not output_file_id:
        logger.warning(f"No output file found for batch ID {batch_id}.")
        if batch.error_file_id:
            logger.error(f"Batch ID {batch_id} encountered an error.")
            batch_error_file_id = batch.error_file_id
            retrieve_save_track_batch_file_content(
                "error", batch_id, batch_error_file_id, tracking_data, record_index
            )
        else:
            logger.error(f"No output or error file found for batch ID {batch_id}.")
    else:

        output_text = retrieve_save_track_batch_file_content(
            "output", batch_id, output_file_id, tracking_data, record_index
        )

        save_output_text_as_json(
            output_text=output_text, path=BATCH_OUTPUT_FILE, record_index=record_index
        )


def check_all_batches(client, tracking_file):
    """
    Check the status of all tracked batches.
    :param client: OpenAI client instance.
    :param tracking_file: Path to the tracking file.
    """
    tracking_data = load_tracking_data(tracking_file)
    if not tracking_data or "batches" not in tracking_data:
        logger.info("No tracked batches found.")
    else:
        logger.info("Checking status for all tracked batches:")
        for batch_id, batch_info in tracking_data["batches"].items():
            logger.info(f"Batch ID: {batch_id}")
            batch = client.batches.retrieve(batch_id)

            batch_status = batch.status
            logger.info(f"Status: {batch_status}")

            input_file = batch_info.get("input_file", None)
            logger.info(f"Input File: {input_file}")

            output_file = batch_info.get("output_file", None)
            logger.info(f"Output File: {output_file}")

            update_tracking_file(
                BATCH_TRACKING_FILE, tracking_data, batch_id, "status", batch_status
            )

            if (batch_status in ["completed", "failed", "expired", "cancelled"]) and (
                output_file is None
            ):
                record_index = batch_info.get("record_index", None)
                retrieve_results(batch, batch_id, tracking_data, record_index)


def update_tracking_file(tracking_file, tracking_data, batch_id, field_name, field_value):
    """
    Update the tracking file with the new field value.
    :param tracking_file: Path to the tracking file.
    :param tracking_data: Parsed JSON data.
    :param batch_id: The ID of the batch to update.
    :param field_name: The field to update.
    :param field_value: The new value for the field.
    """
    with open(tracking_file, "w") as f:
        if tracking_data["batches"][batch_id][field_name] != field_value:
            tracking_data["batches"][batch_id][field_name] = field_value
            json.dump(tracking_data, f, indent=4)
            logger.info(
                f"The field {field_name} has been updated for batch {batch_id} in {tracking_file}"
            )


if __name__ == "__main__":
    client = get_openai_client()
    check_all_batches(client, BATCH_TRACKING_FILE)

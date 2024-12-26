import json

from src.data.generate_synthetic_data.config import (
    OUTPUT_FILE_TREATMENT_PATIENT_DATA,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.config import (
    BATCH_TRACKING_FILE,
    BATCH_ERROR_FILE,
    BATCH_FILE_EXT,
    BATCH_OUTPUT_FILE,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.helpers import (
    save_generated_data_as_json,
)
from src.logging_config import setup_logger
from src.openai_utils.openai_api_handler import get_openai_client

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


def save_file(content, path):
    with open(path, "w") as f:
        f.write(content)


def usage_info(parsed_results):
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


def save_output_as_json_and_csv(output, path):
    logger.info("Parsing batch results...")
    parsed_results = [json.loads(line) for line in output.splitlines()]
    clean_parsed_results = [
        json.loads(result["response"]["body"]["choices"][0]["message"]["content"])
        | {"patient_id": result["custom_id"]}
        for result in parsed_results
    ]

    usage_info(parsed_results)

    # logger.info("Loading original patient data...")
    # original_patient_data = load_patient_data(OUTPUT_FILE_BASIC_PATIENT_DATA)

    logger.info("Merging original and generated data...")
    save_generated_data_as_json(clean_parsed_results, path.with_suffix(".json"))
    logger.info(f"Results saved to {path.with_suffix('.json')} and {path.with_suffix('.csv')}")


def retrieve_results(batch, batch_id, tracking_data):
    logger.info(f"Retrieving results for batch ID: {batch_id}")
    output_file_id = batch.output_file_id
    if not output_file_id:
        logger.warning(f"No output file found for batch ID {batch_id}.")
        if batch.error_file_id:
            logger.error(f"Batch ID {batch_id} encountered an error.")
            batch_error_file_id = batch.error_file_id
            logger.info(f"Retrieving error file {batch_error_file_id} for batch {batch_id}...")
            error_file = client.files.content(batch_error_file_id)
            logger.error(f"Error file content: {error_file.text.strip()}")
            error_file_path = BATCH_ERROR_FILE.with_stem(
                f"{BATCH_ERROR_FILE.stem}_{batch_id}"
            ).with_suffix(BATCH_FILE_EXT)
            logger.info(f"Saving error file to {error_file_path}")
            save_file(error_file.text, error_file_path)
            logger.info(f"Error file saved to {error_file_path}")

            logger.info(f"Updating tracking file with error file path for batch {batch_id}")
            update_tracking_file(
                BATCH_TRACKING_FILE, tracking_data, batch_id, "output_file", str(error_file_path)
            )
        else:
            logger.error(f"No output or error file found for batch ID {batch_id}.")
    else:
        logger.info(f"Retrieving output file for batch {batch_id}...")
        output_file = client.files.content(output_file_id)
        output_file_path = BATCH_OUTPUT_FILE.with_stem(
            f"{BATCH_OUTPUT_FILE.stem}_{batch_id}"
        ).with_suffix(BATCH_FILE_EXT)
        logger.info(f"Saving output file to {output_file_path}")
        save_file(output_file.text, output_file_path)
        logger.info(f"Output file saved to {output_file_path}")

        logger.info(f"Updating tracking file with output file path for batch {batch_id}")
        update_tracking_file(
            BATCH_TRACKING_FILE, tracking_data, batch_id, "output_file", str(output_file_path)
        )

        results_path = OUTPUT_FILE_TREATMENT_PATIENT_DATA.with_stem(
            f"{OUTPUT_FILE_TREATMENT_PATIENT_DATA.stem}_{batch_id}"
        )
        save_output_as_json_and_csv(output_file.text, results_path)


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
            output_file = batch_info.get("output_file", None)
            logger.info(f"Input File: {input_file}")
            logger.info(f"Output File: {output_file}")

            update_tracking_file(
                BATCH_TRACKING_FILE, tracking_data, batch_id, "status", batch_status
            )

            if (batch_status == "completed") and (output_file is None):
                retrieve_results(batch, batch_id, tracking_data)


def update_tracking_file(tracking_file, tracking_data, batch_id, field_name, field_value):
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

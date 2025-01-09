import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.data.generate_synthetic_data.config import OUTPUT_FILE_BASIC_PATIENT_DATA
from src.data.generate_synthetic_data.step_2_treatment_patient_data.config import (
    TEST_MODE,
    TEST_LIMIT,
)
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def build_openai_messages(previous_patient_data_and_treatment: list, current_record: dict) -> list:
    """
    Build OpenAI messages payload for a single record with sequential historical context.

    Args:
        previous_patient_data_and_treatment (list): Treatments and context from previous records for the patient.
        current_record (dict): The specific record for which treatment is generated.

    Returns:
        list: A list of messages formatted for the OpenAI API.
    """
    system_message = {
        "role": "system",
        "content": (
            "You are a medical assistant tasked with generating structured patient data. "
            "Base your recommendations on the provided patient record and treatment history, ensuring clinical accuracy. "
            "Follow the schema enforced by the system."
        ),
    }

    content_patient_history = (
        f"Patient treatment history up to this point: {json.dumps(previous_patient_data_and_treatment)}"
        if previous_patient_data_and_treatment
        else ""
    )
    user_message = {
        "role": "user",
        "content": (
            content_patient_history + "\n"
            "Current patient record: " + json.dumps(current_record) + "\n"
            "Generate data for this record based on the context."
            "Include medications with full details (name, dosage, frequency, duration), treatment history "
            "with logical reasons for medication changes, and concise lifestyle recommendations."
            "Make sure date_started is based on the provided record date."
        ),
    }

    return [system_message, user_message]


def merge_patient_data(patient_data, generated_data):
    """Merge patient data with generated data, adding patient_id."""
    for original, generated in zip(patient_data, generated_data):
        if generated:
            if isinstance(generated, dict):
                record = generated
            else:
                record = generated.model_dump()  # Convert structured data to a dictionary
            record["patient_id"] = original.get("patient_id")  # Add patient_id from input
            yield record


def save_data_as_json(data, data_name: str, path: Path, record_index: int):
    path_with_ext = path.with_stem(f"{path.stem}_{record_index}").with_suffix(".json")

    logger.info(f"Saving {data_name} file to {path_with_ext}")
    logger.info(f"Number of records: {len(data)}")
    logger.info(f"Data: {data}")

    with open(path_with_ext, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"{data_name.capitalize()} file saved to {path_with_ext}")
    return path_with_ext


def save_data_as_jsonl(data, data_name: str, path: Path, record_index: int):
    path_with_ext = path.with_stem(f"{path.stem}_{record_index}").with_suffix(".jsonl")

    logger.info(f"Saving {data_name} file to {path_with_ext}")

    with open(path_with_ext, "w") as f:
        f.write(data)

    logger.info(f"{data_name.capitalize()} file saved to {path_with_ext}")
    return path_with_ext


def load_patient_data(csv_path: Path):
    """Load patient data from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV file {csv_path} does not exist.")

    data = pd.read_csv(csv_path)
    data = data.fillna("None")
    return data


def load_json_data(path: Path):
    """Load JSON data from a file."""
    if not path.exists():
        raise FileNotFoundError(f"Input JSON file {path} does not exist.")

    with open(path, "r") as f:
        data = json.load(f)
    return data


def group_records_by_patient(patient_records):
    """
    Group patient records by their patient_id.

    Args:
        patient_records (list): List of patient records (dictionaries).

    Returns:
        dict: A dictionary where keys are patient IDs and values are lists of records for that patient.
    """
    grouped_data = defaultdict(list)
    for record in patient_records:
        grouped_data[record["patient_id"]].append(record)
    return grouped_data


def load_and_group_patient_data():
    logger.info(f"Loading patient data from: {OUTPUT_FILE_BASIC_PATIENT_DATA}")
    patient_data = load_patient_data(OUTPUT_FILE_BASIC_PATIENT_DATA).to_dict(orient="records")
    if TEST_MODE:
        logger.info(f"Running in test mode, limiting records to {TEST_LIMIT}.")
        patient_data = patient_data[: TEST_LIMIT + 1]
    else:
        logger.info("Running full dataset generation. Number of records: {len(patient_data)}")

    logger.info("Grouping records by patient...")
    grouped_records = group_records_by_patient(patient_data)
    logger.info(f"Number of patients: {len(grouped_records)}")
    return grouped_records

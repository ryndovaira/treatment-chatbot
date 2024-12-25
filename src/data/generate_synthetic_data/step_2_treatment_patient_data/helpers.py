import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

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
            "Ensure the output adheres to the specified JSON schema."
        ),
    }

    # Historical context (if available)
    if previous_patient_data_and_treatment:
        user_message = {
            "role": "user",
            "content": (
                "Generate data for this patient record. "
                "Include medications with details (name, dosage, frequency, duration), treatment history "
                "with reasons for medication changes, and lifestyle recommendations. "
                "Patient treatment history up to this point: "
                + json.dumps(previous_patient_data_and_treatment)
                + "\n"
                "Current patient record: " + json.dumps(current_record) + "\n"
                "Ensure the output adheres to the specified JSON schema."
            ),
        }
    else:
        # No history available
        user_message = {
            "role": "user",
            "content": (
                "Generate data for this patient record. "
                "Include medications with details (name, dosage, frequency, duration), treatment history "
                "with reasons for medication changes, and lifestyle recommendations. "
                "Current patient record: " + json.dumps(current_record) + "\n"
                "Ensure the output adheres to the specified JSON schema."
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


def save_generated_data_as_json(serializable_data, output_path: Path):
    """Save generated data to a JSON file."""
    logger.info(f"Saving generated data to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(serializable_data, f, indent=4)


def load_patient_data(csv_path: Path):
    """Load patient data from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV file {csv_path} does not exist.")

    return pd.read_csv(csv_path).to_dict(orient="records")


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

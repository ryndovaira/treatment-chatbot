import json
from pathlib import Path

import pandas as pd

from src.logging_config import setup_logger

logger = setup_logger(__name__)


def build_openai_messages(patient_record: dict) -> list:
    """
    Build the OpenAI messages payload for generating structured patient data.

    Args:
        patient_record (dict): A single patient record containing input data.

    Returns:
        list: A list of messages formatted for the OpenAI API.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a medical assistant tasked with generating structured patient data. "
                "Ensure the output adheres to the specified JSON schema."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate data for this patient record. "
                "Include medications with details (name, dosage, frequency, duration), treatment history "
                "with reasons for medication changes, and lifestyle recommendations."
                f"Patient record: {patient_record}"
            ),
        },
    ]


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

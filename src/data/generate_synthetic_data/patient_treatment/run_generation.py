import json
from pathlib import Path

import pandas as pd

from src.data.generate_synthetic_data.patient_treatment.patient_data_generator import (
    process_patient_data,
    validate_model_support,
)
from src.data.generate_synthetic_data.synthetic_data_config import DATA_DIR

# Path configuration

INPUT_FILE = DATA_DIR / "basic_patient_data.csv"
LOG_FILE_G = INPUT_FILE.parent / "generation_log.txt"

# Output for processed data
OUTPUT_FILE = DATA_DIR.parents[1] / "processed" / "patient_generated_data.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Configuration variables
TEST_MODE = True  # Set to True for testing or False for full data generation
TEST_LIMIT = 5  # Number of records to process in test mode
OPENAI_MODEL = "gpt-4o"  # Replace with the configured model
LOG_ERRORS = True  # Whether to log errors to a file


def load_patient_data(csv_path: Path):
    """Load patient data from a CSV file."""
    return pd.read_csv(csv_path).to_dict(orient="records")


def save_generated_data(output_data, output_path: Path):
    """Save generated data to a JSON file."""
    with open(output_path, "w") as f:
        serializable_data = [data.dict() if data else None for data in output_data]
        json.dump(serializable_data, f, indent=4)


def log_sample_outputs(data):
    """Log sample outputs for quick feedback."""
    if data:
        print("Sample Generated Data:")
        for idx, patient in enumerate(data[:3], 1):
            print(f"  {idx}. Patient ID: {patient.get('patient_id', 'N/A')}")
            if patient.get("current_medications"):
                print(
                    f"     Medications: {[med['name'] for med in patient['current_medications']]}"
                )
            if patient.get("lifestyle_recommendations"):
                print(f"     Lifestyle: {patient['lifestyle_recommendations']}")
    else:
        print("No data generated to preview.")


def validate_longitudinal_data(data):
    """Validate longitudinal data consistency."""
    print("Validating longitudinal data...")
    for patient in data:
        if "records" in patient and len(patient["records"]) > 1:
            print(f"  Patient {patient['patient_id']} has {len(patient['records'])} records.")


def main():
    print(f"Loading patient data from: {INPUT_FILE}")
    patient_data = load_patient_data(INPUT_FILE)

    # Validate the model's compatibility
    print(f"Validating OpenAI model: {OPENAI_MODEL}")
    try:
        validate_model_support(OPENAI_MODEL)
        print("Model validation passed.")
    except ValueError as e:
        print(f"Model validation failed: {e}")
        return

    # Determine the mode (test or full)
    if TEST_MODE:
        print("Running in test mode...")
        generated_data, errors = process_patient_data(
            patient_data, test_mode=True, test_limit=TEST_LIMIT
        )
        print("Test dataset generation completed!")
    else:
        print("Running full dataset generation...")
        generated_data, errors = process_patient_data(patient_data, test_mode=False)
        print("Full dataset generation completed!")

    # Validate and log outputs
    print(f"Saving generated data to: {OUTPUT_FILE}")
    save_generated_data(generated_data, OUTPUT_FILE)
    log_sample_outputs(generated_data)

    if TEST_MODE:
        validate_longitudinal_data(generated_data)

    if errors:
        print(f"Logging {len(errors)} errors to: {LOG_FILE_G}")
        with open(LOG_FILE_G, "w") as log_file:
            for error in errors:
                log_file.write(error + "\n")

    print("Data generation completed!")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import pandas as pd

from src.data.generate_synthetic_data.patient_treatment.patient_data_generator import (
    process_patient_data,
)
from src.data.generate_synthetic_data.synthetic_data_config import DATA_DIR

# Paths
INPUT_FILE = DATA_DIR / "basic_patient_data.csv"
LOG_FILE_G = INPUT_FILE.parent / "generation_log.txt"
OUTPUT_FILE = DATA_DIR.parents[1] / "processed" / "patient_generated_data.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Configuration
TEST_MODE = True  # Set to True for testing or False for full data generation
TEST_LIMIT = 5  # Number of records to process in test mode


def load_patient_data(csv_path: Path):
    """Load patient data from a CSV file."""
    return pd.read_csv(csv_path).to_dict(orient="records")


def save_generated_data(output_data, output_path: Path):
    """Save generated data to a JSON file."""
    with open(output_path, "w") as f:
        serializable_data = [data.model_dump() for data in output_data if data]
        json.dump(serializable_data, f, indent=4)


def main():
    print(f"Loading patient data from: {INPUT_FILE}")
    patient_data = load_patient_data(INPUT_FILE)

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

    print(f"Saving generated data to: {OUTPUT_FILE}")
    save_generated_data(generated_data, OUTPUT_FILE)

    print("Data generation completed!")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import pandas as pd

from src.data.generate_synthetic_data.patient_treatment.patient_data_generator import (
    process_patient_data,
)

# Path configuration
DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "raw" / "private"
INPUT_FILE = DATA_DIR / "basic_patient_data.csv"
LOG_FILE_G = INPUT_FILE.parent / "generation_log.txt"

# Output for processed data
OUTPUT_FILE = DATA_DIR.parents[1] / "processed" / "patient_generated_data.json"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_patient_data(csv_path: Path):
    """Load patient data from a CSV file."""
    return pd.read_csv(csv_path).to_dict(orient="records")


def save_generated_data(output_data, output_path: Path):
    """Save generated data to a JSON file."""
    with open(output_path, "w") as f:
        serializable_data = [data.dict() if data else None for data in output_data]
        json.dump(serializable_data, f, indent=4)


def main():
    print(f"Loading patient data from: {INPUT_FILE}")
    patient_data = load_patient_data(INPUT_FILE)

    print("Generating structured data for patients...")
    generated_data, errors = process_patient_data(patient_data, test_mode=False)

    print(f"Saving generated data to: {OUTPUT_FILE}")
    save_generated_data(generated_data, OUTPUT_FILE)

    if errors:
        print(f"Logging {len(errors)} errors to: {LOG_FILE_G}")
        with open(LOG_FILE_G, "w") as log_file:
            for error in errors:
                log_file.write(error + "\n")

    print("Full dataset generation completed!")


if __name__ == "__main__":
    main()

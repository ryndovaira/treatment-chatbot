import json
from pathlib import Path

import pandas as pd

from src.data.generate_synthetic_data.config import (
    OUTPUT_FILE_TREATMENT_PATIENT_DATA,
    OUTPUT_FILE_BASIC_PATIENT_DATA,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.config import (
    TEST_LIMIT,
    TEST_MODE,
    LOG_FILE_NAME,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.openai_structured_data_generator import (
    process_patient_data,
)
from src.logging_config import setup_logger

logger = setup_logger(__name__, file_name=LOG_FILE_NAME)


def load_patient_data(csv_path: Path):
    """Load patient data from a CSV file."""
    return pd.read_csv(csv_path).to_dict(orient="records")


def save_generated_data_as_json(patient_data, generated_data, output_path: Path):
    """Save generated data to a JSON file with patient_id added."""
    logger.info(f"Saving generated data to: {output_path}")
    serializable_data = []

    for original, generated in zip(patient_data, generated_data):
        if generated:
            record = generated.model_dump()  # Convert structured data to JSON
            record["patient_id"] = original.get("patient_id")  # Add patient_id
            serializable_data.append(record)

    try:
        with open(output_path, "w") as f:
            json.dump(serializable_data, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to save JSON file: {e}")


def save_generated_data_as_csv(patient_data, generated_data, output_path: Path):
    """Save generated data to a CSV file with patient_id added."""
    logger.info(f"Saving generated data to: {output_path}")
    rows = []

    for original, generated in zip(patient_data, generated_data):
        if generated:
            base_row = {
                "patient_id": original.get("patient_id"),
                "lifestyle_recommendations": "; ".join(generated.lifestyle_recommendations),
            }
            # Flatten current medications
            for med in generated.current_medications:
                med_row = base_row.copy()
                med_row.update(
                    {
                        "medication_name": med.name,
                        "dosage": med.dosage,
                        "frequency": med.frequency,
                        "duration": med.duration,
                    }
                )
                rows.append(med_row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def main():
    logger.info(f"Loading patient data from: {OUTPUT_FILE_BASIC_PATIENT_DATA}")
    patient_data = load_patient_data(OUTPUT_FILE_BASIC_PATIENT_DATA)

    if TEST_MODE:
        logger.info("Running in test mode...")
        generated_data, errors = process_patient_data(
            patient_data, test_mode=True, test_limit=TEST_LIMIT
        )
        logger.info("Test dataset generation completed!")
    else:
        logger.info("Running full dataset generation...")
        generated_data, errors = process_patient_data(patient_data, test_mode=False)
        logger.info("Full dataset generation completed!")

    logger.info(f"Saving generated data")
    save_generated_data_as_json(
        patient_data, generated_data, OUTPUT_FILE_TREATMENT_PATIENT_DATA.with_suffix(".json")
    )
    save_generated_data_as_csv(
        patient_data, generated_data, OUTPUT_FILE_TREATMENT_PATIENT_DATA.with_suffix(".csv")
    )

    logger.info("Data generation completed!")


if __name__ == "__main__":
    main()

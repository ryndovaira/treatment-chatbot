from src.data.generate_synthetic_data.config import (
    OUTPUT_FILE_TREATMENT_PATIENT_DATA,
    OUTPUT_FILE_BASIC_PATIENT_DATA,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.config import (
    TEST_LIMIT,
    TEST_MODE,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.helpers import (
    save_generated_data_as_json,
    load_patient_data,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.openai_sync.openai_structured_data_generator import (
    process_patient_data,
)
from src.logging_config import setup_logger

logger = setup_logger(__name__)


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
        generated_data, OUTPUT_FILE_TREATMENT_PATIENT_DATA.with_suffix(".json")
    )

    logger.info("Data generation completed!")


if __name__ == "__main__":
    main()

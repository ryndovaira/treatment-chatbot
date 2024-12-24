from generate_batch_input import prepare_batch_file
from send_batch import submit_batch
from src.data.generate_synthetic_data.config import OUTPUT_FILE_BASIC_PATIENT_DATA
from src.data.generate_synthetic_data.step_2_treatment_patient_data.config import (
    TEST_MODE,
    TEST_LIMIT,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.helpers import load_patient_data
from src.data.generate_synthetic_data.step_2_treatment_patient_data.openai_async.config import (
    BATCH_INPUT_FILE,
)
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def main():
    logger.info("Starting batch generation process...")

    # Step 1: Load patient data
    logger.info(f"Loading patient data from: {OUTPUT_FILE_BASIC_PATIENT_DATA}")
    patient_data = load_patient_data(OUTPUT_FILE_BASIC_PATIENT_DATA)

    # Step 2: Limit data for test mode
    if TEST_MODE:
        logger.info(f"Running in test mode. Limiting records to {TEST_LIMIT}.")
        patient_data = patient_data[:TEST_LIMIT]

    # Step 3: Prepare batch input file
    prepare_batch_file(patient_data, BATCH_INPUT_FILE)
    logger.info(f"Batch input file created: {BATCH_INPUT_FILE}")

    # Step 4: Submit batch request
    batch_id = submit_batch(BATCH_INPUT_FILE)
    logger.info(f"Batch request submitted with ID: {batch_id}")


if __name__ == "__main__":
    main()

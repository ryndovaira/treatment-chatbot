import json
from pathlib import Path

from batch_utils import retrieve_batch_results
from src.data.generate_synthetic_data.config import OUTPUT_FILE_BASIC_PATIENT_DATA
from src.data.generate_synthetic_data.step_2_treatment_patient_data.helpers import (
    load_patient_data,
    save_generated_data_as_json,
    save_generated_data_as_csv,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.openai_async.config import (
    BATCH_OUTPUT_FILE,
)
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def save_results(batch_id, output_path: Path):
    logger.info(f"Retrieving results for batch ID: {batch_id}")
    results = retrieve_batch_results(batch_id)

    logger.info("Parsing batch results...")
    parsed_results = [json.loads(line) for line in results.splitlines()]

    logger.info("Loading original patient data...")
    original_patient_data = load_patient_data(OUTPUT_FILE_BASIC_PATIENT_DATA)

    logger.info("Merging original and generated data...")
    save_generated_data_as_json(
        original_patient_data, parsed_results, output_path.with_suffix(".json")
    )
    save_generated_data_as_csv(
        original_patient_data, parsed_results, output_path.with_suffix(".csv")
    )
    logger.info(
        f"Results saved to {output_path.with_suffix('.json')} and {output_path.with_suffix('.csv')}"
    )


def main():
    # Example batch ID and output path
    batch_id = "your-batch-id"
    save_results(batch_id, BATCH_OUTPUT_FILE)


if __name__ == "__main__":
    main()

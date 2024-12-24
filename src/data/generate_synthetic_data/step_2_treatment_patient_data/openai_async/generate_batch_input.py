import json

from src.config import OPENAI_MAX_TOKENS, OPENAI_MODEL, OPENAI_TEMPERATURE
from src.data.generate_synthetic_data.config import OUTPUT_FILE_BASIC_PATIENT_DATA
from src.data.generate_synthetic_data.step_2_treatment_patient_data.helpers import (
    build_openai_messages,
    load_patient_data,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.openai_async.send_batch import (
    BATCH_FILE_PATH,
)


def prepare_batch_file(patient_records, batch_file_path):
    """
    Prepare a .jsonl file for the Batch API.

    :param patient_records: List of patient records or input data.
    :param batch_file_path: Path to the output .jsonl file.
    """
    with open(batch_file_path, "w") as f:
        for idx, record in enumerate(patient_records):
            messages = build_openai_messages(record)
            batch_request = {
                "custom_id": f"patient-{record['patient_id']}-record-{record['record_id']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "max_tokens": OPENAI_MAX_TOKENS,
                    "temperature": OPENAI_TEMPERATURE,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": PatientData.model_json_schema(),
                    },
                },
            }
            f.write(json.dumps(batch_request) + "\n")
    print(f"Batch file created at {batch_file_path}")


if __name__ == "__main__":
    # Read data from CSV
    patient_data = load_patient_data(OUTPUT_FILE_BASIC_PATIENT_DATA)

    # Prepare the batch file
    prepare_batch_file(patient_data, BATCH_FILE_PATH)

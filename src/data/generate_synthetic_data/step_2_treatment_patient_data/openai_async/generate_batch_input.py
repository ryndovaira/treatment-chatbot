import json
from pathlib import Path

from src.data.generate_synthetic_data.config import OUTPUT_FILE_BASIC_PATIENT_DATA
from src.data.generate_synthetic_data.step_2_treatment_patient_data.config import (
    BATCH_INPUT_FILE,
    PATIENT_RECORD_INDEX,
    BATCH_OUTPUT_FILE,
    PATIENT_DATA_AND_TREATMENT,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.helpers import (
    build_openai_messages,
    load_patient_data,
    load_json_data,
    save_data_as_json,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.patient_data_models import (
    PatientData,
)
from src.env_config import OPENAI_MAX_TOKENS, OPENAI_MODEL, OPENAI_TEMPERATURE
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def get_patient_records_filtered_by_index() -> tuple:
    """
    Get patient records filtered by the record index.
    :return: Tuple of current patient records and previous patient data and treatment.
    """
    patient_records = load_patient_data(OUTPUT_FILE_BASIC_PATIENT_DATA)
    if PATIENT_RECORD_INDEX == 1:
        return (
            patient_records[patient_records["record_id"] == PATIENT_RECORD_INDEX].to_dict(
                orient="records"
            ),
            [],
        )
    else:
        previous_context_patient_record_index = PATIENT_RECORD_INDEX - 1

        previous_batch_input_file = BATCH_INPUT_FILE.with_stem(
            f"{BATCH_INPUT_FILE.stem}_{previous_context_patient_record_index}"
        ).with_suffix(".json")

        previous_batch_input = load_json_data(previous_batch_input_file)

        previous_batch_output_file = BATCH_OUTPUT_FILE.with_stem(
            f"{BATCH_OUTPUT_FILE.stem}_{previous_context_patient_record_index}"
        ).with_suffix(".json")
        previous_patient_treatment = load_json_data(previous_batch_output_file)

        patient_data_with_treatment = merge_patient_data_with_treatment(
            previous_batch_input, previous_patient_treatment
        )

        current_patient_records = patient_records[
            patient_records["record_id"] == PATIENT_RECORD_INDEX
        ].to_dict(orient="records")

        if PATIENT_RECORD_INDEX > 2:
            previous_patients_data_and_treatment = load_json_data(
                PATIENT_DATA_AND_TREATMENT.with_stem(
                    f"{PATIENT_DATA_AND_TREATMENT.stem}_{previous_context_patient_record_index}"
                ).with_suffix(".json")
            )
            merged_patient_data_with_treatment = (
                patient_data_with_treatment + previous_patients_data_and_treatment
            )
            return current_patient_records, merged_patient_data_with_treatment
        else:
            return current_patient_records, patient_data_with_treatment


def merge_two_patient_data_with_treatment(
    patient_data_with_treatment_1, patient_data_with_treatment_2
) -> list:
    # Create a lookup dictionary for the second list
    patient_data_with_treatment_2_dict = {
        item["patient_id"]: item for item in patient_data_with_treatment_2
    }

    # Merge the two lists by patient_id
    merged_patient_data = []
    for item in patient_data_with_treatment_1:
        patient_id = item["patient_id"]
        if patient_id in patient_data_with_treatment_2_dict:
            # Merge the dictionaries from both lists
            merged_item = {**item, **patient_data_with_treatment_2_dict[patient_id]}
        else:
            # If no match in the second list, include the item as is
            merged_item = item
        merged_patient_data.append(merged_item)

    # Add any items from the second list that are not in the first list
    for patient_id, item in patient_data_with_treatment_2_dict.items():
        if patient_id not in {entry["patient_id"] for entry in patient_data_with_treatment_1}:
            merged_patient_data.append(item)

    return merged_patient_data


def merge_patient_data_with_treatment(patients_data, patient_treatment) -> list:
    """
    Merge patient data with treatment data.
    :param patients_data: List of patient data.
    :param patient_treatment: List of patient treatment data.
    :return: List of merged patient data.
    """
    # Create a dictionary for quick lookups from the treatments_data by patient_id
    treatments_dict = {t["patient_id"]: t for t in patient_treatment}

    # Merge the data
    merged_data = []
    for patient in patients_data:
        patient_id = patient["patient_id"]
        if patient_id in treatments_dict:
            # Combine the dictionaries
            merged_patient = {**patient, **treatments_dict[patient_id]}
        else:
            # If no treatment data is found, include the patient data as-is
            logger.error(f"No treatment data found for patient {patient_id}", exc_info=True)
            raise ValueError(f"No treatment data found for patient {patient_id}")
        merged_data.append(merged_patient)
    return merged_data


def prepare_batch_file(
    current_patient_data, previous_patients_data_and_treatment, batch_file_path: Path
):
    """
    Prepare a .jsonl file for the Batch API.

    :param current_patient_data: List of patient data.
    :param batch_file_path: Path to the output .jsonl file.
    """
    schema_with_name = {
        "name": "PatientDataSchema",
        "schema": PatientData.model_json_schema(),
    }

    batch_input = []
    for idx, patient_data_all in enumerate(current_patient_data):
        record = patient_data_all.copy()
        patient_id = record.pop("patient_id", None)
        record_id = record.pop("record_id", None)
        previous_data_and_treatment = [
            d for d in previous_patients_data_and_treatment if d["patient_id"] == patient_id
        ]
        cleaned_previous_data_and_treatment = [
            {k: v for k, v in record.items() if k not in {"patient_id", "record_id"}}
            for record in previous_data_and_treatment
        ]

        messages = build_openai_messages(
            previous_patient_data_and_treatment=cleaned_previous_data_and_treatment,
            current_record=record,
        )
        batch_request = {
            "custom_id": f"patient_id-{patient_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": OPENAI_MODEL,
                "messages": messages,
                "max_tokens": OPENAI_MAX_TOKENS,
                "temperature": OPENAI_TEMPERATURE,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": schema_with_name,
                },
            },
        }
        batch_input.append(json.dumps(batch_request))

    logger.info(f"Batch input created with {len(batch_input)} records.")
    logger.info(f"Batch input: {batch_input}")

    batch_file_path = batch_file_path.with_stem(f"{batch_file_path.stem}_{PATIENT_RECORD_INDEX}")
    with open(batch_file_path, "w") as f:
        f.write("\n".join(batch_input))
    print(f"Batch file created at {batch_file_path}")


if __name__ == "__main__":
    patient_records, previous_patients_data_and_treatment = get_patient_records_filtered_by_index()

    save_data_as_json(
        data=patient_records,
        data_name="current patient data",
        path=BATCH_INPUT_FILE,
        record_index=PATIENT_RECORD_INDEX,
    )

    save_data_as_json(
        data=previous_patients_data_and_treatment,
        data_name="previous patient data with treatment",
        path=PATIENT_DATA_AND_TREATMENT,
        record_index=PATIENT_RECORD_INDEX,
    )

    prepare_batch_file(
        patient_records,
        previous_patients_data_and_treatment,
        BATCH_INPUT_FILE.with_suffix(".jsonl"),
    )

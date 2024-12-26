from tqdm import tqdm

from src.config import OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS
from src.data.generate_synthetic_data.config import (
    OUTPUT_FILE_TREATMENT_PATIENT_DATA,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.helpers import (
    build_openai_messages,
    save_generated_data_as_json,
    load_and_group_patient_data,
)
from src.data.generate_synthetic_data.step_2_treatment_patient_data.patient_data_models import (
    PatientData,
)
from src.logging_config import setup_logger
from src.openai_utils.openai_api_handler import get_openai_client
from src.openai_utils.openai_token_count_and_cost import (
    estimate_total_price,
    calculate_price,
    calculate_token_count,
)

logger = setup_logger(__name__)


def validate_model_support(model):
    """Validate the configured model to ensure it supports Structured Outputs."""
    if not model.startswith("gpt-4o") and not model.startswith("o1"):
        raise ValueError(
            f"The configured model '{model}' is not supported for Structured Outputs. "
            f"Please use a model like 'gpt-4o', 'gpt-4o-mini', 'o1', or their supported variants."
        )


def track_token_usage(messages, completion, model):
    """Calculate input/output tokens and costs for a single API call."""
    # Calculate input tokens and cost
    token_estimation = estimate_total_price(messages, model=model)
    input_tokens = token_estimation["input_tokens"]
    input_cost = token_estimation["input_price"]

    # Calculate output tokens and cost
    output_tokens = calculate_token_count(
        [{"role": "assistant", "content": completion.choices[0].message.content}],
        model,
    )
    output_cost = calculate_price(output_tokens, model, input=False)

    return {
        "input_tokens": input_tokens,
        "input_cost": input_cost,
        "output_tokens": output_tokens,
        "output_cost": output_cost,
    }


def generate_patient_additional_data(grouped_patient_data):
    """
    Generate structured patient data using OpenAI for each record, informed by patient history.

    Args:
        grouped_patient_data (dict): Dictionary of patient_id to their list of records.

    Returns:
        tuple: List of structured patient data and list of error messages.
    """
    client = get_openai_client()
    generated_data = []

    for index, patient_records in tqdm(
        grouped_patient_data.items(), desc="Processing Patients", unit="patient"
    ):
        previous_patient_data_treatment = []
        for current_record in patient_records:
            record = current_record.copy()
            patient_id = record.pop("patient_id", None)
            record_id = record.pop("record_id", None)
            messages = build_openai_messages(previous_patient_data_treatment, record)
            try:
                completion = client.beta.chat.completions.parse(
                    model=OPENAI_MODEL,
                    messages=messages,
                    max_tokens=OPENAI_MAX_TOKENS,
                    temperature=OPENAI_TEMPERATURE,
                    response_format=PatientData,
                )

                output = completion.choices[0].message.parsed.model_dump()
                patient_data_and_treatment = output | record
                previous_patient_data_treatment.append(patient_data_and_treatment)
                generated_data.append(
                    patient_data_and_treatment | {"patient_id": patient_id, "record_id": record_id}
                )
            except Exception as e:
                error_msg = f"Error generating data for patient_id {patient_id} and record_id {record_id}: {e}"
                generated_data.append(None)
                logger.error(error_msg)
                with open("error_log.txt", "a") as log_file:
                    log_file.write(error_msg + "\n")

    return generated_data


def process_patient_data(patient_records):
    """Process patient data in test or full mode."""
    validate_model_support(OPENAI_MODEL)

    return generate_patient_additional_data(patient_records)


def save_generated_data(generated_data):
    logger.info(f"Saving generated data")
    save_generated_data_as_json(
        generated_data, OUTPUT_FILE_TREATMENT_PATIENT_DATA.with_suffix(".json")
    )


def main():
    logger.info("Starting data generation...")

    grouped_records = load_and_group_patient_data()
    generated_data = process_patient_data(grouped_records)
    save_generated_data(generated_data)

    logger.info("Data generation completed!")


if __name__ == "__main__":
    main()

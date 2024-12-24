from tqdm import tqdm

from src.config import OPENAI_MODEL, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE
from src.data.generate_synthetic_data.step_2_treatment_patient_data.helpers import (
    build_openai_messages,
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


def generate_patient_additional_data(
    patient_records, model=OPENAI_MODEL, max_tokens=OPENAI_MAX_TOKENS, log_errors=True
):
    """Generate structured patient data using OpenAI."""
    client = get_openai_client()
    structured_data = []
    errors = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    for record in tqdm(patient_records, desc="Processing Patients", unit="patient"):
        messages = build_openai_messages(record)
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=OPENAI_TEMPERATURE,
                response_format=PatientData,
            )
            # Track tokens and costs
            token_usage = track_token_usage(messages, completion, model)
            total_input_tokens += token_usage["input_tokens"]
            total_output_tokens += token_usage["output_tokens"]
            total_cost += token_usage["input_cost"] + token_usage["output_cost"]

            structured_data.append(completion.choices[0].message.parsed)
        except Exception as e:
            error_msg = (
                f"Error generating data for record {record.get('patient_id', 'unknown')}: {e}"
            )
            errors.append(error_msg)
            structured_data.append(None)
            logger.error(error_msg)
            if log_errors:
                with open("error_log.txt", "a") as log_file:
                    log_file.write(error_msg + "\n")

    logger.info(f"Total Input Tokens: {total_input_tokens}")
    logger.info(f"Total Output Tokens: {total_output_tokens}")
    logger.info(f"Total Cost: ${total_cost:.6f}")
    return structured_data, errors


def process_patient_data(patient_records, test_mode=True, test_limit=5):
    """Process patient data in test or full mode."""
    # Centralized model validation
    validate_model_support(OPENAI_MODEL)

    if test_mode:
        logger.info(f"Running in test mode, limiting records to {test_limit}.")
        patient_records = patient_records[:test_limit]

    return generate_patient_additional_data(patient_records)

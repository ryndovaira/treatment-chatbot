from pydantic import BaseModel

from src.config import OPENAI_MODEL, OPENAI_MAX_TOKENS
from src.logging_config import setup_logger
from src.openai_utils.openai_api_handler import get_openai_client

logger = setup_logger(__name__)


class PatientData(BaseModel):
    """Schema for OpenAI structured output."""

    current_medications: str
    treatment_history: str
    lifestyle_recommendations: str


def validate_model_support(model: str):
    """Validate if the model is properly set for Structured Outputs."""
    if not model.startswith("gpt-4o") and not model.startswith("o1"):
        raise ValueError(
            f"The configured model '{model}' is not supported for Structured Outputs. "
            f"Please use a model like 'gpt-4o', 'gpt-4o-mini', 'o1', or their supported variants."
        )


def generate_patient_additional_data(
    patient_records, model=OPENAI_MODEL, max_tokens=OPENAI_MAX_TOKENS, log_errors=False
):
    client = get_openai_client()
    structured_data = []
    errors = []

    validate_model_support(model)

    for record in patient_records:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical assistant tasked with generating additional patient data. "
                    "Ensure your output adheres to the given JSON schema and is accurate."
                ),
            },
            {
                "role": "user",
                "content": f"Generate the additional data for the following patient record: {record}",
            },
        ]
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=PatientData,
                max_tokens=max_tokens,
            )
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

    return structured_data, errors


def process_patient_data(patient_records, test_mode=True, test_limit=5):
    """
    Process patient data with OpenAI API for testing or full generation.
    """
    if test_mode:
        logger.info("Running in test mode, limiting records.")
        patient_records = patient_records[:test_limit]

    return generate_patient_additional_data(patient_records)

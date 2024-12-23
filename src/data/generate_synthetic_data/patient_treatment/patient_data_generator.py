from typing import List, Optional

from pydantic import BaseModel, Field

from src.config import OPENAI_MODEL, OPENAI_MAX_TOKENS
from src.logging_config import setup_logger
from src.openai_utils.openai_api_handler import get_openai_client

logger = setup_logger(__name__)


class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication")
    dosage: str = Field(..., description="Dosage of the medication (e.g., '500 mg')")
    frequency: str = Field(..., description="Frequency of intake (e.g., 'Once daily')")
    duration: str = Field(..., description="Duration of usage (e.g., '6 months', 'Ongoing')")


class TreatmentHistoryEntry(BaseModel):
    date_started: str = Field(
        ..., description="Start date of the treatment (ISO format YYYY-MM-DD)"
    )
    medications: List[Medication] = Field(
        ..., description="List of medications used during this period"
    )
    reason_for_change: Optional[str] = Field(
        None, description="Reason for changing medications, if applicable"
    )


class PatientData(BaseModel):
    current_medications: List[Medication] = Field(
        ..., description="List of current medications with details"
    )
    treatment_history: List[TreatmentHistoryEntry] = Field(
        ..., description="Patient's treatment history"
    )
    lifestyle_recommendations: List[str] = Field(
        ..., description="List of lifestyle recommendations for the patient"
    )


def validate_model_support(model: str):
    """Validate the configured model to ensure it supports Structured Outputs."""
    if not model.startswith("gpt-4o") and not model.startswith("o1"):
        raise ValueError(
            f"The configured model '{model}' is not supported for Structured Outputs. "
            f"Please use a model like 'gpt-4o', 'gpt-4o-mini', 'o1', or their supported variants."
        )


def log_patient_data(input_data, output_data, errors):
    """Log input, output, and error data for patient processing."""
    logger.info("=== Patient Data Log Start ===")
    for idx, input_record in enumerate(input_data, start=1):
        patient_id = input_record.get("patient_id", "unknown")
        record_id = input_record.get("record_id", "unknown")
        record_date = input_record.get("record_date", "unknown")

        logger.info(
            f"Processing Patient ID: {patient_id}, Record ID: {record_id}, Date: {record_date}"
        )
        logger.info(f"Input Data: {input_record}")

        output_record = output_data[idx - 1] if idx - 1 < len(output_data) else None
        error = errors[idx - 1] if idx - 1 < len(errors) else None

        if output_record:
            logger.info(f"Output Record: {output_record}")
            logger.info("Output Summary:")
            logger.info(
                f"- Current Medications: {[med.name for med in output_record.current_medications]}"
            )
            logger.info(f"- Lifestyle Recommendations: {output_record.lifestyle_recommendations}")
            logger.info("- Treatment Changes:")
            for th in output_record.treatment_history:
                logger.info(
                    f"  - Date Started: {th.date_started}, Reason for Change: {th.reason_for_change}"
                )
        elif error:
            logger.error(f"Error: {error}")
        else:
            logger.warning("No output or error recorded.")

    logger.info("=== Patient Data Log End ===")


def generate_patient_additional_data(
    patient_records, model=OPENAI_MODEL, max_tokens=OPENAI_MAX_TOKENS, log_errors=False
):
    """Generate structured patient data using OpenAI."""
    client = get_openai_client()
    structured_data = []
    errors = []

    validate_model_support(model)

    for record in patient_records:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical assistant tasked with generating structured patient data. "
                    "Ensure the output adheres to the specified JSON schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Generate additional data for this patient record. "
                    "Include medications with details (name, dosage, frequency, duration), treatment history "
                    "with reasons for medication changes, and lifestyle recommendations."
                    f"Patient record: {record}"
                ),
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

    log_patient_data(patient_records, structured_data, errors)
    return structured_data, errors


def process_patient_data(patient_records, test_mode=True, test_limit=5):
    """Process patient data in test or full mode."""
    if test_mode:
        logger.info(f"Running in test mode, limiting records ({test_limit}).")
        patient_records = patient_records[:test_limit]

    return generate_patient_additional_data(patient_records)

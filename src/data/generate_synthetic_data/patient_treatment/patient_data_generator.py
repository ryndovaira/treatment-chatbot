from typing import List, Optional

from pydantic import BaseModel, Field

from src.config import OPENAI_MODEL, OPENAI_MAX_TOKENS
from src.logging_config import setup_logger
from src.openai_utils.openai_api_handler import get_openai_client

logger = setup_logger(__name__)


class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication")
    dosage: str = Field(..., description="Dosage of the medication")
    frequency: str = Field(..., description="Frequency of intake")
    duration: str = Field(..., description="Duration of usage")


class TreatmentHistoryEntry(BaseModel):
    date_started: str = Field(..., description="Start date of the treatment")
    medications: List[Medication] = Field(
        ..., description="List of medications used during this period"
    )
    reason_for_change: Optional[str] = Field(
        None, description="Reason for changing medications, if applicable"
    )


class PatientData(BaseModel):
    current_medications: List[Medication] = Field(
        ...,
        description="List of current medications with details like name, dosage, frequency, and duration",
    )
    treatment_history: List[TreatmentHistoryEntry] = Field(
        ...,
        description="Patient's treatment history with start dates, medications, and reasons for medication changes",
    )
    lifestyle_recommendations: List[str] = Field(
        ..., description="List of lifestyle recommendations for the patient"
    )


def validate_model_support(model: str):
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

    return structured_data, errors


def process_patient_data(patient_records, test_mode=True, test_limit=5):
    if test_mode:
        logger.info("Running in test mode, limiting records.")
        patient_records = patient_records[:test_limit]

    return generate_patient_additional_data(patient_records)

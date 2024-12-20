from typing import List

from pydantic import BaseModel, Field

from src.config import OPENAI_MODEL, OPENAI_MAX_TOKENS
from src.logging_config import setup_logger
from src.openai_utils.openai_api_handler import get_openai_client

logger = setup_logger(__name__)


class PatientData(BaseModel):
    """Schema for OpenAI structured output."""

    current_medications: List[dict] = Field(
        ...,
        description="List of current medications with details like name, dosage, frequency, and duration",
        example=[
            {"name": "Metformin", "dosage": "500mg", "frequency": "BID", "duration": "2 years"}
        ],
    )
    treatment_history: List[dict] = Field(
        ...,
        description=(
            "Patient's treatment history with start dates, medications, and reasons for medication changes"
        ),
        example=[
            {
                "date_started": "2020-01-01",
                "medications": [
                    {
                        "name": "Metformin",
                        "dosage": "500mg",
                        "frequency": "BID",
                        "duration": "2 years",
                    },
                    {
                        "name": "Insulin",
                        "dosage": "10 units",
                        "frequency": "QD",
                        "duration": "6 months",
                    },
                ],
                "reason_for_change": "Switched to insulin after blood sugar levels remained uncontrolled",
            }
        ],
    )
    lifestyle_recommendations: List[str] = Field(
        ...,
        description="List of lifestyle recommendations for the patient",
        example=[
            "Adopt a balanced diet focusing on whole grains, vegetables, and lean proteins",
            "Engage in at least 150 minutes of moderate aerobic activity weekly",
        ],
    )

    class Config:
        json_schema_extra = {
            "type": "object",
            "properties": {
                "current_medications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the medication"},
                            "dosage": {"type": "string", "description": "Dosage of the medication"},
                            "frequency": {"type": "string", "description": "Frequency of intake"},
                            "duration": {"type": "string", "description": "Duration of usage"},
                        },
                        "required": ["name", "dosage", "frequency", "duration"],
                    },
                },
                "treatment_history": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date_started": {
                                "type": "string",
                                "description": "Start date of the treatment",
                            },
                            "medications": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Name of the medication",
                                        },
                                        "dosage": {
                                            "type": "string",
                                            "description": "Dosage of the medication",
                                        },
                                        "frequency": {
                                            "type": "string",
                                            "description": "Frequency of intake",
                                        },
                                        "duration": {
                                            "type": "string",
                                            "description": "Duration of usage",
                                        },
                                    },
                                    "required": ["name", "dosage", "frequency", "duration"],
                                },
                            },
                            "reason_for_change": {
                                "type": ["string", "null"],
                                "description": "Reason for changing medications, if applicable",
                            },
                        },
                        "required": ["date_started", "medications"],
                    },
                },
                "lifestyle_recommendations": {
                    "type": "array",
                    "items": {"type": "string", "description": "Lifestyle advice"},
                },
            },
            "required": ["current_medications", "treatment_history", "lifestyle_recommendations"],
        }


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
                    "You are a medical assistant tasked with generating structured patient data. "
                    "Ensure the output adheres to the specified JSON schema."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Generate the additional data for the following patient record. "
                    "Ensure that the medications are listed with their name, dosage, frequency, and duration. "
                    "Provide treatment history with reasons for changes and include lifestyle recommendations. "
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
    """
    Process patient data with OpenAI API for testing or full generation.
    """
    if test_mode:
        logger.info("Running in test mode, limiting records.")
        patient_records = patient_records[:test_limit]

    return generate_patient_additional_data(patient_records)

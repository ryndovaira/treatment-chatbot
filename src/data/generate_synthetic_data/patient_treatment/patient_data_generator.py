from pydantic import BaseModel

from src.config import OPENAI_MODEL, OPENAI_MAX_TOKENS
from src.openai_utils.openai_api_handler import get_openai_client


# # Fetch OpenAI settings
# OPENAI_MODEL = get_env_var("OPENAI_MODEL", default="gpt-4o-latest")
# OPENAI_TEMPERATURE = float(get_env_var("OPENAI_TEMPERATURE", default=0.5))
# OPENAI_MAX_TOKENS = int(get_env_var("OPENAI_MAX_TOKENS", default=100))


class PatientData(BaseModel):
    """Schema for OpenAI structured output."""

    current_medications: str
    treatment_history: str
    lifestyle_recommendations: str


def generate_patient_additional_data(
    patient_records, model=OPENAI_MODEL, max_tokens=OPENAI_MAX_TOKENS
):
    """
    Generate additional patient data using OpenAI's API with structured output.
    """
    client = get_openai_client()
    structured_data = []

    for record in patient_records:
        messages = [
            {
                "role": "system",
                "content": "You are a medical assistant tasked with generating additional patient data based on the input.",
            },
            {
                "role": "user",
                "content": f"Generate the additional data for the following patient record: {record}",
            },
        ]
        try:
            completion = client.beta.chat.completions.parse(
                model=model, messages=messages, response_format=PatientData, max_tokens=max_tokens
            )
            structured_data.append(completion.choices[0].message.parsed)
        except Exception as e:
            print(f"Error generating data for record {record['patient_id']}: {e}")
            structured_data.append(None)

    return structured_data


def process_patient_data(patient_records, test_mode=True, test_limit=5):
    """
    Process patient data with OpenAI API for testing or full generation.
    """
    if test_mode:
        patient_records = patient_records[:test_limit]

    return generate_patient_additional_data(patient_records)

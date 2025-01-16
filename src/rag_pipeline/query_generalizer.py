from typing import Dict, List

from src.logging_config import setup_logger

logger = setup_logger(__name__)

# Feature prioritization categories
FEATURE_PRIORITIZATION = {
    "critical": [
        "symptoms",
        "symptom_severity",
        "co_morbidities",
        "age",
        "gender",
        "ethnicity",
    ],
    "secondary": [
        "bmi",
        "blood_pressure_systolic_mm_hg",
        "blood_pressure_diastolic_mm_hg",
        "cholesterol_mg_dl",
        "triglycerides_mg_dl",
    ],
    "contextual": [
        "pregnancy_status",
        "hba1c_percent",
        "fasting_glucose_mg_dl",
        "postprandial_glucose_mg_dl",
        "kidney_function_gfr",
    ],
}


def prioritize_features(patient_data: Dict[str, any]) -> List[str]:
    """
    Prioritize and filter patient features for generalization based on importance.

    Args:
        patient_data (Dict[str, any]): Input patient data with all available fields.

    Returns:
        List[str]: A list of prioritized and formatted features for generalization.
    """
    prioritized_features = []

    # Extract features in order of priority
    for priority, features in FEATURE_PRIORITIZATION.items():
        for feature in features:
            if feature in patient_data and patient_data[feature]:
                prioritized_features.append(f"{feature}: {patient_data[feature]}")

    # Handle unknown features dynamically
    known_features = set(
        feature for features in FEATURE_PRIORITIZATION.values() for feature in features
    )
    for feature, value in patient_data.items():
        if feature not in known_features and value:
            logger.warning(f"Unknown feature '{feature}' found. Including in the output.")
            prioritized_features.append(f"{feature}: {value}")

    return prioritized_features


def prepare_patient_data(patient_data: Dict[str, any]) -> str:
    """
    Prepare a formatted string of patient data for generalization.

    Args:
        patient_data (Dict[str, any]): Input patient data.

    Returns:
        str: Formatted string of prioritized patient data.
    """
    features = prioritize_features(patient_data)
    patient_data_str = "; ".join(features)
    logger.debug(f"Prepared patient data: {patient_data_str}")
    return patient_data_str


def generalize_query(patient_data_str: str, base_query: str) -> str:
    """
    Generate a generalized query for public data retrieval.

    Args:
        patient_data_str (str): Formatted patient data string.
        base_query (str): Base query text.

    Returns:
        str: A generalized query with prioritized patient context.
    """
    if not patient_data_str:
        return base_query  # Fallback to the base query if no features are available

    generalized_query = f"{patient_data_str}. {base_query}"
    logger.debug(f"Generalized query: {generalized_query}")
    return generalized_query


# Example usage
if __name__ == "__main__":
    patient_info = {
        "age": 30,
        "gender": "Female",
        "ethnicity": "Asian",
        "pregnancy_status": "None",
        "weight_kg": 66.9,
        "height_cm": 175.7,
        "bmi": 21.7,
        "hba1c_percent": 4.9,
        "fasting_glucose_mg_dl": 86.6,
        "postprandial_glucose_mg_dl": 111.5,
        "cholesterol_mg_dl": 128.1,
        "hdl_mg_dl": 47.3,
        "ldl_mg_dl": 126.0,
        "triglycerides_mg_dl": 147.2,
        "blood_pressure_systolic_mm_hg": 91.2,
        "blood_pressure_diastolic_mm_hg": 67.2,
        "kidney_function_gfr": 110.5,
        "symptoms": "Frequent urination, Blurred vision, Thirst",
        "symptom_severity": "Moderate",
        "co_morbidities": "Obesity, Peripheral neuropathy",
        "new_feature": "Additional data example",  # Demonstrates unknown feature handling
    }

    # Prepare patient data and generalize query
    patient_data_str = prepare_patient_data(patient_info)
    logger.info(f"Patient data string: {patient_data_str}")

    # Save to artifacts for debugging or reuse
    artifacts_path = "artifacts/patient_data.txt"
    with open(artifacts_path, "w", encoding="utf-8") as file:
        file.write(patient_data_str)
    logger.info(f"Patient data saved to {artifacts_path}")

    # Generate query
    base_query = "What is the recommended treatment?"
    generalized_query = generalize_query(patient_data_str, base_query)
    logger.info(f"Generalized Query: {generalized_query}")
    print("Generalized Query:", generalized_query)

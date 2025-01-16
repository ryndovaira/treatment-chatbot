from typing import Dict, List


def prioritize_features(patient_data: Dict[str, any]) -> List[str]:
    """
    Prioritize and filter patient features for generalization based on importance.

    Args:
        patient_data (Dict[str, any]): Input patient data with all available fields.

    Returns:
        List[str]: A list of prioritized and formatted features for generalization.
    """
    # Critical features
    critical_features = [
        "symptoms",
        "symptom_severity",
        "co_morbidities",
        "age",
        "gender",
        "ethnicity",
    ]

    # Secondary features
    secondary_features = [
        "bmi",
        "blood_pressure_systolic_mm_hg",
        "blood_pressure_diastolic_mm_hg",
        "cholesterol_mg_dl",
        "triglycerides_mg_dl",
    ]

    # Contextual features
    contextual_features = [
        "pregnancy_status",
        "hba1c_percent",
        "fasting_glucose_mg_dl",
        "postprandial_glucose_mg_dl",
        "kidney_function_gfr",
    ]

    prioritized_features = []

    # Extract critical features
    for feature in critical_features:
        if feature in patient_data and patient_data[feature]:
            prioritized_features.append(f"{feature}: {patient_data[feature]}")

    # Extract secondary features
    for feature in secondary_features:
        if feature in patient_data and patient_data[feature]:
            prioritized_features.append(f"{feature}: {patient_data[feature]}")

    # Extract contextual features
    for feature in contextual_features:
        if feature in patient_data and patient_data[feature]:
            prioritized_features.append(f"{feature}: {patient_data[feature]}")

    return prioritized_features


def generalize_query(patient_data: Dict[str, any], base_query: str) -> str:
    """
    Generate a generalized query for public data retrieval.

    Args:
        patient_data (Dict[str, any]): Input patient data.
        base_query (str): Base query text (e.g., "What is the recommended treatment for Type 2 diabetes?").

    Returns:
        str: A generalized query with prioritized patient context.
    """
    features = prioritize_features(patient_data)
    generalized_features = "; ".join(features)
    return f"{generalized_features}. {base_query}"


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
    }
    base_query = "What is the recommended treatment?"
    generalized_query = generalize_query(patient_info, base_query)
    print("Generalized Query:", generalized_query)

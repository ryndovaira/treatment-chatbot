from src.rag_pipeline.query_generalizer import (
    prioritize_features,
    prepare_patient_data,
    generalize_query,
)


def test_prioritize_features_with_known_features():
    patient_data = {
        "age": 30,
        "gender": "Female",
        "ethnicity": "Asian",
        "bmi": 21.7,
        "symptoms": "Frequent urination",
    }
    expected = [
        "symptoms: Frequent urination",
        "age: 30",
        "gender: Female",
        "ethnicity: Asian",
        "bmi: 21.7",
    ]
    assert prioritize_features(patient_data) == expected


def test_prioritize_features_with_unknown_features(caplog):
    patient_data = {
        "age": 30,
        "gender": "Female",
        "ethnicity": "Asian",
        "unknown_feature": "Unknown value",
    }
    expected = [
        "age: 30",
        "gender: Female",
        "ethnicity: Asian",
        "unknown_feature: Unknown value",
    ]
    with caplog.at_level("WARNING"):
        assert prioritize_features(patient_data) == expected
        assert "Unknown feature 'unknown_feature' found. Including in the output." in caplog.text


def test_prepare_patient_data_with_valid_data():
    patient_data = {
        "age": 30,
        "gender": "Female",
        "ethnicity": "Asian",
    }
    expected = "age: 30; gender: Female; ethnicity: Asian"
    assert prepare_patient_data(patient_data) == expected


def test_generalize_query_with_patient_data():
    patient_data_str = "age: 30; gender: Female; ethnicity: Asian"
    base_query = "What is the recommended treatment?"
    expected = "age: 30; gender: Female; ethnicity: Asian. What is the recommended treatment?"
    assert generalize_query(patient_data_str, base_query) == expected


def test_generalize_query_without_patient_data():
    patient_data_str = ""
    base_query = "What is the recommended treatment?"
    assert generalize_query(patient_data_str, base_query) == base_query

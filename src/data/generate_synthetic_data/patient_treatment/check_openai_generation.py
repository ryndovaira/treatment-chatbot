from src.data.generate_synthetic_data.patient_treatment.patient_data_generator import (
    process_patient_data,
)

# Example patient records for testing with longitudinal data
test_records = [
    # Patient with 3 longitudinal records
    {
        "patient_id": 1,
        "age": 45,
        "gender": "Male",
        "ethnicity": "Caucasian",
        "bmi": 26.1,
        "records": [
            {"date": "2022-01-15", "hba1c_percent": 7.2, "blood_pressure": "130/80"},
            {"date": "2022-06-20", "hba1c_percent": 6.8, "blood_pressure": "125/78"},
            {"date": "2023-01-10", "hba1c_percent": 6.5, "blood_pressure": "120/75"},
        ],
    },
    # Patient with only one record
    {
        "patient_id": 2,
        "age": 32,
        "gender": "Female",
        "ethnicity": "African Am.",
        "bmi": 25.7,
        "records": [
            {"date": "2023-01-15", "hba1c_percent": 8.0, "blood_pressure": "140/90"},
        ],
    },
]

if __name__ == "__main__":
    print("Running in test mode with longitudinal data...")
    structured_data = process_patient_data(test_records, test_mode=True)
    print(structured_data)

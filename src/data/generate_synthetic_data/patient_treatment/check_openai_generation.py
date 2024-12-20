from src.data.generate_synthetic_data.patient_treatment.patient_data_generator import (
    process_patient_data,
)

# Example patient records for testing
test_records = [
    {"patient_id": 1, "age": 45, "gender": "Male", "ethnicity": "Caucasian", "bmi": 26.1},
    {"patient_id": 2, "age": 32, "gender": "Female", "ethnicity": "African Am.", "bmi": 25.7},
]

if __name__ == "__main__":
    print("Running in test mode...")
    structured_data = process_patient_data(test_records, test_mode=True)
    print(structured_data)

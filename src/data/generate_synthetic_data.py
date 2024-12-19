import random
from datetime import datetime, timedelta

import pandas as pd

from synthetic_data_config import (
    OUTPUT_FILE,
    GENDERS,
    ETHNICITIES,
    PREGNANCY_STATUS,
    SYMPTOMS,
    SYMPTOM_SEVERITY,
    CO_MORBIDITIES,
    LAB_RANGES,
    NUM_PATIENTS,
    RANDOM_SEED,
)

# Set random seed for reproducibility
random.seed(RANDOM_SEED)


# Generate Demographics
def generate_demographics(patient_id):
    age = random.randint(0, 100)
    gender = random.choice(GENDERS)
    ethnicity = random.choice(ETHNICITIES)
    pregnancy_status = None
    if gender == "Female" and 18 <= age <= 45:
        pregnancy_status = random.choice(PREGNANCY_STATUS)

    weight_kg = round(random.uniform(30, 180), 1)
    height_cm = round(random.uniform(140, 200), 1)
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)

    return {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "pregnancy_status": pregnancy_status,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "bmi": bmi,
    }


# Adjust lab result ranges based on demographics
def adjust_lab_ranges(age, gender, pregnancy_status):
    adjusted_ranges = LAB_RANGES.copy()
    if pregnancy_status == "Pregnant":
        adjusted_ranges.update(
            {
                "hba1c_percent": (4.5, 5.7),
                "triglycerides_mg_dl": (150, 300),
                "kidney_function_gfr": (90, 140),
            }
        )
    if age > 60:
        adjusted_ranges.update(
            {
                "kidney_function_gfr": (50, 90),
                "blood_pressure_systolic_mm_hg": (110, 140),
                "blood_pressure_diastolic_mm_hg": (70, 90),
            }
        )
    if gender == "Female" and age > 50:
        adjusted_ranges["hdl_mg_dl"] = (50, 90)
    elif gender == "Male":
        adjusted_ranges["hdl_mg_dl"] = (40, 60)
    return adjusted_ranges


# Generate Lab Results
def generate_lab_results(age, gender, pregnancy_status):
    lab_ranges = adjust_lab_ranges(age, gender, pregnancy_status)
    return {key: round(random.uniform(*value), 1) for key, value in lab_ranges.items()}


# Generate Symptoms and Co-Morbidities
def generate_conditions():
    symptoms = random.sample(SYMPTOMS, random.randint(1, 3))
    symptom_severity = random.choice(SYMPTOM_SEVERITY)
    co_morbidities = random.sample(CO_MORBIDITIES, random.randint(0, 2))
    return {
        "symptoms": ", ".join(symptoms),
        "symptom_severity": symptom_severity,
        "co_morbidities": ", ".join(co_morbidities),
    }


# Generate Longitudinal Data
def generate_longitudinal_data():
    record_date = datetime.today() - timedelta(days=random.randint(0, 365 * 5))
    record_order = random.randint(1, 5)
    return {
        "record_date": record_date.strftime("%Y-%m-%d"),
        "record_order": record_order,
    }


# Generate a single patient record
def generate_patient_data(patient_id):
    demographics = generate_demographics(patient_id)
    lab_results = generate_lab_results(
        demographics["age"], demographics["gender"], demographics["pregnancy_status"]
    )
    conditions = generate_conditions()
    longitudinal_data = generate_longitudinal_data()

    return {**demographics, **lab_results, **conditions, **longitudinal_data}


# Generate dataset
def generate_dataset():
    data = [generate_patient_data(patient_id) for patient_id in range(1, NUM_PATIENTS + 1)]
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to {OUTPUT_FILE}")


# Run the script
if __name__ == "__main__":
    generate_dataset()

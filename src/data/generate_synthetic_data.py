import csv
import random
from pathlib import Path

from synthetic_data_config import (
    AGE_RANGES,
    GENDERS,
    ETHNICITIES,
    PREGNANCY_STATUSES,
    WEIGHT_RANGE,
    HEIGHT_RANGE,
    NUM_PATIENTS,
)


def generate_patient_data(output_path):
    """
    Generate a CSV file with basic patient demographic data.

    Args:
        output_path (str or Path): The directory where the CSV file will be saved.
    """
    output_file = Path(output_path) / "basic_patient_data.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "patient_id",
                "age",
                "gender",
                "ethnicity",
                "pregnancy_status",
                "weight_kg",
                "height_cm",
                "bmi",
            ]
        )

        for patient_id in range(1, NUM_PATIENTS + 1):
            # Determine the age group
            age_group = random.choices(
                population=["children", "adults", "elderly"],
                weights=[0.1, 0.7, 0.2],  # Adjust proportions as needed
                k=1,
            )[0]
            age = random.randint(*AGE_RANGES[age_group])

            gender = random.choice(GENDERS)
            ethnicity = random.choice(ETHNICITIES)
            weight_kg = round(random.uniform(*WEIGHT_RANGE), 1)
            height_cm = round(random.uniform(*HEIGHT_RANGE), 1)
            bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)

            if gender == "Female" and 15 <= age <= 50:
                pregnancy = random.choice(PREGNANCY_STATUSES)
            else:
                pregnancy = None

            writer.writerow(
                [patient_id, age, gender, ethnicity, pregnancy, weight_kg, height_cm, bmi]
            )


# Generate data

# Paths are relative to the project structure
project_root = Path(__file__).resolve().parents[2]
data_raw_private_dir = project_root / "data" / "raw"

generate_patient_data(data_raw_private_dir)
print(f"Data saved to {data_raw_private_dir / 'basic_patient_data.csv'}")

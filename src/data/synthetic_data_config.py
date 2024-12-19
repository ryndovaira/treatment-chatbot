from pathlib import Path

# Path configuration
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "private"
OUTPUT_FILE = DATA_DIR / "basic_patient_data.csv"

# Configuration
NUM_PATIENTS = 1500
RANDOM_SEED = 42  # Seed for reproducibility

# Gender categories
GENDERS = ["Male", "Female"]

# Ethnicities
ETHNICITIES = ["Asian", "Caucasian", "African American", "Hispanic", "Other"]

# Pregnancy status options
PREGNANCY_STATUS = ["Pregnant", "Not Pregnant", None]

# Symptoms
SYMPTOMS = [
    "Fatigue",
    "Frequent urination",
    "Blurred vision",
    "Weight loss",
    "Thirst",
    "Slow-healing wounds",
    "Tingling or numbness in extremities",
    "Dry mouth",
]

# Symptom severity levels
SYMPTOM_SEVERITY = ["Mild", "Moderate", "Severe"]

# Co-morbidities
CO_MORBIDITIES = [
    "Hypertension",
    "Obesity",
    "Coronary artery disease",
    "Chronic kidney disease",
    "Peripheral neuropathy",
]

# Lab result ranges (defaults)
LAB_RANGES = {
    "hba1c_percent": (4.0, 6.5),
    "fasting_glucose_mg_dl": (70, 100),
    "postprandial_glucose_mg_dl": (90, 140),
    "cholesterol_mg_dl": (125, 200),
    "hdl_mg_dl": (40, 60),
    "ldl_mg_dl": (70, 130),
    "triglycerides_mg_dl": (50, 150),
    "blood_pressure_systolic_mm_hg": (90, 120),
    "blood_pressure_diastolic_mm_hg": (60, 80),
    "kidney_function_gfr": (90, 120),
}

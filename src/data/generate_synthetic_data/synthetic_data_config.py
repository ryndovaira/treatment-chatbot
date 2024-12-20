from pathlib import Path

# Path configuration
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "private"
OUTPUT_FILE = DATA_DIR / "basic_patient_data.csv"
LOG_FILE_G = OUTPUT_FILE.parent / "generation_log.txt"

# Additional directories for verification
VERIFICATION_DIR = DATA_DIR / "verification_results"
LOG_FILE_V = VERIFICATION_DIR / "synthetic_data_verificaion_log.txt"

PLOTS_DIR = VERIFICATION_DIR / "plots"
VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
NUM_PATIENTS = 750  # Number of synthetic patients to generate
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
    "hba1c_percent": (4.0, 6.5),  # % of hemoglobin with glucose
    "fasting_glucose_mg_dl": (70, 100),  # mg/dL
    "postprandial_glucose_mg_dl": (90, 140),  # mg/dL
    "cholesterol_mg_dl": (125, 200),  # mg/dL
    "hdl_mg_dl": (40, 60),  # mg/dL
    "ldl_mg_dl": (70, 130),  # mg/dL
    "triglycerides_mg_dl": (50, 150),  # mg/dL
    "blood_pressure_systolic_mm_hg": (90, 120),  # mmHg
    "blood_pressure_diastolic_mm_hg": (60, 80),  # mmHg
    "kidney_function_gfr": (90, 120),  # mL/min/1.73m^2
}


# Additional validations for lab ranges
def validate_lab_ranges():
    for key, (low, high) in LAB_RANGES.items():
        if low >= high:
            raise ValueError(f"Invalid range for {key}: ({low}, {high})")


# Run validations on import
validate_lab_ranges()

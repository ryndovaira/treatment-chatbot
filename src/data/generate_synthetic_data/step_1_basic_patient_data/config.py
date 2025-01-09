from src.config import LOG_DIR

NUM_PATIENTS = 100  # Number of synthetic patients to generate
GENDERS = ["Male", "Female"]
ETHNICITIES = ["Asian", "Caucasian", "African American", "Hispanic", "Other"]
PREGNANCY_STATUS = ["Pregnant", "Not Pregnant", None]
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
SYMPTOM_SEVERITY = ["Mild", "Moderate", "Severe"]
CO_MORBIDITIES = [
    "Hypertension",
    "Obesity",
    "Coronary artery disease",
    "Chronic kidney disease",
    "Peripheral neuropathy",
]
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


def validate_lab_ranges():
    for key, (low, high) in LAB_RANGES.items():
        if low >= high:
            raise ValueError(f"Invalid range for {key}: ({low}, {high})")


LOG_FILE_NAME = "step_1_basic_patient_data_generation"

VERIFICATION_DIR = LOG_DIR / "step_1_basic_patient_data_verification_results"
LOG_FILE_NAME_VERIFICATION = LOG_DIR / "step_1_basic_patient_data_verification"
PLOTS_DIR = VERIFICATION_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Run validations on import
validate_lab_ranges()

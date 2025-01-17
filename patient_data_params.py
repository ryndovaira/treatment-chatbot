def compute_bmi(weight_kg, height_cm):
    return round(weight_kg / ((height_cm / 100) ** 2), 1)


GENDERS = ["Male", "Female"]
AGE_RANGE = (0, 100)
WEIGHT_KG_RANGE = (30, 180)
HEIGHT_CM_RANGE = (140, 200)
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
    "fasting_glucose_mg_dl": (70.0, 100.0),  # mg/dL
    "postprandial_glucose_mg_dl": (90.0, 140.0),  # mg/dL
    "cholesterol_mg_dl": (125.0, 200.0),  # mg/dL
    "hdl_mg_dl": (40.0, 60.0),  # mg/dL
    "ldl_mg_dl": (70.0, 130.0),  # mg/dL
    "triglycerides_mg_dl": (50.0, 150.0),  # mg/dL
    "blood_pressure_systolic_mm_hg": (90.0, 120.0),  # mmHg
    "blood_pressure_diastolic_mm_hg": (60.0, 80.0),  # mmHg
    "kidney_function_gfr": (90.0, 120.0),  # mL/min/1.73m^2
}
LOG_FILE_NAME = "step_1_basic_patient_data_generation"

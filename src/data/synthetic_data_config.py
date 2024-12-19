# The number of patient records to generate. Default is 1500.
NUM_PATIENTS = 1500

# Age ranges for different demographic groups
AGE_RANGES = {"children": (1, 17), "adults": (18, 64), "elderly": (65, 90)}

# Gender options
GENDERS = ["Male", "Female"]

# Ethnic backgrounds
ETHNICITIES = ["Asian", "Caucasian", "African American", "Hispanic", "Other"]

# Pregnancy status (only applicable for females of childbearing age)
PREGNANCY_STATUSES = ["Pregnant", "Not Pregnant", None]

# Weight (kg) and height (cm) ranges
WEIGHT_RANGE = (40, 180)  # Updated to include extreme obesity cases
HEIGHT_RANGE = (140, 200)

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

# Severity levels for symptoms
SYMPTOM_SEVERITY = ["Mild", "Moderate", "Severe"]

# Lab results ranges (for meaningful distributions)
LAB_RANGES = {
    "hba1c_percent": (4.0, 14.0),
    "fasting_glucose_mg_dl": (70, 200),
    "postprandial_glucose_mg_dl": (90, 300),
    "cholesterol_mg_dl": (120, 300),
    "hdl_mg_dl": (40, 100),
    "ldl_mg_dl": (50, 200),
    "triglycerides_mg_dl": (50, 300),
    "blood_pressure_systolic_mm_hg": (90, 180),
    "blood_pressure_diastolic_mm_hg": (60, 120),
    "kidney_function_gfr": (10, 120),  # ml/min/1.73m²
}

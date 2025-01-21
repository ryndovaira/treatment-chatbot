from paths_and_constants import LOG_DIR
from src.patient_data_params import LAB_RANGES

NUM_PATIENTS = 100  # Number of synthetic patients to generate


def validate_lab_ranges():
    for key, (low, high) in LAB_RANGES.items():
        if low >= high:
            raise ValueError(f"Invalid range for {key}: ({low}, {high})")


VERIFICATION_DIR = LOG_DIR / "step_1_basic_patient_data_verification_results"
LOG_FILE_NAME_VERIFICATION = LOG_DIR / "step_1_basic_patient_data_verification"
PLOTS_DIR = VERIFICATION_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Run validations on import
validate_lab_ranges()

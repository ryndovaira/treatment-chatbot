from pathlib import Path

RANDOM_SEED = 42  # Seed for reproducibility

# Path configuration
DATA_RAW_PRIVATE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "private"
OUTPUT_FILE_BASIC_PATIENT_DATA = DATA_RAW_PRIVATE_DIR / "basic_patient_data.csv"
OUTPUT_FILE_TREATMENT_PATIENT_DATA = DATA_RAW_PRIVATE_DIR / "treatment_patient_data"

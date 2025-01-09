from pathlib import Path

TEST_MODE = False  # Set to True for testing or False for full data generation
TEST_LIMIT = 1  # Number of records to process in test mode

# Batch API configuration

# Dynamically determine the base directory of the async module
BASE_DIR = Path(__file__).resolve().parent

ASYNC_DIR = BASE_DIR / "openai_async"

# Artifacts Directory
BATCH_ARTIFACTS_DIR = ASYNC_DIR / "artifacts"
BATCH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
BATCH_INPUT_FILE = BATCH_ARTIFACTS_DIR / "batch_input"
BATCH_OUTPUT_FILE = BATCH_ARTIFACTS_DIR / "batch_output"
BATCH_ERROR_FILE = BATCH_ARTIFACTS_DIR / "batch_error"
BATCH_TRACKING_FILE = BATCH_ARTIFACTS_DIR / "batch_tracking.json"

PATIENT_DATA_AND_TREATMENT = BATCH_ARTIFACTS_DIR / "patient_data_and_treatment"
# Patient record index
PATIENT_RECORD_INDEX = 6

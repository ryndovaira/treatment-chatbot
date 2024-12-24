from pathlib import Path

# Dynamically determine the base directory of the async module
BASE_DIR = Path(__file__).resolve().parent

# Artifacts Directory
BATCH_ARTIFACTS_DIR = BASE_DIR / "artifacts"
BATCH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
BATCH_INPUT_FILE = BATCH_ARTIFACTS_DIR / "batch_input.jsonl"  # Input JSONL file for Batch API
BATCH_OUTPUT_FILE = (
    BATCH_ARTIFACTS_DIR / "batch_results.jsonl"
)  # Output JSONL file for Batch API results
BATCH_TRACKING_FILE = BATCH_ARTIFACTS_DIR / "batch_tracking.json"  # Metadata tracking file

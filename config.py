from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RAW_PUBLIC_DATA_DIR = BASE_DIR / "data" / "raw" / "public"
PROCESSED_PUBLIC_DATA_FILE = BASE_DIR / "data" / "processed" / "public_data_processed.json"
PROCESSED_PUBLIC_DATA_PICKLE = BASE_DIR / "data" / "processed" / "public_data_processed.pkl"
METADATA_FILE = RAW_PUBLIC_DATA_DIR / "metadata.json"

EMBEDDINGS_OUTPUT_DIR = BASE_DIR / "data" / "embeddings"
EMBEDDINGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_FAISS_DIR = BASE_DIR / "data" / "embeddings" / "public_faiss_index"
PUBLIC_FAISS_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_FAISS_INDEX_PATH = BASE_DIR / "data" / "embeddings" / "public_faiss_index" / "index.faiss"


PRIVATE_DATA_JSON = BASE_DIR / "data" / "raw" / "private" / "patient_data_and_treatment.json"
PRIVATE_FAISS_DIR = BASE_DIR / "data" / "embeddings" / "private_faiss_index"
PRIVATE_FAISS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "text-embedding-ada-002"
TOKEN_LIMIT_PER_MINUTE = 950_000

DEBUG = True

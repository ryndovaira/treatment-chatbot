import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from tqdm import tqdm

from src.config import OPENAI_API_KEY
from src.logging_config import setup_logger

# Logger setup
logger = setup_logger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
FAISS_INDEX_PATH = BASE_DIR / "data" / "embeddings" / "public_faiss_index" / "index.faiss"
METADATA_PATH = BASE_DIR / "data" / "embeddings" / "public_faiss_index" / "index.pkl"
PROCESSED_PUBLIC_DATA_PICKLE = BASE_DIR / "data" / "processed" / "public_data_processed.pkl"

# Debug flag
DEBUG = False  # Set to True for debug mode


def verify_faiss_against_pickle():
    """Verify FAISS index correctness based on the processed pickle file."""
    # Step 1: Load processed pickle data
    if not PROCESSED_PUBLIC_DATA_PICKLE.exists():
        logger.error(f"Processed data pickle file not found at {PROCESSED_PUBLIC_DATA_PICKLE}")
        return
    with PROCESSED_PUBLIC_DATA_PICKLE.open("rb") as f:
        processed_data = pickle.load(f)
    logger.info(f"Loaded {len(processed_data)} entries from {PROCESSED_PUBLIC_DATA_PICKLE}.")

    # Step 2: Handle debug mode
    if DEBUG:
        logger.warning("DEBUG mode is enabled. Verifying only the first 10 entries.")
        processed_data = processed_data[:10]

    # Step 3: Load FAISS index
    if not FAISS_INDEX_PATH.exists():
        logger.error(f"FAISS index file not found at {FAISS_INDEX_PATH}")
        return
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_PATH.parent), embeddings=embeddings, allow_dangerous_deserialization=True
    )
    logger.info(f"FAISS index loaded. Vector count: {vectorstore.index.ntotal}")

    # Step 4: Validate vector count
    if vectorstore.index.ntotal != len(processed_data):
        logger.error(
            f"FAISS vector count ({vectorstore.index.ntotal}) does not match processed data count ({len(processed_data)})."
        )
        return
    logger.info("FAISS vector count matches processed data count.")

    # Step 5: Validate metadata alignment
    unmatched_ids = []
    logger.info("Validating metadata alignment...")
    for doc in tqdm(processed_data, desc="Verifying metadata alignment", unit="docs"):
        doc_id = doc.metadata["id"]
        results = vectorstore.similarity_search(doc.page_content, k=1)
        if not results or results[0].metadata["id"] != doc_id:
            unmatched_ids.append(doc_id)

    if unmatched_ids:
        logger.error(f"Metadata mismatch for the following IDs: {unmatched_ids}")
    else:
        logger.info("All metadata entries in FAISS align with the processed data.")

    # Step 6: Test query
    test_query = "Type 2 diabetes prevention guidelines"
    logger.info(f"Testing query: {test_query}")
    results = vectorstore.similarity_search(test_query, k=3)
    logger.info("Top 3 query results:")
    for i, result in enumerate(results, 1):
        logger.info(f"Result {i}:")
        logger.info(f"  Page Content: {result.page_content[:200]}...")
        logger.info(f"  Metadata: {result.metadata}")


if __name__ == "__main__":
    verify_faiss_against_pickle()

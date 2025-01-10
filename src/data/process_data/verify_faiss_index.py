import pickle
from pathlib import Path

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import OPENAI_API_KEY
from src.logging_config import setup_logger

# Logger setup
logger = setup_logger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
FAISS_INDEX_PATH = BASE_DIR / "data" / "embeddings" / "public_faiss_index" / "index.faiss"
METADATA_PATH = BASE_DIR / "data" / "embeddings" / "public_faiss_index" / "index.pkl"
PROCESSED_PUBLIC_DATA_FILE = BASE_DIR / "data" / "processed" / "public_data_processed.json"


def verify_faiss_index():
    """Verify the FAISS index and metadata."""
    # Load metadata
    if not METADATA_PATH.exists():
        logger.error(f"Metadata file not found at {METADATA_PATH}")
        return
    with METADATA_PATH.open("rb") as f:
        metadata = pickle.load(f)
    logger.info(f"Loaded {len(metadata)} metadata entries from {METADATA_PATH}.")

    # Load FAISS index with explicit deserialization flag
    if not FAISS_INDEX_PATH.exists():
        logger.error(f"FAISS index file not found at {FAISS_INDEX_PATH}")
        return

    # Use OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_PATH.parent), embeddings=embeddings, allow_dangerous_deserialization=True
    )
    logger.info(f"FAISS index loaded. Vector count: {vectorstore.index.ntotal}")

    # Verify metadata consistency
    logger.info("Verifying metadata consistency...")
    for i, meta in enumerate(metadata[:5]):  # Check first 5 entries
        logger.info(f"Metadata {i + 1}: {meta}")

    # Compare metadata count with FAISS vector count
    if len(metadata) != vectorstore.index.ntotal:
        logger.warning(
            f"Metadata count ({len(metadata)}) does not match FAISS vector count ({vectorstore.index.ntotal})."
        )
    else:
        logger.info("Metadata count matches FAISS vector count.")

    # Test query
    test_query = "Type 2 diabetes prevention guidelines"
    logger.info(f"Testing query: {test_query}")
    results = vectorstore.similarity_search(test_query, k=3)

    # Log query results
    logger.info("Top 3 query results:")
    for i, result in enumerate(results, 1):
        logger.info(f"Result {i}:")
        logger.info(f"  Page Content: {result.page_content[:1000]}...")
        logger.info(f"  Metadata: {result.metadata}")


if __name__ == "__main__":
    verify_faiss_index()

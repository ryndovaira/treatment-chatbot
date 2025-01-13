import json

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from tqdm import tqdm

from src.config import OPENAI_API_KEY
from src.logging_config import setup_logger

logger = setup_logger(__name__)


# Constants
DEBUG = True  # Set True for testing smaller datasets


# Helper Functions
def prepare_private_documents(data):
    """
    Prepare documents for embedding by concatenating relevant fields.
    Each entry will be converted into a text format suitable for embedding.
    """
    documents = []
    for entry in data:
        patient_id = entry.get("patient_id", "unknown")
        text = (
            f"Patient ID: {patient_id}\n"
            f"Age: {entry.get('age')}, Gender: {entry.get('gender')}, Ethnicity: {entry.get('ethnicity')}\n"
            f"Symptoms: {entry.get('symptoms')} (Severity: {entry.get('symptom_severity')})\n"
            f"Co-morbidities: {entry.get('co_morbidities')}\n"
            f"Current Medications: {', '.join([med['name'] for med in entry.get('current_medications', [])])}\n"
            f"Treatment History: {entry.get('treatment_history', '')}\n"
            f"Lifestyle Recommendations: {entry.get('lifestyle_recommendations', '')}\n"
        )
        documents.append({"id": patient_id, "text": text, "metadata": entry})
    return documents


# Main Function
def embed_private_data():
    """Generate embeddings for private data and save to a FAISS index."""
    # Step 1: Load private data
    if not PRIVATE_DATA_JSON.exists():
        logger.error(f"Private data file not found at {PRIVATE_DATA_JSON}")
        return
    with PRIVATE_DATA_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        logger.warning("No private data entries found.")
        return

    # Step 2: Prepare documents
    logger.info(f"Preparing private data documents for embedding...")
    documents = prepare_private_documents(data)

    # Step 3: Debug mode
    if DEBUG:
        logger.warning("DEBUG mode is enabled. Processing only the first 10 documents.")
        documents = documents[:10]

    logger.info(f"Total documents to process: {len(documents)}")

    # Step 4: Generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    logger.info("Generating embeddings and creating FAISS index...")
    vectorstore = FAISS.from_texts(
        [doc["text"] for doc in tqdm(documents, desc="Embedding texts")],
        embedding=embeddings,
        metadatas=[doc["metadata"] for doc in documents],
    )

    # Step 5: Save FAISS index
    logger.info("Saving FAISS index and metadata...")
    vectorstore.save_local(str(PRIVATE_FAISS_DIR))
    logger.info(f"Private FAISS index saved to {PRIVATE_FAISS_DIR}")

    # Validation
    if vectorstore.index.ntotal != len(documents):
        logger.error(
            f"Mismatch: FAISS index count ({vectorstore.index.ntotal}) does not match document count ({len(documents)})."
        )
    else:
        logger.info("FAISS index successfully created and validated.")


if __name__ == "__main__":
    embed_private_data()

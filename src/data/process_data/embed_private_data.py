import json

from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from paths_and_constants import PRIVATE_DATA_JSON, DEBUG, PRIVATE_FAISS_DIR
from src.env_config import PRIVATE_EMBEDDING_MODEL
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def prepare_private_documents(data):
    """
    Prepare documents for embedding by concatenating relevant fields.
    Each entry will be converted into a text format suitable for embedding.
    """
    logger.info(f"Preparing private data documents for embedding...")

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
    logger.info(f"Total documents to process: {len(documents)}")

    if DEBUG:
        logger.warning("DEBUG mode is enabled. Processing only the first 10 documents.")
        documents = documents[:10]

    return documents


def load_private_data():
    if not PRIVATE_DATA_JSON.exists():
        logger.error(f"Private data file not found at {PRIVATE_DATA_JSON}")

    with PRIVATE_DATA_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        logger.error("No private data entries found.")

    return data


def generate_embeddings(documents):
    model = SentenceTransformer(PRIVATE_EMBEDDING_MODEL)
    logger.info(f"Generating embeddings using model: {PRIVATE_EMBEDDING_MODEL}")
    vectorstore = FAISS.from_texts(
        [doc["text"] for doc in tqdm(documents, desc="Embedding texts")],
        embedding=lambda texts: model.encode(texts, batch_size=32, show_progress_bar=True),
        metadatas=[doc["metadata"] for doc in documents],
    )
    return vectorstore


def save_faiss_index(vectorstore):
    logger.info("Saving FAISS index and metadata...")
    vectorstore.save_local(str(PRIVATE_FAISS_DIR))
    logger.info(f"Private FAISS index saved to {PRIVATE_FAISS_DIR}")


def validate_vector_count(documents, vectorstore):
    if vectorstore.index.ntotal != len(documents):
        logger.error(
            f"Mismatch: FAISS index count ({vectorstore.index.ntotal}) does not match document count ({len(documents)})."
        )
    else:
        logger.info("FAISS index successfully created and validated.")


def embed_private_data():
    """Generate embeddings for private data and save to a FAISS index."""
    data = load_private_data()

    documents = prepare_private_documents(data)

    vectorstore = generate_embeddings(documents)

    save_faiss_index(vectorstore)

    validate_vector_count(documents, vectorstore)


if __name__ == "__main__":
    embed_private_data()

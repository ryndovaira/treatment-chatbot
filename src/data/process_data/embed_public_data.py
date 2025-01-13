import pickle
from pathlib import Path

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from tiktoken import encoding_for_model
from tqdm import tqdm

from src.config import OPENAI_API_KEY
from src.logging_config import setup_logger

# Logger setup
logger = setup_logger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
PROCESSED_PUBLIC_DATA_PICKLE = BASE_DIR / "data" / "processed" / "public_data_processed.pkl"
EMBEDDINGS_OUTPUT_DIR = BASE_DIR / "data" / "embeddings"
EMBEDDINGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
MODEL_NAME = "text-embedding-ada-002"
TOKEN_LIMIT_PER_MINUTE = 950_000  # Slightly below OpenAI's 1,000,000 TPM limit for safety
DEBUG = True  # Set to True to enable debug mode


# Helper functions
def calculate_token_count(text: str, model: str = MODEL_NAME) -> int:
    """Calculate the token count of a given text using the specified model's tokenizer."""
    tokenizer = encoding_for_model(model)
    return len(tokenizer.encode(text))


def batch_documents(documents, token_limit):
    """Batch documents such that total tokens per batch do not exceed the token limit."""
    batches = []
    current_batch = []
    current_tokens = 0

    for doc in documents:
        token_count = calculate_token_count(doc.page_content)
        if current_tokens + token_count > token_limit:
            # Save current batch and reset
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(doc)
        current_tokens += token_count

    if current_batch:
        batches.append(current_batch)  # Add the final batch

    return batches


# Main function
def embed_public_data():
    """Embed processed public data and save to FAISS."""
    # Load processed data
    with PROCESSED_PUBLIC_DATA_PICKLE.open("rb") as f:
        documents = pickle.load(f)

    if not documents:
        logger.warning("No documents found for embedding.")
        return

    # Debug mode: Limit documents
    if DEBUG:
        logger.warning("Running in DEBUG mode: Processing only 10 documents.")
        documents = documents[:10]

    logger.info(f"Loaded {len(documents)} documents for embedding.")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Batch documents
    logger.info("Batching documents to respect token limits...")
    batches = batch_documents(documents, TOKEN_LIMIT_PER_MINUTE)
    logger.info(f"Created {len(batches)} batches.")

    # Process each batch
    vectorstore = None
    for batch in tqdm(batches, desc="Processing batch", unit="batch"):
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            new_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(new_store)

    # Save FAISS index
    output_file = EMBEDDINGS_OUTPUT_DIR / "public_faiss_index"
    vectorstore.save_local(str(output_file))
    logger.info(f"Embeddings saved to {output_file}.")


if __name__ == "__main__":
    embed_public_data()

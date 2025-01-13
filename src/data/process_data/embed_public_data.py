import pickle

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from tiktoken import encoding_for_model
from tqdm import tqdm

from src.config import OPENAI_API_KEY
from src.data.process_data.config import (
    MODEL_NAME,
    PROCESSED_PUBLIC_DATA_PICKLE,
    DEBUG,
    TOKEN_LIMIT_PER_MINUTE,
    PUBLIC_FAISS_DIR,
)
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def calculate_token_count(text: str, model: str = MODEL_NAME) -> int:
    """Calculate the token count of a given text using the specified model's tokenizer."""
    tokenizer = encoding_for_model(model)
    return len(tokenizer.encode(text))


def batch_documents(documents, token_limit):
    """Batch documents to respect token limits."""
    logger.info("Batching documents to respect token limits...")
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

    logger.info(f"Created {len(batches)} batches.")
    return batches


def load_processed_data():
    """Load processed public data for embedding."""
    with PROCESSED_PUBLIC_DATA_PICKLE.open("rb") as f:
        documents = pickle.load(f)

    if DEBUG:
        logger.warning("Running in DEBUG mode: Processing only 10 documents.")
        documents = documents[:10]

    if not documents:
        logger.error("No documents found for embedding.")
    else:
        logger.info(f"Loaded {len(documents)} documents for embedding.")

    return documents


def process_batches(batches, embeddings):
    """Process batches of documents and create a FAISS index."""
    vectorstore = None
    for batch in tqdm(batches, desc="Processing batch", unit="batch"):
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            new_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(new_store)
    return vectorstore


def save_faiss_index(vectorstore):
    """Save the FAISS index to disk."""
    vectorstore.save_local(str(PUBLIC_FAISS_DIR))
    logger.info(f"Embeddings saved to {PUBLIC_FAISS_DIR}.")


def embed_public_data():
    """Embed processed public data and save to FAISS."""
    documents = load_processed_data()

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    batches = batch_documents(documents, TOKEN_LIMIT_PER_MINUTE)

    vectorstore = process_batches(batches, embeddings)

    save_faiss_index(vectorstore)


if __name__ == "__main__":
    embed_public_data()

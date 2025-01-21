import json
import pickle
import unicodedata

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm

from paths_and_constants import (
    RAW_PUBLIC_DATA_DIR,
    PROCESSED_PUBLIC_DATA_FILE,
    PROCESSED_PUBLIC_DATA_PICKLE,
    METADATA_FILE,
)
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def generate_unique_id(source: str, file_name: str, page: int) -> str:
    """Generate a unique ID for each document chunk."""
    return f"{source}_{file_name}_{page}"


def save_processed_documents_as_json(all_documents, file_path):
    with PROCESSED_PUBLIC_DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump([doc.model_dump() for doc in all_documents], f, indent=4)


def save_processed_documents_as_pickle(all_documents, file_path):
    with PROCESSED_PUBLIC_DATA_PICKLE.open("wb") as f:
        pickle.dump(all_documents, f)


def save_processed_documents(all_documents):
    save_processed_documents_as_json(all_documents, PROCESSED_PUBLIC_DATA_FILE)
    save_processed_documents_as_pickle(all_documents, PROCESSED_PUBLIC_DATA_PICKLE)
    logger.info(
        f"Processed data saved to {PROCESSED_PUBLIC_DATA_FILE} and {PROCESSED_PUBLIC_DATA_PICKLE}."
    )


def load_metadata():
    with METADATA_FILE.open("r", encoding="utf-8") as f:
        metadata = {entry["file_name"]: entry for entry in json.load(f)}
    logger.info(f"Loaded metadata for {len(metadata)} PDF files.")
    return metadata


def split_documents(splitter, documents, metadata, pdf_file):
    all_docs = []
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        # Normalize text to handle special characters
        normalized_text = unicodedata.normalize("NFKD", chunk.page_content)
        chunk.page_content = normalized_text

        # Add metadata and generate unique ID
        file_metadata = metadata[pdf_file.name]
        chunk.metadata.update(file_metadata)
        chunk.metadata["page"] = chunk.metadata.get("page", 0)
        unique_id = generate_unique_id(
            source=file_metadata["source"],
            file_name=file_metadata["file_name"],
            page=chunk.metadata["page"],
        )
        chunk.metadata["id"] = unique_id
        all_docs.append(chunk)
    return all_docs


def preprocess_public_data():
    """Parse PDFs, attach metadata, and save processed data."""
    metadata = load_metadata()

    all_documents = []
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    pdfs = list(RAW_PUBLIC_DATA_DIR.glob("*.pdf"))
    logger.info(f"Found {len(list(pdfs))} PDF files in {RAW_PUBLIC_DATA_DIR}.")
    for pdf_file in tqdm(pdfs, desc="Processing PDFs", unit="file"):
        if pdf_file.name not in metadata:
            logger.warning(f"No metadata for {pdf_file.name}, skipping.")
            continue

        logger.info(f"Processing {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        try:
            documents = loader.load()  # Load the document as a list of `Document` objects
        except Exception as e:
            logger.error(f"Failed to load {pdf_file.name}: {e}")
            continue

        try:
            all_documents.extend(split_documents(splitter, documents, metadata, pdf_file))
        except Exception as e:
            logger.error(f"Error splitting {pdf_file.name}: {e}")

    save_processed_documents(all_documents)


if __name__ == "__main__":
    preprocess_public_data()

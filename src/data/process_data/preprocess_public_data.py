import json
from pathlib import Path

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from src.logging_config import setup_logger

# Logger setup
logger = setup_logger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
RAW_PUBLIC_DATA_DIR = BASE_DIR / "data" / "raw" / "public"
PROCESSED_PUBLIC_DATA_FILE = BASE_DIR / "data" / "processed" / "public_data_processed.json"
METADATA_FILE = RAW_PUBLIC_DATA_DIR / "metadata.json"


def preprocess_public_data():
    """Parse PDFs, attach metadata, and save processed data."""
    # Load metadata
    with METADATA_FILE.open("r", encoding="utf-8") as f:
        metadata = {entry["file_name"]: entry for entry in json.load(f)}

    all_documents = []
    splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    for pdf_file in RAW_PUBLIC_DATA_DIR.glob("*.pdf"):
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

        # Split the document into chunks
        try:
            chunks = splitter.split_documents(
                documents
            )  # Pass the list of `Document` objects directly
            for chunk in chunks:
                chunk.metadata.update(metadata[pdf_file.name])
                all_documents.append(chunk.model_dump())
        except Exception as e:
            logger.error(f"Error splitting {pdf_file.name}: {e}")

    # Save processed documents
    with PROCESSED_PUBLIC_DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_documents, f, indent=4)

    logger.info(f"Processed data saved to {PROCESSED_PUBLIC_DATA_FILE}.")


if __name__ == "__main__":
    preprocess_public_data()

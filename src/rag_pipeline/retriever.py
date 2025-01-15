from pathlib import Path

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import PUBLIC_FAISS_DIR, PRIVATE_FAISS_DIR, RETRIEVAL_TOP_N
from src.config import OPENAI_API_KEY
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def load_faiss_index(index_dir: Path, embeddings: OpenAIEmbeddings) -> FAISS:
    """
    Load a FAISS index from the specified directory using the given embeddings.

    Args:
        index_dir (Path): Path to the directory containing the FAISS index.
        embeddings (Embeddings): The embedding model used to create the FAISS index.

    Returns:
        FAISS: The loaded FAISS retriever.
    """
    if not index_dir.exists():
        logger.error(f"FAISS index directory not found: {index_dir}")
        raise FileNotFoundError(f"Directory {index_dir} does not exist.")

    logger.info(f"Loading FAISS index from {index_dir}...")
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)


def retrieve_context(query: str, public_retriever: FAISS, private_retriever: FAISS, top_n):
    """
    Retrieve context from public and private FAISS retrievers.

    Args:
        query (str): The user query.
        public_retriever (FAISS): The public FAISS retriever.
        private_retriever (FAISS): The private FAISS retriever.
        top_n (int): Number of top results to retrieve.

    Returns:
        dict: Retrieved contexts with metadata from both retrievers.
    """
    logger.info(f"Retrieving context for query: {query}")

    public_results = public_retriever.similarity_search(query, k=top_n)
    private_results = private_retriever.similarity_search(query, k=top_n)

    logger.info(f"Retrieved {len(public_results)} results from public data.")
    logger.info(f"Retrieved {len(private_results)} results from private data.")

    return {
        "public_results": [
            {"text": res.page_content, "metadata": res.metadata} for res in public_results
        ],
        "private_results": [
            {"text": res.page_content, "metadata": res.metadata} for res in private_results
        ],
    }


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Load the indexes
    public_retriever = load_faiss_index(PUBLIC_FAISS_DIR, embeddings)
    private_retriever = load_faiss_index(PRIVATE_FAISS_DIR, embeddings)

    # Example usage
    patient_info = {
        "age": 23,
        "gender": "Female",
        "ethnicity": "Caucasian",
        "weight_kg": 120,
        "height_cm": 160,
        "hba1c_percent": 4.5,
        "cholesterol_mg_dl": 160.8,
        "triglycerides_mg_dl": 57.4,
    }
    query = f"{patient_info}\nWhat is the recommended treatment for Type 2 diabetes?"
    results = retrieve_context(query, public_retriever, private_retriever, RETRIEVAL_TOP_N)
    print("Public Results:")
    for result in results["public_results"]:
        print(f"- {result['text']} (Source: {result['metadata']})")

    print("\nPrivate Results:")
    for result in results["private_results"]:
        print(f"- {result['text']} (Source: {result['metadata']})")

    output_path = Path("full_answer.txt")
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"Question: {query}\n\n")
        f.write("Public Results:\n")
        for result in results["public_results"]:
            f.write(f"- {result['text']} (Source: {result['metadata']})\n")
        f.write("\nPrivate Results:\n")
        for result in results["private_results"]:
            f.write(f"- {result['text']} (Source: {result['metadata']})\n")

    logger.info(f"Results saved to {output_path}")

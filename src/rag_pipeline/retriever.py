import os

from langchain_community.vectorstores import FAISS

from src.logging_config import setup_logger

logger = setup_logger(__name__)


def load_faiss_index(index_dir):
    """
    Load a FAISS index from the specified directory.

    Args:
        index_dir (str): Path to the directory containing the FAISS index.

    Returns:
        FAISS: The loaded FAISS retriever.
    """
    if not os.path.exists(index_dir):
        logger.error(f"FAISS index directory not found: {index_dir}")
        raise FileNotFoundError(f"Directory {index_dir} does not exist.")

    logger.info(f"Loading FAISS index from {index_dir}...")
    return FAISS.load_local(index_dir)


def retrieve_context(query, public_retriever, private_retriever, top_n=5):
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
    # Configuration paths for the indexes
    PUBLIC_FAISS_DIR = "data/embeddings/public_faiss_index"
    PRIVATE_FAISS_DIR = "data/embeddings/private_faiss_index"

    # Load the indexes
    public_retriever = load_faiss_index(PUBLIC_FAISS_DIR)
    private_retriever = load_faiss_index(PRIVATE_FAISS_DIR)

    # Example usage
    query = "What is the recommended treatment for Type 2 diabetes?"
    results = retrieve_context(query, public_retriever, private_retriever)
    print("Public Results:")
    for result in results["public_results"]:
        print(f"- {result['text']} (Source: {result['metadata']})")

    print("\nPrivate Results:")
    for result in results["private_results"]:
        print(f"- {result['text']} (Source: {result['metadata']})")

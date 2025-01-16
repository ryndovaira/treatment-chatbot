import json
from pathlib import Path
from typing import Dict, Any, List

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import PUBLIC_FAISS_DIR, PRIVATE_FAISS_DIR, RETRIEVAL_TOP_N
from query_generalizer import generalize_query
from src.config import OPENAI_API_KEY
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def load_faiss_index(index_dir: Path, embeddings: OpenAIEmbeddings) -> FAISS:
    """
    Load a FAISS index from the specified directory using the given embeddings.

    Args:
        index_dir (Path): Path to the directory containing the FAISS index.
        embeddings (OpenAIEmbeddings): The embedding model used to create the FAISS index.

    Returns:
        FAISS: The loaded FAISS retriever.
    """
    if not index_dir.exists():
        logger.error(f"FAISS index directory not found: {index_dir}")
        raise FileNotFoundError(f"Directory {index_dir} does not exist.")

    logger.info(f"Loading FAISS index from {index_dir}...")
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)


def retrieve_context(
    query: str, public_retriever: FAISS, private_retriever: FAISS, top_n: int
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve context from public and private FAISS retrievers.

    Args:
        query (str): The generalized query.
        public_retriever (FAISS): Public FAISS retriever.
        private_retriever (FAISS): Private FAISS retriever.
        top_n (int): Number of top results to retrieve.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Retrieved contexts with metadata from both retrievers.
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

    # Load FAISS indexes
    public_retriever = load_faiss_index(PUBLIC_FAISS_DIR, embeddings)
    private_retriever = load_faiss_index(PRIVATE_FAISS_DIR, embeddings)

    # Example usage
    patient_info = {
        "age": 30,
        "gender": "Female",
        "ethnicity": "Asian",
        "pregnancy_status": "None",
        "weight_kg": 66.9,
        "height_cm": 175.7,
        "bmi": 21.7,
        "hba1c_percent": 4.9,
        "fasting_glucose_mg_dl": 86.6,
        "postprandial_glucose_mg_dl": 111.5,
        "cholesterol_mg_dl": 128.1,
        "hdl_mg_dl": 47.3,
        "ldl_mg_dl": 126.0,
        "triglycerides_mg_dl": 147.2,
        "blood_pressure_systolic_mm_hg": 91.2,
        "blood_pressure_diastolic_mm_hg": 67.2,
        "kidney_function_gfr": 110.5,
        "symptoms": "Frequent urination, Blurred vision, Thirst",
        "symptom_severity": "Moderate",
        "co_morbidities": "Obesity, Peripheral neuropathy",
    }
    base_query = "What is the recommended treatment?"
    generalized_query = generalize_query(patient_info, base_query)

    results = retrieve_context(
        generalized_query, public_retriever, private_retriever, RETRIEVAL_TOP_N
    )

    print("Public Results:")
    for result in results["public_results"]:
        print(f"- {result['text']} (Source: {result['metadata']})")

    print("\nPrivate Results:")
    for result in results["private_results"]:
        print(f"- {result['text']} (Source: {result['metadata']})")

    with open("artifacts/public_docs.json", "w") as f:
        f.write(json.dumps(results["public_results"], indent=4))
    with open("artifacts/private_docs.json", "w") as f:
        f.write(json.dumps(results["private_results"], indent=4))

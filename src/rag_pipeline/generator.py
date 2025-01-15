from pathlib import Path
from typing import Dict, Any

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from config import PUBLIC_FAISS_DIR, PRIVATE_FAISS_DIR, RAG_MODEL_NAME
from retriever import load_faiss_index
from src.config import OPENAI_API_KEY
from src.logging_config import setup_logger

# Setup logger
logger = setup_logger(__name__)


def create_rag_chain(retriever: FAISS) -> RetrievalQA:
    """
    Create a Retrieval-Augmented Generation (RAG) chain.

    Args:
        retriever (FAISS): The retriever for fetching relevant documents.

    Returns:
        RetrievalQA: A RetrievalQA chain combining retrieval and generation.
    """
    llm = ChatOpenAI(model_name=RAG_MODEL_NAME, openai_api_key=OPENAI_API_KEY)
    logger.info(f"Initialized ChatOpenAI model: {RAG_MODEL_NAME}")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever.as_retriever(),
        return_source_documents=True,
    )


def generate_answer(
    query: str, public_chain: RetrievalQA, private_chain: RetrievalQA
) -> Dict[str, Any]:
    """
    Generate answers for a query using both public and private RAG chains.

    Args:
        query (str): The user query.
        public_chain (RetrievalQA): The public RAG chain.
        private_chain (RetrievalQA): The private RAG chain.

    Returns:
        Dict[str, Any]: A dictionary containing answers and source documents from both chains.
    """
    logger.info(f"Generating answers for query: {query}")

    public_response = public_chain({"query": query})
    private_response = private_chain({"query": query})

    return {
        "public_answer": public_response.get("result", ""),
        "public_sources": public_response.get("source_documents", []),
        "private_answer": private_response.get("result", ""),
        "private_sources": private_response.get("source_documents", []),
    }


def save_results(query: str, results: Dict[str, Any], output_path: Path) -> None:
    """
    Save the query results to a file.

    Args:
        query (str): The user query.
        results (Dict[str, Any]): The results including answers and sources.
        output_path (Path): Path to save the results.
    """
    logger.info(f"Saving results to {output_path}...")

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\n")
        f.write("Public Answer:\n")
        f.write(f"{results['public_answer']}\n\n")
        f.write("Public Sources:\n")
        for doc in results["public_sources"]:
            f.write(f"- {doc.page_content[:200]}... (Source: {doc.metadata})\n")
        f.write("\nPrivate Answer:\n")
        f.write(f"{results['private_answer']}\n\n")
        f.write("Private Sources:\n")
        for doc in results["private_sources"]:
            f.write(f"- {doc.page_content[:200]}... (Source: {doc.metadata})\n")

    logger.info(f"Results successfully saved to {output_path}")


if __name__ == "__main__":
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Load FAISS retrievers
    public_retriever = load_faiss_index(PUBLIC_FAISS_DIR, embeddings)
    private_retriever = load_faiss_index(PRIVATE_FAISS_DIR, embeddings)

    # Create RAG chains
    public_chain = create_rag_chain(public_retriever)
    private_chain = create_rag_chain(private_retriever)

    # Example query
    query = "What is the recommended treatment for Type 2 diabetes?"

    # Generate answers
    results = generate_answer(query, public_chain, private_chain)

    # Save results
    output_file = Path("full_answer_with_sources.txt")
    save_results(query, results, output_file)

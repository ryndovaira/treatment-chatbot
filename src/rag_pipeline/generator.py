from pathlib import Path
from typing import Dict, Any, List

from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_openai.chat_models import ChatOpenAI

from paths_and_constants import RAG_MODEL_NAME
from src.config import OPENAI_API_KEY
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def dict_to_human_readable(info: Dict[str, Any]) -> str:
    """
    Convert a dictionary into a human-readable string for query augmentation.

    Args:
        info (Dict[str, Any]): The patient information.

    Returns:
        str: Human-readable representation of the dictionary.
    """
    return ", ".join(f"{key}: {value}" for key, value in info.items())


def create_rag_chain(retriever: Any) -> RetrievalQA:
    """
    Create a Retrieval-Augmented Generation (RAG) chain.

    Args:
        retriever (Any): The retriever for fetching relevant documents.

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


def summarize_retrieved_documents(documents: List[Document], llm: ChatOpenAI) -> str:
    """
    Summarize the retrieved documents using an LLM.

    Args:
        documents (List[Document]): A list of retrieved documents.
        llm (ChatOpenAI): The language model used for summarization.

    Returns:
        str: The summarized text.
    """
    logger.info(f"Summarizing {len(documents)} documents.")
    combined_text = "\n".join([doc.page_content for doc in documents])
    prompt = f"Summarize the following information:\n\n{combined_text}"
    response = llm.predict(prompt)
    logger.info("Generated summary using LLM.")
    return response


def generate_summary_answer(
    query: str, private_context: str, context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a summary answer based on retrieved contexts and private information.

    Args:
        query (str): The user query.
        private_context (str): The private information added to the query.
        context (Dict[str, Any]): Retrieved public and private documents.

    Returns:
        Dict[str, Any]: Summarized answer and source documents.
    """
    augmented_query = f"{private_context}. {query}"
    llm = ChatOpenAI(model_name=RAG_MODEL_NAME, openai_api_key=OPENAI_API_KEY)

    public_summary = summarize_retrieved_documents(context["public_sources"], llm)
    private_summary = summarize_retrieved_documents(context["private_sources"], llm)

    return {
        "query": augmented_query,
        "public_summary": public_summary,
        "private_summary": private_summary,
        "public_sources": context["public_results"],
        "private_sources": context["private_results"],
    }


if __name__ == "__main__":
    from retriever import load_faiss_index, retrieve_context
    from langchain_community.embeddings import OpenAIEmbeddings
    from paths_and_constants import PUBLIC_FAISS_DIR, PRIVATE_FAISS_DIR

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Load retrievers
    public_retriever = load_faiss_index(PUBLIC_FAISS_DIR, embeddings)
    private_retriever = load_faiss_index(PRIVATE_FAISS_DIR, embeddings)

    # Example patient information
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
    private_context = dict_to_human_readable(patient_info)

    # User query
    query = "What is the recommended treatment for Type 2 diabetes?"

    # Retrieve documents
    context = retrieve_context(
        query, public_retriever, private_retriever, top_n_public=5, top_n_private=3
    )

    # Generate summary answer
    results = generate_summary_answer(query, private_context, context)

    # Save results
    output_path = Path("summary_with_sources.json")
    with output_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {output_path}")

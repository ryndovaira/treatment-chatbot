import json
from typing import Dict, List, Any

from langchain_openai.chat_models import ChatOpenAI

from config import RAG_MODEL_NAME
from src.config import OPENAI_API_KEY
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def generate_summary(llm: ChatOpenAI, documents: List[Dict[str, Any]], source_type: str) -> str:
    """
    Generate a summary for a set of documents using a language model.

    Args:
        llm (ChatOpenAI): The language model for summarization.
        documents (List[Dict[str, Any]]): The documents to summarize.
        source_type (str): Type of documents (public or private).

    Returns:
        str: A summary of the provided documents.
    """
    if not documents:
        logger.warning(f"No documents provided for {source_type} summarization.")
        return f"No {source_type} data to summarize."

    logger.info(f"Generating summary for {source_type} documents...")
    context = "\n\n".join([doc["text"] for doc in documents])
    summary_prompt = (
        f"Summarize the following {source_type} documents and provide key insights:\n\n{context}"
    )

    summary = llm.invoke(summary_prompt)
    logger.info(f"Summary generated for {source_type} documents.")
    return summary


def generate_combined_summary(public_summary: str, private_summary: str) -> str:
    """
    Generate a combined summary using the public and private summaries.

    Args:
        public_summary (str): Summary of public documents.
        private_summary (str): Summary of private documents.

    Returns:
        str: Combined summary integrating both public and private insights.
    """
    logger.info("Generating combined summary...")
    combined_prompt = (
        f"Integrate the following summaries into a unified insight:\n\n"
        f"Public Summary:\n{public_summary}\n\nPrivate Summary:\n{private_summary}\n\n"
        f"Provide actionable recommendations and key takeaways."
    )

    llm = ChatOpenAI(model_name=RAG_MODEL_NAME, openai_api_key=OPENAI_API_KEY)
    combined_summary = llm.predict(combined_prompt)
    logger.info("Combined summary generated.")
    return combined_summary


if __name__ == "__main__":
    # Example usage
    llm = ChatOpenAI(model_name=RAG_MODEL_NAME, openai_api_key=OPENAI_API_KEY)

    with open("artifacts/public_docs.json", "r") as f:
        public_docs = json.load(f)
        logger.info(f"Loaded {len(public_docs)} public documents.")
        logger.debug(f"Example public document: {public_docs[0]}")

    with open("artifacts/private_docs.json", "r") as f:
        private_docs = json.load(f)
        logger.info(f"Loaded {len(private_docs)} private documents.")
        logger.debug(f"Example private document: {private_docs[0]}")

    public_summary = generate_summary(llm, public_docs, source_type="public")
    logger.info(f"Public Summary:\n{public_summary}\n")

    private_summary = generate_summary(llm, private_docs, source_type="private")
    logger.info(f"Private Summary:\n{private_summary}\n")

    combined_summary = generate_combined_summary(public_summary, private_summary)
    logger.info(f"Combined Summary:\n{combined_summary}")

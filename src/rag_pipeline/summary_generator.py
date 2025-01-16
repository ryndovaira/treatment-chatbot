import json
from typing import Dict, List, Any

from langchain_openai.chat_models import ChatOpenAI

from config import RAG_MODEL_NAME
from src.config import OPENAI_API_KEY
from src.logging_config import setup_logger

logger = setup_logger(__name__)


def generate_summary(
    llm: ChatOpenAI, documents: List[Dict[str, Any]], source_type: str, target_case: str
) -> str:
    """
    Generate a concise, actionable summary for a set of documents using a language model.

    Args:
        llm (ChatOpenAI): The language model for summarization.
        documents (List[Dict[str, Any]]): The documents to summarize.
        source_type (str): Type of documents (e.g., "private" or "public").
        target_case (str): Target case description to guide summarization.

    Returns:
        str: A refined, concise summary of the provided documents.
    """
    if not documents:
        logger.warning(f"No documents provided for {source_type} summarization.")
        return f"No data to summarize for {source_type}."

    logger.info(f"Generating summary for {source_type} documents...")

    if source_type == "private":
        prompt = (
            f"Target case details:\n{target_case}\n\n"
            f"Summarize the following patient histories. Do not repeat the target case details."
            f"Identify similarities and differences with the target case, "
            f"highlight treatment outcomes, and discuss why certain treatments worked or failed:\n\n"
            f"{[doc['text'] for doc in documents]}\n\n"
            f"Provide a structured summary with key patterns, actionable insights, and specific recommendations for the target case."
        )
    else:  # For public data
        prompt = (
            f"Target case details:\n{target_case}\n\n"
            f"Summarize the following article samples. Do not repeat the target case details."
            f"Extract actionable recommendations, highlight population-specific guidelines "
            f"or treatments relevant to the target case, and discuss recent research breakthroughs:\n\n"
            f"{[doc['text'] for doc in documents]}\n\n"
            f"Provide a concise, actionable summary with insights tailored to the target case."
        )

    summary = llm.invoke(prompt)
    summary_content = summary.content
    logger.info(f"Summary generated for {len(documents)} documents.")
    return summary_content


def generate_combined_summary(public_summary: str, private_summary: str, target_case: str) -> str:
    """
    Generate a consolidated actionable summary combining public and private insights.

    Args:
        public_summary (str): Summary of public documents.
        private_summary (str): Summary of private documents.
        target_case (str): Target case description to guide consolidation.

    Returns:
        str: Combined actionable summary.
    """
    logger.info("Generating actionable combined summary...")
    prompt = (
        f"Target case details:\n{target_case}\n\n"
        f"Public Summary:\n{public_summary}\n\n"
        f"Private Summary:\n{private_summary}\n\n"
        f"Integrate these insights to provide a unified treatment plan for the target case. "
        f"Do not repeat the target case details."
        f"Highlight actionable steps, considerations, and evidence supporting the recommendations. "
        f"Structure the output as follows:\n"
        f"1. Key Findings from Public Data\n"
        f"2. Key Findings from Private Data\n"
        f"3. Integrated Recommendations\n"
        f"4. Additional Considerations or Unresolved Questions."
    )
    llm = ChatOpenAI(model_name=RAG_MODEL_NAME, openai_api_key=OPENAI_API_KEY)
    combined_summary = llm.invoke(prompt)
    combined_summary_content = combined_summary.content
    logger.info("Combined summary generated.")
    return combined_summary_content


if __name__ == "__main__":
    # Example usage
    llm = ChatOpenAI(model_name=RAG_MODEL_NAME, openai_api_key=OPENAI_API_KEY)

    # Load public documents
    with open("artifacts/public_docs.json", "r", encoding="utf-8") as f:
        public_docs = json.load(f)
        logger.info(f"Loaded {len(public_docs)} public documents.")

    # Load private documents
    with open("artifacts/private_docs.json", "r", encoding="utf-8") as f:
        private_docs = json.load(f)
        logger.info(f"Loaded {len(private_docs)} private documents.")

    # Example target case
    target_case = "symptoms: Frequent urination, Blurred vision, Thirst; symptom_severity: Moderate; co_morbidities: Obesity, Peripheral neuropathy; age: 30; gender: Female; ethnicity: Asian; bmi: 21.7; blood_pressure_systolic_mm_hg: 91.2; blood_pressure_diastolic_mm_hg: 67.2; cholesterol_mg_dl: 128.1; triglycerides_mg_dl: 147.2; pregnancy_status: None; hba1c_percent: 4.9; fasting_glucose_mg_dl: 86.6; postprandial_glucose_mg_dl: 111.5; kidney_function_gfr: 110.5"

    # Generate summaries
    public_summary = generate_summary(
        llm, public_docs, source_type="public", target_case=target_case
    )
    logger.info(f"Public Summary:\n{public_summary}\n")

    private_summary = generate_summary(
        llm, private_docs, source_type="private", target_case=target_case
    )
    logger.info(f"Private Summary:\n{private_summary}\n")

    # Generate combined summary
    combined_summary = generate_combined_summary(public_summary, private_summary, target_case)
    logger.info(f"Combined Summary:\n{combined_summary}")

    # save results to file in artifacts folder
    with open("artifacts/combined_summary.md", "w", encoding="utf-8") as file:
        file.write(combined_summary)
    with open("artifacts/public_summary.md", "w", encoding="utf-8") as file:
        file.write(public_summary)
    with open("artifacts/private_summary.md", "w", encoding="utf-8") as file:
        file.write(private_summary)

from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import BaseModel

from paths_and_constants import PUBLIC_FAISS_DIR, PRIVATE_FAISS_DIR, RAG_MODEL_NAME
from query_generalizer import prepare_patient_data, generalize_query
from query_logger import log_query
from retriever import load_faiss_index, retrieve_context
from src.env_config import OPENAI_API_KEY
from src.logging_config import setup_logger
from summary_generator import generate_summary, generate_combined_summary

# Initialize FastAPI application
app = FastAPI()

# Logger setup
logger = setup_logger(__name__)

# Embedding model initialization
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load retrievers
public_retriever = load_faiss_index(PUBLIC_FAISS_DIR, embeddings)
private_retriever = load_faiss_index(PRIVATE_FAISS_DIR, embeddings)
llm = ChatOpenAI(model_name=RAG_MODEL_NAME, openai_api_key=OPENAI_API_KEY)


class QueryRequest(BaseModel):
    """
    Request model for the `/query` endpoint.
    """

    patient_data: Dict[str, Any]
    base_query: str


class QueryResponse(BaseModel):
    """
    Response model for the `/query` endpoint.
    """

    public_summary: str
    private_summary: str
    combined_summary: str
    public_sources: List[Dict[str, Any]]
    private_sources: List[Dict[str, Any]]


@app.post("/query", response_model=QueryResponse)
def query_rag_pipeline(request: QueryRequest):
    """
    Query the RAG pipeline and return the results.

    Args:
        request (QueryRequest): The query request containing the patient's data and query.

    Returns:
        QueryResponse: The results containing summaries and source documents.
    """
    try:
        # Prepare patient data
        patient_data_str = prepare_patient_data(request.patient_data)
        generalized_query = generalize_query(patient_data_str, request.base_query)
        logger.info(f"Generalized Query: {generalized_query}")

        # Retrieve documents
        retrieved_context = retrieve_context(
            generalized_query, public_retriever, private_retriever, top_n=5
        )

        # Generate summaries
        public_summary = generate_summary(
            llm=llm,
            documents=retrieved_context["public_results"],
            source_type="public",
            target_case=patient_data_str,
        )
        private_summary = generate_summary(
            llm=llm,
            documents=retrieved_context["private_results"],
            source_type="private",
            target_case=patient_data_str,
        )

        # Generate combined summary
        combined_summary = generate_combined_summary(
            public_summary=public_summary,
            private_summary=private_summary,
            target_case=patient_data_str,
        )

        # Log the query and results
        results = {
            "public_summary": public_summary,
            "private_summary": private_summary,
            "combined_summary": combined_summary,
            "public_sources": retrieved_context["public_results"],
            "private_sources": retrieved_context["private_results"],
        }
        log_query(generalized_query, results)

        return results

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the query.")


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)

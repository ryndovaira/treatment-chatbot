from typing import List, Dict, Any

from fastapi import FastAPI
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import BaseModel

from config import PUBLIC_FAISS_DIR, PRIVATE_FAISS_DIR
from generator import create_rag_chain, generate_answer
from query_logger import log_query
from retriever import load_faiss_index
from src.config import OPENAI_API_KEY
from src.logging_config import setup_logger

# Initialize FastAPI application
app = FastAPI()

# Logger setup
logger = setup_logger(__name__)

# Embedding model initialization
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load retrievers
public_retriever = load_faiss_index(PUBLIC_FAISS_DIR, embeddings)
private_retriever = load_faiss_index(PRIVATE_FAISS_DIR, embeddings)

# Create RAG chains
public_chain = create_rag_chain(public_retriever)
private_chain = create_rag_chain(private_retriever)


class QueryRequest(BaseModel):
    """
    Request model for the `/query` endpoint.
    """

    query: str


class SourceDocument(BaseModel):
    """
    Representation of a single source document in the response.
    """

    text: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    """
    Response model for the `/query` endpoint.
    """

    public_answer: str
    public_sources: List[SourceDocument]
    private_answer: str
    private_sources: List[SourceDocument]


@app.get("/")
def welcome_message() -> Dict[str, str]:
    """
    Root endpoint to verify API status.

    Returns:
        Dict[str, str]: Welcome message.
    """
    return {"message": "Welcome to the Diabetes Treatment RAG API"}


#
# @app.post("/query", response_model=QueryResponse)
# def query_rag_pipeline(request: QueryRequest) -> QueryResponse:
#     """
#     Handles user queries and returns responses from both public and private RAG pipelines.
#
#     Args:
#         request (QueryRequest): The user query.
#
#     Returns:
#         QueryResponse: Answers and source documents from both pipelines.
#
#     Raises:
#         HTTPException: If an error occurs during query processing.
#     """
#     logger.info(f"Received query: {request.query}")
#     try:
#         # Generate answers using the RAG pipeline
#         results = generate_answer(request.query, public_chain, private_chain)
#         return QueryResponse(
#             public_answer=results["public_answer"],
#             public_sources=[
#                 SourceDocument(text=doc.page_content, metadata=doc.metadata)
#                 for doc in results["public_sources"]
#             ],
#             private_answer=results["private_answer"],
#             private_sources=[
#                 SourceDocument(text=doc.page_content, metadata=doc.metadata)
#                 for doc in results["private_sources"]
#             ],
#         )
#     except Exception as e:
#         logger.error(f"Error processing query: {e}")
#         raise HTTPException(status_code=500, detail="An error occurred while processing the query.")


@app.post("/query")
def query_rag_pipeline(request: QueryRequest):
    """
    Query the RAG pipeline and return the results.

    Args:
        request (QueryRequest): The query request containing the user's query.

    Returns:
        Dict: The results containing answers and sources.
    """
    results = generate_answer(request.query, public_chain, private_chain)

    # Log the query and results
    log_query(request.query, results)

    return results


if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)

from pathlib import Path

import requests
import streamlit as st

# Configuration
API_URL = "http://localhost:8000/query"
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)


# Utility Functions
def send_query_to_api(query: str):
    """
    Send the user query to the FastAPI backend.

    Args:
        query (str): The user query.

    Returns:
        dict: API response or None if an error occurred.
    """
    try:
        response = requests.post(API_URL, json={"query": query})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None


def display_results(results: dict):
    """
    Display the API response results on the Streamlit frontend.

    Args:
        results (dict): The API response containing answers and sources.
    """
    st.subheader("Results")

    # Public Answer
    st.markdown("### Public Answer")
    st.write(results.get("public_answer", "No public answer available."))

    st.markdown("### Public Sources")
    public_sources = results.get("public_sources", [])
    for source in public_sources:
        st.markdown(f"- **Excerpt**: {source['text'][:200]}...")
        st.markdown(f"  **Metadata**: {source['metadata']}")

    # Private Answer
    st.markdown("### Private Answer")
    st.write(results.get("private_answer", "No private answer available."))

    st.markdown("### Private Sources")
    private_sources = results.get("private_sources", [])
    for source in private_sources:
        st.markdown(f"- **Excerpt**: {source['text'][:200]}...")
        st.markdown(f"  **Metadata**: {source['metadata']}")


# Main Streamlit App
def main():
    st.set_page_config(page_title="Diabetes Treatment Chatbot", layout="wide")
    st.title("Diabetes Treatment Chatbot")
    st.markdown(
        """
        **Retrieve evidence-based diabetes treatment recommendations**. Enter your query below to get responses from
        public guidelines and private clinical data.
        """
    )

    # Input Section
    query = st.text_area("Enter your query", height=150)

    if st.button("Submit Query"):
        if not query.strip():
            st.error("Query cannot be empty.")
        else:
            with st.spinner("Fetching responses..."):
                response = send_query_to_api(query)
                if response:
                    display_results(response)
                else:
                    st.error("Failed to fetch responses. Please try again.")


if __name__ == "__main__":
    main()

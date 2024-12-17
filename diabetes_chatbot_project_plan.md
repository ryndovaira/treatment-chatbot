# Project Plan: Diabetes Treatment Support Chatbot with RAG Pipeline

## 1. Project Scope and Goals

- **Objective**: Develop a **Retrieval-Augmented Generation (RAG)** chatbot for evidence-based diabetes treatment
  recommendations.
- **Public Data**: Clinical guidelines, articles, and mock patient data for lab results, demographics, and symptoms.
- **Simulated Private Data**: Patient records, lab tests, and individual case studies.
- **Models**:
    - **LLaMA** (configurable for simulated private data).
    - **ChatGPT API** (configurable for public data).
- **Deployment**: Configurable deployment on-premise or cloud (e.g., Hetzner, AWS, Google Cloud, Azure).
- **Development**: Local machine (Windows + WSL Ubuntu) with GPU support.

---

## 2. Project Phases

### Phase 1: Environment Setup

- **Tools**:
    - WSL Ubuntu for GPU utilization.
    - NVIDIA drivers and CUDA installed on WSL.
    - **Dependencies**:
        - Python: `transformers`, `bitsandbytes`, `pandas`, `numpy`, `torch`, `faiss-cpu`, `langchain`,
          `pinecone-client`.
        - Docker for containerization.
        - FastAPI for the backend API.
        - Streamlit for frontend testing.
    - Local GPU check: Verify GPU compatibility with **PyTorch** and **CUDA**.
- **Questions**:
    - What GPU model does your machine have (for accurate CUDA setup)?

---

### Phase 2: Data Collection and Preprocessing

- **Public Data**:
    - Diabetes guidelines: ADA Standards of Care, clinical publications.
    - Research articles (PDF, web scraped).
    - Datasets: UCI Pima Indians Diabetes Dataset, WHO, CDC datasets.
- **Mock Data Generation**:
    - Use `faker` and `pandas` to create:
        - Patient demographics: Age, gender, BMI, lifestyle habits.
        - Lab results: HbA1c, glucose levels, cholesterol.
        - Symptoms: Fatigue, frequent urination, blurred vision.
        - Treatments: Medications, diet recommendations.
- **Data Format**:
    - Structured: CSV, JSON.
    - Unstructured: PDFs, markdown files.

---

### Phase 3: RAG Pipeline Development

1. **Data Processing and Embedding**:
    - Preprocess text data (tokenization, cleaning).
    - Convert documents into embeddings using:
        - **Sentence-Transformers** (efficient Hugging Face models).
        - Use quantization (`bitsandbytes`) for memory-efficient processing.
2. **Vector Database**:
    - Prototype with **FAISS** (local).
    - For cloud deployment, use **Pinecone** for scalable storage.
3. **Retriever**:
    - Implement a retriever to fetch relevant embeddings.
4. **Generator**:
    - Integrate the generator models:
        - **LLaMA 3 8B** locally for public data.
        - **ChatGPT-3.5 API** for simulated private data.
5. **LangChain Integration**:
    - Use **LangChain** to orchestrate the RAG workflow: Retriever → Generator.

---

### Phase 4: Fine-Tuning Models

- **LLaMA Fine-Tuning (Private Data)**:
    - **Environment**:
        - Local fine-tuning using **LoRA/QLoRA** for efficiency.
        - Tools: Hugging Face Transformers, `bitsandbytes`.
    - **Cloud Backup**:
        - Explore fine-tuning and serving on **Google Vertex AI**, AWS SageMaker, or Azure ML.
    - **Steps**:
        - Load the base LLaMA model (configurable version) for fine-tuning.
        - Train on **mock private data** for personalization.
        - Validate performance with `lm-evaluation-harness`.

- **OpenAI ChatGPT (Public Data)**:
    - Fine-tuning is not applicable to proprietary models.
    - Adjust prompts, temperature, and other settings for better alignment with the task.
    - Validate outputs to ensure tone and relevance align with expectations.
- **Environment**:
    - Local fine-tuning using **LoRA/QLoRA** (for efficiency).
    - Tools: Hugging Face Transformers, `bitsandbytes`.
- **Cloud Backup**:
    - Prepare for fine-tuning on **Google Vertex AI**.
- **Steps**:
    - Load the base LLaMA model (configurable version) for fine-tuning.
    - Train on **mock private data** for personalization.
    - Validate fine-tuning performance with lm-evaluation-harness.

---

### Phase 5: Backend API Development

- **Framework**: FastAPI.
- **Endpoints**:
    - `/query`: Accepts patient input, retrieves relevant data, and generates recommendations.
    - `/status`: Health check for debugging.
- **Testing**:
    - Local API testing with Postman or Curl.

---

### Phase 6: Frontend Development

- **Tool**: Streamlit.
- **Features**:
    - Input fields: Symptoms, lab values, demographics.
    - Output: Treatment recommendations and reasoning.
- **Testing**:
    - Connect frontend to the FastAPI backend for live results.

---

### Phase 7: Deployment Preparation

- **Local Deployment**:
    - Dockerize the API and Streamlit app.
    - Ensure compatibility with WSL Ubuntu (test locally).
- **Cloud Deployment**:
    - Google Vertex AI for fine-tuning and serving.
    - Use `Docker` for consistent container deployment.
    - Prepare alternative deployment plans for AWS EC2.
- **Adaptable Configurations**:
    - Switch between local, Google Vertex AI, and AWS using environment variables or YAML files.

---

### Phase 8: Testing, Monitoring, and Evaluation

- **Testing**:
    - Unit tests: Pytest for backend and data pipelines.
    - API response checks for tone and relevance.
    - Manual validation of chatbot answers.
- **Monitoring**:
    - Logs: Integrate basic logging for errors and API usage.
    - Scalability checks: Simulate cloud deployments for load testing.

---

### Phase 9: Documentation

- **Code Documentation**:
    - Clean, modular code with comments.
- **Project Documentation**:
    - README with:
        - Setup instructions (local and cloud).
        - Description of RAG pipeline.
        - Usage examples.
    - Architecture diagrams (using tools like Draw.io).
- **Portfolio Enhancement**:
    - Highlight the adaptability between **local** and **cloud environments**.
    - Emphasize the use of state-of-the-art techniques (LoRA/QLoRA, LangChain).

---

## Technologies Summary

| **Category**             | **Tool/Library**                              |
|--------------------------|-----------------------------------------------|
| **Programming Language** | Python                                        |
| **Data Processing**      | Pandas, NumPy                                 |
| **Model Framework**      | Hugging Face Transformers                     |
| **Optimization**         | bitsandbytes, LoRA/QLoRA                      |
| **RAG Pipeline**         | LangChain                                     |
| **Vector Database**      | FAISS (local), Pinecone (cloud)               |
| **Backend**              | FastAPI                                       |
| **Frontend**             | Streamlit                                     |
| **Containerization**     | Docker, Docker Compose                        |
| **Deployment Platforms** | Local (WSL Ubuntu), Google Vertex AI, AWS EC2 |
| **Testing**              | Pytest, manual testing                        |

---

## Questions for Clarification

1. **GPU Details**: What is the model of your GPU to confirm compatibility with CUDA and LoRA/QLoRA performance?
2. **Mock Data**: Would you like the mock patient data to include any specific demographics or conditions (e.g., type 1
   vs. type 2 diabetes)?
3. **Evaluation Metrics**: Should we include any specific metrics for chatbot response quality, like BLEU or ROUGE
   scores?

---

## Next Steps

1. Confirm all technologies, steps, and components in the plan.
2. Start with **Phase 1: Environment Setup** for local development.

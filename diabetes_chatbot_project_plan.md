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

#### Mock Data Design

The mock data will include:

- **Edge Cases**: Extreme glucose levels, unusual symptoms, and mixed comorbidities (e.g., hypertension, obesity).
- **Types of Diabetes**: Type 1, Type 2, and gestational diabetes.
- **Diverse Demographics**:
    - Age Groups: Children, adults, and elderly.
    - Gender: Male and female.
    - Lifestyle: Sedentary, active, smokers, non-smokers.
- **Longitudinal Data**:
    - Simple tracking of lab values over time for selected patients. Example:
      ```json
      {"patient_id": 123, "age": 45, "HbA1c_over_time": [8.2, 7.5, 6.8], "treatment": "Metformin"}
      ```
- **Exclusions**: Anonymized clinical text records (e.g., doctor-patient conversations) are excluded for simplicity.

---

### Phase 3: RAG Pipeline Development

The **RAG (Retrieval-Augmented Generation)** pipeline combines dynamic information retrieval with model generation to
ensure responses are **accurate, up-to-date, and contextually grounded**.

1. **Data Processing and Embedding**:
    - Preprocess text data (tokenization, cleaning).
    - Convert documents into embeddings using:
        - **Sentence-Transformers** (efficient Hugging Face models for embedding generation).
        - Use quantization (`bitsandbytes`) for memory-efficient processing on large datasets.

2. **Vector Database**:
    - Store and retrieve document embeddings for fast and accurate similarity search.
    - **Local Deployment**: Use **FAISS** for prototyping.
    - **Cloud Deployment**: Use **Pinecone** for scalable vector storage.

3. **Retriever**:
    - Implement a retriever to query the vector database and fetch the most relevant document embeddings.
    - These retrieved documents serve as **context** for the generator.

4. **Generator**:
    - Integrate generator models to generate grounded responses using the retrieved context:
        - **LLaMA**: Handles **simulated private data** for HIPAA-like compliance.
        - **ChatGPT API**: Handles **public data**, including clinical guidelines and research articles.

5. **LangChain Integration**:
    - Use **LangChain** to seamlessly orchestrate the RAG workflow:
        - **Retriever**: Fetches relevant documents.
        - **Generator**: Produces answers conditioned on the retrieved documents.
    - Pass both the **retrieved context** and the **user query** to the generator for improved response accuracy.

#### Explainability Features

The chatbot will include explainability features to enhance trust and usability:

- **Summary of Recommendations**: Clear and concise output for users.
- **Relevant Guidelines**: Highlight retrieved portions of medical guidelines as evidence.
- **Top N Similar Patients**: Examples of similar cases and their treatments.
- **Top N Articles**: Latest research articles for special cases, with links to the source.
- **Confidence Scores**: Display a confidence level for each recommendation (e.g., "85% confident").

---

### Phase 4: Fine-Tuning Models

#### **Fine-Tuning and RAG: Why Use Both?**

Fine-tuning and RAG work together to ensure the chatbot produces accurate, domain-specific, and up-to-date responses:

- **Fine-Tuning**:
    - Adapts the generator models to domain-specific tasks (e.g., understanding diabetes guidelines, terminology, and
      patient scenarios).
    - Makes the models task-aware and more effective for structured medical data.

- **RAG**:
    - Dynamically retrieves **external, up-to-date context** (e.g., guidelines, similar cases, research articles).
    - Grounds the model’s outputs in reliable sources, reducing hallucinations and improving factual accuracy.

By combining **static learning** (fine-tuning) with **dynamic retrieval** (RAG), the chatbot benefits from both
pre-trained domain knowledge and live, contextually relevant information.

---

#### **LLaMA Fine-Tuning (Private Data)**

- **Purpose**: Adapt LLaMA to generate accurate, HIPAA-compliant treatment recommendations based on patient records and
  lab results.
- **Environment**:
    - Local fine-tuning using **LoRA/QLoRA** for efficient training.
    - Tools: Hugging Face Transformers, `bitsandbytes`.
- **Cloud Backup**:
    - Explore fine-tuning and hosting on **Google Vertex AI**, AWS SageMaker, or Azure ML.
- **Steps**:
    - Load the base LLaMA model (configurable version) for fine-tuning.
    - Train on **mock private patient data**, including demographics, lab tests, and treatment histories.
    - Validate fine-tuning performance using `lm-evaluation-harness` and patient-specific test prompts.

---

#### **OpenAI ChatGPT Fine-Tuning (Public Data)**

- **Purpose**: Fine-tune ChatGPT models (e.g., GPT-3.5) to align responses with **clinical guidelines and medical
  research**.
- **Dataset Preparation**:
    - Prepare a fine-tuning dataset in **JSONL format** with prompt-completion pairs:
      ```json
      {"prompt": "What is the recommended HbA1c target for Type 2 diabetes?", 
       "completion": "For most adults with Type 2 diabetes, the HbA1c target is below 7.0%, as per ADA guidelines."}
      ```
    - Ensure data is high-quality, relevant, and adheres to OpenAI’s token and size guidelines.

- **Steps**:
    - Upload the fine-tuning dataset to OpenAI using the CLI or API.
    - Initiate the fine-tuning job and monitor progress through OpenAI tools.
    - Validate outputs post fine-tuning to ensure:
        - **Clinical alignment**: Responses match ADA guidelines and research findings.
        - **Tone and relevance**: Answers are concise, professional, and evidence-based.

---

### **Combining Fine-Tuning and RAG**

The system will use **both fine-tuning and RAG** to achieve optimal results:

1. **Fine-Tuning**:
    - LLaMA becomes task-specific for personalized, patient-level recommendations.
    - OpenAI GPT becomes aligned with public medical guidelines.

2. **RAG**:
    - Dynamically retrieves relevant external documents to provide real-time, factually accurate context.
    - Retrieved context improves the generator’s output by reducing hallucinations and enhancing accuracy.

3. **Final Workflow**:
    - **Retrieve**: Use a retriever to fetch relevant medical guidelines or patient histories.
    - **Augment**: Pass the retrieved context into the generator model as input.
    - **Generate**: Use fine-tuned LLaMA or GPT models to generate final, grounded treatment recommendations.

#### Example Workflow

- **User Input**: "What treatment is best for a 45-year-old male with HbA1c of 8.2%?"
- **Retriever**: Fetches relevant ADA guidelines recommending metformin as the first-line treatment.
- **Generator** (Fine-Tuned LLaMA or ChatGPT):
  ```text
  Recommendation: Metformin is the first-line treatment for Type 2 diabetes with HbA1c above 7.5%.
  Confidence Score: 90%
  Guideline Source: ADA Standards of Care, 2024.
  Top 3 Similar Cases: Patients with HbA1c 8.0%-8.5% responded well to Metformin + lifestyle changes.
  Latest Research: Article 'SGLT2 inhibitors for early intervention', Journal of Diabetes Care, 2024.
  ```

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

#### Evaluation Metrics

To assess chatbot performance, the following metrics will be used:

- **Classification Metrics**:
    - Accuracy and F1-score for outputs like drug recommendations and critical key phrases.
- **Response Quality**:
    - BLEU or ROUGE scores to measure alignment with gold-standard answers.
- **Performance**:
    - Response latency to ensure acceptable wait times.
- **Feedback**:
    - A basic like/dislike feedback system to identify areas for improvement.

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
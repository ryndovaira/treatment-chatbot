# **Treatment Chatbot for Diabetes (Diabetes Mellitus) Support**

This project develops a **Retrieval-Augmented Generation (RAG)** chatbot to provide evidence-based **diabetes treatment
recommendations**.
The chatbot integrates **public medical guidelines** with simulated **private clinical data**, while
adhering to real-world considerations like HIPAA compliance.

The main focus of the project is **learning**, including:

- Building a modular and configurable pipeline.
- Exploring multiple deployment options (Hetzner, AWS, Google Cloud, and Azure).
- Experimenting with both on-premise-like and cloud-based LLaMA deployments.

---

## **Overview**

The chatbot:

- **Private Data**: Processed using **LLaMA 3** (simulating on-premise hosting).
- **Public Data**: Retrieved and processed via **OpenAI's ChatGPT API**.
- **Deployment Options**:
    - **Hetzner Shared vCPU** for initial, low-cost deployment.
    - **AWS/Google Cloud/Azure** for scalable, production-ready deployments.
- **LLaMA Deployment Strategies**:
    1. **Fine-tune LLaMA locally**, then load and host it on cloud platforms (e.g., AWS EC2, Google Compute Engine,
       Azure VMs).
    2. Use cloud-based **MLOps services** like Google Vertex AI, AWS SageMaker, or Azure ML for fine-tuning and serving.

This flexibility ensures a robust learning experience while simulating a real-world production pipeline.

---

## **Table of Contents**

1. [Project Structure](#project-structure)
2. [Key Features](#key-features)
3. [Requirements](#requirements)
4. [Usage](#usage)
5. [Deployment Options](#deployment-options)
6. [Configurations](#configurations)
7. [Data Gathering](#data-gathering)
8. [Future Work](#future-work)
9. [About the Author](#about-the-author)

---

## **Project Structure**

```plaintext
treatment-chatbot/
│
├── .gitignore            # Ignores unnecessary files
├── .env                  # Environment variables for models and APIs
├── .env.example          # Template for environment variables
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
│
├── data/                 # Data files
│   ├── raw/              # Raw public guidelines and mock private data
│   └── processed/        # Preprocessed data for model usage
│
├── src/                  # Source code
│   ├── __init__.py       # Marks src as a Python package
│   ├── data/             # Data loaders and preprocessors
│   ├── models/           # Model loading, fine-tuning, and inference logic
│   ├── pipeline/         # RAG pipeline implementation
│   ├── api/              # FastAPI backend
│   ├── frontend/         # Streamlit frontend for user interaction
│   ├── utils/            # Logging, helper functions, and environment management
│   └── config/           # Configurable YAML/JSON files for dynamic settings
│
├── tests/                # Unit and integration tests
│   ├── test_pipeline.py  # Tests for RAG pipeline components
│   └── test_api.py       # API endpoint tests
│
├── scripts/              # Scripts for data generation and model fine-tuning
│   ├── generate_mock_data.py  # Mock private data generation
│   └── fine_tune_llama.py     # Fine-tuning script for LLaMA models
│
├── logs/                 # Logging outputs for debugging
├── Dockerfile            # Container setup for deployment
└── docker-compose.yml    # Multi-service deployment (API + frontend)
```

---

## **Key Features**

1. **Retrieval-Augmented Generation (RAG) Pipeline**:
    - Combines public data (via ChatGPT API) and private data (via LLaMA 3) to generate accurate, evidence-based
      recommendations.
    - Modular design with **FAISS** for local vector search and **Pinecone** for cloud-based indexing.

2. **Simulated On-Premise Design**:
    - **LLaMA 3** models are assumed to run "on-premise" for private data processing.
    - Practical deployment uses rented servers or cloud solutions.

3. **Public Data Processing**:
    - Integrates with **OpenAI API (ChatGPT 3.5)** to process public data and guidelines.

4. **Deployment Options**:
    - **Hetzner Shared vCPU**: Cost-effective for initial deployment.
    - **AWS/Google Cloud/Azure**: Scalable, production-ready options for learning cloud deployments.

5. **LLaMA Deployment Strategies**:
    - Fine-tune locally → Load and host the model on AWS EC2/Google Compute/Azure VM.
    - Use cloud-based **MLOps services** (e.g., Google Vertex AI, AWS SageMaker, or Azure ML).

6. **Configurable Setup**:
    - Model versions, API keys, and deployment options are configurable via `.env` files and YAML configurations in
      `config/`.

7. **Interactive User Interface**:
    - **Streamlit** frontend for easy chatbot interaction.
    - **FastAPI** backend for efficient model serving.

---

## **Requirements**

- **Python**: 3.12
- **CUDA**: 12.7
- **GPU**: NVIDIA GeForce MX450

Install dependencies:

```bash
pip install -r requirements.txt
```

## Cloning the Repository with Git LFS

Ensure you have Git LFS installed before cloning:

```bash
git lfs install
git clone <repository-url>


---

## **Usage**

### **1. Setup Configurations**
Copy `.env.example` to `.env` and update the following variables:
```dotenv
CHATGPT_API_KEY=your_openai_api_key
CHATGPT_MODEL=gpt-3.5-turbo
LLAMA_MODEL_PATH=./models/llama-3-8B
LLAMA_MODEL_VERSION=8B
```

---

### **2. Run the Application**

1. Start the **FastAPI backend**:
   ```bash
   python src/api/main.py
   ```

2. Launch the **Streamlit interface**:
   ```bash
   streamlit run src/frontend/app.py
   ```

Access the chatbot at: `http://localhost:8501`

---

## **Deployment Options**

1. **Hetzner Shared vCPU**:
    - Use Docker for a simple and low-cost deployment:
      ```bash
      docker-compose up --build
      ```

2. **AWS/Google Cloud/Azure**:
    - Deploy the FastAPI backend and Streamlit frontend on EC2, Google Compute, or Azure VMs.
    - Alternatively, use MLOps tools like Google Vertex AI or AWS SageMaker for fine-tuning and hosting LLaMA models.

3. **Switch Between Deployments**:
    - Use environment variables in `.env` to toggle deployment configurations.

---

## Data Gathering

This project involves gathering both public and simulated private data to support the development of the diabetes
treatment chatbot.

### Public Data

We collect:

- **Clinical Guidelines**: e.g., ADA Standards of Care, NICE guidelines.
- **Research Articles**: e.g., PubMed Central and bioRxiv papers.
- **Open Datasets**: e.g., UCI Machine Learning Repository and WHO statistics.

### Simulated Private Data

We generate:

- **Patient Demographics**: Age, gender, BMI, and more.
- **Lab Results**: HbA1c, glucose, cholesterol levels.
- **Symptoms and Treatments**: Including edge cases and longitudinal data.

For detailed data sources, methods, and examples, refer to the [Project Plan](./diabetes_chatbot_project_plan.md).


---

## **Future Work**

1. **Performance Optimization**:
    - Evaluate latency and resource usage across different deployment strategies.

2. **Advanced Monitoring**:
    - Integrate logging and monitoring for API performance and model outputs.

3. **Larger Models**:
    - Experiment with LLaMA 13B or other larger fine-tuned models for improved accuracy.

---

## **About the Author**

This project was created by **Irina Ryndova**, a Machine Learning Engineer and Data Scientist.

- **GitHub**: [ryndovaira](https://github.com/ryndovaira)
- **Email**: [ryndovaira@gmail.com](mailto:ryndovaira@gmail.com)

---

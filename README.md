# SorcererXStreme

**SorcererXStreme** is a serverless backend infrastructure designed for AI-driven spiritual and metaphysical consultation. It leverages **AWS Lambda**, **Amazon Bedrock (Nova Pro)**, and **Pinecone** to deliver features ranging from RAG-based chatbots to complex Vietnamese Horoscope (Tử Vi) calculations.

## System Architecture

The project operates on a Serverless Architecture using AWS Lambda functions, triggered by API Gateway or S3 events. It integrates a custom calculation engine with Generative AI to provide personalized spiritual readings.

### Repository Structure
This project follows a Monorepo pattern, housing three distinct microservices triggered by specific paths.

SorcererXStreme/
├── .github/workflows/           # CI/CD Pipelines
│   ├── deploy_chatbot.yml       # Deploy Chatbot Service
│   ├── deploy_embedding.yml     # Deploy Embedding Service
│   └── deploy_metaphysical.yml  # Deploy Metaphysical Service
├── lambda/
│   ├── chatbot/                 # Chatbot Service (Python 3.13)
│   │   ├── lambda_function.py
│   │   ├── requirements.txt
│   │   └── tests/
│   ├── embedding/               # Knowledge Engine (Python 3.13)
│   │   ├── lambda_function.py
│   │   └── requirements.txt
│   └── metaphysical/            # Calculation Engine (Python 3.10)
│       ├── lasotuvi/            # Custom Library: Vietnamese Horoscope logic
│       ├── lambda_function.py
│       ├── prompts.py           # AI Prompts (Tarot, Astrology, Tu Vi)
│       ├── requirements.txt
│       └── tests/

---

## Services Overview

### 1. Chatbot Service (`lambda/chatbot`)
**Runtime:** Python 3.13
The central orchestrator for user interactions.
* **Intent Detection:** Routes queries between General Chit-chat, Tarot, and Horoscope contexts.
* **RAG Integration:** Queries **Pinecone** to retrieve context-aware spiritual knowledge.
* **GenAI:** Connects to **AWS Bedrock** (Amazon Nova Micro/Pro) for final response generation.

### 2. Embedding Service (`lambda/embedding`)
**Runtime:** Python 3.13
The ETL pipeline for knowledge management.
* **Trigger:** Processes `.jsonl` datasets uploaded to S3.
* **Vectorization:** Generates embeddings (e.g., Cohere Multilingual) via AWS Bedrock.
* **Dual-Sync Storage:**
    * **DynamoDB:** Stores raw content and metadata (preserving Vietnamese accents).
    * **Pinecone:** Stores vector embeddings for semantic search.

### 3. Metaphysical Service (`lambda/metaphysical`)
**Runtime:** Python 3.10
A specialized computational engine for spiritual domains. It uses a custom-built library (`lasotuvi`) to perform astronomical calculations without relying on external APIs.
* **Vietnamese Horoscope (Tử Vi):** Calculates Lunar dates, arranges Stars (An Sao), and generates Heaven/Earth Boards (Thiên Bàn/Địa Bàn).
* **Tarot:** Analyzes 3-card spreads (Past/Present/Future) using AI-driven prompts.
* **Astrology & Numerology:** Computes Zodiac signs, compatibility scores, and Life Path Numbers.
* **AI Interpretation:** Uses **Amazon Nova Pro** to synthesize calculation results into natural language using dynamic templates from `prompts.py`.

---

## Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Compute** | AWS Lambda | Serverless execution environment. |
| **LLM** | Amazon Nova Pro | Generative AI models via AWS Bedrock. |
| **Vector DB** | Pinecone | Semantic search for RAG. |
| **NoSQL DB** | Amazon DynamoDB | Key-value storage for metadata and contexts. |
| **CI/CD** | GitHub Actions | Automated testing, packaging, and deployment. |

---

## Environment Variables

The following environment variables are required in the AWS Lambda configuration:

| Variable | Service(s) | Description |
| :--- | :--- | :--- |
| `BEDROCK_REGION` | All | AWS Region (e.g., `ap-southeast-1`). |
| `BEDROCK_MODEL_ID` | Chatbot, Metaphysical | Model ID (e.g., `amazon.nova-pro-v1:0`). |
| `DYNAMODB_TABLE_NAME` | Embedding, Metaphysical | Name of the DynamoDB table. |
| `PINECONE_API_KEY` | Chatbot, Embedding | API Key for Pinecone Vector DB. |
| `PINECONE_HOST` | Chatbot, Embedding | Pinecone Index URL. |

---

## Deployment (CI/CD)

Deployment is fully automated via **GitHub Actions** using a path-filtering strategy to optimize resources.

### Workflow Logic
1.  **Triggers:** The pipeline runs only when files in a specific `lambda/<service>` directory are modified.
2.  **Continuous Integration (Test):**
    * Sets up the specific Python environment (3.13 or 3.10).
    * Installs dependencies and runs `pytest` (mocking AWS/Pinecone services).
3.  **Continuous Deployment (Deploy):**
    * **Packaging:** Installs dependencies targeting `manylinux2014_x86_64` for AWS Linux compatibility.
    * **Optimization:** Strips `__pycache__` to reduce zip size.
    * **Metaphysical Special Handling:** Automatically bundles the local `lasotuvi/` library and `prompts.py` into the deployment package.
    * **Update:** Deploys the code to AWS Lambda using AWS CLI.

### Local Testing Command
```bash
# Example for Metaphysical Service
cd lambda/metaphysical
pip install -r requirements.txt
pytest
```

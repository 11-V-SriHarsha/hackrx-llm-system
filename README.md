# 🚀 HackRx 6.0 – Intelligent Query–Retrieval System

A blazing-fast, secure, and accurate API that uses **LLMs + vector search** to answer user queries from large documents like insurance policies and contracts.
This is our official submission for the **Bajaj Finserv HackRx 6.0 Hackathon** 🏆

---

## 🧠 Key Features

* Retrieval-Augmented Generation (RAG) pipeline
* Real-time PDF processing from URL
* Context-based answering using **LLaMA 4 Scout** (Groq)
* Temporary vector index per request using **Pinecone**
* Hugging Face embeddings: `BAAI/bge-small-en-v1.5`
* Token-secured REST API using **FastAPI**
* Auto-cleanup of all resources post request

---

## 🛠️ Tech Stack

| Layer           | Technology                        |
| --------------- | --------------------------------- |
| API Framework   | FastAPI                           |
| LLM Inference   | Groq LPU™ + LLaMA 4 Scout         |
| Vector Store    | Pinecone                          |
| Embeddings      | HuggingFace Sentence Transformers |
| PDF Parsing     | PyPDF via LangChain               |
| Prompt Chaining | LangChain                         |

---

## 📆 System Architecture (RAG Pipeline)

```
User Question
      │
      ▼
[Document URL] ──▶ [PDF Processor] ──▶ [Text Chunks]
                                         │
                                         ▼
                              [Embeddings + Pinecone]
                                         │
                                         ▼
                            [Context Retriever → LLM]
                                         │
                                         ▼
                                🔁 Answer Generated
```

---

## ⚙️ API: `/api/v1/hackrx/run`

* **Method:** `POST`
* **Auth:** `Authorization: Bearer <HACKRX_BEARER_TOKEN>`
* **Content-Type:** `application/json`

### ✅ Request Body

```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "Does this policy cover maternity expenses?"
  ]
}
```

### ✅ Response

```json
{
  "answers": [
    "The grace period for premium payment is 30 days.",
    "No, this policy explicitly excludes maternity expenses."
  ]
}
```

---

## 🧚 How to Test

### 🔁 With cURL

```bash
curl -X POST http://localhost:8000/api/v1/hackrx/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer HACKRX_BEARER_TOKEN" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
    "questions": [
      "What is the grace period for premium payment under this policy?",
      "What is the waiting period for PED coverage?"
    ]
  }'
```

### 📫 With Postman

1. Open Postman and create a new `POST` request.
2. URL: `http://localhost:8000/api/v1/hackrx/run`
3. Headers:

   * `Content-Type`: `application/json`
   * `Authorization`: `Bearer HACKRX_BEARER_TOKEN`
4. Body → raw → JSON:

```json
{
  "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
  "questions": [
    "What is the grace period for premium payment under this policy?",
    "What is the waiting period for PED coverage?"
  ]
}
```

5. Click **Send** to receive the response.

---

## ⚙️ Setup & Installation

### 🧱 Prerequisites

* Python 3.10+
* Pinecone API Key
* Groq API Key

### 📦 1. Clone the Repo

```bash
git clone https://github.com/your-username/hackrx-query-system.git
cd hackrx-query-system
```

### 🔁 2. Create & Activate Virtual Environment

```bash
# Create
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
```

### 📥 3. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 🔐 4. Setup `.env` File

Create a file called `.env` in the project root and paste:
#### ✅ Sample `.env`

```env
PINECONE_API_KEY=your_real_pinecone_api_key
GROQ_API_KEY=your_real_groq_api_key
HACKRX_BEARER_TOKEN=..... (from HackRx dashboard)
```

---

### 🚀 5. Run the Server

```bash
python run.py
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

---

## 🩺 Health Check

* **GET** `/health`
  ✅ Verifies Pinecone connection, Groq LLM, environment config, and FastAPI uptime.

---

## 🧾 License

**MIT** — free to use with attribution.

---


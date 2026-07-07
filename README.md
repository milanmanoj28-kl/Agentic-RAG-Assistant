# Agentic RAG Assistant

An agentic AI system that combines Retrieval-Augmented Generation (RAG) with autonomous tool-calling — the agent reasons over its own knowledge base of documents *or* searches the web, depending on the question, and remembers conversation context across turns.

## Overview

This project extends a traditional RAG pipeline into a true agentic system using LangChain's agent framework. Instead of always retrieving from a fixed document set, the agent autonomously decides which tool to use:

- **Document Search** — retrieves relevant chunks from uploaded documents using vector similarity search
- **Web Search** — searches the web for general knowledge questions outside the document scope

The system is served as a REST API, supports live document uploads (auto-embedded into the knowledge base without restarting), and is fully containerized with Docker.

## Architecture

User Question
│
▼
FastAPI /chat endpoint
│
▼
LangChain Agent (Groq LLM)
│
├──► Tool: document_search  (ChromaDB + Sentence Transformers)
└──► Tool: web_search       (Tavily API)
│
▼
Reasoned Answer

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangChain (1.0, `create_agent`) |
| LLM | Groq (Llama 3.1 8B Instant) |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Web Search Tool | Tavily API |
| API Framework | FastAPI |
| Containerization | Docker |

## Features

- **Agentic tool routing** — the LLM decides whether to search documents or the web based on the question
- **Conversational memory** — maintains context across multiple turns in a session
- **Dynamic document upload** — `/upload` endpoint accepts PDF/TXT files, chunks and embeds them into the vector store instantly, no restart required
- **REST API** — clean `/chat` and `/upload` endpoints with structured request/response models (Pydantic)
- **Dockerized** — fully containerized for consistent, portable deployment

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/chat` | Ask a question; agent responds using the appropriate tool |
| POST | `/upload` | Upload a PDF/TXT document to add to the knowledge base |

## Running Locally

### Prerequisites
- Python 3.13
- Groq API key ([console.groq.com](https://console.groq.com))
- Tavily API key ([tavily.com](https://tavily.com))

### Setup

```bash
git clone https://github.com/your-username/agentic-rag-assistant.git
cd agentic-rag-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here

Add documents to the `documents/` folder, then build the vector store:
```bash
python vector_store.py
```

Run the API:
```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive API documentation.

### Running with Docker

```bash
docker build -t agentic-rag-assistant .
docker run -p 8000:8000 --env-file .env agentic-rag-assistant
```

## Example

**Request** (`POST /chat`):
```json
{
  "question": "What projects has the candidate worked on?"
}
```

**Response:**
```json
{
  "answer": "Based on the documents, the candidate has worked on..."
}
```

## What This Demonstrates

- Building and orchestrating LLM agents with tool-calling and memory (LangChain 1.0)
- Designing RAG pipelines with vector databases and embedding models
- REST API development with FastAPI and Pydantic
- Containerizing AI applications with Docker for portable deployment

## Future Improvements

- Add authentication (JWT) for multi-user support
- Move conversation memory to persistent storage (Redis)
- Add async document processing via Celery for large file uploads

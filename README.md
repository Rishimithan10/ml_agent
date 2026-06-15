# Production LLM System: RAG + LLaMA 3B + LangGraph Agent

An end-to-end, production-grade LLM system built in stages — starting from a
Retrieval-Augmented Generation (RAG) pipeline and evolving into a multi-tool
LangGraph ReAct agent, fine-tuned with QLoRA, deployed via FastAPI + Docker on
Render, monitored with MLflow, and quantized to GGUF for efficient local inference.

---

## Architecture Overview

```
Question → LangGraph Agent → Router → [Pinecone | Tavily | Code Executor]
              ↓
         Fine-tuned LLaMA 3B (QLoRA)
              ↓
       Self-Evaluation + Retry Loop
              ↓
         MLflow Logging → Answer
```

---

## Stage 1 — RAG Pipeline

Built an end-to-end Retrieval-Augmented Generation pipeline using HuggingFace
embeddings (`all-MiniLM-L6-v2`), Pinecone vector database, and LLaMA 3B for
context-grounded response generation.

**Hybrid Retrieval** — combines dense vector search (cosine similarity) and
BM25 sparse retrieval with score fusion (α = 0.5) for improved context quality.

| Metric | Value |
|---|---|
| Faithfulness (RAGAS) | 1.0 |
| Answer Relevancy (RAGAS) | 1.0 |
| Context Precision (RAGAS) | 0.85 |
| End-to-end query latency | ~570 ms |
| BM25 retrieval latency | 0.0002 s |

---

## Stage 2 — Fine-Tuning (QLoRA)

Fine-tuned LLaMA 3B using **QLoRA** on **279 domain-specific ML Q&A pairs**
indexed in Pinecone, embedding domain expertise directly into model weights
while retaining RAG retrieval for grounding.

- Base model: LLaMA 3B (Instruct)
- Method: QLoRA (4-bit quantized base + LoRA adapters)
- Training data: 279 curated ML Q&A pairs
- Outcome: Improved response quality on specialized/domain queries

---

## Stage 3 — LangGraph ReAct Agent

Extended the fine-tuned RAG pipeline into a **multi-tool LangGraph agent**
implementing the **ReAct reasoning pattern** with dynamic tool routing,
self-evaluation, and automatic retries.

**Tools available to the agent:**

| Tool | Purpose |
|---|---|
| Pinecone Vector Search | Searches 279 indexed ML responses (`all-MiniLM-L6-v2` embeddings) |
| Tavily Web Search | Live web results for recent/current information |
| Python Code Executor | Executes Python code blocks and returns output |

**Agent flow:**
1. Question received
2. Router selects the best tool based on intent
3. Selected tool executes and returns context
4. LLM generates a grounded answer
5. Self-evaluation loop scores answer quality
6. If quality is low → automatic retry (max 2 iterations)

---

## Stage 4 — FastAPI + Docker + Render Deployment

Deployed the agent as a production REST API.

- **FastAPI** — request/response validation via Pydantic, async endpoints
- **Docker** — single container packaging app, model, and dependencies
- **Render** — public hosting with health checks

**Endpoints:**

| Endpoint | Description |
|---|---|
| `POST /ask` | Returns answer, tool used, quality score, response time |
| `GET /health` | Service health check |
| `GET /docs` | Interactive Swagger UI |

| Metric | Value |
|---|---|
| Average latency (deployed) | 2.1 s |
| p95 latency (deployed) | 2.9 s |
| Base model serving | Groq API (GPU-backed inference) |

> Base model serving via Groq was selected over local GPU inference due to
> hardware and latency constraints on the deployment environment.

---

## Stage 5 — MLflow Experiment Tracking

Integrated **MLflow** (hosted on Dagshub) for full observability into every
agent run.

**Logged per query:**
- Question / prompt
- Tool selected (Pinecone / Tavily / Code Executor)
- Number of retry iterations
- Quality score (0.0 – 1.0)
- Latency (seconds)
- Full answer and retrieved context as text artifacts

Used to analyze deployment behavior including cold-start latency and request
queueing patterns.

---

## Stage 6 — Quantization (GGUF)

Quantized the fine-tuned QLoRA adapter (merged with base LLaMA 3B) to **GGUF**
format using `llama.cpp` for efficient local/edge inference.

- Merged LoRA adapter into base model weights
- Converted merged model to GGUF (F16)
- Quantized to **Q4_K_M** for reduced size with minimal quality loss
- Uploaded quantized model to HuggingFace Hub

---

## Stage 7 — Evaluation Dashboard

Built a **Streamlit** dashboard for real-time monitoring of agent performance,
visualizing:
- Tool usage distribution (Pinecone vs Tavily vs Code Executor)
- Quality score trends over time
- Latency distribution
- Retry rate analysis

---

## Tech Stack

**Languages:** Python
**LLMs & ML:** PyTorch, HuggingFace Transformers, LangGraph, QLoRA (PEFT), RAG
**Retrieval:** Pinecone (vector DB), BM25, Sentence-Transformers (`all-MiniLM-L6-v2`)
**Agent Tools:** Tavily Search API, Python code execution
**Serving:** Groq API (LLaMA 3B inference)
**MLOps:** FastAPI, Docker, Render, MLflow (Dagshub), GGUF (llama.cpp)
**Evaluation:** RAGAS, Streamlit

---

## Results Summary

| Stage | Metric | Result |
|---|---|---|
| RAG | Faithfulness | 1.0 |
| RAG | Answer Relevancy | 1.0 |
| RAG | Context Precision | 0.85 |
| RAG | End-to-end latency | ~570 ms |
| RAG | BM25 retrieval latency | 0.0002 s |
| Deployment | Average latency | 2.1 s |
| Deployment | p95 latency | 2.9 s |
| Fine-tuning | Training data | 279 ML Q&A pairs |
| Forecasting (sibling project) | XGBoost MAPE | 5.91% |

---

## Links

- **GitHub:** [github.com/Rishimithan10/ml_agent](https://github.com/Rishimithan10/RagApplication)
- **LinkedIn:** [linkedin.com/in/rishimithan-k](https://linkedin.com/in/rishimithan-k)

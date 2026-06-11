import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from agent import run_agent
from model import load_model

load_dotenv()

# ── Lifespan — runs on startup and shutdown ───────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — load model once
    print("Starting up — loading model...")
    load_model()
    print("Server ready!")
    yield
    # Shutdown
    print("Shutting down...")

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="ML Systems Agent API",
    description="Domain expert agent for GPU computing and ML systems",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Pydantic Schemas ──────────────────────────────────────────
class QuestionRequest(BaseModel):
    question  : str = Field(..., min_length=5, max_length=500)
    # ... = required field
    # min_length, max_length = validation

class AnswerResponse(BaseModel):
    answer        : str
    tool_used     : str
    quality_score : float
    iterations    : int
    response_time : float

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Check if server is running"""
    return {
        "status"  : "healthy",
        "model"   : "Llama-3.2-3B-Instruct (fine-tuned)",
        "version" : "1.0.0",
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Send a question, get an answer from the agent"""
    try:
        start_time = time.time()

        # Run the agent
        result = run_agent(request.question)

        response_time = round(time.time() - start_time, 2)

        return AnswerResponse(
            answer        = result["answer"],
            tool_used     = result["tool_used"],
            quality_score = result["quality_score"],
            iterations    = result["iterations"],
            response_time = response_time,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {
        "message" : "ML Systems Agent API",
        "docs"    : "/docs",
        "health"  : "/health",
        "ask"     : "POST /ask"
    }
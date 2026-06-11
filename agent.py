import os
import torch
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from dotenv import load_dotenv
from model import get_model
import re

load_dotenv()

# ── Initialize clients ────────────────────────────────────────
pc       = Pinecone(api_key=os.getenv("PINECONE_KEY"))
index    = pc.Index(os.getenv("INDEX_NAME"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
tavily   = TavilyClient(api_key=os.getenv("TAVILY_KEY"))

# ── Agent State ───────────────────────────────────────────────
class AgentState(TypedDict):
    question      : str
    tool_used     : Optional[str]
    context       : Optional[str]
    answer        : Optional[str]
    quality_score : Optional[float]
    iterations    : int
    final_answer  : Optional[str]

# ── Nodes ─────────────────────────────────────────────────────
def router(state: AgentState) -> AgentState:
    question = state["question"].lower()

    # Detect code execution intent
    code_signals = [
        "```python",           # explicit code block
        "execute",             # "execute this"
        "run this",            # "run this code"
        "run the code",
        "calculate",           # "calculate the result"
        "compute",             # "compute this"
        "what is the output",  # "what is the output of"
        "what does this print",
        "result =",            # contains assignment
        "import ",             # contains import statement
    ]

    # Detect web search intent
    web_signals = [
        "latest", "recent",
        "2025", "2026",
        "news", "today",
        "current", "now",
        "update",
    ]

    if any(signal in question for signal in code_signals):
        tool = "code_executor"
    elif any(signal in question for signal in web_signals):
        tool = "web_search"
    else:
        tool = "pinecone_search"

    print(f"[Router] Selected: {tool}")
    return {**state, "tool_used": tool}


def execute_tool(state: AgentState) -> AgentState:
    tool = state.get("tool_used", "pinecone_search")
    if tool == "pinecone_search":
        return pinecone_search(state)
    elif tool == "web_search":
        return web_search(state)
    elif tool == "code_executor":
        return code_executor(state)
    return state


def pinecone_search(state: AgentState) -> AgentState:
    query_vector = embedder.encode(state["question"]).tolist()
    results      = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
    )
    chunks = [
        m["metadata"].get("text", "")
        for m in results["matches"]
        if m["score"] > 0.5
    ]
    context = "\n\n".join(chunks) if chunks else "No relevant context found."
    return {**state, "tool_used": "pinecone_search", "context": context}


def web_search(state: AgentState) -> AgentState:
    results = tavily.search(query=state["question"], max_results=3)
    chunks  = [r["content"] for r in results["results"]]
    context = "\n\n".join(chunks)
    return {**state, "tool_used": "web_search", "context": context}


def code_executor(state: AgentState) -> AgentState:
    question = state["question"]

    # Try to find code block with backticks first
    code_match = re.search(r"```python(.*?)```", question, re.DOTALL)

    if code_match:
        code = code_match.group(1).strip()
    else:
        # Try to extract raw code without backticks
        # Remove common prefixes like "run this:", "calculate:", etc.
        prefixes = [
            "run this code:", "run this:", "execute this:",
            "calculate:", "compute:", "what is the output of:",
            "run:", "execute:",
        ]
        code = question
        for prefix in prefixes:
            if prefix in code.lower():
                # Take everything after the prefix
                idx  = code.lower().index(prefix)
                code = code[idx + len(prefix):].strip()
                break

    print(f"[Code] Executing: {code[:100]}")

    try:
        exec_globals = {}
        exec(code, exec_globals)
        # Try common output variable names
        output = (
            exec_globals.get("result") or
            exec_globals.get("output") or
            exec_globals.get("answer") or
            "Code executed successfully"
        )
        output = str(output)
    except Exception as e:
        output = f"Error: {str(e)}"

    print(f"[Code] Output: {output[:100]}")
    return {**state, "tool_used": "code_executor", "context": output}

from model import generate_response

def generate_answer(state: AgentState) -> AgentState:
    prompt = f"""Use this context to answer the question accurately.

    Context:
    {state['context']}

    Question:
    {state['question']}

    Answer:"""

    answer = generate_response(prompt)
    return {**state, "answer": answer}

# def generate_answer(state: AgentState) -> AgentState:
#     model, tokenizer = get_model()

#     prompt = f"""<|system|>
# You are an expert ML systems assistant specializing in GPU computing,
# CUDA kernels, and transformer architectures.
# Use the provided context to answer accurately.

# Context:
# {state['context']}
# <|user|>
# {state['question']}
# <|assistant|>
# """
#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=1024,
#     ).to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=300,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
#     answer     = tokenizer.decode(new_tokens, skip_special_tokens=True)
#     return {**state, "answer": answer}


def evaluate_answer(state: AgentState) -> AgentState:
    answer  = state["answer"]
    context = state["context"]
    score   = 0.0

    # Uncertainty check
    uncertainty_phrases = [
        "i couldn't find", "i don't know",
        "no specific information", "without more context",
        "i'm not sure", "cannot determine",
    ]
    if any(p in answer.lower() for p in uncertainty_phrases):
        return {**state, "quality_score": 0.2}

    # Length check
    if len(answer.strip()) > 20:
        score += 0.3

    # Domain keyword check
    keywords = [
        "cuda", "gpu", "memory", "thread", "kernel",
        "transformer", "attention", "model", "warp",
        "bandwidth", "retrieval", "embedding", "vector",
    ]
    score += min(sum(1 for k in keywords if k in answer.lower()) * 0.1, 0.4)

    # Grounding check
    if context and context != "No relevant context found.":
        overlap = len(set(context.lower().split()) & set(answer.lower().split()))
        score  += min(overlap / 20, 0.3)

    return {**state, "quality_score": min(score, 1.0)}


def should_retry(state: AgentState) -> str:
    score      = state.get("quality_score", 0)
    iterations = state.get("iterations", 0)
    if score >= 0.7 or iterations >= 2:
        return "end"
    return "retry"


def select_retry_tool(state: AgentState) -> AgentState:
    current = state.get("tool_used", "pinecone_search")
    next_tool = "web_search" if current == "pinecone_search" else "pinecone_search"
    return {**state, "tool_used": next_tool, "iterations": state.get("iterations", 0) + 1}


def finalize(state: AgentState) -> AgentState:
    return {**state, "final_answer": state["answer"]}


# ── Build Graph ───────────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("router",       router)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("generate",     generate_answer)
    graph.add_node("evaluate",     evaluate_answer)
    graph.add_node("retry",        select_retry_tool)
    graph.add_node("finalize",     finalize)

    graph.set_entry_point("router")
    graph.add_edge("router",       "execute_tool")
    graph.add_edge("execute_tool", "generate")
    graph.add_edge("generate",     "evaluate")
    graph.add_edge("retry",        "execute_tool")
    graph.add_edge("finalize",     END)

    graph.add_conditional_edges(
        "evaluate",
        should_retry,
        {"end": "finalize", "retry": "retry"}
    )

    return graph.compile()


# ── Run agent function ────────────────────────────────────────
agent = build_agent()

def run_agent(question: str) -> dict:
    initial_state = AgentState(
        question      = question,
        tool_used     = None,
        context       = None,
        answer        = None,
        quality_score = None,
        iterations    = 0,
        final_answer  = None,
    )
    final_state = agent.invoke(initial_state)
    return {
        "answer"        : final_state["final_answer"],
        "tool_used"     : final_state.get("tool_used", "N/A"),
        "quality_score" : final_state.get("quality_score", 0),
        "iterations"    : final_state.get("iterations", 0),
    }
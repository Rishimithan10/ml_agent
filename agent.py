# import os
# import torch
# import time
# from typing import TypedDict, Optional
# from langgraph.graph import StateGraph, END
# from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer
# from tavily import TavilyClient
# from dotenv import load_dotenv
# from model import get_model, generate_response
# import re
# import mlflow
# import dagshub

# load_dotenv()

# # ── MLflow Setup ──────────────────────────────────────────────
# os.environ["DAGSHUB_USER_TOKEN"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


# dagshub.init(repo_owner="rishimithan", repo_name="ml_agent", mlflow=True)
# mlflow.set_experiment("ml_agent")



# # ── Initialize clients ────────────────────────────────────────
# pc       = Pinecone(api_key=os.getenv("PINECONE_KEY"))
# index    = pc.Index(os.getenv("INDEX_NAME"))
# embedder = SentenceTransformer("all-MiniLM-L6-v2")
# tavily   = TavilyClient(api_key=os.getenv("TAVILY_KEY"))

# # ── Agent State ───────────────────────────────────────────────
# class AgentState(TypedDict):
#     question      : str
#     tool_used     : Optional[str]
#     context       : Optional[str]
#     answer        : Optional[str]
#     quality_score : Optional[float]
#     iterations    : int
#     final_answer  : Optional[str]
#     latency       : Optional[float]

# # ── Nodes ─────────────────────────────────────────────────────
# def router(state: AgentState) -> AgentState:
#     question = state["question"].lower()

#     code_signals = [
#         "```python", "execute", "run this", "run the code",
#         "calculate", "compute", "what is the output",
#         "what does this print", "result =", "import ",
#     ]
#     web_signals = [
#         "latest", "recent", "2025", "2026",
#         "news", "today", "current", "now", "update",
#     ]

#     if any(signal in question for signal in code_signals):
#         tool = "code_executor"
#     elif any(signal in question for signal in web_signals):
#         tool = "web_search"
#     else:
#         tool = "pinecone_search"

#     print(f"[Router] Selected: {tool}")
#     return {**state, "tool_used": tool}


# def execute_tool(state: AgentState) -> AgentState:
#     tool = state.get("tool_used", "pinecone_search")
#     if tool == "pinecone_search":
#         return pinecone_search(state)
#     elif tool == "web_search":
#         return web_search(state)
#     elif tool == "code_executor":
#         return code_executor(state)
#     return state


# def pinecone_search(state: AgentState) -> AgentState:
#     start        = time.time()
#     query_vector = embedder.encode(state["question"]).tolist()
#     results      = index.query(
#         vector=query_vector,
#         top_k=3,
#         include_metadata=True,
#     )
#     chunks = [
#         m["metadata"].get("text", "")
#         for m in results["matches"]
#         if m["score"] > 0.5
#     ]
#     context = "\n\n".join(chunks) if chunks else "No relevant context found."
#     latency = time.time() - start
#     return {**state, "tool_used": "pinecone_search", "context": context, "latency": latency}


# def web_search(state: AgentState) -> AgentState:
#     start   = time.time()
#     results = tavily.search(query=state["question"], max_results=3)
#     chunks  = [r["content"] for r in results["results"]]
#     context = "\n\n".join(chunks)
#     latency = time.time() - start
#     return {**state, "tool_used": "web_search", "context": context, "latency": latency}


# def code_executor(state: AgentState) -> AgentState:
#     start    = time.time()
#     question = state["question"]

#     code_match = re.search(r"```python(.*?)```", question, re.DOTALL)
#     if code_match:
#         code = code_match.group(1).strip()
#     else:
#         prefixes = [
#             "run this code:", "run this:", "execute this:",
#             "calculate:", "compute:", "what is the output of:",
#             "run:", "execute:",
#         ]
#         code = question
#         for prefix in prefixes:
#             if prefix in code.lower():
#                 idx  = code.lower().index(prefix)
#                 code = code[idx + len(prefix):].strip()
#                 break

#     print(f"[Code] Executing: {code[:100]}")

#     try:
#         exec_globals = {}
#         exec(code, exec_globals)
#         output = (
#             exec_globals.get("result") or
#             exec_globals.get("output") or
#             exec_globals.get("answer") or
#             "Code executed successfully"
#         )
#         output = str(output)
#     except Exception as e:
#         output = f"Error: {str(e)}"

#     print(f"[Code] Output: {output[:100]}")
#     latency = time.time() - start
#     return {**state, "tool_used": "code_executor", "context": output, "latency": latency}


# def generate_answer(state: AgentState) -> AgentState:
#     prompt = f"""Use this context to answer the question accurately.

#     Context:
#     {state['context']}

#     Question:
#     {state['question']}

#     Answer:"""

#     answer = generate_response(prompt)
#     return {**state, "answer": answer}


# def evaluate_quality(state: AgentState) -> AgentState:
#     """Score answer quality 0-1 based on simple heuristics"""
#     answer = state.get("answer", "")

#     if not answer or len(answer) < 20:
#         score = 0.0
#     elif "error" in answer.lower() or "i don't know" in answer.lower():
#         score = 0.3
#     elif len(answer) > 100:
#         score = 0.9
#     else:
#         score = 0.7

#     print(f"[Evaluator] Quality score: {score}")
#     return {**state, "quality_score": score}

# # def generate_answer(state: AgentState) -> AgentState:
# #     model, tokenizer = get_model()

# #     prompt = f"""<|system|>
# # You are an expert ML systems assistant specializing in GPU computing,
# # CUDA kernels, and transformer architectures.
# # Use the provided context to answer accurately.

# # Context:
# # {state['context']}
# # <|user|>
# # {state['question']}
# # <|assistant|>
# # """
# #     inputs = tokenizer(
# #         prompt,
# #         return_tensors="pt",
# #         truncation=True,
# #         max_length=1024,
# #     ).to(model.device)

# #     with torch.no_grad():
# #         outputs = model.generate(
# #             **inputs,
# #             max_new_tokens=300,
# #             temperature=0.7,
# #             top_p=0.9,
# #             do_sample=True,
# #             pad_token_id=tokenizer.eos_token_id,
# #         )

# #     new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
# #     answer     = tokenizer.decode(new_tokens, skip_special_tokens=True)
# #     return {**state, "answer": answer}


# def should_retry(state: AgentState) -> str:
#     """Retry if quality is low and iterations < 2"""
#     if state.get("quality_score", 0) < 0.5 and state.get("iterations", 0) < 2:
#         print("[Evaluator] Low quality — retrying...")
#         return "retry"
#     return "done"


# def finalize(state: AgentState) -> AgentState:
#     """Log everything to MLflow"""
#     with mlflow.start_run():
#         mlflow.log_param("question",    state["question"])
#         mlflow.log_param("tool_used",   state.get("tool_used", "unknown"))
#         mlflow.log_param("iterations",  state.get("iterations", 1))

#         mlflow.log_metric("quality_score", state.get("quality_score", 0.0))
#         mlflow.log_metric("latency_sec",   round(state.get("latency", 0.0), 4))

#         mlflow.log_text(state.get("answer", ""), "answer.txt")
#         mlflow.log_text(state.get("context", ""), "context.txt")

#     print("[MLflow] Run logged.")
#     return {**state, "final_answer": state.get("answer")}


# # ── Build Graph ───────────────────────────────────────────────
# def build_agent():
#     graph = StateGraph(AgentState)

#     graph.add_node("router",          router)
#     graph.add_node("execute_tool",    execute_tool)
#     graph.add_node("generate_answer", generate_answer)
#     graph.add_node("evaluate_quality",evaluate_quality)
#     graph.add_node("finalize",        finalize)

#     graph.set_entry_point("router")
#     graph.add_edge("router",          "execute_tool")
#     graph.add_edge("execute_tool",    "generate_answer")
#     graph.add_edge("generate_answer", "evaluate_quality")

#     graph.add_conditional_edges(
#         "evaluate_quality",
#         should_retry,
#         {
#             "retry": "router",   # low quality → retry with router
#             "done":  "finalize", # good quality → finalize + log
#         }
#     )

#     graph.add_edge("finalize", END)
#     return graph.compile()


# agent = build_agent()


# def run_agent(question: str) -> dict:
#     initial_state = AgentState(
#         question      = question,
#         tool_used     = None,
#         context       = None,
#         answer        = None,
#         quality_score = None,
#         iterations    = 0,
#         final_answer  = None,
#         latency       = None,
#     )
#     result = agent.invoke(initial_state)
#     return {
#         "answer"       : result.get("final_answer"),
#         "tool_used"    : result.get("tool_used"),
#         "quality_score": result.get("quality_score"),
#         "latency"      : result.get("latency"),
#         "iterations"   : result.get("iterations"),
#     }

import os
import torch
import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from dotenv import load_dotenv
from model import get_model, generate_response
import re
import mlflow
import dagshub

load_dotenv()

# ── MLflow + DagsHub Setup ────────────────────────────────────
os.environ["DAGSHUB_USER_TOKEN"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
dagshub.init(repo_owner="rishimithan", repo_name="ml_agent", mlflow=True)
mlflow.set_experiment("ml_agent")

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
    latency       : Optional[float]
    start_time    : Optional[float]    

# ── Nodes ─────────────────────────────────────────────────────
def router(state: AgentState) -> AgentState:
    question = state["question"].lower()

    code_signals = [
        "```python", "execute", "run this", "run the code",
        "calculate", "compute", "what is the output",
        "what does this print", "result =", "import ",
    ]
    web_signals = [
        "latest", "recent", "2025", "2026",
        "news", "today", "current", "now", "update",
    ]

    if any(signal in question for signal in code_signals):
        tool = "code_executor"
    elif any(signal in question for signal in web_signals):
        tool = "web_search"
    else:
        tool = "pinecone_search"

    print(f"[Router] Selected: {tool}")

    # Record start time on first pass only
    start_time = state.get("start_time") or time.time()
    return {**state, "tool_used": tool, "start_time": start_time}


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
    start        = time.time()
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
    latency = round(time.time() - start, 4)
    print(f"[Pinecone] Retrieved {len(chunks)} chunks in {latency}s")
    return {**state, "tool_used": "pinecone_search", "context": context, "latency": latency}


def web_search(state: AgentState) -> AgentState:
    start   = time.time()
    results = tavily.search(query=state["question"], max_results=3)
    chunks  = [r["content"] for r in results["results"]]
    context = "\n\n".join(chunks)
    latency = round(time.time() - start, 4)
    print(f"[Web] Retrieved {len(chunks)} results in {latency}s")
    return {**state, "tool_used": "web_search", "context": context, "latency": latency}


def code_executor(state: AgentState) -> AgentState:
    start    = time.time()
    question = state["question"]

    code_match = re.search(r"```python(.*?)```", question, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        prefixes = [
            "run this code:", "run this:", "execute this:",
            "calculate:", "compute:", "what is the output of:",
            "run:", "execute:",
        ]
        code = question
        for prefix in prefixes:
            if prefix in code.lower():
                idx  = code.lower().index(prefix)
                code = code[idx + len(prefix):].strip()
                break

    try:
        exec_globals = {}
        exec(code, exec_globals)
        output = (
            exec_globals.get("result") or
            exec_globals.get("output") or
            exec_globals.get("answer") or
            "Code executed successfully"
        )
        output = str(output)
    except Exception as e:
        output = f"Error: {str(e)}"

    latency = round(time.time() - start, 4)
    print(f"[Code] Output: {output[:100]} in {latency}s")
    return {**state, "tool_used": "code_executor", "context": output, "latency": latency}


def generate_answer(state: AgentState) -> AgentState:
    prompt = f"""Use this context to answer the question accurately.
If context is insufficient say so clearly.

Context:
{state['context']}

Question:
{state['question']}

Answer:"""

    answer = generate_response(prompt)
    print(f"[Generator] Answer: {answer[:80]}...")
    return {**state, "answer": answer}


def evaluate_quality(state: AgentState) -> AgentState:
    answer = state.get("answer", "")

    if not answer or len(answer) < 20:
        score = 0.0
    elif any(p in answer.lower() for p in [
        "i don't know", "i couldn't find",
        "no specific information", "i'm not sure"
    ]):
        score = 0.3
    elif len(answer) > 100:
        score = 0.9
    else:
        score = 0.7

    print(f"[Evaluator] Quality score: {score}")
    return {**state, "quality_score": score}


def should_retry(state: AgentState) -> str:
    if state.get("quality_score", 0) < 0.5 and state.get("iterations", 0) < 2:
        print("[Evaluator] Low quality — retrying...")
        return "retry"
    return "done"


def increment_retry(state: AgentState) -> AgentState:
    """Increment iterations before retrying"""
    new_iterations = state.get("iterations", 0) + 1
    print(f"[Retry] Attempt {new_iterations}")
    return {**state, "iterations": new_iterations}


def finalize(state: AgentState) -> AgentState:
    """Log everything to MLflow → DagsHub"""
    total_latency = round(time.time() - state.get("start_time", time.time()), 4)

    with mlflow.start_run():
        # Parameters
        mlflow.log_param("question_preview", state["question"][:100])
        mlflow.log_param("question_length",  len(state["question"]))
        mlflow.log_param("tool_used",        state.get("tool_used", "unknown"))
        mlflow.log_param("iterations",       state.get("iterations", 0))

        # Metrics
        mlflow.log_metric("quality_score",   state.get("quality_score", 0.0))
        mlflow.log_metric("tool_latency",    state.get("latency", 0.0))
        mlflow.log_metric("total_latency",   total_latency)
        mlflow.log_metric("answer_length",   len(state.get("answer", "")))

        # Artifacts
        mlflow.log_text(state.get("question", ""), "question.txt")
        mlflow.log_text(state.get("answer",   ""), "answer.txt")
        mlflow.log_text(state.get("context",  ""), "context.txt")

    print(f"[MLflow] Logged to DagsHub. Total: {total_latency}s")
    return {**state, "final_answer": state.get("answer")}


# ── Build Graph ───────────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("router",           router)
    graph.add_node("execute_tool",     execute_tool)
    graph.add_node("generate_answer",  generate_answer)
    graph.add_node("evaluate_quality", evaluate_quality)
    graph.add_node("increment_retry",  increment_retry)
    graph.add_node("finalize",         finalize)

    graph.set_entry_point("router")
    graph.add_edge("router",           "execute_tool")
    graph.add_edge("execute_tool",     "generate_answer")
    graph.add_edge("generate_answer",  "evaluate_quality")
    graph.add_edge("increment_retry",  "router")
    graph.add_edge("finalize",         END)

    graph.add_conditional_edges(
        "evaluate_quality",
        should_retry,
        {
            "retry": "increment_retry",
            "done" : "finalize",
        }
    )

    return graph.compile()


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
        latency       = None,
        start_time    = None,
    )
    result = agent.invoke(initial_state)
    return {
        "answer"        : result.get("final_answer"),
        "tool_used"     : result.get("tool_used"),
        "quality_score" : result.get("quality_score"),
        "latency"       : result.get("latency"),
        "iterations"    : result.get("iterations"),
    }
# import torch
# import os
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
# )
# from peft import PeftModel
# from huggingface_hub import login
# from dotenv import load_dotenv

# load_dotenv()

# # ── Global variables ──────────────────────────────────────────
# # Loaded once at startup, reused for every request
# model     = None
# tokenizer = None

# def load_model():
#     """Load fine-tuned model into memory"""
#     global model, tokenizer

#     print("Logging into HuggingFace...")
#     login(os.getenv("HF_TOKEN"))

#     MODEL_NAME  = os.getenv("MODEL_NAME")
#     ADAPTER_DIR = os.getenv("ADAPTER_DIR")

#     # Quantization config
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float32,
#         bnb_4bit_use_double_quant=True,
#     )

#     print("Loading base model...")
#     base_model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=bnb_config,
#         device_map="auto",
#         trust_remote_code=True,
#     )

#     print("Loading LoRA adapter...")
#     model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
#     model = model.merge_and_unload()
#     model.eval()

#     print("Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     tokenizer.pad_token = tokenizer.eos_token

#     print("Model ready!")
#     return model, tokenizer


def get_model():
    """Return loaded model — used by other modules"""
    global model, tokenizer
    if model is None:
        load_model()
    return model, tokenizer

# model.py
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_KEY"))

def generate_response(prompt: str) -> str:
    """Call Groq API instead of local model"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role"   : "system",
                "content": "You are an expert ML systems assistant specializing in GPU computing, CUDA kernels, and transformer architectures."
            },
            {
                "role"   : "user",
                "content": prompt
            }
        ],
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message.content

def load_model():
    """Verify Groq connection at startup"""
    test = generate_response("Say OK")
    print(f"Groq connected: {test}")

def get_model():
    return None, None    # not used with Groq
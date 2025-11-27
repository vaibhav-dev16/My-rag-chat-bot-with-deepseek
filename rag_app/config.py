import os
import warnings
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file (if present)
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL")


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers based only on the given document.\n"
    "Keep answers concise and clear."
)

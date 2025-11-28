import os
import warnings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
# Here i am loading environment variables from a .env file and also providing default values for configuration settings used in the RAG chat application.
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")
LLM_MODEL = os.getenv("DEEPSEEK_MODEL")


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# System prompt for the AI assistant
SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers based only on the given document.\n"
    "Keep answers concise and clear."
)

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 4

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

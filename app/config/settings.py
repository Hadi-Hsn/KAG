import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# OpenAI Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# ChromaDB Configuration
CHROMA_HOST = os.getenv("CHROMA_SERVER_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_SERVER_PORT", "8001"))

# Application Configuration
DATA_DIR = "data"
COLLECTION_NAME = "rag_collection"
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# Pushover Configuration (for usage notifications)
PUSHOVER_USER = os.getenv("PUSHOVER_USER", "")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

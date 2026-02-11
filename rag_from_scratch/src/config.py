import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# --- Project Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Data file paths
RAW_SALARIES_PATH = RAW_DATA_DIR / "salaries.csv"
PROCESSED_SALARIES_PATH = PROCESSED_DATA_DIR / "cleaned_salaries.csv"
FAISS_INDEX_PATH = PROCESSED_DATA_DIR / "faiss_index.bin"

# --- Model Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# LLM Configuration
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0  # Deterministic output for grounded answers

# --- Retrieval Configuration ---
DEFAULT_TOP_K = 5  # Number of chunks to retrieve

# --- Logging Configuration ---
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def setup_logging(name: str = __name__) -> logging.Logger:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    return logging.getLogger(name)


# --- Mapping Configurations ---
EXPERIENCE_MAPPING = {
    'EN': 'Entry-level', 
    'MI': 'Mid-level', 
    'SE': 'Senior-level', 
    'EX': 'Executive-level'
}

EMPLOYMENT_MAPPING = {
    'FT': 'Full-time', 
    'PT': 'Part-time',
    'CT': 'Contract', 
    'FL': 'Freelance'
}

COMPANY_SIZE_MAPPING = {
    'S': 'small', 
    'M': 'medium', 
    'L': 'large'
}

REMOTE_RATIO_MAPPING = {
    100: 'remote', 
    50: 'hybrid', 
    0: 'on-site'
}

# --- API Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

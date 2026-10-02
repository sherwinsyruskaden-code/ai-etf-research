import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# API Credentials
GROWW_API_KEY = os.getenv("GROWW_API_KEY", "")
GROWW_API_SECRET = os.getenv("GROWW_API_SECRET", "")

# Database settings
POSTGRES_USER = os.getenv("POSTGRES_USER", "etf_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "etf_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "etf_research")

DEFAULT_DB_URL = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)

# Pipeline Configuration
HISTORICAL_DAYS = int(os.getenv("HISTORICAL_DAYS", "3650"))
TOP_ETF_COUNT = int(os.getenv("TOP_ETF_COUNT", "10"))
TIMEZONE = os.getenv("TIMEZONE", "Asia/Kolkata")

# Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
SQL_DIR = BASE_DIR / "sql"

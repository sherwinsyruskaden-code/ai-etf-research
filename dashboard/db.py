"""
Shared database connection for the Streamlit dashboard.
Read-only: queries only — no writes, no model training.
"""
import os
from pathlib import Path
from functools import lru_cache

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, Engine

# Load .env from the project root (two levels up from dashboard/)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_ROOT / ".env")

_DEFAULT_URL = (
    "postgresql+psycopg://etf_user:etf_password@localhost:5433/etf_research"
)

def get_db_url() -> str:
    return os.getenv("DATABASE_URL", _DEFAULT_URL)


@st.cache_resource
def get_engine() -> Engine:
    """Cached SQLAlchemy engine (read-only queries)."""
    url = get_db_url()
    return create_engine(url, echo=False, pool_pre_ping=True)

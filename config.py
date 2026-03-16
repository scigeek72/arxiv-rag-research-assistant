"""
ArXiv Research Paper RAG System  —  Configuration  (v4)
========================================================
This is a GENERIC RAG system for any arXiv research topic.
To switch topics, change TOPIC_NAME and SEARCH_QUERY — everything else
(data directories, UI labels, system prompts) adapts automatically.

Examples of topics you can use:
  TOPIC_NAME = "Text-to-SQL"         → SQL & NLP research
  TOPIC_NAME = "Transformer Models"  → deep learning architecture papers
  TOPIC_NAME = "Cancer Biology"      → biomedical research papers
  TOPIC_NAME = "Climate Change"      → environmental science papers
  TOPIC_NAME = "Quantum Computing"   → quantum physics & CS papers

API Key loading strategy (tried in order, first match wins):
  1. This project's own .env file    (place a .env file here)
  2. ~/Dropbox/LegalApp/.env         (your existing project)
  3. ~/Dropbox/LegalDocs/.env        (your existing project)
  4. Shell environment variable      (export OPENAI_API_KEY=sk-...)
  5. Falls back to empty string      (USE_OPENAI will then fail validation)
"""

import os
from pathlib import Path


# ── python-dotenv: auto-load .env from first matching path ────────────────────
def _load_dotenv() -> str:
    """
    Search a priority-ordered list of .env file locations.
    Load the first one found and return its path (or '' if none found).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return ""   # python-dotenv not installed — fall back to os.environ

    _this_dir = Path(__file__).resolve().parent
    _home     = Path.home()

    candidate_paths = [
        _this_dir / ".env",                              # 1. this project
        _home / "Dropbox" / "LegalApp"  / ".env",       # 2. LegalApp
        _home / "Dropbox" / "LegalDocs" / ".env",       # 3. LegalDocs
    ]

    for path in candidate_paths:
        if path.exists():
            load_dotenv(dotenv_path=path, override=False)
            # override=False: already-set shell vars are not overwritten.
            return str(path)

    return ""   # no .env found; rely on shell environment


_dotenv_source = _load_dotenv()   # runs once at import time


# ═════════════════════════════════════════════════════════════════════════════
class Config:
    # =========================================================================
    # ★  TOPIC CONFIGURATION  ★
    #    ↓↓  CHANGE THESE TWO FIELDS TO SWITCH RESEARCH DOMAINS  ↓↓
    # =========================================================================

    # Short display name for the research topic — used in UI labels,
    # window titles, system prompts, and log messages.
    TOPIC_NAME: str = "Transformer Models"

    # One-sentence description of the topic — used in LLM system prompts
    # so the model knows what domain it is helping with.
    TOPIC_DESCRIPTION: str = "advancements and architectures in transformer-based deep learning models"

    # Auto-derived URL-safe slug — used to name the data directory.
    # Do NOT edit this line; it updates automatically from TOPIC_NAME.
    TOPIC_SLUG: str = (
        TOPIC_NAME.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(":", "")
    )

    # Example questions shown in the UI chat sidebar.
    # Customise these to match your topic.
    EXAMPLE_QUESTIONS: list = [
        f"What are the main challenges in {TOPIC_NAME}?",
        f"What are recent advances in {TOPIC_NAME}?",
        f"What evaluation metrics are used for {TOPIC_NAME}?",
        f"Compare different approaches to {TOPIC_NAME}",
    ]

    # =========================================================================
    # DIRECTORY SETTINGS
    #   Data is stored under  arxiv_<topic_slug>_data/
    #   e.g. "Text-to-SQL" → arxiv_text_to_sql_data/
    #        "Cancer Biology" → arxiv_cancer_biology_data/
    # =========================================================================
    DATA_DIR:     str = f"arxiv_{TOPIC_SLUG}_data"
    PDF_DIR:      str = os.path.join(DATA_DIR, "pdfs")
    TEXT_DIR:     str = os.path.join(DATA_DIR, "text")
    VECTOR_DB_DIR:str = os.path.join(DATA_DIR, "vector_db")
    LOG_DIR:      str = os.path.join(DATA_DIR, "logs")
    CACHE_DIR:    str = os.path.join(DATA_DIR, "cache")
    METADATA_DIR: str = os.path.join(DATA_DIR, "metadata")

    # =========================================================================
    # LLM PROVIDER TOGGLE
    #   USE_OPENAI = False  →  local LM Studio (default)
    #   USE_OPENAI = True   →  OpenAI API (requires OPENAI_API_KEY)
    # =========================================================================
    USE_OPENAI: bool = True

    # --- OpenAI settings (active when USE_OPENAI = True) ---------------------
    OPENAI_API_KEY: str         = os.getenv("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL: str      = "gpt-4o-mini"           # gpt-4o / o3-mini
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small" # or -large

    # --- Local LM Studio settings (active when USE_OPENAI = False) -----------
    LM_STUDIO_API_BASE: str   = "http://localhost:1234/v1"
    LM_STUDIO_MODEL_NAME: str = "local-model"   # auto-detected at runtime

    # =========================================================================
    # EMBEDDING SETTINGS
    #   USE_OPENAI_EMBEDDINGS = False  → BAAI/bge-large-en-v1.5  (free, local)
    #   USE_OPENAI_EMBEDDINGS = True   → text-embedding-3-small   (needs key)
    #
    #   NOTE: Changing the embedding model requires a full index rebuild:
    #         python build_rag_index_v4.py --rebuild
    # =========================================================================
    USE_OPENAI_EMBEDDINGS: bool = True

    # Local model options (best quality → fastest):
    #   "BAAI/bge-large-en-v1.5"  ← default, best retrieval quality
    #   "all-mpnet-base-v2"         good quality, medium speed
    #   "all-MiniLM-L6-v2"         fastest, lower quality
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"

    # =========================================================================
    # TEXT PROCESSING
    # =========================================================================
    CHUNK_SIZE: int    = 1000
    CHUNK_OVERLAP: int = 200
    BATCH_SIZE: int    = 50

    # =========================================================================
    # DOWNLOAD SETTINGS
    # =========================================================================
    MAX_PAPERS_DEFAULT: int = 200
    DOWNLOAD_TIMEOUT: int   = 30
    DOWNLOAD_DELAY: float   = 0.5
    MAX_RETRIES: int        = 3

    # =========================================================================
    # RAG QUERY SETTINGS
    # =========================================================================
    RETRIEVAL_K: int       = 10    # candidates retrieved before re-ranking
    RERANK_TOP_K: int      = 5     # kept after cross-encoder re-ranking
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int    = 1024

    # --- Hybrid Search (BM25 keyword + Semantic vector) ----------------------
    USE_HYBRID_SEARCH: bool = True
    BM25_WEIGHT: float      = 0.4
    SEMANTIC_WEIGHT: float  = 0.6

    # --- Cross-Encoder Re-ranking --------------------------------------------
    USE_RERANKER: bool  = False   # disabled: requires PyTorch>=2.4 (base env has 2.0.1)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- HyDE (Hypothetical Document Embeddings) -----------------------------
    USE_HYDE: bool = True

    # --- Multi-hop Chain-of-Thought ------------------------------------------
    USE_MULTIHOP: bool     = True
    MULTIHOP_MAX_HOPS: int = 3

    # --- Persistent SQLite Query Cache ---------------------------------------
    USE_PERSISTENT_CACHE: bool = True
    CACHE_DB_FILE: str = os.path.join(DATA_DIR, "cache", "query_cache.db")
    CACHE_MAX_SIZE: int = 500

    # =========================================================================
    # ARXIV SEARCH QUERY
    #   ↓↓  CHANGE THIS when you change TOPIC_NAME  ↓↓
    #
    #   arXiv search syntax reference:
    #   https://arxiv.org/help/api/user-manual#query_details
    #
    #   Other topic examples:
    #     Cancer biology:
    #       '(ti:"cancer" AND (abs:"tumor" OR abs:"oncology"))'
    #     Transformer models:
    #       '(ti:"transformer" OR ti:"attention mechanism" OR abs:"self-attention")'
    #     Climate change:
    #       '(ti:"climate change" OR abs:"global warming" OR abs:"carbon emissions")'
    #     Quantum computing:
    #       '(ti:"quantum computing" OR ti:"qubit" OR abs:"quantum algorithm")'
    # =========================================================================
    SEARCH_QUERY: str = (
        '(ti:"transformer" OR ti:"attention mechanism" OR '
        'ti:"vision transformer" OR ti:"large language model" OR '
        'ti:"self-attention" OR ti:"ViT" OR ti:"BERT" OR ti:"GPT" OR '
        'ti:"diffusion transformer" OR ti:"multimodal transformer") AND '
        '(abs:"transformer" OR abs:"attention" OR abs:"self-attention")'
    )

    # =========================================================================
    # LOGGING SETTINGS
    # =========================================================================
    LOG_LEVEL: str  = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str   = os.path.join(DATA_DIR, "logs", "rag_system.log")

    # =========================================================================
    # CLASS METHODS
    # =========================================================================

    @classmethod
    def create_directories(cls) -> None:
        """Create all required directories if they don't exist."""
        for d in [cls.DATA_DIR, cls.PDF_DIR, cls.TEXT_DIR,
                  cls.VECTOR_DB_DIR, cls.LOG_DIR,
                  cls.CACHE_DIR, cls.METADATA_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def validate_openai_config(cls) -> tuple:
        """Return (is_valid: bool, message: str)."""
        if cls.USE_OPENAI or cls.USE_OPENAI_EMBEDDINGS:
            if not cls.OPENAI_API_KEY:
                return False, (
                    "OPENAI_API_KEY is not set.\n"
                    "Options:\n"
                    "  A) Create a .env file in this project folder:  "
                    "OPENAI_API_KEY=sk-...\n"
                    "  B) export OPENAI_API_KEY=sk-...  in your shell\n"
                    "  C) Ensure one of your existing .env files is reachable."
                )
            if not cls.OPENAI_API_KEY.startswith("sk-"):
                return False, "OPENAI_API_KEY does not look valid (must start with 'sk-')."
        return True, "Configuration valid."

    @classmethod
    def get_api_key_source(cls) -> str:
        """Return a human-readable description of where the API key came from."""
        if not _dotenv_source:
            if cls.OPENAI_API_KEY:
                return "shell environment variable"
            return "not found"
        try:
            rel = Path(_dotenv_source).relative_to(Path.home())
            return f"~/{rel}"
        except ValueError:
            return _dotenv_source

    @classmethod
    def get_active_llm_info(cls) -> dict:
        """Return a dict describing the active LLM backend."""
        if cls.USE_OPENAI:
            return {
                "provider": "OpenAI",
                "model":    cls.OPENAI_CHAT_MODEL,
                "api_base": "https://api.openai.com/v1",
            }
        return {
            "provider": "LM Studio (Local)",
            "model":    cls.LM_STUDIO_MODEL_NAME,
            "api_base": cls.LM_STUDIO_API_BASE,
        }

    @classmethod
    def get_active_embedding_info(cls) -> dict:
        """Return a dict describing the active embedding backend."""
        if cls.USE_OPENAI_EMBEDDINGS:
            return {"provider": "OpenAI", "model": cls.OPENAI_EMBEDDING_MODEL}
        return {"provider": "Local (HuggingFace)", "model": cls.EMBEDDING_MODEL_NAME}

    @classmethod
    def get_system_status(cls) -> dict:
        """Return real-time system stats (PDFs, text files, vector DB, cache)."""
        status: dict = {}
        try:
            status["topic"]            = cls.TOPIC_NAME
            status["pdfs_count"]       = (
                len([f for f in os.listdir(cls.PDF_DIR) if f.endswith(".pdf")])
                if os.path.exists(cls.PDF_DIR) else 0
            )
            status["text_files_count"] = (
                len([f for f in os.listdir(cls.TEXT_DIR) if f.endswith(".txt")])
                if os.path.exists(cls.TEXT_DIR) else 0
            )
            status["vector_db_exists"] = (
                os.path.exists(cls.VECTOR_DB_DIR)
                and bool(os.listdir(cls.VECTOR_DB_DIR))
            )
            if status["vector_db_exists"]:
                db_size = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _dn, files in os.walk(cls.VECTOR_DB_DIR)
                    for f in files
                )
                status["vector_db_size_mb"] = round(db_size / (1024 * 1024), 2)
            else:
                status["vector_db_size_mb"] = 0.0

            status["cache_entries"] = 0
            if os.path.exists(cls.CACHE_DB_FILE):
                import sqlite3
                with sqlite3.connect(cls.CACHE_DB_FILE) as conn:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM query_cache"
                    ).fetchone()
                    status["cache_entries"] = row[0] if row else 0

            status["llm_provider"]       = "OpenAI" if cls.USE_OPENAI else "LM Studio"
            status["embedding_provider"] = "OpenAI" if cls.USE_OPENAI_EMBEDDINGS else "Local"
            status["active_llm_model"]   = (
                cls.OPENAI_CHAT_MODEL if cls.USE_OPENAI else cls.LM_STUDIO_MODEL_NAME
            )
            status["active_embed_model"] = (
                cls.OPENAI_EMBEDDING_MODEL
                if cls.USE_OPENAI_EMBEDDINGS
                else cls.EMBEDDING_MODEL_NAME
            )
        except Exception as e:
            status["error"] = str(e)
        return status

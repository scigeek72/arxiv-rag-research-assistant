"""
ArXiv Research Paper RAG  —  Query Engine  v3
==============================================
Generic query engine for any arXiv research topic.
Change Config.TOPIC_NAME to switch domains — prompts update automatically.

New in v3 (over query_rag_lmstudio_v2.py):
  ✅  OpenAI / LM Studio toggle (single Config flag)
  ✅  OpenAI / local embedding toggle
  ✅  Hybrid Search  —  BM25 keyword + semantic vector (Reciprocal Rank Fusion)
  ✅  HyDE           —  Hypothetical Document Embeddings for better recall
  ✅  Cross-Encoder Re-ranking  —  ms-marco MiniLM for precision
  ✅  Persistent SQLite Cache   —  survives restarts, evicts LRU
  ✅  Multi-hop Chain-of-Thought—  decomposes complex queries
  ✅  Enriched source display   —  title, authors, year per citation
  ✅  Backward-compatible wrappers for Gradio / CLI
"""

import os
import json
import time
import sqlite3
import hashlib
import logging
import datetime
from typing import Optional, List, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document
from openai import OpenAI

from config import Config


# ─── Logging ─────────────────────────────────────────────────────────────────

def _setup_logging():
    Config.create_directories()
    log_file = os.path.join(Config.LOG_DIR, "rag_queries_v3.log")
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)

logger = _setup_logging()


# ─── Persistent SQLite Cache ─────────────────────────────────────────────────

class PersistentCache:
    """
    SQLite-backed query cache.
    • Survives application restarts.
    • LRU eviction (evicts 50 oldest when full).
    • Keyed by SHA-256(query + year_filter + mode).
    """

    def __init__(self, db_path: str, max_size: int = 500):
        self.db_path  = db_path
        self.max_size = max_size
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    query     TEXT NOT NULL,
                    response  TEXT NOT NULL,
                    created   REAL NOT NULL,
                    accessed  REAL NOT NULL,
                    hit_count INTEGER DEFAULT 1
                )
            """)
            c.commit()

    @staticmethod
    def _make_key(query: str, year_filter, mode: str) -> str:
        raw = f"{query.strip()}||{year_filter}||{mode}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str,
            year_filter: Optional[int] = None,
            mode: str = "auto") -> Optional[str]:
        key = self._make_key(query, year_filter, mode)
        now = time.time()
        with sqlite3.connect(self.db_path) as c:
            row = c.execute(
                "SELECT response FROM query_cache WHERE cache_key=?", (key,)
            ).fetchone()
            if row:
                c.execute(
                    "UPDATE query_cache SET accessed=?, hit_count=hit_count+1 "
                    "WHERE cache_key=?", (now, key)
                )
                c.commit()
                logger.info(f"Cache HIT: {query[:60]}")
                return row[0]
        return None

    def set(self, query: str, response: str,
            year_filter: Optional[int] = None,
            mode: str = "auto"):
        key = self._make_key(query, year_filter, mode)
        now = time.time()
        with sqlite3.connect(self.db_path) as c:
            count = c.execute(
                "SELECT COUNT(*) FROM query_cache"
            ).fetchone()[0]
            if count >= self.max_size:
                c.execute(
                    "DELETE FROM query_cache WHERE cache_key IN "
                    "(SELECT cache_key FROM query_cache "
                    " ORDER BY accessed ASC LIMIT 50)"
                )
            c.execute(
                "INSERT OR REPLACE INTO query_cache "
                "(cache_key, query, response, created, accessed) "
                "VALUES (?,?,?,?,?)",
                (key, query[:500], response, now, now),
            )
            c.commit()
        logger.debug(f"Cached: {query[:60]}")

    def clear(self):
        with sqlite3.connect(self.db_path) as c:
            c.execute("DELETE FROM query_cache")
            c.commit()
        logger.info("Persistent cache cleared")

    def stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as c:
            count = c.execute(
                "SELECT COUNT(*) FROM query_cache"
            ).fetchone()[0]
            hits = c.execute(
                "SELECT SUM(hit_count) FROM query_cache"
            ).fetchone()[0] or 0
        return {"entries": count, "total_hits": hits, "db_path": self.db_path}


# ─── Embedding Factory ────────────────────────────────────────────────────────

def _get_embeddings():
    if Config.USE_OPENAI_EMBEDDINGS:
        valid, msg = Config.validate_openai_config()
        if not valid:
            raise ValueError(f"OpenAI config error: {msg}")
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError("Run: pip install langchain-openai")
        logger.info(f"Embeddings: OpenAI {Config.OPENAI_EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=Config.OPENAI_EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
        )
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    logger.info(f"Embeddings: local {Config.EMBEDDING_MODEL_NAME}")
    return SentenceTransformerEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)


# ─── LLM Client Factory ──────────────────────────────────────────────────────

def setup_llm_client():
    """
    Return (client, model_name) for the active LLM provider.
    Raises ConnectionError if provider is unreachable.
    """
    if Config.USE_OPENAI:
        valid, msg = Config.validate_openai_config()
        if not valid:
            raise ValueError(f"OpenAI config error: {msg}")
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        model  = Config.OPENAI_CHAT_MODEL
        try:
            client.models.retrieve(model)   # quick connectivity check
        except Exception:
            # models.retrieve may fail for some models; list is safer
            try:
                client.models.list()
            except Exception as e:
                raise ConnectionError(f"Cannot reach OpenAI API: {e}")
        print(f"✅ OpenAI API connected  | Model: {model}")
        logger.info(f"LLM: OpenAI {model}")
        return client, model
    else:
        client = OpenAI(base_url=Config.LM_STUDIO_API_BASE, api_key="not-needed")
        try:
            models = client.models.list()
            model = (models.data[0].id if models.data
                     else Config.LM_STUDIO_MODEL_NAME)
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach LM Studio at {Config.LM_STUDIO_API_BASE}: {e}"
            )
        print(f"✅ LM Studio connected   | Model: {model}")
        logger.info(f"LLM: LM Studio {model}")
        return client, model


# ─── Vector Store + Retrievers ───────────────────────────────────────────────

def load_vector_store(persist_directory: str):
    logger.info(f"Loading vector store from {persist_directory}")
    embeddings = _get_embeddings()
    return Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)


def _load_all_docs_from_chroma(db) -> List[Document]:
    """Fetch all documents from ChromaDB (needed for BM25 index)."""
    try:
        result = db._collection.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(result["documents"], result["metadatas"])
        ]
        logger.info(f"Loaded {len(docs)} docs from Chroma for BM25")
        return docs
    except Exception as e:
        logger.warning(f"Could not load docs for BM25: {e}")
        return []


def build_retrievers(db, all_docs: Optional[List[Document]] = None):
    """
    Build a retriever:
      • Hybrid (EnsembleRetriever)  if USE_HYBRID_SEARCH and docs available
      • Semantic only               otherwise
    """
    semantic = db.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})

    if not Config.USE_HYBRID_SEARCH or not all_docs:
        logger.info("Retrieval mode: semantic only")
        return semantic

    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever

        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = Config.RETRIEVAL_K

        ensemble = EnsembleRetriever(
            retrievers=[bm25, semantic],
            weights=[Config.BM25_WEIGHT, Config.SEMANTIC_WEIGHT],
        )
        logger.info(
            f"Retrieval mode: hybrid "
            f"(BM25={Config.BM25_WEIGHT}, Semantic={Config.SEMANTIC_WEIGHT})"
        )
        return ensemble
    except ImportError:
        logger.warning("rank-bm25 not installed; falling back to semantic search.")
        print("⚠️  rank-bm25 not installed → using semantic search. "
              "Run: pip install rank-bm25")
        return semantic


# ─── Cross-Encoder Re-ranker ─────────────────────────────────────────────────

_reranker = None   # lazy singleton

def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading re-ranker: {Config.RERANKER_MODEL}")
            print(f"🎯 Loading cross-encoder re-ranker ({Config.RERANKER_MODEL})…")
            _reranker = CrossEncoder(Config.RERANKER_MODEL)
        except ImportError:
            logger.warning("sentence-transformers not found; re-ranking disabled.")
            _reranker = None
    return _reranker


def rerank_documents(query: str, docs: List[Document]) -> List[Document]:
    """Re-rank docs with cross-encoder, return top RERANK_TOP_K."""
    if not docs:
        return docs
    reranker = _get_reranker()
    if reranker is None:
        return docs[: Config.RERANK_TOP_K]
    try:
        pairs  = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        result = [d for _, d in ranked[: Config.RERANK_TOP_K]]
        logger.info(f"Re-ranked {len(docs)} → kept {len(result)}")
        return result
    except Exception as e:
        logger.warning(f"Re-ranking failed: {e}")
        return docs[: Config.RERANK_TOP_K]


# ─── HyDE ─────────────────────────────────────────────────────────────────────

def generate_hypothetical_doc(query: str, client, model: str) -> str:
    """
    HyDE: generate a short hypothetical research excerpt for the query,
    then embed THAT instead of the query for retrieval.
    """
    prompt = (
        "Write a concise 2-3 sentence research paper excerpt that would "
        f"directly answer this question about {Config.TOPIC_NAME}:"
        f"\n\nQuestion: {query}\n\nExcerpt:"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        hyp = resp.choices[0].message.content.strip()
        logger.info(f"HyDE: generated {len(hyp)}-char hypothetical doc")
        return hyp
    except Exception as e:
        logger.warning(f"HyDE failed → using original query: {e}")
        return query


# ─── Multi-hop Decomposition ─────────────────────────────────────────────────

def decompose_query(query: str, client, model: str) -> List[str]:
    """
    Decompose a complex query into ≤ MULTIHOP_MAX_HOPS simpler sub-questions.
    Returns [query] if the question is already simple.
    """
    prompt = (
        f"Given this research question:\n\"{query}\"\n\n"
        f"If it requires multiple reasoning steps, break it into at most "
        f"{Config.MULTIHOP_MAX_HOPS} simpler sub-questions.\n"
        "If it is already simple, output only the original question.\n"
        "Output ONLY a numbered list (1. … 2. … ), no extra text."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        raw   = resp.choices[0].message.content.strip()
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        sub_qs = []
        for line in lines:
            if line and line[0].isdigit() and ". " in line:
                sub_qs.append(line.split(". ", 1)[1].strip())
            elif line:
                sub_qs.append(line)
        if not sub_qs:
            return [query]
        sub_qs = sub_qs[: Config.MULTIHOP_MAX_HOPS]
        logger.info(f"Multi-hop: {len(sub_qs)} sub-questions for '{query[:60]}'")
        return sub_qs
    except Exception as e:
        logger.warning(f"Query decomposition failed: {e}")
        return [query]


# ─── Context Formatting ───────────────────────────────────────────────────────

def format_context(docs: List[Document]) -> str:
    pieces = []
    for i, doc in enumerate(docs):
        m      = doc.metadata
        pid    = m.get("paper_id", "Unknown")
        year   = m.get("paper_year", "")
        title  = m.get("title", "")
        authors= m.get("authors", "")

        hdr = f"[Source {i+1}: {pid}"
        if year:
            hdr += f" ({year})"
        if title:
            hdr += f" — {title[:80]}"
        hdr += "]"
        if authors:
            hdr += f"\n  Authors: {authors[:100]}"
        pieces.append(f"{hdr}\n{doc.page_content}")
    return "\n\n---\n\n".join(pieces)


# ─── LLM Call Helper ─────────────────────────────────────────────────────────

def _call_llm(system_msg: str, user_msg: str,
              client, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=Config.LLM_TEMPERATURE,
        max_tokens=Config.LLM_MAX_TOKENS,
    )
    return resp.choices[0].message.content.strip()


# ─── Internal Retrieval Helper ───────────────────────────────────────────────

def _retrieve(query: str, retriever, db,
              client, model: str,
              paper_year: Optional[int],
              use_hyde: bool) -> List[Document]:
    retrieval_query = query
    if use_hyde:
        print("🔮 HyDE: generating hypothetical document…")
        retrieval_query = generate_hypothetical_doc(query, client, model)

    try:
        if paper_year and db:
            return db.similarity_search(
                retrieval_query,
                k=Config.RETRIEVAL_K,
                filter={"paper_year": paper_year},
            )
        return retriever.invoke(retrieval_query)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []


# ─── Main RAG Pipeline ───────────────────────────────────────────────────────

def rag_query_v3(
    query: str,
    retriever,
    llm_client,
    model_name: str,
    paper_year: Optional[int] = None,
    use_cache: bool = True,
    mode: str = "auto",    # "auto" | "standard" | "hyde" | "multihop"
    cache: Optional[PersistentCache] = None,
    db=None,
) -> Dict[str, Any]:
    """
    Full v3 RAG pipeline.

    Returns dict:
      answer        : str
      sources       : List[dict]   — paper_id, year, title, authors, snippet
      sub_questions : List[str]
      mode_used     : str
      cache_hit     : bool
      response_time : float (seconds)
    """
    t0 = time.time()
    logger.info(f"RAG v3 | mode={mode} year={paper_year} | {query[:80]}")

    # ── 1. Check cache ────────────────────────────────────────────────────────
    if use_cache and cache:
        cached = cache.get(query, paper_year, mode)
        if cached:
            try:
                result = json.loads(cached)
                result["cache_hit"]     = True
                result["response_time"] = round(time.time() - t0, 3)
                return result
            except Exception:
                pass   # corrupted entry → re-generate

    # ── 2. Decide active mode ────────────────────────────────────────────────
    if mode == "auto":
        complex_kws = ["compare", "difference", "how does", "why does",
                       "explain", "vs ", "versus", "what are the steps",
                       "trade-off", "tradeoff", "disadvantage"]
        is_complex  = any(kw in query.lower() for kw in complex_kws)
        active_mode = (
            "multihop" if (Config.USE_MULTIHOP and is_complex) else
            "hyde"     if Config.USE_HYDE else
            "standard"
        )
    else:
        active_mode = mode

    # ── 3. Retrieve ──────────────────────────────────────────────────────────
    sub_questions  = [query]
    retrieved_docs: List[Document] = []

    if active_mode == "multihop":
        print("🔀 Multi-hop: decomposing query…")
        sub_questions = decompose_query(query, llm_client, model_name)
        print(f"   Sub-questions: {sub_questions}")
        for sq in sub_questions:
            retrieved_docs.extend(
                _retrieve(sq, retriever, db, llm_client, model_name,
                          paper_year, use_hyde=False)
            )
        # deduplicate
        seen, uniq = set(), []
        for d in retrieved_docs:
            h = hashlib.md5(d.page_content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                uniq.append(d)
        retrieved_docs = uniq
    else:
        use_hyde_flag = (active_mode == "hyde") and Config.USE_HYDE
        retrieved_docs = _retrieve(
            query, retriever, db, llm_client, model_name,
            paper_year, use_hyde=use_hyde_flag,
        )

    # ── 4. Re-rank ───────────────────────────────────────────────────────────
    if Config.USE_RERANKER and retrieved_docs:
        print(f"🎯 Re-ranking {len(retrieved_docs)} chunks…")
        retrieved_docs = rerank_documents(query, retrieved_docs)

    # ── 5. Build context ─────────────────────────────────────────────────────
    if retrieved_docs:
        context = format_context(retrieved_docs)
        print(f"✅ Using {len(retrieved_docs)} chunks as context")
    else:
        context = "No relevant documents found in the knowledge base."
        print("⚠️  No relevant documents found.")

    # ── 6. Prompt & generate ─────────────────────────────────────────────────
    print(f"💬 Generating answer ({active_mode} mode, "
          f"{'OpenAI' if Config.USE_OPENAI else 'LM Studio'})…")

    system_msg = (
        f"You are an expert AI research assistant specialising in {Config.TOPIC_NAME} "
        f"({Config.TOPIC_DESCRIPTION}). "
        "Answer questions accurately using the provided research paper excerpts.\n"
        "Guidelines:\n"
        "1. Base answers primarily on the provided context.\n"
        "2. Cite sources as [Source N] when referencing specific findings.\n"
        "3. State clearly if the context is insufficient.\n"
        "4. Compare multiple papers when relevant.\n"
        "5. Be technically precise about methodologies and evaluation metrics."
    )

    if active_mode == "multihop" and len(sub_questions) > 1:
        user_msg = (
            f"Original question: {query}\n\n"
            f"I decomposed it into sub-questions:\n"
            + "\n".join(f"  {i+1}. {q}" for i, q in enumerate(sub_questions))
            + f"\n\nContext from research papers:\n{context}\n\n"
            "Synthesise a comprehensive answer that builds on all sub-questions."
        )
    else:
        user_msg = (
            f"Context from research papers:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Provide a thorough, well-cited answer."
        )

    try:
        answer = _call_llm(system_msg, user_msg, llm_client, model_name)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        answer = f"❌ Error generating answer: {e}"

    # ── 7. Build result dict ─────────────────────────────────────────────────
    sources = []
    for doc in retrieved_docs:
        m = doc.metadata
        sources.append({
            "paper_id":  m.get("paper_id", "Unknown"),
            "paper_year":m.get("paper_year", ""),
            "title":     m.get("title", ""),
            "authors":   m.get("authors", ""),
            "paper_url": m.get("paper_url", ""),
            "snippet":   (doc.page_content[:200] + "…"
                          if len(doc.page_content) > 200
                          else doc.page_content),
        })

    result = {
        "answer":        answer,
        "sources":       sources,
        "sub_questions": sub_questions,
        "mode_used":     active_mode,
        "cache_hit":     False,
        "response_time": round(time.time() - t0, 3),
    }

    # ── 8. Store in cache ────────────────────────────────────────────────────
    if use_cache and cache:
        cache.set(query, json.dumps(result), paper_year, active_mode)

    logger.info(
        f"Done in {result['response_time']}s | mode={active_mode} "
        f"| sources={len(sources)}"
    )
    return result


# ─── Backward-Compatible Wrapper (for Gradio / old scripts) ──────────────────

def rag_query_enhanced(
    query: str,
    retriever,
    llm_client,
    llm_model_name: str,
    paper_year: Optional[int] = None,
    use_cache: bool = True,
) -> str:
    """Return answer string only — used by gradio_interface.py."""
    cache = (PersistentCache(Config.CACHE_DB_FILE, Config.CACHE_MAX_SIZE)
             if Config.USE_PERSISTENT_CACHE else None)
    result = rag_query_v3(
        query=query,
        retriever=retriever,
        llm_client=llm_client,
        model_name=llm_model_name,
        paper_year=paper_year,
        use_cache=use_cache,
        cache=cache,
    )
    return result["answer"]


# ─── Global Singletons (used by public API + Desktop App) ────────────────────

_db:          Optional[Chroma]           = None
_retriever                                = None
_llm_client:  Optional[OpenAI]           = None
_model_name:  str                        = ""
_cache:       Optional[PersistentCache]  = None


def initialize_rag(vector_db_dir: Optional[str] = None) -> bool:
    """
    Initialize all RAG components.
    Call once on startup (app, CLI, or Gradio).
    Returns True on success.
    """
    global _db, _retriever, _llm_client, _model_name, _cache

    vdir = vector_db_dir or Config.VECTOR_DB_DIR
    try:
        print("📚 Loading vector database…")
        _db = load_vector_store(vdir)

        all_docs = (
            _load_all_docs_from_chroma(_db) if Config.USE_HYBRID_SEARCH else None
        )
        _retriever = build_retrievers(_db, all_docs)

        print("🔌 Setting up LLM client…")
        _llm_client, _model_name = setup_llm_client()

        _cache = (
            PersistentCache(Config.CACHE_DB_FILE, Config.CACHE_MAX_SIZE)
            if Config.USE_PERSISTENT_CACHE else None
        )

        logger.info("RAG v3 initialised successfully")
        return True
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        print(f"❌ Init error: {e}")
        return False


def rag_query(
    query: str,
    year_filter: Optional[int] = None,
    mode: str = "auto",
) -> Dict[str, Any]:
    """Public query entry-point used by the desktop app and CLI."""
    if not _retriever or not _llm_client:
        return {
            "answer":        "RAG system not initialised. Call initialize_rag() first.",
            "sources":       [],
            "sub_questions": [query],
            "mode_used":     "error",
            "cache_hit":     False,
            "response_time": 0.0,
        }
    return rag_query_v3(
        query=query,
        retriever=_retriever,
        llm_client=_llm_client,
        model_name=_model_name,
        paper_year=year_filter,
        use_cache=True,
        mode=mode,
        cache=_cache,
        db=_db,
    )


# ─── CLI Pretty-print ────────────────────────────────────────────────────────

def _print_result(result: Dict, year: Optional[int] = None):
    print(f"\n{'=' * 65}")
    if year:
        print(f"📅 Year filter : {year}")
    print(
        f"🔧 Mode: {result['mode_used']}  |  "
        f"Cache: {'HIT ⚡' if result['cache_hit'] else 'MISS'}  |  "
        f"Time: {result['response_time']}s"
    )
    if len(result["sub_questions"]) > 1:
        print(f"🔀 Sub-questions:")
        for i, q in enumerate(result["sub_questions"], 1):
            print(f"   {i}. {q}")

    print(f"\n💬 Answer:\n{result['answer']}")

    if result["sources"]:
        seen, printed = set(), 0
        print(f"\n📚 Sources ({len(result['sources'])} chunks):")
        for src in result["sources"]:
            pid = src["paper_id"]
            if pid not in seen:
                seen.add(pid)
                yr  = f" ({src['paper_year']})" if src["paper_year"] else ""
                ttl = f" — {src['title'][:60]}" if src["title"] else ""
                url = f"\n      {src['paper_url']}" if src["paper_url"] else ""
                print(f"  • {pid}{yr}{ttl}{url}")
    print("=" * 65)


# ─── CLI Entry-point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print(f"  {Config.TOPIC_NAME} — RAG Query System  v3")
    print("=" * 65)
    print(f"  Topic         : {Config.TOPIC_NAME}")
    print(f"  LLM           : {'OpenAI (' + Config.OPENAI_CHAT_MODEL + ')' if Config.USE_OPENAI else 'LM Studio (' + Config.LM_STUDIO_MODEL_NAME + ')'}")
    print(f"  Embeddings    : {'OpenAI' if Config.USE_OPENAI_EMBEDDINGS else Config.EMBEDDING_MODEL_NAME}")
    print(f"  Hybrid Search : {'ON' if Config.USE_HYBRID_SEARCH else 'OFF'}")
    print(f"  HyDE          : {'ON' if Config.USE_HYDE else 'OFF'}")
    print(f"  Re-ranking    : {'ON' if Config.USE_RERANKER else 'OFF'}")
    print(f"  Multi-hop     : {'ON' if Config.USE_MULTIHOP else 'OFF'}")
    print(f"  Persist Cache : {'ON' if Config.USE_PERSISTENT_CACHE else 'OFF'}")
    print()

    if not initialize_rag():
        print("Initialisation failed. Exiting.")
        exit(1)

    HELP = """
Commands:
  /year <YYYY> <question>    — filter by publication year
  /mode <auto|standard|hyde|multihop>  — change retrieval mode
  /clear_cache               — clear persistent cache
  /cache_stats               — show cache statistics
  /status                    — show system status
  /help                      — show this message
  exit                       — quit
"""
    print(HELP)

    current_mode = "auto"
    while True:
        try:
            raw = input("Your Query: ").strip()
            if not raw:
                continue

            if raw.lower() in ("exit", "/quit"):
                print("Goodbye!")
                break
            elif raw == "/help":
                print(HELP)
            elif raw == "/clear_cache":
                if _cache:
                    _cache.clear()
                print("✅ Cache cleared.")
            elif raw == "/cache_stats":
                if _cache:
                    s = _cache.stats()
                    print(f"  entries    : {s['entries']}")
                    print(f"  total hits : {s['total_hits']}")
                else:
                    print("Cache not enabled.")
            elif raw == "/status":
                s = Config.get_system_status()
                print("\n📊 System Status:")
                for k, v in s.items():
                    print(f"  {k}: {v}")
            elif raw.startswith("/mode "):
                current_mode = raw.split()[1]
                print(f"✅ Mode: {current_mode}")
            elif raw.startswith("/year "):
                parts = raw[6:].split(" ", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    year, q = int(parts[0]), parts[1]
                    _print_result(
                        rag_query(q, year_filter=year, mode=current_mode),
                        year=year,
                    )
                else:
                    print("Usage: /year <YYYY> <question>")
            else:
                _print_result(rag_query(raw, mode=current_mode))

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            logger.error(f"CLI error: {e}")

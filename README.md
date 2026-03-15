# ArXiv Research Paper RAG System

A **generic, topic-agnostic** Retrieval-Augmented Generation (RAG) system that lets you chat with a collection of arXiv research papers. Ask natural-language questions and receive answers grounded in real academic literature.

The system ships pre-configured for **Text-to-SQL** papers, but switching to any other research domain (cancer biology, transformer models, climate science, quantum computing…) requires changing **just two lines** in `config.py`.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [What's New in v4](#whats-new-in-v4)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
   - [Switching Research Topics](#switching-research-topics)
   - [Choosing Your LLM — OpenAI vs LM Studio](#choosing-your-llm)
   - [API Key Setup (.env)](#api-key-setup)
6. [Step 1 — Build the Knowledge Base](#step-1--build-the-knowledge-base)
7. [Step 2 — Run the App](#step-2--run-the-app)
   - [Option A: Desktop App (PyQt6) — Recommended](#option-a-desktop-app-pyqt6--recommended)
   - [Option B: Web App (Gradio)](#option-b-web-app-gradio)
   - [Option C: CLI v3 (query_rag_v3.py)](#option-c-cli-v3)
   - [Option D: Legacy CLI (query_rag_lmstudio_v2.py)](#option-d-legacy-cli)
8. [File Structure](#file-structure)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)
11. [Acknowledgements](#acknowledgements)

---

## How It Works

```
┌─────────────────────────────── BUILD (once) ──────────────────────────────────┐
│                                                                                 │
│  config.py          arXiv API          PDFs           ChromaDB                 │
│  SEARCH_QUERY  ──►  search    ──►  download  ──►  embed & index               │
│  TOPIC_NAME                         + OCR           + rich metadata            │
│                                                   (title/authors/abstract)     │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────── QUERY (each time) ─────────────────────────────┐
│                                                                                 │
│  User question                                                                  │
│       │                                                                         │
│       ▼                                                                         │
│  [SQLite cache] ──hit──► return instantly                                       │
│       │ miss                                                                    │
│       ▼                                                                         │
│  HyDE: generate hypothetical answer  ──►  embed it                             │
│       │                                                                         │
│       ▼                                                                         │
│  Hybrid retrieval: BM25 (0.4) + Semantic (0.6)  via Reciprocal Rank Fusion    │
│       │                                                                         │
│       ▼                                                                         │
│  Cross-encoder re-ranking  ──►  top-5 chunks                                   │
│       │                                                                         │
│       ▼                                                                         │
│  Multi-hop (if complex): decompose ──► sub-queries ──► synthesise              │
│       │                                                                         │
│       ▼                                                                         │
│  LLM (OpenAI GPT-4o-mini  OR  local LM Studio)  ──►  cited answer             │
│       │                                                                         │
│       ▼                                                                         │
│  [Save to SQLite cache]                                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## What's New in v4

| Feature | v3 (old) | v4 (current) |
|---|---|---|
| **Topic-agnostic** | ❌ hardcoded Text-to-SQL | ✅ change 2 lines in config.py |
| **LLM provider** | LM Studio only | ✅ OpenAI **or** LM Studio (toggle) |
| **Embeddings** | MiniLM (local only) | ✅ OpenAI / BGE-large / MiniLM |
| **Retrieval** | semantic only | ✅ Hybrid BM25 + Semantic (RRF) |
| **Query expansion** | ❌ | ✅ HyDE |
| **Re-ranking** | ❌ | ✅ Cross-encoder ms-marco-MiniLM |
| **Complex queries** | ❌ | ✅ Multi-hop chain-of-thought |
| **Cache** | in-memory LRU (lost on restart) | ✅ Persistent SQLite |
| **Metadata** | paper_id + year | ✅ title + authors + abstract + DOI + URL |
| **Desktop app** | ❌ | ✅ PyQt6 (+ Tkinter fallback) |
| **Web app** | Gradio (basic) | ✅ Gradio (topic-aware) |
| **API key loading** | env var only | ✅ auto-discovers from sibling projects |
| **Rebuild flag** | ❌ | ✅ `--rebuild` wipes & recreates index |

---

## Prerequisites

| Tool | Purpose | Required for |
|---|---|---|
| Python 3.9+ | runtime | everything |
| pip | package installer | everything |
| OpenAI API key | cloud LLM + embeddings | OpenAI mode |
| LM Studio | local LLM server | local mode |
| Internet connection | arXiv downloads | build step only |

> **Note:** If you cannot run local LLMs (limited hardware), use **OpenAI mode** — set `USE_OPENAI = True` in `config.py`. Cost for 200 papers: ~$0.50 for embeddings, ~$0.002 per query.

---

## Installation

```bash
# 1. Navigate to the project folder
cd nl2sql_RAG_app

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements_v4.txt
```

<details>
<summary>What's installed</summary>

```
arxiv, requests          — arXiv paper search & download
PyMuPDF                  — PDF → text extraction
langchain + community    — RAG orchestration framework
langchain-openai         — OpenAI embeddings & LLM wrapper
sentence-transformers    — local BGE / MiniLM embeddings + cross-encoder re-ranking
chromadb                 — vector database
rank-bm25                — BM25 keyword retrieval
openai                   — LLM API client (works for both OpenAI & LM Studio)
PyQt6                    — desktop GUI
gradio                   — web UI
python-dotenv            — .env file loading
tqdm, numpy              — progress bars, numerics
```
</details>

---

## Configuration

All configuration lives in **`config.py`**. You rarely need to touch any other file.

### Switching Research Topics

```python
# config.py  ← THE ONLY TWO LINES YOU NEED TO CHANGE

TOPIC_NAME: str = "Text-to-SQL"
# Examples:
# TOPIC_NAME = "Cancer Biology"
# TOPIC_NAME = "Transformer Models"
# TOPIC_NAME = "Climate Change"
# TOPIC_NAME = "Quantum Computing"

SEARCH_QUERY: str = (
    '(ti:"Text-to-SQL" OR abs:"Text-to-SQL" OR ...)'
)
# Replace with an appropriate arXiv query for your topic.
# arXiv search syntax: https://arxiv.org/help/api/user-manual#query_details
#
# Examples:
#   Cancer biology:
#     '(ti:"cancer" AND (abs:"tumor" OR abs:"oncology"))'
#   Transformer models:
#     '(ti:"transformer" OR ti:"attention mechanism" OR abs:"self-attention")'
#   Climate change:
#     '(ti:"climate change" OR abs:"global warming" OR abs:"carbon emissions")'
```

Everything else — the data directory, window titles, system prompts, example questions — **adapts automatically** from `TOPIC_NAME`.

---

### Choosing Your LLM

#### OpenAI (cloud) — recommended if you cannot run local models

```python
# config.py
USE_OPENAI = True                          # use OpenAI API
USE_OPENAI_EMBEDDINGS = True               # use OpenAI embeddings  (optional)
OPENAI_CHAT_MODEL = "gpt-4o-mini"         # or "gpt-4o", "o3-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
```

#### LM Studio (local) — free, private, no internet needed at query time

```python
# config.py
USE_OPENAI = False                         # use local LM Studio
USE_OPENAI_EMBEDDINGS = False              # use local BGE-large embeddings
LM_STUDIO_API_BASE = "http://localhost:1234/v1"
```

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a GGUF model (e.g., Qwen2.5 7B Instruct, Mistral 7B)
3. Load the model → click **Local Server** tab → **Start Server**

#### Embedding model options (local)

```python
# Best quality (default):
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Faster, good quality:
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# Fastest (original, lower quality):
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
```

> ⚠️ **Changing the embedding model requires a full index rebuild:**
> `python build_rag_index_v4.py --rebuild`

---

### API Key Setup

The system finds your OpenAI API key automatically using the following priority order:

```
1. This project's own .env file       ← highest priority
2. Sibling project .env files         ← configurable in config.py
3. Shell environment variable         ← lowest priority
```

**Option A — project-specific `.env` file (recommended)**
```bash
cp .env.example .env
# Open .env and set:  OPENAI_API_KEY=sk-...
```

**Option B — reuse a key from another project**
If your API key already lives in another project's `.env`, add that path to the `_load_dotenv()` function in `config.py`:
```python
# config.py  —  _load_dotenv()
candidate_paths = [
    _this_dir / ".env",                          # 1. this project (always first)
    Path.home() / "my_other_project" / ".env",   # 2. add your own paths here
]
```
The first path that exists and contains `OPENAI_API_KEY` wins; the rest are ignored.

**Option C — shell environment variable**
```bash
export OPENAI_API_KEY=sk-...
```

To verify which source was loaded at runtime:
```python
from config import Config
print(Config.get_api_key_source())
# → "/path/to/the/.env/that/was/used"
```

> 🔒 `.env` is listed in `.gitignore` — your key will never be accidentally committed.

---

## Step 1 — Build the Knowledge Base

Run **once** (or again whenever you want to add more papers). Downloads PDFs, extracts text, chunks it, generates embeddings, and stores everything in ChromaDB.

```bash
# Download up to 200 papers (default)
python build_rag_index_v4.py

# Limit to a specific year
python build_rag_index_v4.py --year 2024

# Custom date range
python build_rag_index_v4.py --start-date 2023-01-01 --end-date 2023-12-31

# Limit paper count
python build_rag_index_v4.py --max-papers 50

# Wipe existing index and rebuild from scratch
python build_rag_index_v4.py --rebuild

# Combine options
python build_rag_index_v4.py --year 2024 --max-papers 100
```

**What happens during the build:**

```
📊 System Status:
  📚  PDFs downloaded    : 0
  🗂️   Vector DB exists   : No

Found 187 papers from arXiv.
After date filter: 187 papers remain.
💾 Saved paper metadata for 187 papers total.

📥 Downloading PDFs: 100%|████████████| 187/187
📝 Extracting text:  100%|████████████| 187/187

✂️  Chunking documents... → 4,231 chunks
🏷️  Enriching chunk metadata with paper info...
🔍 Dedup: removed 312 duplicates → 3,919 unique chunks
🔢 Initialising embedding model (first run downloads weights)...
💾 Writing to ChromaDB...  Batch 1/1 (3,919 chunks)…

🎉 Knowledge base build complete!
   Topic     : Text-to-SQL
   Vector DB : arxiv_text_to_sql_data/vector_db
   Metadata  : arxiv_text_to_sql_data/metadata/papers.json
```

> ⏱️ **First run:** Embedding model weights are downloaded (~500MB for BGE-large). Subsequent runs reuse cached weights.

---

## Step 2 — Run the App

Four interfaces are available, from most to least feature-rich:

---

### Option A: Desktop App (PyQt6) — Recommended

A full native desktop application with a chat interface, source panel, settings dialog, and system status tab.

```bash
python desktop_app.py
```

**Features:**
- 💬 Chat with conversation history and styled message bubbles
- 📚 Live source panel — shows cited papers with arXiv links after each answer
- 📅 Year filter and query mode selector (auto / standard / HyDE / multihop) in the toolbar
- ⚙️ Settings dialog — flip OpenAI/LM Studio, enter API key, adjust K, temperature, toggles
- 📊 Status tab — live DB size, cache entries, active model info, feature toggle overview
- 💡 Example questions pulled from `Config.EXAMPLE_QUESTIONS`
- ⚡ Non-blocking worker thread — UI never freezes during LLM calls
- 🖥️ Tkinter fallback — if PyQt6 is not installed, launches a simpler Tkinter window automatically

**Query modes:**

| Mode | When to use |
|---|---|
| `auto` | System picks the best mode (default) |
| `standard` | Direct semantic search — fastest |
| `hyde` | Hypothetical Document Embedding — better for vague questions |
| `multihop` | Decomposes complex questions — best for "compare X and Y" style queries |

**Screenshot layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ 🤖 Text-to-SQL Research Assistant  (RAG v3)    File  Help   │
├─────────────────────────────────────────────────────────────┤
│ 📅 Year: [All Years ▼]  🔧 Mode: [auto ▼]   [🗑 Clear Chat] │
├──────────────────────────────────┬──────────────────────────┤
│                                  │ 📚 Sources               │
│  You: What are challenges...     │                          │
│                                  │ 📄 2301.12345 (2023)     │
│  Assistant: Based on [Source 1]  │ Schema Linking in NL2SQL │
│  the main challenges are...      │ Authors: Smith et al.    │
│                                  │ Open on arXiv ↗          │
│                                  │ ─────────────────────    │
│                                  │ 📄 2302.67890 (2023)     │
│                                  │ ...                      │
├──────────────────────────────────┴──────────────────────────┤
│  [Ask anything about Text-to-SQL research…     ] [Ask ➤]   │
│                                                              │
│ 💡 Examples: [Main challenges...] [Schema linking?] [...]   │
└─────────────────────────────────────────────────────────────┘
│ Done  |  mode=hyde  |  time=4.2s  |  cache=MISS            │
└─────────────────────────────────────────────────────────────┘
```

---

### Option B: Web App (Gradio)

Browser-based chat interface. No installation of PyQt6 needed.

```bash
python gradio_interface.py
# Opens automatically at http://localhost:7860
```

**Features:**
- Chat interface with retry / undo / clear buttons
- Year filter dropdown (sidebar)
- Example questions sidebar
- Help and Status commands (`help`, `status`, `clear cache`)
- `share=True` in the launch call for a temporary public link

**Accessing manually:** [http://localhost:7860](http://localhost:7860)

> ⚠️ The Gradio interface imports from `query_rag_lmstudio_v2.py` (legacy). For the full v3 pipeline (HyDE, hybrid search, re-ranking, multi-hop), use the Desktop App or CLI v3.

---

### Option C: CLI v3

Full v3 pipeline in the terminal. Great for scripting, SSH sessions, or automation.

```bash
python query_rag_v3.py
```

**Startup output:**
```
═════════════════════════════════════════════════════════════
  Text-to-SQL — RAG Query System  v3
═════════════════════════════════════════════════════════════
  Topic         : Text-to-SQL
  LLM           : OpenAI (gpt-4o-mini)
  Embeddings    : BAAI/bge-large-en-v1.5
  Hybrid Search : ON
  HyDE          : ON
  Re-ranking    : ON
  Multi-hop     : ON
  Persist Cache : ON

📚 Loading vector database…
✅ OpenAI API connected  | Model: gpt-4o-mini
```

**Available commands:**

```bash
# Standard question
Your Query: What are the main challenges in Text-to-SQL?

# Filter by year
Your Query: /year 2023 How does schema linking work?

# Change query mode
Your Query: /mode multihop

# Show cache statistics
Your Query: /cache_stats

# Clear persistent cache
Your Query: /clear_cache

# System status
Your Query: /status

# Help
Your Query: /help

# Quit
Your Query: exit
```

**Sample response:**
```
═════════════════════════════════════════════════════════════
🔧 Mode: hyde  |  Cache: MISS  |  Time: 3.8s
🔮 HyDE: generating hypothetical document…
🎯 Re-ranking 10 chunks…
✅ Using 5 chunks as context

💬 Answer:
The main challenges in Text-to-SQL include schema linking [Source 1],
complex nested queries [Source 2], and cross-domain generalisation [Source 3]...

📚 Sources (3 papers):
  • 2301.12345 (2023) — Schema Linking for Text-to-SQL
      https://arxiv.org/abs/2301.12345
  • 2302.67890 (2023) — Handling Complex Queries
  • 2303.11111 (2023) — Cross-domain Generalisation
═════════════════════════════════════════════════════════════
```

---

### Option D: Legacy CLI

The original CLI from v2. Still works, uses simpler retrieval (no HyDE / re-ranking).

```bash
python query_rag_lmstudio_v2.py
```

**Available commands:**
```bash
Your Query: What are the main challenges?
Your Query: /year 2023 Latest BERT approaches?
Your Query: /clear_cache
Your Query: /help
Your Query: exit
```

> 💡 Use this only if you need to run with LM Studio and don't want to set up OpenAI. For best results, use **CLI v3** or the **Desktop App**.

---

## File Structure

```
nl2sql_RAG_app/
│
├── config.py                      ★ Central config — change topic & LLM here
├── .env.example                   ★ Copy to .env and add your API key
├── .gitignore                     Protects .env and data directories
│
├── build_rag_index_v4.py          ★ Build/update the knowledge base  (use this)
├── query_rag_v3.py                ★ CLI query engine with all v4 features
├── desktop_app.py                 ★ PyQt6 desktop app (+ Tkinter fallback)
├── gradio_interface.py            ★ Gradio web interface
│
├── query_rag_lmstudio_v2.py       Legacy CLI (LM Studio, simpler pipeline)
├── build_rag_index_with_time_range_v3.py   Legacy build script
├── build_rag_index_with_time_range_v2.py   Original build script
├── query_rag_lmstudio.py          Original CLI
│
├── requirements_v4.txt            ★ Current dependencies  (use this)
├── requirements_updated.txt       Legacy
├── requirements.txt               Original
│
├── README.md                      This file
├── README_updated.md              Old README (kept for reference)
│
└── arxiv_<topic_slug>_data/       Auto-created data directory
    ├── pdfs/                      Downloaded PDF files
    ├── text/                      Extracted plain-text files
    ├── vector_db/                 ChromaDB vector index
    ├── cache/
    │   └── query_cache.db         Persistent SQLite query cache
    ├── metadata/
    │   └── papers.json            Full paper metadata (title, authors, DOI…)
    └── logs/
        ├── rag_system.log         Build process logs
        ├── rag_queries_v3.log     v3 query logs
        └── rag_queries.log        Legacy query logs
```

> **Data directory naming:** The folder is auto-named from `TOPIC_NAME`.
> `"Text-to-SQL"` → `arxiv_text_to_sql_data/`
> `"Cancer Biology"` → `arxiv_cancer_biology_data/`

---

## Advanced Features

### HyDE — Hypothetical Document Embeddings

Before retrieving, the LLM generates a short hypothetical research excerpt that *would* answer your question. That excerpt is embedded and used for retrieval instead of the raw query — significantly improving recall for vague or indirect questions.

Toggle: `Config.USE_HYDE = True/False`

---

### Hybrid Search (BM25 + Semantic)

Combines keyword-based BM25 search with dense vector search using **Reciprocal Rank Fusion**:

```
final_rank = 0.4 × BM25_rank  +  0.6 × Semantic_rank
```

BM25 catches exact technical terms (SQL keywords, model names); semantic search catches paraphrased concepts.

Toggle: `Config.USE_HYBRID_SEARCH = True/False`
Weights: `Config.BM25_WEIGHT`, `Config.SEMANTIC_WEIGHT`

---

### Cross-Encoder Re-ranking

After retrieval, a cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores each `(query, chunk)` pair together — much more accurate than cosine similarity. The top `RERANK_TOP_K` chunks are kept.

Toggle: `Config.USE_RERANKER = True/False`

---

### Multi-hop Chain-of-Thought

For complex questions containing words like *"compare"*, *"difference"*, *"explain"*, *"vs"*, the system:
1. Decomposes the question into ≤ 3 sub-questions
2. Retrieves chunks independently for each sub-question
3. Synthesises all answers into one cohesive response

Toggle: `Config.USE_MULTIHOP = True/False`

---

### Persistent SQLite Cache

Query results are stored in `arxiv_<topic>_data/cache/query_cache.db`. Identical queries (same text + year filter + mode) return instantly without calling the LLM.

- LRU eviction when cache exceeds `CACHE_MAX_SIZE` (default 500 entries)
- Survives application restarts
- Clear via `/clear_cache` command or Settings dialog

Toggle: `Config.USE_PERSISTENT_CACHE = True/False`

---

## Troubleshooting

**`OPENAI_API_KEY not set` error**
→ Check `Config.get_api_key_source()` — it tells you exactly which .env file was loaded.
→ Ensure the key starts with `sk-` and your `.env` file is in the right path.

**`Cannot reach OpenAI API`**
→ Check internet connection.
→ Verify the key is valid at [platform.openai.com](https://platform.openai.com).

**`Cannot reach LM Studio`**
→ Open LM Studio → Local Server tab → Start Server.
→ Verify `LM_STUDIO_API_BASE = "http://localhost:1234/v1"` in `config.py`.
→ Ensure a model is loaded in LM Studio.

**`Vector DB is empty / no results`**
→ Run `python build_rag_index_v4.py` first.
→ Check `arxiv_<topic>_data/logs/rag_system.log` for build errors.

**`No papers found after date filter`**
→ Try a wider date range or remove `--year`.
→ Verify `SEARCH_QUERY` in `config.py` is correct for your topic.

**Poor answer quality**
→ Increase `RETRIEVAL_K` (more candidates before re-ranking).
→ Switch to a better embedding model (`BAAI/bge-large-en-v1.5`).
→ Switch to a stronger LLM (`gpt-4o` instead of `gpt-4o-mini`).
→ Use `multihop` mode for complex multi-part questions.

**Changed embedding model, now getting poor results**
→ Rebuild the index: `python build_rag_index_v4.py --rebuild`
→ The embedding model used at build time and query time **must match**.

**`PyQt6` not found — desktop app won't launch**
→ `pip install PyQt6`
→ The app will automatically fall back to the Tkinter version if PyQt6 is missing.

**ChromaDB batch errors on large collections**
→ Reduce `MAX_PAPERS_DEFAULT` in `config.py` or use `--max-papers 100`.
→ The build script automatically batches at 5,000 chunks per write.

**Slow first query after restart**
→ Normal — the cross-encoder model is lazy-loaded on first use (~1–2 seconds).
→ Subsequent queries are faster (model stays in memory).

---

## Acknowledgements

| Library | Purpose |
|---|---|
| [arxiv](https://pypi.org/project/arxiv/) | arXiv paper search & metadata |
| [PyMuPDF](https://pypi.org/project/PyMuPDF/) | PDF text extraction |
| [LangChain](https://github.com/langchain-ai/langchain) | RAG orchestration framework |
| [sentence-transformers](https://www.sbert.net/) | Local embeddings + cross-encoder re-ranking |
| [ChromaDB](https://www.trychroma.com/) | Vector database |
| [rank-bm25](https://github.com/dorianbrown/rank_bm25) | BM25 keyword retrieval |
| [openai](https://pypi.org/project/openai/) | OpenAI & LM Studio API client |
| [PyQt6](https://pypi.org/project/PyQt6/) | Desktop GUI framework |
| [Gradio](https://gradio.app/) | Web interface framework |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | .env file loading |
| [tqdm](https://tqdm.github.io/) | Progress bars |
| [LM Studio](https://lmstudio.ai/) | Local LLM server |

---

*Feel free to extend and adapt this project for your own research domain!*

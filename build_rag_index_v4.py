"""
ArXiv Research Paper RAG  —  Build Index  v4
=============================================
Generic knowledge base builder for any arXiv research topic.
Change Config.TOPIC_NAME + Config.SEARCH_QUERY to switch domains.

Improvements over v3:
  • OpenAI text-embedding-3-small / text-embedding-3-large support
  • BGE large embeddings (BAAI/bge-large-en-v1.5) — best local quality
  • Enriched metadata stored per chunk:
      title, authors, abstract, categories, doi, paper_url, paper_year
  • Paper metadata persisted to JSON (metadata/papers.json) for later look-ups
  • --rebuild flag to wipe and recreate the vector DB from scratch
  • All other v3 features: logging, tqdm, duplicate detection, batch processing
"""

import arxiv
import requests
import os
import json
import time
import fitz          # PyMuPDF
import argparse
import datetime
import hashlib
import logging
from pathlib import Path
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.docstore.document import Document

from config import Config


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging():
    Config.create_directories()
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)

logger = setup_logging()


# ─── Embedding Factory ────────────────────────────────────────────────────────

def get_embeddings():
    """Return the appropriate embedding model based on config."""
    if Config.USE_OPENAI_EMBEDDINGS:
        valid, msg = Config.validate_openai_config()
        if not valid:
            raise ValueError(f"OpenAI config error: {msg}")
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError("Run: pip install langchain-openai")
        logger.info(f"Using OpenAI embeddings: {Config.OPENAI_EMBEDDING_MODEL}")
        print(f"🔗 Using OpenAI embeddings  ({Config.OPENAI_EMBEDDING_MODEL})")
        return OpenAIEmbeddings(
            model=Config.OPENAI_EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
        )
    else:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        logger.info(f"Using local embeddings: {Config.EMBEDDING_MODEL_NAME}")
        print(f"🔗 Using local embeddings   ({Config.EMBEDDING_MODEL_NAME})")
        return SentenceTransformerEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)


# ─── Argument Parsing ────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=f"Build '{Config.TOPIC_NAME}' RAG knowledge base from arXiv papers (v4)."
    )
    parser.add_argument("--start-date", type=str,
                        help="Start date YYYY-MM-DD (inclusive).")
    parser.add_argument("--end-date", type=str,
                        help="End date YYYY-MM-DD (inclusive).")
    parser.add_argument("--year", type=int,
                        help="Shorthand for a full calendar year, e.g. --year 2024.")
    parser.add_argument("--max-papers", type=int,
                        default=Config.MAX_PAPERS_DEFAULT,
                        help=f"Max papers to download (default {Config.MAX_PAPERS_DEFAULT}).")
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe existing vector DB and rebuild from scratch.")
    args = parser.parse_args()

    date_range = None
    if args.year:
        date_range = {
            "start": datetime.datetime(args.year, 1, 1, tzinfo=datetime.timezone.utc),
            "end":   datetime.datetime(args.year, 12, 31, 23, 59, 59,
                                       tzinfo=datetime.timezone.utc),
        }
    elif args.start_date or args.end_date:
        if args.end_date and not args.start_date:
            print("ERROR: --end-date requires --start-date.")
            exit(1)
        date_range = {}
        if args.start_date:
            date_range["start"] = datetime.datetime.strptime(
                args.start_date, "%Y-%m-%d"
            ).replace(tzinfo=datetime.timezone.utc)
        if args.end_date:
            end = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
            date_range["end"] = datetime.datetime(
                end.year, end.month, end.day, 23, 59, 59,
                tzinfo=datetime.timezone.utc,
            )
        else:
            today = datetime.datetime.now()
            date_range["end"] = datetime.datetime(
                today.year, today.month, today.day, 23, 59, 59,
                tzinfo=datetime.timezone.utc,
            )

    return args.max_papers, date_range, args.rebuild


# ─── Status ──────────────────────────────────────────────────────────────────

def print_system_status():
    s = Config.get_system_status()
    print("\n📊 System Status:")
    print(f"  📚  PDFs downloaded    : {s.get('pdfs_count', 0)}")
    print(f"  📄  Text files         : {s.get('text_files_count', 0)}")
    print(f"  🗂️   Vector DB exists   : {'Yes' if s.get('vector_db_exists') else 'No'}")
    print(f"  💾  DB size            : {s.get('vector_db_size_mb', 0)} MB")
    print(f"  🤖  LLM provider       : {s.get('llm_provider', '?')}")
    print(f"  🔢  Embedding provider : {s.get('embedding_provider', '?')}")
    print()


# ─── arXiv Search (v4: full metadata) ────────────────────────────────────────

def search_arxiv(query, max_results=200,
                 sort_criterion=arxiv.SortCriterion.Relevance,
                 date_range=None):
    """
    Search arXiv and return a list of rich paper dicts with full metadata.
    v4 adds: title, authors, abstract, categories, doi, paper_url.
    """
    logger.info(f"arXiv search: '{query[:80]}', max={max_results}")
    if date_range:
        s = date_range.get("start", "")
        e = date_range.get("end", "")
        print(f"Filtering papers {s} → {e}")

    client = arxiv.Client()
    paper_list = []

    try:
        search = arxiv.Search(query=query, max_results=max_results,
                              sort_by=sort_criterion)
        results = list(client.results(search))
        print(f"Found {len(results)} papers from arXiv.")
        logger.info(f"arXiv returned {len(results)} results")

        for result in results:
            if date_range:
                pub = result.published
                if pub.tzinfo:
                    pub = pub.astimezone(datetime.timezone.utc).replace(tzinfo=None)

                start = date_range.get("start")
                end   = date_range.get("end")
                if start and start.tzinfo:
                    start = start.astimezone(datetime.timezone.utc).replace(tzinfo=None)
                if end and end.tzinfo:
                    end = end.astimezone(datetime.timezone.utc).replace(tzinfo=None)

                if start and pub < start:
                    continue
                if end and pub > end:
                    continue

            paper_id = result.entry_id.split("/")[-1]
            authors_str = ", ".join(
                a.name for a in result.authors[:5]
            ) + (" et al." if len(result.authors) > 5 else "")

            paper_list.append({
                "id":         paper_id,
                "title":      result.title.strip(),
                "authors":    authors_str,
                "abstract":   result.summary.replace("\n", " ").strip()[:600],
                "categories": ", ".join(result.categories),
                "doi":        result.doi or "",
                "paper_url":  result.entry_id,
                "published":  result.published.strftime("%Y-%m-%d"),
            })

        print(f"After date filter: {len(paper_list)} papers remain.")
        logger.info(f"Returning {len(paper_list)} papers after filtering")

    except Exception as e:
        logger.error(f"arXiv search error: {e}")
        print(f"ERROR during arXiv search: {e}")

    return paper_list


def save_paper_metadata(paper_list):
    """Persist enriched paper metadata to JSON for later reference."""
    meta_file = os.path.join(Config.METADATA_DIR, "papers.json")
    existing = {}
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    for p in paper_list:
        existing[p["id"]] = p

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved metadata for {len(existing)} papers → {meta_file}")
    print(f"💾 Saved paper metadata for {len(existing)} papers total.")
    return existing


def load_paper_metadata() -> dict:
    """Load the stored paper metadata dict (id → metadata)."""
    meta_file = os.path.join(Config.METADATA_DIR, "papers.json")
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


# ─── Download PDFs ────────────────────────────────────────────────────────────

def download_papers(paper_list, save_dir):
    """Download PDFs for paper_list (list of dicts with 'id')."""
    if not paper_list:
        print("No papers to download.")
        return

    ids = [p["id"] for p in paper_list]
    logger.info(f"Downloading {len(ids)} papers to {save_dir}")
    print(f"\nDownloading {len(ids)} papers...")

    client = arxiv.Client()
    search = arxiv.Search(id_list=ids, query="", max_results=len(ids))
    results = list(client.results(search))

    if len(results) != len(ids):
        print(f"⚠️  Metadata found for {len(results)}/{len(ids)} IDs.")
        logger.warning(f"Metadata mismatch: {len(results)}/{len(ids)}")

    downloaded, failed = 0, []

    for result in tqdm(results, desc="📥 Downloading PDFs"):
        pid = result.entry_id.split("/")[-1]
        pdf_path = os.path.join(save_dir, f"{pid}.pdf")

        if os.path.exists(pdf_path):
            downloaded += 1
            continue

        if result.pdf_url:
            try:
                resp = requests.get(result.pdf_url, stream=True,
                                    timeout=Config.DOWNLOAD_TIMEOUT)
                resp.raise_for_status()
                with open(pdf_path, "wb") as fh:
                    for chunk in resp.iter_content(8192):
                        fh.write(chunk)
                downloaded += 1
                logger.debug(f"Downloaded {pid}")
            except Exception as e:
                logger.error(f"Failed {pid}: {e}")
                failed.append(pid)
        else:
            logger.warning(f"No PDF URL for {pid}")
            failed.append(pid)

        time.sleep(Config.DOWNLOAD_DELAY)

    print(f"\n✅ Downloaded {downloaded}/{len(ids)} papers.")
    if failed:
        print(f"❌ Failed ({len(failed)}): {failed[:10]}")
    logger.info(f"Download complete: {downloaded} ok, {len(failed)} failed")


# ─── PDF → Text ───────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str | None:
    try:
        with fitz.open(pdf_path) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception as e:
        logger.error(f"PDF extraction error {pdf_path}: {e}")
        return None


def process_all_pdfs(pdf_dir: str, text_dir: str):
    """Convert all PDFs to cleaned .txt files."""
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    print(f"\n📄 Extracting text from {len(pdf_files)} PDFs...")
    ok = 0

    for filename in tqdm(pdf_files, desc="📝 Extracting text"):
        txt_name = filename.replace(".pdf", ".txt")
        txt_path = os.path.join(text_dir, txt_name)
        if os.path.exists(txt_path):
            ok += 1
            continue

        text = extract_text_from_pdf(os.path.join(pdf_dir, filename))
        if text:
            cleaned = "\n".join(l.strip() for l in text.split("\n") if l.strip())
            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write(cleaned)
            ok += 1
            logger.debug(f"Extracted {filename}")
        else:
            logger.error(f"Failed to extract {filename}")

    print(f"✅ Text extraction done: {ok}/{len(pdf_files)} files.")
    logger.info(f"Text extraction complete: {ok}/{len(pdf_files)}")


# ─── Deduplication ───────────────────────────────────────────────────────────

def remove_duplicate_chunks(docs):
    seen, unique, n_dup = set(), [], 0
    for doc in docs:
        h = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
        else:
            n_dup += 1
    print(f"🔍 Dedup: removed {n_dup} duplicates → {len(unique)} unique chunks.")
    logger.info(f"Dedup: {n_dup} removed, {len(unique)} kept")
    return unique


# ─── Vector DB Creation (v4: enriched metadata) ──────────────────────────────

def create_vector_database(text_dir: str, persist_directory: str,
                           paper_metadata: dict, rebuild: bool = False):
    """
    Create / update ChromaDB with enriched metadata per chunk.
    paper_metadata: dict of paper_id → {title, authors, abstract, …}
    rebuild: if True, wipe the existing DB first.
    """
    logger.info(f"Vector DB creation started. Model: {Config.EMBEDDING_MODEL_NAME}")

    # ── Optionally wipe existing DB ──
    if rebuild and os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
        os.makedirs(persist_directory, exist_ok=True)
        print("🗑️  Wiped existing vector database for rebuild.")
        logger.info("Vector DB wiped for rebuild")

    # ── Load text documents ──
    print(f"\n📚 Loading text files from '{text_dir}'...")
    loader = DirectoryLoader(text_dir, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"   Loaded {len(documents)} documents.")
    logger.info(f"Loaded {len(documents)} text documents")

    if not documents:
        print("⚠️  No documents found. Skipping vector DB creation.")
        return None

    # ── Chunk ──
    print("✂️  Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
    )
    docs = splitter.split_documents(documents)
    print(f"   Created {len(docs)} chunks.")
    logger.info(f"Created {len(docs)} chunks")

    # ── Enrich metadata (v4 improvement) ──
    print("🏷️  Enriching chunk metadata with paper info...")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "")
        paper_id = os.path.basename(source).replace(".txt", "") if source else f"unknown_{i}"

        # Extract year from arXiv ID (YYMM.NNNNN format)
        paper_year = None
        if paper_id and "." in paper_id:
            try:
                yy = int(paper_id.split(".")[0][:2])
                paper_year = 2000 + yy if yy <= 30 else 1900 + yy
            except (ValueError, IndexError):
                pass

        # Pull full metadata from saved JSON (v4)
        paper_info = paper_metadata.get(paper_id, {})

        doc.metadata.update({
            "paper_id":    paper_id,
            "paper_year":  paper_year,
            "chunk_index": i,
            "source_type": "arxiv_paper",
            "chunk_length": len(doc.page_content),
            # v4 enrichments:
            "title":       paper_info.get("title", ""),
            "authors":     paper_info.get("authors", ""),
            "abstract":    paper_info.get("abstract", "")[:300],  # trim for Chroma
            "categories":  paper_info.get("categories", ""),
            "doi":         paper_info.get("doi", ""),
            "paper_url":   paper_info.get("paper_url", ""),
            "published":   paper_info.get("published", ""),
        })

    logger.info("Metadata enrichment complete")

    # ── Deduplicate ──
    docs = remove_duplicate_chunks(docs)
    if not docs:
        print("⚠️  No chunks remain after dedup. Aborting.")
        return None

    # ── Embeddings ──
    print("🔢 Initialising embedding model (first run downloads weights)...")
    embeddings = get_embeddings()

    # ── Write to ChromaDB in batches ──
    MAX_CHROMA_BATCH = 5000
    print(f"💾 Writing to ChromaDB at '{persist_directory}'...")
    logger.info(f"Writing {len(docs)} chunks to ChromaDB")

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        db = Chroma(persist_directory=persist_directory,
                    embedding_function=embeddings)
        for i in range(0, len(docs), MAX_CHROMA_BATCH):
            batch = docs[i: i + MAX_CHROMA_BATCH]
            batch_no = i // MAX_CHROMA_BATCH + 1
            total_batches = (len(docs) + MAX_CHROMA_BATCH - 1) // MAX_CHROMA_BATCH
            print(f"   Batch {batch_no}/{total_batches}  ({len(batch)} chunks)…")
            db.add_documents(batch)
            logger.info(f"Added batch {batch_no}/{total_batches}")
    else:
        # Fresh DB — create in batches manually
        db = Chroma.from_documents(
            docs[:MAX_CHROMA_BATCH], embeddings,
            persist_directory=persist_directory,
        )
        for i in range(MAX_CHROMA_BATCH, len(docs), MAX_CHROMA_BATCH):
            batch = docs[i: i + MAX_CHROMA_BATCH]
            batch_no = i // MAX_CHROMA_BATCH + 1
            total_batches = (len(docs) + MAX_CHROMA_BATCH - 1) // MAX_CHROMA_BATCH
            print(f"   Batch {batch_no}/{total_batches}  ({len(batch)} chunks)…")
            db.add_documents(batch)
            logger.info(f"Added batch {batch_no}/{total_batches}")

    db.persist()
    print(f"\n✅ Vector database saved to '{persist_directory}'.")
    logger.info("Vector database persisted successfully")
    return db


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"=== Build RAG Index v4 started | topic: {Config.TOPIC_NAME} ===")
    print("=" * 60)
    print(f"  {Config.TOPIC_NAME} — Knowledge Base Builder  v4")
    print("=" * 60)

    Config.create_directories()
    print_system_status()

    # Validate OpenAI config if needed
    valid, msg = Config.validate_openai_config()
    if not valid:
        print(f"❌ {msg}")
        exit(1)

    max_papers, date_range, do_rebuild = parse_arguments()
    logger.info(f"Args: max_papers={max_papers}, rebuild={do_rebuild}")

    # 1. Search arXiv with full metadata
    found_papers = search_arxiv(
        Config.SEARCH_QUERY,
        max_results=max_papers,
        sort_criterion=arxiv.SortCriterion.Relevance,
        date_range=date_range,
    )

    if not found_papers:
        print("No papers found. Cannot proceed.")
        logger.error("No papers found; aborting")
        exit(1)

    # 2. Save enriched metadata to JSON
    all_metadata = save_paper_metadata(found_papers)

    # 3. Download PDFs
    download_papers(found_papers, Config.PDF_DIR)

    # 4. Extract text
    process_all_pdfs(Config.PDF_DIR, Config.TEXT_DIR)

    # 5. Build / update vector database
    db = create_vector_database(
        Config.TEXT_DIR,
        Config.VECTOR_DB_DIR,
        paper_metadata=all_metadata,
        rebuild=do_rebuild,
    )

    if db:
        print(f"\n🎉 {Config.TOPIC_NAME} knowledge base build complete!")
        print(f"   Topic     : {Config.TOPIC_NAME}")
        print(f"   Vector DB : {Config.VECTOR_DB_DIR}")
        print(f"   Metadata  : {Config.METADATA_DIR}/papers.json")
        logger.info("Build complete")
        print_system_status()
    else:
        print("❌ Build failed.")
        logger.error("Build failed")

    logger.info(f"=== Build RAG Index v4 ended | topic: {Config.TOPIC_NAME} ===")

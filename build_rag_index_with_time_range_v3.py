import arxiv
import requests
import os
import time
import fitz  # PyMuPDF
import argparse
import datetime
import hashlib
# === ADDED FOR IMPROVEMENTS ===
import logging
from pathlib import Path
import functools
from tqdm import tqdm
# === END ADDITION ===
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.docstore.document import Document

# === ADDED: Import configuration ===
from config import Config
# === END ADDITION ===

# === ADDED: Setup logging system ===
def setup_logging():
    """Setup logging configuration."""
    Config.create_directories()  # Ensure log directory exists
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()
# === END ADDITION ===

# --- Command Line Argument Parsing ---

def parse_arguments():
    """Parse command line arguments for time range and paper count."""
    parser = argparse.ArgumentParser(description='Build a RAG knowledge base from arXiv papers on Text-to-SQL.')
    
    # Time range arguments
    parser.add_argument('--start-date', type=str, 
                        help='Start date for paper search (YYYY-MM-DD format). Papers published on or after this date will be included.')
    parser.add_argument('--end-date', type=str, 
                        help='End date for paper search (YYYY-MM-DD format). Papers published on or before this date will be included.')
    parser.add_argument('--year', type=int, 
                        help='Specific year to filter papers (e.g., 2024). This is a shorthand alternative to setting start-date and end-date.')
    
    # Paper count argument
    parser.add_argument('--max-papers', type=int, default=Config.MAX_PAPERS_DEFAULT,  # MODIFIED: Use config
                        help=f'Maximum number of papers to download (default: {Config.MAX_PAPERS_DEFAULT})')
    
    args = parser.parse_args()
    
    # Process dates if provided
    date_range = None
    if args.year:
        date_range = {
            'start': datetime.datetime(args.year, 1, 1, tzinfo=datetime.timezone.utc),
            'end': datetime.datetime(args.year, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc)
        }
        logger.info(f"Filtering papers for year {args.year}")
    elif args.start_date or args.end_date:
        date_range = {}
        
        if args.end_date and not args.start_date:
            error_msg = "Error: When specifying an end date (--end-date), you must also specify a start date (--start-date)."
            logger.error(error_msg)
            print(error_msg)
            print("This prevents downloading an excessive number of papers.")
            print("Example: --start-date 2023-01-01 --end-date 2023-12-31")
            exit(1)
        
        if args.start_date:
            try:
                naive_start = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
                date_range['start'] = naive_start.replace(tzinfo=datetime.timezone.utc)
                logger.info(f"Start date set to {args.start_date}")
            except ValueError:
                error_msg = f"Error: Invalid start date format. Please use YYYY-MM-DD format."
                logger.error(error_msg)
                print(error_msg)
                exit(1)
                
            if args.end_date:
                try:
                    naive_end = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
                    naive_end = datetime.datetime(naive_end.year, naive_end.month, naive_end.day, 23, 59, 59)
                    date_range['end'] = naive_end.replace(tzinfo=datetime.timezone.utc)
                    logger.info(f"End date set to {args.end_date}")
                except ValueError:
                    error_msg = f"Error: Invalid end date format. Please use YYYY-MM-DD format."
                    logger.error(error_msg)
                    print(error_msg)
                    exit(1)
            else:
                current_date = datetime.datetime.now()
                current_end = datetime.datetime(current_date.year, current_date.month, current_date.day, 23, 59, 59)
                date_range['end'] = current_end.replace(tzinfo=datetime.timezone.utc)
                logger.info(f"No end date specified. Using current date ({current_date.strftime('%Y-%m-%d')}) as end date.")
                print(f"No end date specified. Using current date ({current_date.strftime('%Y-%m-%d')}) as end date.")
    
    return args.max_papers, date_range

# === ADDED: Status reporting function ===
def print_system_status():
    """Print current system status."""
    status = Config.get_system_status()
    print(f"\n📊 System Status:")
    print(f"📚 Papers downloaded: {status.get('pdfs_count', 0)}")
    print(f"📄 Text files processed: {status.get('text_files_count', 0)}")
    print(f"🔍 Vector database exists: {'Yes' if status.get('vector_db_exists') else 'No'}")
    if status.get('vector_db_size_mb'):
        print(f"💾 Database size: {status.get('vector_db_size_mb')} MB")
    logger.info(f"System status: {status}")
# === END ADDITION ===

# === MODIFIED: Use config for duplicate detection ===
def get_text_hash(text):
    """Create a unique hash fingerprint for text content."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def remove_duplicate_chunks(docs):
    """Remove duplicate document chunks based on content hash."""
    logger.info(f"Starting duplicate detection on {len(docs)} chunks")
    seen_hashes = set()
    unique_docs = []
    duplicates_found = 0
    
    for doc in docs:
        content_hash = get_text_hash(doc.page_content)
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
        else:
            duplicates_found += 1
    
    message = f"Removed {duplicates_found} duplicate chunks out of {len(docs)} total"
    print(message)
    logger.info(message)
    return unique_docs
# === END MODIFICATION ===

# === ADDED: Batch processing for large document sets ===
def process_documents_in_batches(documents, batch_size=None):
    """Process documents in batches to manage memory usage."""
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    
    logger.info(f"Processing {len(documents)} documents in batches of {batch_size}")
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        logger.debug(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        yield batch
# === END ADDITION ===

# --- Function to Search arXiv by Keyword ---

def search_arxiv(query, max_results=200, sort_criterion=arxiv.SortCriterion.Relevance, date_range=None):
    """
    Searches arXiv for papers based on a keyword query and returns a list of paper dictionaries.
    """
    logger.info(f"Starting arXiv search with query: '{query}', max_results: {max_results}")
    
    if date_range:
        start_str = date_range.get('start', 'earliest').strftime("%Y-%m-%d") if isinstance(date_range.get('start', 'earliest'), datetime.datetime) else 'earliest'
        end_str = date_range.get('end', 'latest').strftime("%Y-%m-%d") if isinstance(date_range.get('end', 'latest'), datetime.datetime) else 'latest'
        print(f"Filtering papers from {start_str} to {end_str}")
        logger.info(f"Date range filter: {start_str} to {end_str}")
    
    client = arxiv.Client()
    paper_list = []

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion
        )

        results = list(client.results(search))

        if not results:
            message = "No results found for the specified query."
            print(message)
            logger.warning(message)
        else:
            print(f"Found {len(results)} papers matching the query.")
            logger.info(f"Found {len(results)} papers matching the query")
            
            filtered_results = []
            for result in results:
                include_paper = True
                
                if date_range:
                    published_date = result.published
                    
                    if published_date.tzinfo is not None:
                        published_date = published_date.astimezone(datetime.timezone.utc).replace(tzinfo=None)
                    
                    start_date = None
                    end_date = None
                    
                    if 'start' in date_range:
                        start_date = date_range['start']
                        if start_date.tzinfo is not None:
                            start_date = start_date.astimezone(datetime.timezone.utc).replace(tzinfo=None)
                    
                    if 'end' in date_range:
                        end_date = date_range['end']
                        if end_date.tzinfo is not None:
                            end_date = end_date.astimezone(datetime.timezone.utc).replace(tzinfo=None)
                    
                    if start_date and published_date < start_date:
                        include_paper = False
                    if end_date and published_date > end_date:
                        include_paper = False
                
                if include_paper:
                    filtered_results.append(result)
                    paper_list.append({
                        'id': result.entry_id.split('/')[-1],
                        'title': result.title,
                        'published': result.published
                    })
            
            if len(filtered_results) < len(results):
                message = f"After date filtering: {len(filtered_results)} papers remain within the specified date range."
                print(message)
                logger.info(message)

    except Exception as e:
        error_msg = f"An error occurred during arXiv search: {e}"
        print(error_msg)
        print("Please check your internet connection and ensure the 'arxiv' library is installed correctly.")
        logger.error(error_msg)

    logger.info(f"Returning {len(paper_list)} papers from search")
    return paper_list

# --- Function to Download PDFs by ID ---

def download_papers(ids, save_dir):
    """Downloads PDFs for given arXiv IDs with improved error handling and logging."""
    if not ids:
        message = "No paper IDs provided for download."
        print(message)
        logger.warning(message)
        return

    logger.info(f"Starting download of {len(ids)} papers to {save_dir}")
    print(f"\nAttempting to download {len(ids)} papers by ID...")
    
    client = arxiv.Client()
    search = arxiv.Search(id_list=ids, query="", max_results=len(ids))
    downloaded_count = 0
    failed_downloads = []

    results = list(client.results(search))

    if len(results) != len(ids):
        warning_msg = f"Warning: Found metadata for only {len(results)} out of {len(ids)} requested IDs."
        print(warning_msg)
        logger.warning(warning_msg)

    # === ADDED: Progress bar for downloads ===
    for result in tqdm(results, desc="Downloading papers"):
        paper_id_short = result.entry_id.split('/')[-1]
        pdf_path = os.path.join(save_dir, f"{paper_id_short}.pdf")

        if os.path.exists(pdf_path):
            print(f"PDF for ID {paper_id_short} already exists. Skipping download.")
            logger.debug(f"Skipping existing file: {paper_id_short}")
            downloaded_count += 1
            continue

        if result.pdf_url:
            try:
                print(f"Downloading {paper_id_short} from {result.pdf_url}...")
                logger.debug(f"Downloading {paper_id_short}")
                
                response = requests.get(result.pdf_url, stream=True, timeout=Config.DOWNLOAD_TIMEOUT)
                response.raise_for_status()
                
                with open(pdf_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                print(f"Successfully downloaded {paper_id_short}")
                logger.info(f"Successfully downloaded {paper_id_short}")
                downloaded_count += 1
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Error downloading {paper_id_short}: {e}"
                print(error_msg)
                logger.error(error_msg)
                failed_downloads.append(paper_id_short)
            except Exception as e:
                error_msg = f"An unexpected error occurred downloading {paper_id_short}: {e}"
                print(error_msg)
                logger.error(error_msg)
                failed_downloads.append(paper_id_short)
        else:
            error_msg = f"No PDF URL found for ID {paper_id_short}"
            print(error_msg)
            logger.warning(error_msg)
            failed_downloads.append(paper_id_short)

        time.sleep(Config.DOWNLOAD_DELAY)  # MODIFIED: Use config value
    # === END ADDITION ===

    final_msg = f"Finished downloading. Successfully downloaded {downloaded_count} out of {len(ids)} requested papers."
    print(f"\n{final_msg}")
    logger.info(final_msg)
    
    if failed_downloads:
        print("\nCould not download papers for the following IDs:")
        logger.warning(f"Failed to download {len(failed_downloads)} papers: {failed_downloads}")
        for failed_id in failed_downloads:
            print(f"- {failed_id}")

# --- Step 3: Extract Text from PDFs ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF using PyMuPDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def process_all_pdfs(pdf_dir, text_dir):
    """Processes all PDFs in a directory and saves extracted text."""
    logger.info(f"Starting PDF text extraction from {pdf_dir} to {text_dir}")
    print("\nStarting text extraction from PDFs...")
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    processed_count = 0
    
    # === ADDED: Progress bar for PDF processing ===
    for filename in tqdm(pdf_files, desc="Extracting text from PDFs"):
        pdf_path = os.path.join(pdf_dir, filename)
        text_filename = filename.replace(".pdf", ".txt")
        text_path = os.path.join(text_dir, text_filename)

        if os.path.exists(text_path):
            print(f"Text file for {filename} already exists. Skipping extraction.")
            logger.debug(f"Skipping existing text file: {text_filename}")
            processed_count += 1
            continue

        print(f"Extracting text from {filename}...")
        logger.debug(f"Processing PDF: {filename}")
        text = extract_text_from_pdf(pdf_path)
        
        if text:
            cleaned_text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            print(f"Saved text to {text_filename}")
            logger.info(f"Successfully extracted text from {filename} -> {text_filename}")
            processed_count += 1
        else:
            logger.error(f"Failed to extract text from {filename}")
            print(f"Failed to extract text from {filename}")
    # === END ADDITION ===

    final_msg = f"Finished text extraction. Processed {processed_count} text files."
    print(f"\n{final_msg}")
    logger.info(final_msg)

# === MODIFIED: Enhanced vector database creation with metadata ===
def create_vector_database(text_dir, persist_directory, embedding_model_name):
    """Creates and persists a ChromaDB vector store from text files with enhanced metadata."""
    logger.info(f"Starting vector database creation using model '{embedding_model_name}'")
    print(f"\nStarting vector database creation using model '{embedding_model_name}'...")

    print(f"Loading documents from text files in '{text_dir}'...")
    logger.info(f"Loading documents from {text_dir}")
    
    loader = DirectoryLoader(text_dir, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} documents from text files.")
    logger.info(f"Loaded {len(documents)} documents from text files")

    if not documents:
        message = "No documents loaded from text files. Skipping vector database creation."
        print(message)
        logger.warning(message)
        return None

    print("Chunking documents for vector store...")
    logger.info("Starting document chunking")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,      # MODIFIED: Use config
        chunk_overlap=Config.CHUNK_OVERLAP, # MODIFIED: Use config
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks for vector store.")
    logger.info(f"Created {len(docs)} chunks for vector store")

    # === ADDED: Enhanced metadata for documents ===
    for i, doc in enumerate(docs):
        source_file = doc.metadata.get('source', '')
        paper_id = os.path.basename(source_file).replace('.txt', '') if source_file else f'unknown_{i}'
        
        # Extract year from paper ID if possible (arXiv format: YYMM.NNNNN)
        paper_year = None
        if paper_id and '.' in paper_id:
            try:
                year_part = paper_id.split('.')[0]
                if len(year_part) >= 2:
                    year_prefix = int(year_part[:2])
                    # Convert YY to YYYY (assuming 2000s for 00-30, 1900s for 31-99)
                    paper_year = 2000 + year_prefix if year_prefix <= 30 else 1900 + year_prefix
            except (ValueError, IndexError):
                pass
        
        doc.metadata.update({
            'paper_id': paper_id,
            'chunk_index': i,
            'source_type': 'arxiv_paper',
            'paper_year': paper_year,
            'chunk_length': len(doc.page_content)
        })
    
    logger.info("Enhanced document metadata with paper IDs and years")
    # === END ADDITION ===

    docs = remove_duplicate_chunks(docs)

    if not docs:
        message = "No chunks created from documents. Skipping vector database creation."
        print(message)
        logger.warning(message)
        return None

    print("Initializing embedding model...")
    logger.info(f"Initializing embedding model: {embedding_model_name}")
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    print(f"Creating and persisting ChromaDB to '{persist_directory}'...")
    logger.info(f"Creating vector database at {persist_directory}")

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Loading existing vector database...")
        logger.info("Loading existing vector database")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        if docs:
            print(f"Adding {len(docs)} new chunks to the existing database...")
            logger.info(f"Adding {len(docs)} new chunks to existing database")
            
            # === ADDED: Batch processing for large document sets ===
            batch_size = Config.BATCH_SIZE * 100  # Use larger batches for vector DB (5000)
            max_chroma_batch = 5000  # ChromaDB's safe batch limit
            effective_batch_size = min(batch_size, max_chroma_batch)
            
            for i in range(0, len(docs), effective_batch_size):
                batch = docs[i:i + effective_batch_size]
                print(f"Adding batch {i//effective_batch_size + 1}/{(len(docs) + effective_batch_size - 1)//effective_batch_size} ({len(batch)} chunks)...")
                logger.info(f"Adding batch {i//effective_batch_size + 1} with {len(batch)} chunks")
                db.add_documents(batch)
            # === END ADDITION ===
            
            print("New chunks added.")
    else:
        print("Creating a new vector database...")
        logger.info("Creating new vector database")
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persist_directory
        )
        print("New vector database created.")

    db.persist()
    final_msg = f"Vector database persisted to '{persist_directory}'."
    print(final_msg)
    logger.info(final_msg)
    return db
# === END MODIFICATION ===

# --- Main Execution ---

if __name__ == "__main__":
    logger.info("=== Starting Text-to-SQL RAG Knowledge Base Build ===")
    print("--- Building Text-to-SQL RAG Knowledge Base ---")
    
    # === ADDED: Create directories and show initial status ===
    Config.create_directories()
    print_system_status()
    # === END ADDITION ===
    
    max_papers, date_range = parse_arguments()
    
    logger.info(f"Configuration: max_papers={max_papers}, date_range={date_range}")
    
    # MODIFIED: Use config for search query
    found_papers = search_arxiv(Config.SEARCH_QUERY, max_results=max_papers, 
                               sort_criterion=arxiv.SortCriterion.Relevance,
                               date_range=date_range)

    if not found_papers:
        message = "No papers found matching the search query. Cannot proceed with download and indexing."
        print(message)
        logger.error(message)
    else:
        paper_ids_to_download = [paper['id'] for paper in found_papers]
        print(f"\nProceeding to download {len(paper_ids_to_download)} papers...")
        logger.info(f"Proceeding to download {len(paper_ids_to_download)} papers")

        # MODIFIED: Use config paths
        download_papers(paper_ids_to_download, Config.PDF_DIR)
        process_all_pdfs(Config.PDF_DIR, Config.TEXT_DIR)
        
        vector_db = create_vector_database(Config.TEXT_DIR, Config.VECTOR_DB_DIR, Config.EMBEDDING_MODEL_NAME)

        if vector_db:
            success_msg = f"\nRAG knowledge base setup complete. Vector database stored at: {Config.VECTOR_DB_DIR}"
            print(success_msg)
            print("You can now use this database for retrieval using query_rag_lmstudio.py.")
            logger.info("RAG knowledge base setup completed successfully")
            
            # === ADDED: Final status report ===
            print_system_status()
            # === END ADDITION ===
        else:
            error_msg = "Failed to build the RAG knowledge base."
            print(f"\n{error_msg}")
            logger.error(error_msg)
    
    logger.info("=== Text-to-SQL RAG Knowledge Base Build Complete ===")

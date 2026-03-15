import arxiv
import requests
import os
import time
import fitz  # PyMuPDF
import argparse
import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.docstore.document import Document # Import Document class

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
    parser.add_argument('--max-papers', type=int, default=200,
                        help='Maximum number of papers to download (default: 200)')
    
    args = parser.parse_args()
    
    # Process dates if provided
    date_range = None
    if args.year:
        # If year is provided, set date range for the entire year
        # Use UTC timezone to match arXiv's datetime objects
        import pytz
        utc = pytz.UTC
        date_range = {
            'start': utc.localize(datetime.datetime(args.year, 1, 1)),
            'end': utc.localize(datetime.datetime(args.year, 12, 31, 23, 59, 59))
        }
    elif args.start_date or args.end_date:
        date_range = {}
        
        # Check if only end_date is provided (without start_date) - this is an error
        if args.end_date and not args.start_date:
            print("Error: When specifying an end date (--end-date), you must also specify a start date (--start-date).")
            print("This prevents downloading an excessive number of papers.")
            print("Example: --start-date 2023-01-01 --end-date 2023-12-31")
            exit(1)
        
        # Import pytz for timezone handling
        import pytz
        utc = pytz.UTC
        
        # Process start_date
        if args.start_date:
            try:
                naive_start = datetime.datetime.strptime(args.start_date, "%Y-%m-%d")
                date_range['start'] = utc.localize(naive_start)
            except ValueError:
                print(f"Error: Invalid start date format. Please use YYYY-MM-DD format.")
                exit(1)
                
            # Process end_date within the start_date block
            if args.end_date:
                try:
                    naive_end = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
                    # Set to end of day
                    naive_end = datetime.datetime(naive_end.year, naive_end.month, naive_end.day, 23, 59, 59)
                    date_range['end'] = utc.localize(naive_end)
                except ValueError:
                    print(f"Error: Invalid end date format. Please use YYYY-MM-DD format.")
                    exit(1)
            else:
                # No end_date provided, use current date as default
                current_date = datetime.datetime.now()
                current_end = datetime.datetime(current_date.year, current_date.month, current_date.day, 23, 59, 59)
                date_range['end'] = utc.localize(current_end)
                print(f"No end date specified. Using current date ({current_date.strftime('%Y-%m-%d')}) as end date.")
    
    return args.max_papers, date_range

# --- Configuration ---

# Directory to save downloaded PDFs and the text files
data_dir = "arxiv_text_to_sql_data"
pdf_dir = os.path.join(data_dir, "pdfs")
text_dir = os.path.join(data_dir, "text")
vector_db_dir = os.path.join(data_dir, "vector_db")

os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(text_dir, exist_ok=True)
os.makedirs(vector_db_dir, exist_ok=True)

# Embedding Model - A good local model for semantic similarity
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Or 'all-mpnet-base-v2' (larger, potentially better)

# --- Function to Search arXiv by Keyword ---

def search_arxiv(query, max_results=200, sort_criterion=arxiv.SortCriterion.Relevance, date_range=None):
    """
    Searches arXiv for papers based on a keyword query and returns a list of paper dictionaries.

    Args:
        query (str): The search query string (e.g., 'ti:"Text-to-SQL" OR abs:"Text-to-SQL"').
        max_results (int): The maximum number of search results to retrieve.
        sort_criterion (arxiv.SortCriterion): The criterion to sort the search results.
        date_range (dict, optional): Dictionary containing 'start' and/or 'end' datetime objects to filter results.

    Returns:
        list: A list of dictionaries, where each dictionary contains 'id', 'title', and 'published' date.
              Returns an empty list if no results are found or an error occurs.
    """
    print(f"Searching arXiv for '{query}' with max_results={max_results}...")
    
    # If date range is provided, print the range
    if date_range:
        start_str = date_range.get('start', 'earliest').strftime("%Y-%m-%d") if isinstance(date_range.get('start', 'earliest'), datetime.datetime) else 'earliest'
        end_str = date_range.get('end', 'latest').strftime("%Y-%m-%d") if isinstance(date_range.get('end', 'latest'), datetime.datetime) else 'latest'
        print(f"Filtering papers from {start_str} to {end_str}")
    
    client = arxiv.Client()
    paper_list = []

    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion
        )

        # Fetch all results up to max_results
        results = list(client.results(search))

        if not results:
            print("No results found for the specified query.")
        else:
            print(f"Found {len(results)} papers matching the query.")
            
            # Filter results by date range if provided
            filtered_results = []
            for result in results:
                include_paper = True
                
                if date_range:
                    # Ensure result.published is timezone-aware
                    # If it's naive, assume it's UTC
                    published_date = result.published
                    if published_date.tzinfo is None:
                        import pytz
                        published_date = pytz.UTC.localize(published_date)
                    
                    # Check if paper is within the date range
                    if 'start' in date_range and published_date < date_range['start']:
                        include_paper = False
                    if 'end' in date_range and published_date > date_range['end']:
                        include_paper = False
                
                if include_paper:
                    filtered_results.append(result)
                    paper_list.append({
                        'id': result.entry_id.split('/')[-1],
                        'title': result.title,
                        'published': result.published
                    })
                    # Optional: print found papers during search
                    # print(f"Found: {result.title} (ID: {result.entry_id.split('/')[-1]}) Published: {result.published.strftime('%Y-%m-%d')}")
            
            if len(filtered_results) < len(results):
                print(f"After date filtering: {len(filtered_results)} papers remain within the specified date range.")

    except Exception as e:
        print(f"An error occurred during arXiv search: {e}")
        print("Please check your internet connection and ensure the 'arxiv' library is installed correctly.")

    return paper_list

# --- Function to Download PDFs by ID (Original Logic) ---

def download_papers(ids, save_dir):
    """Downloads PDFs for given arXiv IDs."""
    if not ids:
        print("No paper IDs provided for download.")
        return

    print(f"\nAttempting to download {len(ids)} papers by ID...")
    client = arxiv.Client()
    # Use an empty query with id_list to fetch by ID
    # Fetch metadata for the given IDs
    search = arxiv.Search(id_list=ids, query="", max_results=len(ids))
    downloaded_count = 0
    failed_downloads = []

    # Convert generator to list to iterate multiple times if needed (though not strictly needed here)
    results = list(client.results(search))

    if len(results) != len(ids):
        print(f"Warning: Found metadata for only {len(results)} out of {len(ids)} requested IDs.")
        # You might want to identify which IDs were not found

    for result in results:
        paper_id_short = result.entry_id.split('/')[-1]
        pdf_path = os.path.join(save_dir, f"{paper_id_short}.pdf")

        if os.path.exists(pdf_path):
            print(f"PDF for ID {paper_id_short} already exists. Skipping download.")
            downloaded_count += 1
            continue

        if result.pdf_url:
            try:
                print(f"Downloading {paper_id_short} from {result.pdf_url}...")
                response = requests.get(result.pdf_url, stream=True, timeout=30) # Added timeout
                response.raise_for_status() # Raise an exception for bad status codes
                with open(pdf_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Successfully downloaded {paper_id_short}")
                downloaded_count += 1
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {paper_id_short}: {e}")
                failed_downloads.append(paper_id_short)
            except Exception as e:
                 print(f"An unexpected error occurred downloading {paper_id_short}: {e}")
                 failed_downloads.append(paper_id_short)
        else:
            print(f"No PDF URL found for ID {paper_id_short}")
            failed_downloads.append(paper_id_short)


        time.sleep(0.5) # Be polite to the arXiv API

    print(f"\nFinished downloading. Successfully downloaded {downloaded_count} out of {len(ids)} requested papers.")
    if failed_downloads:
         print("\nCould not download papers for the following IDs:")
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
        print(f"Error extracting text from {pdf_path}: {e}")
        return None

def process_all_pdfs(pdf_dir, text_dir):
    """Processes all PDFs in a directory and saves extracted text."""
    print("\nStarting text extraction from PDFs...")
    processed_count = 0
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            text_filename = filename.replace(".pdf", ".txt")
            text_path = os.path.join(text_dir, text_filename)

            if os.path.exists(text_path):
                 print(f"Text file for {filename} already exists. Skipping extraction.")
                 processed_count += 1
                 continue

            print(f"Extracting text from {filename}...")
            text = extract_text_from_pdf(pdf_path)
            if text:
                # Basic cleaning: remove excessive newlines/whitespace
                cleaned_text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)
                print(f"Saved text to {text_filename}")
                processed_count += 1
            else:
                print(f"Failed to extract text from {filename}")

    print(f"\nFinished text extraction. Processed {processed_count} text files.")


# --- Step 4, 5, & 6: Create Embeddings and Vector Store ---

def create_vector_database(text_dir, persist_directory, embedding_model_name):
    """Creates and persists a ChromaDB vector store from text files."""
    print(f"\nStarting vector database creation using model '{embedding_model_name}'...")

    # Use Langchain's DirectoryLoader to load documents from the text directory
    print(f"Loading documents from text files in '{text_dir}'...")
    # Ensure the loader is configured to load .txt files
    loader = DirectoryLoader(text_dir, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from text files.")

    if not documents:
        print("No documents loaded from text files. Skipping vector database creation.")
        return None

    # Chunking using Langchain's integrated process
    print("Chunking documents for vector store...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Experiment with chunk size
        chunk_overlap=200, # Experiment with overlap
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks for vector store.")

    if not docs:
        print("No chunks created from documents. Skipping vector database creation.")
        return None

    # Initialize the embedding model
    print("Initializing embedding model...")
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # Create and persist the vector store
    print(f"Creating and persisting ChromaDB to '{persist_directory}'...")

    # Check if the database already exists and load it, otherwise create a new one
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Loading existing vector database...")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        # Add new documents if any
        if docs:
             print(f"Adding {len(docs)} new chunks to the existing database...")
             # Chroma can handle duplicates based on content/ID if configured, but add_documents
             # might add duplicates if the same document content is presented again.
             # For a robust update, you might need to track processed files or use Chroma's update methods.
             # For simplicity here, we add all new docs.
             db.add_documents(docs)
             print("New chunks added.")
    else:
        print("Creating a new vector database...")
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persist_directory
        )
        print("New vector database created.")


    db.persist()
    print(f"Vector database persisted to '{persist_directory}'.")
    return db

# --- Main Execution ---

if __name__ == "__main__":
    print("--- Building Text-to-SQL RAG Knowledge Base ---")
    
    # Parse command line arguments
    max_papers, date_range = parse_arguments()
    
    # --- Step 1: Search for Papers to Get IDs ---
    # Use a flexible query that includes multiple variants of Text-to-SQL
    search_query = '(ti:"Text-to-SQL" OR abs:"Text-to-SQL" OR ti:"text-to-sql" OR abs:"text-to-sql" OR ' + \
                  'ti:"NL-to-SQL" OR abs:"NL-to-SQL" OR ti:"nl-to-sql" OR abs:"nl-to-sql" OR ' + \
                  'ti:"natural language to SQL" OR abs:"natural language to SQL" OR ' + \
                  '(ti:text AND ti:sql) OR (abs:text AND abs:sql) OR ' + \
                  '(ti:nl AND ti:sql) OR (abs:nl AND abs:sql))'
    # Use the max_papers from command line
    found_papers = search_arxiv(search_query, max_results=max_papers, 
                               sort_criterion=arxiv.SortCriterion.Relevance,
                               date_range=date_range)

    if not found_papers:
        print("No papers found matching the search query. Cannot proceed with download and indexing.")
    else:
        # Extract just the IDs from the search results
        paper_ids_to_download = [paper['id'] for paper in found_papers]
        print(f"\nProceeding to download {len(paper_ids_to_download)} papers...")

        # --- Step 2: Download PDFs by ID ---
        # Using the original download function that takes a list of IDs
        download_papers(paper_ids_to_download, pdf_dir)

        # --- Step 3: Extract Text ---
        process_all_pdfs(pdf_dir, text_dir)

        # Steps 4, 5, & 6: Create Vector Database
        # The function create_vector_database now handles loading, chunking, embedding, and storing
        vector_db = create_vector_database(text_dir, vector_db_dir, EMBEDDING_MODEL_NAME)

        if vector_db:
            print("\nRAG knowledge base setup complete.")
            print(f"Vector database stored at: {vector_db_dir}")
            print("You can now use this database for retrieval using query_rag_lmstudio.py.")

        else:
            print("\nFailed to build the RAG knowledge base.")

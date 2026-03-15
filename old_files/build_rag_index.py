import arxiv
import requests
import os
import time
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---

# 1. Replace with your actual list of 150+ arXiv paper IDs on Text-to-SQL
# You need to compile this list yourself (e.g., by searching arXiv and copying IDs)
# Example: Find relevant papers on arXiv, look for the "arXiv ID" (e.g., 2401.01234)
# and add it to this list.
paper_ids = [
    "2502.14913", # Placeholder - Replace with real IDs
    "2501.10858", # Placeholder - Replace with real IDs
    "2410.01943v1", # Placeholder - Replace with real IDs
    "2412.10138" # Placeholder - Replace with real IDs
    # ... add all your 150+ IDs here
]

if not paper_ids or all([id.startswith(('2204.00001','2305.00002','2401.00003')) for id in paper_ids]):
    print("ERROR: Please replace the placeholder paper_ids list with your actual list of arXiv IDs.")
    exit()

# Directory to save downloaded PDFs and the text files
data_dir = "arxiv_text_to_sql_data"
pdf_dir = os.path.join(data_dir, "pdfs")
text_dir = os.path.join(data_dir, "text")
vector_db_dir = os.path.join(data_dir, "vector_db")

os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(text_dir, exist_ok=True)
os.makedirs(vector_db_dir, exist_ok=True)

# Embedding Model - A good local model for semantic similarity
# 'all-MiniLM-L6-v2' is a good balance of size/performance for many tasks
# Other options: 'all-mpnet-base-v2' (larger, potentially better)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Step 1 & 2: Download PDFs ---

def download_papers(ids, save_dir):
    """Downloads PDFs for given arXiv IDs."""
    print(f"Attempting to download {len(ids)} papers...")
    client = arxiv.Client()
    # Use an empty query with id_list to fetch by ID
    search = arxiv.Search(id_list=ids, query="", max_results=len(ids))
    downloaded_count = 0

    for result in client.results(search):
        paper_id_short = result.entry_id.split('/')[-1]
        pdf_path = os.path.join(save_dir, f"{paper_id_short}.pdf")

        if os.path.exists(pdf_path):
            print(f"PDF for {paper_id_short} already exists. Skipping download.")
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
            except Exception as e:
                 print(f"An unexpected error occurred downloading {paper_id_short}: {e}")
        else:
            print(f"No PDF URL found for {paper_id_short}")

        time.sleep(0.5) # Be polite to the server

    print(f"\nFinished downloading. Successfully downloaded {downloaded_count} out of {len(ids)} papers.")

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


# --- Step 4: Chunk Text ---

def chunk_text_files(text_dir):
    """Reads text files and splits content into chunks."""
    print("\nStarting text chunking...")
    all_chunks = []
    # Recommended splitter for general text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Experiment with chunk size
        chunk_overlap=200, # Experiment with overlap
        length_function=len,
        is_separator_regex=False,
    )

    for filename in os.listdir(text_dir):
        if filename.endswith(".txt"):
            text_path = os.path.join(text_dir, filename)
            try:
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read()

                if not text.strip():
                    print(f"Skipping empty text file: {filename}")
                    continue

                chunks = text_splitter.create_documents([text])
                # Add source metadata to each chunk
                for chunk in chunks:
                    chunk.metadata['source'] = filename
                all_chunks.extend(chunks)
                print(f"Chunked {filename} into {len(chunks)} chunks.")
            except Exception as e:
                 print(f"Error reading or chunking {filename}: {e}")

    print(f"\nFinished chunking. Created a total of {len(all_chunks)} chunks.")
    return all_chunks

# --- Step 5 & 6: Create Embeddings and Vector Store ---

def create_vector_database(chunks, persist_directory, embedding_model_name):
    """Creates and persists a ChromaDB vector store from text chunks."""
    print(f"\nStarting vector database creation using model '{embedding_model_name}'...")
    if not chunks:
        print("No chunks to process. Skipping vector database creation.")
        return None

    # Initialize the embedding model
    print("Initializing embedding model...")
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # Create and persist the vector store
    print(f"Creating and persisting ChromaDB to '{persist_directory}'...")
    # Chroma.from_documents handles chunking and embedding internally if given raw documents
    # But we already chunked, so we use .add_documents (less common but works)
    # A simpler approach might be to give raw text files to Langchain's DirectoryLoader
    # and then use Chroma.from_documents. However, our current setup allows more control.

    # Let's use the cleaner approach with Langchain's DirectoryLoader
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    print("Loading documents from text files...")
    # Load documents from the text directory
    loader = DirectoryLoader(text_dir, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from text files.")

    # Re-chunking using Langchain's integrated process
    print("Re-chunking documents for vector store...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks for vector store.")


    # Create the Chroma DB from the documents and embeddings
    # This automatically splits, embeds, and adds to the DB
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persist_directory
    )

    db.persist()
    print(f"Vector database created and persisted to '{persist_directory}'.")
    return db

# --- Main Execution ---

if __name__ == "__main__":
    print("--- Building Text-to-SQL RAG Knowledge Base ---")

    # Step 1 & 2: Download PDFs
    download_papers(paper_ids, pdf_dir)

    # Step 3: Extract Text
    process_all_pdfs(pdf_dir, text_dir)

    # Step 4: Create Chunks of vectors to be used in Step 5
    chunks = chunk_text_files(text_dir)

    # Steps 5, & 6: Create Vector Database
    # The function create_vector_database now handles loading, chunking, embedding, and storing
    # using Langchain's DirectoryLoader and Chroma.from_documents for simplicity.
    vector_db = create_vector_database(chunks, vector_db_dir, EMBEDDING_MODEL_NAME)

    if vector_db:
        print("\nRAG knowledge base setup complete.")
        print(f"Vector database stored at: {vector_db_dir}")
        print(f"You can now use this database for retrieval.")

        # --- Example Retrieval (Optional - for testing) ---
        print("\n--- Example Retrieval ---")
        query = "What are the main challenges in Text-to-SQL?"
        print(f"Query: '{query}'")

        # To query the database, load it first
        print("Loading vector database for retrieval...")
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        retriever_db = Chroma(persist_directory=vector_db_dir, embedding_function=embeddings)

        # Perform retrieval
        retriever = retriever_db.as_retriever(search_kwargs={"k": 5}) # Get top 5 relevant chunks
        retrieved_docs = retriever.invoke(query) # Use .invoke() with Langchain Expression Language

        print(f"\nRetrieved {len(retrieved_docs)} relevant chunks:")
        for i, doc in enumerate(retrieved_docs):
            print(f"--- Chunk {i+1} (Source: {doc.metadata.get('source', 'N/A')}) ---")
            print(doc.page_content[:500] + "...") # Print first 500 chars
            print("-" * 20)

        print("\nCopy the retrieved chunks and your query into LM Studio to get an answer.")

    else:
        print("\nFailed to build the RAG knowledge base.")
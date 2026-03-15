import os
import time
import datetime
# === ADDED FOR IMPROVEMENTS ===
import logging
import functools
import hashlib
from typing import Optional, List, Dict, Any
# === END ADDITION ===
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

# === ADDED: Import configuration ===
from config import Config
# === END ADDITION ===

# === ADDED: Setup logging ===
def setup_logging():
    """Setup logging for query system."""
    Config.create_directories()
    
    # Create separate log file for queries
    query_log_file = os.path.join(Config.LOG_DIR, "rag_queries.log")
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(query_log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()
# === END ADDITION ===

# === ADDED: Query caching system ===
class QueryCache:
    """Simple query cache to avoid repeated LLM calls for identical queries."""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def _get_query_hash(self, query: str, context_hash: str) -> str:
        """Create hash for query + context combination."""
        combined = f"{query}||{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_context_hash(self, retrieved_docs: List) -> str:
        """Create hash for retrieved context."""
        context_text = "".join([doc.page_content for doc in retrieved_docs])
        return hashlib.md5(context_text.encode()).hexdigest()
    
    def get(self, query: str, retrieved_docs: List) -> Optional[str]:
        """Get cached response if available."""
        context_hash = self._get_context_hash(retrieved_docs)
        query_hash = self._get_query_hash(query, context_hash)
        
        if query_hash in self.cache:
            self.access_times[query_hash] = time.time()
            logger.info(f"Cache hit for query: {query[:50]}...")
            return self.cache[query_hash]
        
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
    
    def set(self, query: str, retrieved_docs: List, response: str):
        """Cache the response."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        context_hash = self._get_context_hash(retrieved_docs)
        query_hash = self._get_query_hash(query, context_hash)
        
        self.cache[query_hash] = response
        self.access_times[query_hash] = time.time()
        logger.debug(f"Cached response for query: {query[:50]}...")

# Initialize global cache
query_cache = QueryCache()
# === END ADDITION ===

# === ADDED: Query metrics tracking ===
def log_query_metrics(query: str, response_time: float, num_chunks_retrieved: int, cache_hit: bool = False):
    """Log query performance metrics."""
    metrics = {
        'timestamp': datetime.datetime.now().isoformat(),
        'query_length': len(query),
        'response_time': round(response_time, 2),
        'chunks_retrieved': num_chunks_retrieved,
        'cache_hit': cache_hit
    }
    logger.info(f"Query metrics: {metrics}")
# === END ADDITION ===

# --- Load the Vector Database and Retriever ---

def load_retriever(persist_directory: str, embedding_model_name: str):
    """Loads the persisted ChromaDB vector store and creates a retriever."""
    logger.info(f"Loading vector database from '{persist_directory}'...")
    print(f"Loading vector database from '{persist_directory}'...")
    
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        # MODIFIED: Use config for retrieval settings
        retriever = db.as_retriever(search_kwargs={"k": Config.RETRIEVAL_K})
        
        print("Vector database loaded successfully.")
        logger.info("Vector database loaded successfully")
        return retriever
    except Exception as e:
        error_msg = f"Error loading vector database: {e}"
        print(error_msg)
        print("Please ensure the index was built correctly by running the previous script.")
        logger.error(error_msg)
        return None

# === ADDED: Auto model detection ===
def auto_detect_model(api_base: str) -> str:
    """Auto-detect the loaded model from LM Studio."""
    try:
        print("🔍 Auto-detecting model from LM Studio...")
        client = OpenAI(base_url=api_base, api_key="not-needed")
        models = client.models.list()
        
        if models.data and len(models.data) > 0:
            model_name = models.data[0].id
            print(f"✅ Auto-detected model: {model_name}")
            logger.info(f"Auto-detected model: {model_name}")
            return model_name
        else:
            fallback = "gpt-3.5-turbo"
            print(f"⚠️  No models found, using fallback: {fallback}")
            logger.warning("No models found via API, using fallback")
            return fallback
            
    except Exception as e:
        fallback = "gpt-3.5-turbo"
        print(f"⚠️  Model detection failed ({e}), using fallback: {fallback}")
        logger.warning(f"Model detection failed: {e}, using fallback")
        return fallback
# === END ADDITION ===

# --- Setup LLM API Client ---

def setup_llm_client(api_base: str):
    """Sets up the OpenAI-compatible client for LM Studio with auto model detection."""
    logger.info(f"Attempting to connect to LM Studio API at {api_base}...")
    print(f"Attempting to connect to LM Studio API at {api_base}...")
    
    try:
        client = OpenAI(base_url=api_base, api_key="not-needed")
        
        # === ADDED: Test connection and auto-detect model ===
        try:
            # Test if we can connect and detect model
            models = client.models.list()
            print("✅ Successfully connected to LM Studio API")
            logger.info("LM Studio API connection successful")
            
            # Auto-detect model
            detected_model = auto_detect_model(api_base)
            
            # Store detected model for later use
            client._detected_model = detected_model
            
        except Exception as e:
            print(f"⚠️  Connected to API but model detection failed: {e}")
            logger.warning(f"Model detection failed: {e}")
            client._detected_model = "gpt-3.5-turbo"
        # === END ADDITION ===
        
        logger.info("LM Studio API client setup successful")
        return client
    except Exception as e:
        error_msg = f"Error setting up LM Studio API client: {e}"
        print(error_msg)
        print("Please ensure LM Studio is running and the local server is started.")
        logger.error(error_msg)
        return None

# === ADDED: Enhanced RAG query function ===
def rag_query_enhanced(query: str, retriever, llm_client, llm_model_name: str, 
                      paper_year: Optional[int] = None, use_cache: bool = True) -> str:
    """
    Enhanced RAG retrieval with metadata filtering, caching, and improved prompting.
    
    Args:
        query: User question
        retriever: Vector database retriever
        llm_client: LLM API client
        llm_model_name: Model name for LLM
        paper_year: Optional year filter for papers
        use_cache: Whether to use query caching
    """
    start_time = time.time()
    
    if not retriever:
        return "Error: RAG retriever is not initialized."
    if not llm_client:
        return "Error: LLM API client is not initialized."

    logger.info(f"Processing enhanced query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
    print(f"\nProcessing query: '{query}'")

    # === ADDED: Enhanced retrieval with metadata filtering ===
    print("Retrieving relevant documents...")
    logger.info("Starting document retrieval")
    
    # Modify search if year filter is provided
    search_kwargs = {"k": Config.RETRIEVAL_K}
    if paper_year:
        search_kwargs["filter"] = {"paper_year": paper_year}
        logger.info(f"Filtering results for year: {paper_year}")
    
    try:
        # Create a new retriever with updated search kwargs if needed
        if paper_year:
            retriever = retriever.vectorstore.as_retriever(search_kwargs=search_kwargs)
        
        retrieved_docs = retriever.invoke(query)
    except Exception as e:
        logger.error(f"Error during document retrieval: {e}")
        retrieved_docs = []
    # === END ADDITION ===

    if not retrieved_docs:
        print("No relevant documents found.")
        logger.warning("No relevant documents found")
        context = "No specific context found from the knowledge base."
        num_chunks = 0
    else:
        num_chunks = len(retrieved_docs)
        print(f"Retrieved {num_chunks} relevant chunks.")
        logger.info(f"Retrieved {num_chunks} relevant chunks")
        
        # === ADDED: Enhanced context formatting with metadata ===
        context_pieces = []
        for i, doc in enumerate(retrieved_docs):
            metadata = doc.metadata
            paper_id = metadata.get('paper_id', 'Unknown')
            paper_year = metadata.get('paper_year', 'Unknown')
            
            context_piece = f"[Source {i+1}: Paper {paper_id}"
            if paper_year != 'Unknown':
                context_piece += f" ({paper_year})"
            context_piece += f"]\n{doc.page_content}\n"
            
            context_pieces.append(context_piece)
        
        context = "\n---\n".join(context_pieces)
        # === END ADDITION ===

    # === ADDED: Check cache before LLM call ===
    cached_response = None
    if use_cache:
        cached_response = query_cache.get(query, retrieved_docs)
        if cached_response:
            response_time = time.time() - start_time
            log_query_metrics(query, response_time, num_chunks, cache_hit=True)
            return cached_response
    # === END ADDITION ===

    # === MODIFIED: Enhanced system message and prompting ===
    system_message = (
        f"You are an AI research assistant expert in the {Config.TOPIC_NAME} domain "
        f"({Config.TOPIC_DESCRIPTION}). "
        "Use the provided context from research papers to answer questions accurately. "
        "Guidelines:\n"
        "1. Base your answer primarily on the provided context\n"
        "2. If the context doesn't contain enough information, clearly state this\n"
        "3. When possible, mention which papers (by ID/year) support your claims\n"
        "4. Be precise about technical details and methodologies\n"
        "5. If multiple papers have different approaches, compare them briefly\n"
        "6. Focus on recent developments when relevant"
    )

    # Enhanced prompt structure
    prompt = f"""Context from research papers:
{context}

Question: {query}

Please provide a comprehensive answer based on the research papers above. If you reference specific findings, mention the paper source when possible."""
    # === END MODIFICATION ===

    # Send prompt to LM Studio API
    print("Sending prompt to LM Studio LLM...")
    logger.info("Sending prompt to LLM")
    
    try:
        # === MODIFIED: Use auto-detected model with fallback ===
        # Try to use auto-detected model, fallback to config if needed
        model_to_use = getattr(llm_client, '_detected_model', llm_model_name)
        if not model_to_use:
            model_to_use = llm_model_name
            
        logger.debug(f"Using model: {model_to_use}")
        # === END MODIFICATION ===
        
        response = llm_client.chat.completions.create(
            model=model_to_use,  # MODIFIED: Use auto-detected model
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=Config.LLM_TEMPERATURE,  # MODIFIED: Use config
            max_tokens=Config.LLM_MAX_TOKENS     # MODIFIED: Use config
        )
        
        llm_response = response.choices[0].message.content
        
        # === ADDED: Cache the response ===
        if use_cache:
            query_cache.set(query, retrieved_docs, llm_response)
        # === END ADDITION ===
        
        response_time = time.time() - start_time
        log_query_metrics(query, response_time, num_chunks, cache_hit=False)
        
        logger.info(f"Successfully generated response in {response_time:.2f}s")
        return llm_response

    except Exception as e:
        error_msg = f"Error calling LM Studio API: {e}"
        print(error_msg)
        print("Please ensure LM Studio is running, the server is started, and a model is loaded.")
        logger.error(error_msg)
        return "Error: Could not get response from LM Studio LLM."

# === ADDED: Backward compatibility function ===
def rag_query(query: str, retriever, llm_client, llm_model_name: str) -> str:
    """Original RAG query function for backward compatibility."""
    return rag_query_enhanced(query, retriever, llm_client, llm_model_name)
# === END ADDITION ===

# === ADDED: Interactive query enhancements ===
def print_help():
    """Print available commands."""
    help_text = f"""
Available commands:
- Regular question: Just type your question about {Config.TOPIC_NAME}
- Filter by year: /year 2023 What are the latest approaches?
- Clear cache: /clear_cache
- Help: /help
- Exit: exit or /quit
"""
    print(help_text)

def parse_command(user_input: str) -> tuple:
    """Parse user input for special commands."""
    user_input = user_input.strip()
    
    if user_input.startswith('/year '):
        parts = user_input[6:].split(' ', 1)
        if len(parts) == 2 and parts[0].isdigit():
            year = int(parts[0])
            query = parts[1]
            return 'query_with_year', (query, year)
    
    elif user_input == '/clear_cache':
        return 'clear_cache', None
    
    elif user_input == '/help':
        return 'help', None
    
    elif user_input.lower() in ['exit', '/quit']:
        return 'exit', None
    
    else:
        return 'query', user_input
# === END ADDITION ===

# --- Main User Interaction Loop ---

if __name__ == "__main__":
    logger.info(f"=== Starting {Config.TOPIC_NAME} RAG Query System ===")
    
    # MODIFIED: Use config values
    rag_retriever = load_retriever(Config.VECTOR_DB_DIR, Config.EMBEDDING_MODEL_NAME)
    lm_client = setup_llm_client(Config.LM_STUDIO_API_BASE)

    if rag_retriever and lm_client:
        print(f"\n--- {Config.TOPIC_NAME} Knowledge Base Chat ---")
        print(f"Connected to RAG index ({Config.VECTOR_DB_DIR}) and LM Studio API ({Config.LM_STUDIO_API_BASE}).")
        print("Enhanced features: Query caching, metadata filtering, improved responses")
        print(f"Type '/help' for available commands or just ask your questions about {Config.TOPIC_NAME}.")
        logger.info("RAG query system initialized successfully")

        while True:
            try:
                user_input = input("\nYour Query: ").strip()
                
                if not user_input:
                    continue
                
                command_type, data = parse_command(user_input)
                
                if command_type == 'exit':
                    print("Goodbye!")
                    logger.info("User ended session")
                    break
                
                elif command_type == 'help':
                    print_help()
                
                elif command_type == 'clear_cache':
                    query_cache.cache.clear()
                    query_cache.access_times.clear()
                    print("Query cache cleared.")
                    logger.info("Query cache cleared by user")
                
                elif command_type == 'query_with_year':
                    query, year = data
                    print(f"Searching papers from {year} for: {query}")
                    logger.info(f"Query with year filter {year}: {query}")
                    
                    llm_response = rag_query_enhanced(
                        query, rag_retriever, lm_client, 
                        getattr(lm_client, '_detected_model', Config.LM_STUDIO_MODEL_NAME), paper_year=year  # MODIFIED: Use auto-detected model
                    )
                    
                    print(f"\nLLM Response (filtered for {year}):")
                    print(llm_response)
                    print("-" * 50)
                
                elif command_type == 'query':
                    query = data
                    logger.info(f"Standard query: {query}")
                    
                    llm_response = rag_query_enhanced(
                        query, rag_retriever, lm_client, getattr(lm_client, '_detected_model', Config.LM_STUDIO_MODEL_NAME)  # MODIFIED: Use auto-detected model
                    )
                    
                    print("\nLLM Response:")
                    print(llm_response)
                    print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                logger.info("User interrupted session")
                break
            except Exception as e:
                error_msg = f"An error occurred: {e}"
                print(error_msg)
                logger.error(error_msg)

    else:
        error_msg = "Could not initialize RAG system or connect to LM Studio API. Exiting."
        print(f"\n{error_msg}")
        logger.error(error_msg)
    
    logger.info(f"=== {Config.TOPIC_NAME} RAG Query System Ended ===")

# **Text-to-SQL Research Papers RAG System**

This project implements a Retrieval-Augmented Generation (RAG) system to explore and query the knowledge contained within a collection of arXiv research papers focused on the Text-to-SQL task.

Instead of manually reading through dozens or hundreds of papers, this system allows you to ask questions about the field, and an open-source Large Language Model (LLM) running locally via LM Studio will provide answers grounded in the content of the papers.

## **Features**

* **Automated Paper Download:** Downloads PDF papers directly from arXiv based on search criteria and time ranges.  
* **Time-Range Filtering:** Filter papers by specific years or custom date ranges.
* **Comprehensive Search:** Uses multiple search terms to find relevant Text-to-SQL papers.
* **Text Extraction:** Extracts text content from downloaded PDF files.  
* **Knowledge Indexing:** Chunks the extracted text and creates vector embeddings.  
* **Vector Database:** Stores text chunks and embeddings in a local ChromaDB instance for efficient semantic search.  
* **Integrated Querying:** Connects to a local LLM server (like LM Studio) via its API, retrieves relevant context from the vector database based on a user query, and sends the query + context to the LLM to generate an informed answer.

### **🆕 Enhanced Features**

* **Smart Duplicate Detection:** Automatically removes duplicate text chunks using content hashing for better database efficiency.
* **Centralized Configuration:** All settings managed through `config.py` for easy customization.
* **Comprehensive Logging:** Detailed logs for debugging and monitoring system performance.
* **Progress Tracking:** Visual progress bars for downloads, text extraction, and processing.
* **Enhanced Metadata:** Documents include paper IDs, publication years, and chunk information.
* **Query Caching:** Intelligent caching system to speed up repeated queries.
* **Year-Based Filtering:** Filter search results by publication year for focused research.
* **Interactive Commands:** Enhanced query interface with special commands.
* **Batch Processing:** Efficient handling of large document sets and ChromaDB limits.
* **System Status Monitoring:** Real-time status dashboard showing database size and file counts.

## **Prerequisites**

Before you begin, ensure you have the following installed:

1. **Python 3.8+:** [Download and Install Python](https://www.python.org/downloads/)  
2. **pip:** The Python package installer (usually comes with Python).  
3. **Git (Optional but Recommended):** For cloning this repository if it were hosted online.  
4. **LM Studio:** [Download and Install LM Studio](https://lmstudio.ai/)  
5. **A GGUF LLM Model:** Download a suitable GGUF model (like Qwen2.5 7B or 14B Instruct) within LM Studio.

## **Setup**

1. **Get the Code:**  
   * If this code were in a Git repository, you would clone it:  
     ```bash
     git clone <repository_url>  
     cd <repository_directory>
     ```

   * Otherwise, ensure you have these files in the same directory:
     - `config.py`
     - `build_rag_index_with_time_range_v2.py` (or latest version v3)
     - `query_rag_lmstudio.py` (or latest version v2)
     - `requirements.txt` (or latest version requirements_updated.txt)

2. **Install Dependencies:** Open your terminal or command prompt, navigate to the project directory, and install the required Python libraries:  
   ```bash
   pip install -r requirements.txt
   ```
   
   Or manually install:
   ```bash
   pip install arxiv requests PyMuPDF langchain langchain-community sentence-transformers chromadb openai tqdm typing-extensions
   ```

   *(Note: PyMuPDF might require some system dependencies on certain OS. Refer to its documentation if pip install fails).*

## **Usage**

### **Step 1: Build the Knowledge Base (Run Once)**

This script downloads the PDFs, extracts text, chunks the text, generates embeddings, and stores them in a local vector database (`arxiv_text_to_sql_data/vector_db`).

**🆕 Enhanced with logging, progress bars, and duplicate detection!**

```bash
# Download papers with default settings (most recent 200 papers)
python build_rag_index_with_time_range_v2.py

# Download papers from a specific year (e.g., 2024)
python build_rag_index_with_time_range_v2.py --year 2024

# Download papers from a specific date range
python build_rag_index_with_time_range_v2.py --start-date 2023-08-01 --end-date 2024-01-31

# Limit the number of papers to download
python build_rag_index_with_time_range_v2.py --max-papers 50

# Combine date range and paper limit
python build_rag_index_with_time_range_v2.py --start-date 2023-01-01 --end-date 2023-12-31 --max-papers 100
```

**Note:** When specifying an end date, you must also specify a start date to prevent downloading an excessive number of papers.

**🆕 What you'll see:**
- System status dashboard showing current database state
- Progress bars for downloads and text extraction
- Duplicate detection results (e.g., "Removed 203 duplicate chunks out of 1847 total")
- Batch processing for large document sets
- Comprehensive logging in `arxiv_text_to_sql_data/logs/`

This process might take some time depending on your internet connection, the number of papers, and your computer's performance, especially the PDF text extraction and embedding generation steps.

### **Step 2: Start LM Studio Local Server**

Open LM Studio, load the desired GGUF model (e.g., Qwen2.5 7B or 14B Instruct), go to the "Local Server" tab (< > icon), and click "Start Server". Note the server address and port (default is http://localhost:1234).

### **Step 3: Query the Knowledge Base**

This script loads the vector database and allows you to enter questions. It retrieves relevant information and sends it to the LM Studio API to generate an answer based on the retrieved context.

**🆕 Enhanced with caching, year filtering, and interactive commands!**

```bash
python query_rag_lmstudio.py
```

#### **🆕 Interactive Commands**

The enhanced query system supports special commands:

```bash
# Regular questions (enhanced with paper citations)
Your Query: What are the main challenges in Text-to-SQL?

# Filter by publication year
Your Query: /year 2023 What are the latest BERT approaches in Text-to-SQL?

# Get help
Your Query: /help

# Clear query cache
Your Query: /clear_cache

# Exit the system
Your Query: exit
```

#### **🆕 Enhanced Response Format**

Responses now include paper citations and metadata:

```
LLM Response:
Based on the research papers, the main challenges include:

[Source 1: Paper 2301.12345 (2023)]
Schema linking remains a significant challenge, where models must correctly map natural language mentions to database schema elements...

[Source 2: Paper 2302.67890 (2023)]  
Complex nested queries pose difficulties for current approaches...
```

Type **exit** and press Enter to quit the query session.

<span style="color: red;">**Important:** Ensure LM Studio is running and the Local Server is started before querying. The system will show connection errors if LM Studio is not available.</span>

## **File Structure**

Your project directory will look like this after setup:

```
your_project/
├── config.py                              # 🆕 Centralized configuration
├── build_rag_index_with_time_range_v2.py  # Enhanced build script
├── query_rag_lmstudio.py                  # Enhanced query script
├── requirements.txt                       # Updated dependencies
└── arxiv_text_to_sql_data/
    ├── pdfs/                              # Downloaded PDF files
    ├── text/                              # Extracted text files
    ├── vector_db/                         # ChromaDB vector database
    └── logs/                              # 🆕 System and query logs
        ├── rag_system.log                 # Build process logs
        └── rag_queries.log                # Query session logs
```

## **Configuration**

**🆕 Centralized Configuration (`config.py`):**

All system settings are now managed in a single configuration file:

```python
class Config:
    # Directory settings
    DATA_DIR = "arxiv_text_to_sql_data"
    
    # Model settings
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    
    # Processing settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # LM Studio settings
    LM_STUDIO_API_BASE = "http://localhost:1234/v1"
    LM_STUDIO_MODEL_NAME = "qwen2.5-coder-32b-instruct"
    
    # Query settings
    RETRIEVAL_K = 5  # Number of chunks to retrieve
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 1024
```

**Legacy Configuration (if not using config.py):**

You can still adjust parameters by editing the Python files directly:

* **build_rag_index_with_time_range_v2.py**:  
  * Command-line arguments for date range and paper count filtering.
  * `EMBEDDING_MODEL_NAME`: The Sentence Transformer model used for embeddings.  
  * `RecursiveCharacterTextSplitter` parameters (`chunk_size`, `chunk_overlap`): Adjust how text is split.
  
* **query_rag_lmstudio.py**:  
  * `EMBEDDING_MODEL_NAME`: Must match the model used for indexing.  
  * `LM_STUDIO_API_BASE`: Address of the LM Studio local server.  
  * `LM_STUDIO_MODEL_NAME`: Model identifier expected by the LM Studio API.  
  * Retriever `search_kwargs={"k": 5}`: Number of relevant chunks to retrieve.  
  * System and user prompt templates: Modify the instructions given to the LLM.  
  * API call parameters (`temperature`, `max_tokens`): Control LLM response generation.

## **Troubleshooting**

* **pip install errors:** Ensure you have Python and pip correctly installed and that your environment is set up correctly (consider using a virtual environment).  

* **Download Errors:** Check your internet connection and verify your search criteria. arXiv might rate-limit requests if you download too many too quickly (the script includes small delays).  

* **PDF Extraction Errors:** Some PDFs might be corrupt or have unusual formatting that PyMuPDF struggles with. Check the console output during `process_all_pdfs`. You might need manual intervention for problematic files or try alternative PDF libraries.  

* **Vector Database Loading Errors:** Ensure `build_rag_index_with_time_range_v2.py` ran completely without errors and that the `vector_db_dir` exists and contains data.

* **🆕 ChromaDB Batch Size Errors:** The system now automatically handles large document sets by processing them in batches. If you see batch-related errors, the system will retry with smaller batch sizes.

* **LM Studio API Connection Errors:**  
  * **🔴 Most Common Issue:** Verify LM Studio is running and the Local Server is started.
  * Verify `LM_STUDIO_API_BASE` matches the server address/port shown in LM Studio.
  * Check your firewall settings.
  * Ensure a model is loaded in LM Studio.
  * Look at query logs: `arxiv_text_to_sql_data/logs/rag_queries.log`

* **🆕 Year Filtering Issues:** 
  * Not all papers may have extractable publication years.
  * Try different years (2023, 2024) if 2025 returns no results.
  * Check logs for year extraction details.

* **Poor RAG Results:**  
  * The quality of text extraction is crucial. Review the `.txt` files created.  
  * Experiment with `chunk_size` and `chunk_overlap` in `config.py`.
  * Try a different `EMBEDDING_MODEL_NAME` (e.g., "all-mpnet-base-v2").
  * Increase the number of retrieved chunks (`RETRIEVAL_K` in `config.py`).
  * Refine the system message and prompt template sent to the LLM.  
  * The capabilities of the chosen GGUF model in LM Studio also play a significant role.

## **🆕 Performance Features**

* **Query Caching:** Repeated queries return instantly from cache
* **Duplicate Detection:** Reduces database size and improves search quality
* **Batch Processing:** Handles large document collections efficiently
* **Progress Tracking:** Visual feedback during long operations
* **Logging:** Comprehensive debugging and performance monitoring

## **🆕 Monitoring & Logs**

The system now provides detailed logging and monitoring:

* **Build Logs:** `arxiv_text_to_sql_data/logs/rag_system.log`
* **Query Logs:** `arxiv_text_to_sql_data/logs/rag_queries.log`
* **System Status:** Real-time dashboard showing database size and file counts
* **Performance Metrics:** Query response times, cache hit rates, and retrieval statistics

## **Acknowledgements**

This project utilizes the following open-source libraries:

* [arxiv](https://pypi.org/project/arxiv/)  
* [requests](https://pypi.org/project/requests/)  
* [PyMuPDF](https://pypi.org/project/PyMuPDF/) (fitz)  
* [Langchain](https://github.com/langchain-ai/langchain)  
* [Sentence-Transformers](https://www.sbert.net/)  
* [ChromaDB](https://www.trychroma.com/)  
* [openai](https://pypi.org/project/openai/) (for API interaction)
* **🆕** [tqdm](https://tqdm.github.io/) (for progress bars)

Special thanks to the creators of LM Studio for providing an easy way to run local LLMs with an API.

Feel free to extend and modify this project!
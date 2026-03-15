# **Text-to-SQL Research Papers RAG System**

This project implements a Retrieval-Augmented Generation (RAG) system to explore and query the knowledge contained within a collection of arXiv research papers focused on the Text-to-SQL task.

Instead of manually reading through dozens or hundreds of papers, this system allows you to ask questions about the field, and an open-source Large Language Model (LLM) running locally via LM Studio will provide answers grounded in the content of the papers.

## **Features**

* **Automated Paper Download:** Downloads PDF papers directly from arXiv based on a list of IDs.  
* **Text Extraction:** Extracts text content from downloaded PDF files.  
* **Knowledge Indexing:** Chunks the extracted text and creates vector embeddings.  
* **Vector Database:** Stores text chunks and embeddings in a local ChromaDB instance for efficient semantic search.  
* **Integrated Querying:** Connects to a local LLM server (like LM Studio) via its API, retrieves relevant context from the vector database based on a user query, and sends the query \+ context to the LLM to generate an informed answer.

## **Prerequisites**

Before you begin, ensure you have the following installed:

1. **Python 3.8+:** [Download and Install Python](https://www.python.org/downloads/)  
2. **pip:** The Python package installer (usually comes with Python).  
3. **Git (Optional but Recommended):** For cloning this repository if it were hosted online.  
4. **LM Studio:** [Download and Install LM Studio](https://lmstudio.ai/)  
5. **A GGUF LLM Model:** Download a suitable GGUF model (like Qwen2.5 7B or 14B Instruct) within LM Studio.  
6. **Your List of arXiv Paper IDs:** Compile a list of the \~150 arXiv IDs for the Text-to-SQL papers you want to include in your knowledge base.

## **Setup**

1. **Get the Code:**  
   * If this code were in a Git repository, you would clone it:  
     git clone \<repository\_url\>  
     cd \<repository\_directory\>

   * Otherwise, make sure you have the two Python files (build\_rag\_index.py and query\_rag\_lmstudio.py) in the same directory.  
2. **Install Dependencies:** Open your terminal or command prompt, navigate to the project directory, and install the required Python libraries:  
   pip install arxiv requests PyMuPDF langchain langchain-community sentence-transformers chromadb openai

   *(Note: PyMuPDF might require some system dependencies on certain OS. Refer to its documentation if pip install fails).*  
3. **Add Your Paper IDs:** Open the build\_rag\_index.py file and replace the placeholder paper\_ids list with your actual list of \~150+ arXiv IDs.  
   \# build\_rag\_index.py  
   \# ...  
   paper\_ids \= \[  
       "YOUR\_PAPER\_ID\_1",  
       "YOUR\_PAPER\_ID\_2",  
       \# Add all your 150+ paper IDs here  
       "YOUR\_PAPER\_ID\_N"  
   \]  
   \# ...

## **Usage**

### **Step 1: Build the Knowledge Base (Run Once)**

This script downloads the PDFs, extracts text, chunks the text, generates embeddings, and stores them in a local vector database (arxiv\_text\_to\_sql\_data/vector\_db).

python build\_rag\_index.py

This process might take some time depending on your internet connection, the number of papers, and your computer's performance, especially the PDF text extraction and embedding generation steps.

### **Step 2: Start LM Studio Local Server**

Open LM Studio, load the desired GGUF model (e.g., Qwen2.5 7B/14B Instruct), go to the "Local Server" tab (\< \> icon), and click "Start Server". Note the server address and port (default is http://localhost:1234).

### **Step 3: Query the Knowledge Base**

This script loads the vector database and allows you to enter questions. It retrieves relevant information and sends it to the LM Studio API to generate an answer based on the retrieved context.

python query\_rag\_lmstudio.py

The script will prompt you to enter your queries. Type your question and press Enter. The script will then print the LLM's response.

Type exit and press Enter to quit the query session.

<span style= "color: red;" > **Important:** Ensure the LM\_STUDIO\_API\_BASE variable in query\_rag\_lmstudio.py matches the address and port shown in your LM Studio's Local Server tab. You might also need to adjust LM\_STUDIO\_MODEL\_NAME depending on how your specific LM Studio version and model are configured.</span>

## **Code Structure**

* build\_rag\_index.py: Handles downloading, text extraction, chunking, embedding, and creating the ChromaDB vector store. Run this first.  
* query\_rag\_lmstudio.py: Handles loading the vector store, taking user queries, performing retrieval, interacting with the LM Studio API, and displaying the results. Run this after building the index and starting the LM Studio server.

## **Configuration**

You can adjust various parameters by editing the Python files:

* **build\_rag\_index.py**:  
  * paper\_ids: The list of arXiv IDs (mandatory).  
  * data\_dir, pdf\_dir, text\_dir, vector\_db\_dir: Output directories.  
  * EMBEDDING\_MODEL\_NAME: The Sentence Transformer model used for embeddings.  
  * RecursiveCharacterTextSplitter parameters (chunk\_size, chunk\_overlap): Adjust how text is split.  
* **query\_rag\_lmstudio.py**:  
  * vector\_db\_dir: Location of the vector database (must match build\_rag\_index.py).  
  * EMBEDDING\_MODEL\_NAME: Must match the model used for indexing.  
  * LM\_STUDIO\_API\_BASE: Address of the LM Studio local server.  
  * LM\_STUDIO\_MODEL\_NAME: Model identifier expected by the LM Studio API (often flexible).  
  * Retriever search\_kwargs={"k": 5}: Number of relevant chunks to retrieve.  
  * System and user prompt templates: Modify the instructions given to the LLM.  
  * API call parameters (temperature, max\_tokens): Control LLM response generation.

## **Troubleshooting**

* **pip install errors:** Ensure you have Python and pip correctly installed and that your environment is set up correctly (consider using a virtual environment).  
* **Download Errors:** Check your internet connection and verify the arXiv IDs are correct. arXiv might rate-limit requests if you download too many too quickly (the script includes small delays).  
* **PDF Extraction Errors:** Some PDFs might be corrupt or have unusual formatting that PyMuPDF struggles with. Check the console output during process\_all\_pdfs. You might need manual intervention for problematic files or try alternative PDF libraries.  
* **Vector Database Loading Errors:** Ensure build\_rag\_index.py ran completely without errors and that the vector\_db\_dir exists and contains data.  
* **LM Studio API Connection Errors:**  
  * Verify LM Studio is running.  
  * Verify the Local Server is started in LM Studio.  
  * Verify LM\_STUDIO\_API\_BASE in query\_rag\_lmstudio.py matches the server address/port.  
  * Check your firewall.  
  * Ensure a model is loaded in LM Studio.  
* **Poor RAG Results:**  
  * The quality of text extraction is crucial. Review the .txt files created.  
  * Experiment with chunk\_size and chunk\_overlap.  
  * Try a different EMBEDDING\_MODEL\_NAME.  
  * Increase the number of retrieved chunks (k).  
  * Refine the system message and prompt template sent to the LLM.  
  * The capabilities of the chosen GGUF model in LM Studio also play a significant role.

## **Acknowledgements**

This project utilizes the following open-source libraries:

* [arxiv](https://pypi.org/project/arxiv/)  
* [requests](https://pypi.org/project/requests/)  
* [PyMuPDF](https://pypi.org/project/PyMuPDF/) (fitz)  
* [Langchain](https://github.com/langchain-ai/langchain)  
* [Sentence-Transformers](https://www.sbert.net/)  
* [ChromaDB](https://www.trychroma.com/)  
* [openai](https://pypi.org/project/openai/) (for API interaction)

Special thanks to the creators of LM Studio for providing an easy way to run local LLMs with an API.

Feel free to extend and modify this project\!
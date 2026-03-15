import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI # Use the OpenAI client for LM Studio API

# --- Configuration ---

# Directory where your vector database is stored
data_dir = "arxiv_text_to_sql_data"
vector_db_dir = os.path.join(data_dir, "vector_db")

# Embedding Model - MUST BE THE SAME ONE USED FOR INDEXING
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Or 'all-mpnet-base-v2', etc.

# LM Studio API Configuration
# This is the default address and port for the LM Studio local server
LM_STUDIO_API_BASE = "http://localhost:1234/v1"
# You might need to specify the model name EXACTLY as it appears in LM Studio
# after you load it. Or set client.model to a placeholder like "model-name"
# if the API doesn't strictly require it. Check LM Studio API documentation.
# For many GGUF models via LM Studio, a placeholder like 'gpt-3.5-turbo' or
# even 'model-name' works because LM Studio routes to the loaded model.
# Let's try 'model-name' first, but be prepared to change it.
LM_STUDIO_MODEL_NAME = "qwen2.5-coder-32b-instruct" # Placeholder, adjust if needed by LM Studio

# --- Load the Vector Database and Retriever ---

def load_retriever(persist_directory, embedding_model_name):
    """Loads the persisted ChromaDB vector store and creates a retriever."""
    print(f"Loading vector database from '{persist_directory}'...")
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        # Create a retriever instance
        retriever = db.as_retriever(search_kwargs={"k": 5}) # Get top 5 relevant chunks
        print("Vector database loaded successfully.")
        return retriever
    except Exception as e:
        print(f"Error loading vector database: {e}")
        print("Please ensure the index was built correctly by running the previous script.")
        return None

# --- Setup LLM API Client ---

def setup_llm_client(api_base):
    """Sets up the OpenAI-compatible client for LM Studio."""
    print(f"Attempting to connect to LM Studio API at {api_base}...")
    try:
        client = OpenAI(base_url=api_base, api_key="not-needed") # API key is not needed for local LM Studio
        # Test connection (optional but good practice)
        # try:
        #     models = client.models.list()
        #     print(f"Successfully connected to LM Studio API. Available models (according to API): {[m.id for m in models.data]}")
        # except Exception as e:
        #      print(f"Connected to API base, but failed to list models: {e}")
        #      print("This might be okay depending on how LM Studio implements the API.")
        return client
    except Exception as e:
        print(f"Error setting up LM Studio API client: {e}")
        print("Please ensure LM Studio is running and the local server is started.")
        return None

# --- RAG Query Function ---

def rag_query(query, retriever, llm_client, llm_model_name):
    """Performs RAG retrieval and sends prompt to LLM via API."""
    if not retriever:
        return "Error: RAG retriever is not initialized."
    if not llm_client:
        return "Error: LLM API client is not initialized."

    print(f"\nProcessing query: '{query}'")

    # Step 7 & 8: Retrieve relevant documents
    print("Retrieving relevant documents...")
    retrieved_docs = retriever.invoke(query) # Use .invoke() with Langchain Expression Language

    if not retrieved_docs:
        print("No relevant documents found.")
        # Optionally send query to LLM anyway, or return a specific message
        context = "No specific context found from the knowledge base."
    else:
        print(f"Retrieved {len(retrieved_docs)} relevant chunks.")
        # Format the retrieved context
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        # Optional: Add source metadata to context string
        # sources = "\nSources:\n" + "\n".join(list(set([doc.metadata.get('source', 'N/A') for doc in retrieved_docs])))
        # context += sources


    # Step 9: Construct prompt for the LLM
    # System message to guide the LLM's behavior
    system_message = (
        "You are an AI assistant expert in the Text-to-SQL domain. "
        "Use the following provided context to answer the user's question. "
        "If you cannot find the answer within the context, state that you cannot answer based on the provided information. "
        "Do not use your prior knowledge about Text-to-SQL beyond understanding the context. "
        "Cite the source files mentioned in the context where possible (if source metadata is included)."
    )

    # Combine context and user query
    prompt = f"""
    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # Send prompt to LM Studio API
    print("Sending prompt to LM Studio LLM...")
    try:
        # Use the chat completions endpoint, which is common for instruction following
        response = llm_client.chat.completions.create(
            model=llm_model_name, # This might be ignored by LM Studio, or needs to match
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt} # Send combined prompt as user message
            ],
            temperature=0.1, # Adjust temperature as needed (lower for more focused answers)
            max_tokens=1024 # Limit the response length
        )
        # Extract and return the LLM's response
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error calling LM Studio API: {e}")
        print("Please ensure LM Studio is running, the server is started, and a model is loaded.")
        return "Error: Could not get response from LM Studio LLM."


# --- Main User Interaction Loop ---

if __name__ == "__main__":
    # Load the RAG retriever
    rag_retriever = load_retriever(vector_db_dir, EMBEDDING_MODEL_NAME)

    # Setup the LM Studio client
    lm_client = setup_llm_client(LM_STUDIO_API_BASE)

    if rag_retriever and lm_client:
        print("\n--- Text-to-SQL Knowledge Base Chat ---")
        print(f"Connected to RAG index ({vector_db_dir}) and LM Studio API ({LM_STUDIO_API_BASE}).")
        print("Type your questions about Text-to-SQL based on the papers.")
        print("Type 'exit' to quit.")

        while True:
            user_query = input("\nYour Query: ")
            if user_query.lower() == 'exit':
                break

            # Get response from the RAG system
            llm_response = rag_query(user_query, rag_retriever, lm_client, LM_STUDIO_MODEL_NAME)

            print("\nLLM Response:")
            print(llm_response)
            print("-" * 40)

    else:
        print("\nCould not initialize RAG system or connect to LM Studio API. Exiting.")
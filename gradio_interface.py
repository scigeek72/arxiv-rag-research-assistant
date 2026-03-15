#!/usr/bin/env python3
"""
ArXiv Research Paper RAG  —  Gradio Web Interface
==================================================
Generic browser-based chat interface. Topic is driven by Config.TOPIC_NAME.
Change TOPIC_NAME in config.py to switch research domains.
"""

import gradio as gr
import time
import logging
from typing import Optional

# === IMPORT YOUR EXISTING RAG SYSTEM ===
from config import Config
from query_rag_lmstudio_v2 import (
    load_retriever, 
    setup_llm_client, 
    rag_query_enhanced,
    auto_detect_model,
    query_cache
)
# === END IMPORTS ===

# === GRADIO-SPECIFIC LOGGING ===
def setup_gradio_logging():
    """Setup logging for Gradio interface."""
    Config.create_directories()
    
    # Create separate log file for Gradio
    gradio_log_file = Config.LOG_DIR + "/gradio_interface.log"
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(gradio_log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_gradio_logging()
# === END LOGGING ===

# === GLOBAL RAG SYSTEM COMPONENTS ===
rag_retriever = None
lm_client = None
detected_model = None

def initialize_rag_system():
    """Initialize the RAG system components once on startup."""
    global rag_retriever, lm_client, detected_model
    
    logger.info("🚀 Initializing RAG system for Gradio interface...")
    
    try:
        # Load vector database and retriever
        print("📚 Loading vector database...")
        rag_retriever = load_retriever(Config.VECTOR_DB_DIR, Config.EMBEDDING_MODEL_NAME)
        
        if not rag_retriever:
            return False, "❌ Failed to load vector database. Please run build_rag_index first."
        
        # Setup LM Studio client
        print("🔌 Connecting to LM Studio...")
        lm_client = setup_llm_client(Config.LM_STUDIO_API_BASE)
        
        if not lm_client:
            return False, "❌ Failed to connect to LM Studio. Please ensure LM Studio is running."
            
        # Get detected model name
        detected_model = getattr(lm_client, '_detected_model', Config.LM_STUDIO_MODEL_NAME)
        
        logger.info("✅ RAG system initialized successfully")
        return True, f"✅ RAG system ready! Using model: {detected_model}"
        
    except Exception as e:
        error_msg = f"❌ Failed to initialize RAG system: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def format_response(response: str, year_filter: Optional[int] = None) -> str:
    """Format the LLM response for better readability in Gradio."""
    if not response or response.startswith("Error:"):
        return response
    
    # Add year filter info if applied
    formatted = response
    if year_filter:
        formatted = f"📅 **Filtered for year {year_filter}**\n\n" + formatted
    
    # Add some basic formatting for better readability
    formatted = formatted.replace('\n\n', '\n\n📝 ')
    
    # Highlight paper sources if present
    if '[Source' in formatted:
        # Make sources more visible
        formatted = formatted.replace('[Source', '\n\n📄 **Source')
        formatted = formatted.replace(']', '**')
    
    return formatted

# === ENHANCED CHAT FUNCTION WITH YEAR FILTERING ===
def enhanced_chat_function(message: str, history: list, year_filter: str, settings: dict) -> tuple:
    """
    Enhanced chat function with year filtering and settings.
    
    Args:
        message: User's input message
        history: Chat history (handled by Gradio)
        year_filter: Selected year filter ("All Years" or specific year)
        settings: Additional settings (temperature, max_tokens, etc.)
    
    Returns:
        Tuple of (response, updated_history)
    """
    if not message or not message.strip():
        return f"Please enter a question about {Config.TOPIC_NAME} research.", history
    
    # Check if RAG system is initialized
    if not rag_retriever or not lm_client:
        return "❌ RAG system not initialized. Please restart the interface.", history
    
    # Parse year filter
    paper_year = None
    if year_filter and year_filter != "All Years":
        try:
            paper_year = int(year_filter)
        except ValueError:
            paper_year = None
    
    # Log the query
    query_info = f"Query: {message[:100]}{'...' if len(message) > 100 else ''}"
    if paper_year:
        query_info += f" | Year filter: {paper_year}"
    logger.info(query_info)
    
    try:
        # Handle special commands (simplified for Gradio)
        message = message.strip()
        
        if message.lower() in ['/help', 'help']:
            help_response = f"""
🤖 **{Config.TOPIC_NAME} Research Assistant Help**

**How to use:**
- Ask any question about {Config.TOPIC_NAME} research
- Use the year filter to focus on specific publication years
- Example: "{Config.EXAMPLE_QUESTIONS[0]}"
- Example: "{Config.EXAMPLE_QUESTIONS[1]}"

**Year Filtering:**
- Select "All Years" for comprehensive results
- Select specific years (2020-2025) to focus on recent work
- Year filtering searches paper metadata for targeted results

**Special commands:**
- Type "clear cache" to clear the response cache
- Type "status" to see system information

**Tips:**
- Be specific in your questions for better results
- Combine year filtering with focused questions
- The system searches through arXiv papers to answer you
- Responses include paper citations when available
            """.strip()
            return help_response, history
        
        elif message.lower() == 'clear cache':
            query_cache.cache.clear()
            query_cache.access_times.clear()
            logger.info("Cache cleared by user")
            return "🗑️ **Cache cleared!** Next queries will fetch fresh results.", history
        
        elif message.lower() == 'status':
            status = Config.get_system_status()
            status_response = f"""
📊 **System Status:**
- 🔬 Topic: {Config.TOPIC_NAME}
- 📚 Papers in database: {status.get('pdfs_count', 0)}
- 📄 Text files processed: {status.get('text_files_count', 0)}
- 🔍 Vector database: {'✅ Ready' if status.get('vector_db_exists') else '❌ Missing'}
- 💾 Database size: {status.get('vector_db_size_mb', 0)} MB
- 🤖 Model: {detected_model}
- 🔗 LLM provider: {status.get('llm_provider', '?')}
- 📅 Year filter: {year_filter}
            """.strip()
            return status_response, history
        
        # Regular query processing with year filtering
        start_time = time.time()
        
        # Get response from RAG system with year filtering
        response = rag_query_enhanced(
            query=message,
            retriever=rag_retriever,
            llm_client=lm_client,
            llm_model_name=detected_model,
            paper_year=paper_year,  # Apply year filter
            use_cache=True
        )
        
        response_time = time.time() - start_time
        logger.info(f"Query processed in {response_time:.2f} seconds")
        
        # Format response for better readability
        formatted_response = format_response(response, paper_year)
        
        return formatted_response, history
        
    except Exception as e:
        error_msg = f"❌ An error occurred: {str(e)}"
        logger.error(f"Enhanced chat function error: {str(e)}")
        return error_msg, history

def create_interface():
    """Create and configure the enhanced Gradio interface with year filtering."""
    
    # Initialize RAG system
    print(f"🚀 Starting {Config.TOPIC_NAME} RAG System...")
    success, message = initialize_rag_system()
    
    if not success:
        print(f"❌ Initialization failed: {message}")
        # Create a simple error interface
        error_interface = gr.Interface(
            fn=lambda x: f"System initialization failed: {message}",
            inputs=gr.Textbox(placeholder="System not ready..."),
            outputs=gr.Textbox(),
            title=f"❌ {Config.TOPIC_NAME} RAG System - Initialization Error"
        )
        return error_interface
    
    print(f"✅ {message}")
    
    # === ENHANCED INTERFACE WITH YEAR FILTERING ===
    with gr.Blocks(
        title=f"🤖 {Config.TOPIC_NAME} Research Assistant",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .message {
            font-size: 16px !important;
            line-height: 1.6 !important;
        }
        .sidebar {
            background-color: #f8f9fa !important;
            padding: 15px !important;
            border-radius: 10px !important;
        }
        """
    ) as interface:
        
        # Header
        gr.Markdown(f"""
        # 🤖 {Config.TOPIC_NAME} Research Assistant

        Ask questions about **{Config.TOPIC_NAME}** research and get answers grounded in arXiv papers!

        **Topic:** {Config.TOPIC_NAME} | **Model:** {detected_model} | **Papers:** {Config.get_system_status().get('pdfs_count', 0)}
        """)
        
        with gr.Row():
            # Main chat area (left side)
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    bubble_full_width=False,
                    show_label=False,
                    placeholder=f"Start by asking a question about {Config.TOPIC_NAME} research!"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder=f"Ask me anything about {Config.TOPIC_NAME} research...",
                        show_label=False,
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    retry_btn = gr.Button("🔄 Retry")
                    undo_btn = gr.Button("↩️ Undo") 
                    clear_btn = gr.Button("🗑️ Clear Chat")
            
            # Sidebar with controls (right side)
            with gr.Column(scale=1, elem_classes=["sidebar"]):
                gr.Markdown("### 🎛️ **Controls**")
                
                # Year Filter
                year_filter = gr.Dropdown(
                    choices=["All Years", "2025", "2024", "2023", "2022", "2021", "2020"],
                    value="All Years",
                    label="📅 Year Filter",
                    info="Filter papers by publication year"
                )
                
                # Quick Actions
                gr.Markdown("### ⚡ **Quick Actions**")
                help_btn = gr.Button("❓ Help", size="sm")
                status_btn = gr.Button("📊 Status", size="sm") 
                cache_btn = gr.Button("🗑️ Clear Cache", size="sm")
                
                # Examples
                gr.Markdown("### 💡 **Example Questions**")
                examples = gr.Examples(
                    examples=[[q] for q in Config.EXAMPLE_QUESTIONS],
                    inputs=[msg],
                    label="Click to try:"
                )
        
        # === ENHANCED CHAT LOGIC ===
        def user_message(message, history):
            """Handle user message submission."""
            if message.strip():
                return "", history + [[message, None]]
            return message, history
        
        def bot_response(history, year_filter_val):
            """Generate bot response with year filtering."""
            if history and history[-1][1] is None:
                user_msg = history[-1][0]
                
                # Use enhanced chat function
                response, _ = enhanced_chat_function(user_msg, history, year_filter_val, {})
                history[-1][1] = response
                
            return history
        
        # === EVENT HANDLERS ===
        # Main chat functionality
        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, [chatbot, year_filter], chatbot
        )
        submit_btn.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, [chatbot, year_filter], chatbot
        )
        
        # Quick action buttons
        help_btn.click(lambda: enhanced_chat_function("help", [], "All Years", {})[0], 
                      outputs=gr.Textbox(visible=False))
        status_btn.click(lambda: enhanced_chat_function("status", [], "All Years", {})[0], 
                        outputs=gr.Textbox(visible=False))
        cache_btn.click(lambda: enhanced_chat_function("clear cache", [], "All Years", {})[0], 
                       outputs=gr.Textbox(visible=False))
        
        # Chat controls
        clear_btn.click(lambda: [], outputs=chatbot, queue=False)
        retry_btn.click(lambda history: bot_response(history, "All Years"), [chatbot], chatbot)
        
        # Year filter change handler
        year_filter.change(
            lambda: gr.Info(f"Year filter updated! Next query will use the selected year filter."),
            queue=False
        )
    
    return interface

# === MAIN APPLICATION ===
if __name__ == "__main__":
    print(f"🌐 Launching {Config.TOPIC_NAME} RAG System Web Interface...")
    print("📝 This will open in your browser automatically.")
    print("🔗 Manual access: http://localhost:7860")
    print("⏹️  Press Ctrl+C to stop the server.")
    
    # Create and launch interface
    interface = create_interface()
    
    # Launch with configuration
    interface.launch(
        share=False,          # Set to True if you want a public link
        server_name="localhost",
        server_port=7860,
        show_error=True,
        quiet=False,
        inbrowser=True        # Automatically open in browser
    )

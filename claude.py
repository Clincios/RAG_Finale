import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get required environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

import time
import hashlib
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler

# Custom callback handler for streaming
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

class PDFChatbotConfig:
    """Configuration class for the chatbot"""
    def __init__(self):
        # Required environment variables
        self.GOOGLE_API_KEY = GOOGLE_API_KEY
        
        # Optional environment variables with defaults
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "10"))
        self.MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
        
        # Static configurations
        self.PDF_DIR = Path("pdfFiles")
        self.VECTOR_DB_DIR = Path("vectorDB")
        self.METADATA_FILE = Path("pdf_metadata.json")
        self.SUPPORTED_FILE_TYPES = ["pdf"]

class PDFProcessor:
    """Handles PDF processing and vectorization"""
    
    def __init__(self, config: PDFChatbotConfig):
        self.config = config
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=config.GOOGLE_API_KEY
        )
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to check if it's already processed"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def load_metadata(self) -> Dict:
        """Load metadata about processed files"""
        if self.config.METADATA_FILE.exists():
            with open(self.config.METADATA_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self, metadata: Dict):
        """Save metadata about processed files"""
        with open(self.config.METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def is_file_processed(self, file_path: Path) -> bool:
        """Check if file has already been processed"""
        metadata = self.load_metadata()
        file_hash = self.get_file_hash(file_path)
        return file_hash in metadata
    
    def process_pdf(self, file_path: Path) -> List:
        """Process PDF and return document chunks"""
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content found in PDF")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            
            # Add metadata to chunks
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'source_file': file_path.name,
                    'chunk_index': i,
                    'processed_at': datetime.now().isoformat()
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
    
    def create_vectorstore(self, documents: List) -> Chroma:
        """Create vector store from documents"""
        try:
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.config.VECTOR_DB_DIR)
            )
            vectorstore.persist()
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise

class ChatbotUI:
    """Handles the Streamlit UI"""
    
    def __init__(self):
        self.config = PDFChatbotConfig()
        self.processor = PDFProcessor(self.config)
        self.setup_directories()
        self.initialize_session_state()
    
    def setup_directories(self):
        """Create necessary directories"""
        for directory in [self.config.PDF_DIR, self.config.VECTOR_DB_DIR]:
            directory.mkdir(exist_ok=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        
        # Enhanced prompt template
        if 'template' not in st.session_state:
            st.session_state.template = """You are an intelligent document assistant specialized in analyzing and answering questions about PDF documents. 
            You provide accurate, detailed, and helpful responses based on the provided context.

            Instructions:
            - Answer questions directly and accurately based on the context
            - If information is not available in the context, clearly state that
            - Provide specific page references when possible
            - Use professional and informative tone
            - For complex queries, break down your response into clear sections

            Context: {context}
            Chat History: {history}

            Question: {question}
            Assistant:"""
        
        if 'prompt' not in st.session_state:
            st.session_state.prompt = PromptTemplate(
                input_variables=["history", "context", "question"],
                template=st.session_state.template,
            )
        
        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferWindowMemory(
                memory_key="history",
                return_messages=True,
                input_key="question",
                k=self.config.MAX_MEMORY_MESSAGES
            )
        
        if 'llm' not in st.session_state:
            st.session_state.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.config.GOOGLE_API_KEY,
                temperature=0.1,
                max_output_tokens=4096,
                model_kwargs={"streaming": True}
            )
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None
        
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
    
    def validate_uploaded_file(self, uploaded_file) -> bool:
        """Validate uploaded file"""
        if uploaded_file is None:
            return False
        
        # Check file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in self.config.SUPPORTED_FILE_TYPES:
            st.error(f"Unsupported file type. Please upload: {', '.join(self.config.SUPPORTED_FILE_TYPES)}")
            return False
        
        # Check file size
        if uploaded_file.size > self.config.MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File too large. Maximum size: {self.config.MAX_FILE_SIZE_MB}MB")
            return False
        
        return True
    
    def save_uploaded_file(self, uploaded_file) -> Path:
        """Save uploaded file to disk"""
        file_path = self.config.PDF_DIR / uploaded_file.name
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        return file_path
    
    def process_uploaded_file(self, uploaded_file):
        """Process the uploaded PDF file"""
        file_path = self.config.PDF_DIR / uploaded_file.name
        
        try:
            # Check if file is already processed
            if self.processor.is_file_processed(file_path):
                st.info("File already processed. Loading existing vector store...")
                st.session_state.vectorstore = Chroma(
                    persist_directory=str(self.config.VECTOR_DB_DIR),
                    embedding_function=self.processor.embeddings
                )
            else:
                with st.status("Processing PDF...", expanded=True) as status:
                    st.write("📄 Reading PDF content...")
                    
                    # Save file
                    if not file_path.exists():
                        file_path = self.save_uploaded_file(uploaded_file)
                    
                    # Process PDF
                    st.write("🔄 Splitting document into chunks...")
                    documents = self.processor.process_pdf(file_path)
                    
                    st.write("🧮 Creating embeddings and vector store...")
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=self.processor.embeddings,
                        persist_directory=str(self.config.VECTOR_DB_DIR)
                    )
                    st.session_state.vectorstore.persist()
                    
                    # Update metadata
                    metadata = self.processor.load_metadata()
                    file_hash = self.processor.get_file_hash(file_path)
                    metadata[file_hash] = {
                        'filename': uploaded_file.name,
                        'processed_at': datetime.now().isoformat(),
                        'chunks_count': len(documents)
                    }
                    self.processor.save_metadata(metadata)
                    
                    status.update(label="✅ PDF processed successfully!", state="complete")
            
            # Add file to processed files
            st.session_state.processed_files.add(uploaded_file.name)
            
            # Create QA chain
            self.create_qa_chain()
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"File processing error: {str(e)}")
    
    def create_qa_chain(self):
        """Create the QA chain"""
        if st.session_state.vectorstore is not None:
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )
            
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type='stuff',
                retriever=retriever,
                verbose=False,
                chain_type_kwargs={
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory,
                }
            )
    
    def display_chat_history(self):
        """Display chat history"""
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["message"])
    
    def handle_user_input(self, user_input: str):
        """Handle user input and generate response"""
        # Add user message to chat history
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.qa_chain.invoke({"query": user_input})
                
                # Simulate streaming response
                full_response = response['result']
                displayed_response = ""
                
                for chunk in full_response.split():
                    displayed_response += chunk + " "
                    time.sleep(0.03)
                    message_placeholder.markdown(displayed_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add assistant message to chat history
                assistant_message = {"role": "assistant", "message": full_response}
                st.session_state.chat_history.append(assistant_message)
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                message_placeholder.markdown(error_message)
                st.error("Please try again or upload a different PDF.")
                logger.error(f"Query processing error: {str(e)}")
    
    def display_sidebar(self):
        """Display sidebar with additional information"""
        with st.sidebar:
            st.header("📊 Chatbot Info")
            
            if st.session_state.processed_files:
                st.subheader("Processed Files:")
                for file in st.session_state.processed_files:
                    st.write(f"📄 {file}")
            
            st.subheader("Features:")
            st.write("✅ PDF document analysis")
            st.write("✅ Intelligent chunking")
            st.write("✅ Conversation memory")
            st.write("✅ Source-based answers")
            st.write("✅ File caching")
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.memory.clear()
                st.rerun()
            
            if st.button("🔄 Reset Application"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    def run(self):
        """Run the main application"""
        st.set_page_config(
            page_title="PDF Chatbot Pro",
            page_icon="📚",
            layout="wide"
        )
        
        st.title("📚 PDF Chatbot Pro")
        st.markdown("Upload a PDF document and ask questions about its content!")
        
        # Display sidebar
        self.display_sidebar()
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=self.config.SUPPORTED_FILE_TYPES,
            help=f"Maximum file size: {self.config.MAX_FILE_SIZE_MB}MB"
        )
        
        # Process uploaded file
        if uploaded_file is not None and self.validate_uploaded_file(uploaded_file):
            if uploaded_file.name not in st.session_state.processed_files:
                self.process_uploaded_file(uploaded_file)
        
        # Display chat interface
        if st.session_state.vectorstore is not None:
            st.success("✅ PDF loaded successfully! You can now ask questions.")
            
            # Display chat history
            self.display_chat_history()
            
            # Chat input
            if user_input := st.chat_input("Ask a question about your PDF..."):
                self.handle_user_input(user_input)
        
        else:
            st.info("👆 Please upload a PDF file to start chatting!")
            
            # Show example questions
            st.subheader("Example questions you can ask:")
            example_questions = [
                "What is the main topic of this document?",
                "Can you summarize the key points?",
                "What are the conclusions mentioned?",
                "Are there any specific recommendations?",
                "What data or statistics are presented?"
            ]
            
            for question in example_questions:
                st.write(f"• {question}")

# Run the application
if __name__ == "__main__":
    app = ChatbotUI()
    app.run()
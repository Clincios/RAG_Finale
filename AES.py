import streamlit as st
import os
import time
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import shutil
import re
from dataclasses import dataclass, asdict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import statistics

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

@dataclass
class ScoringResult:
    """Data class for storing scoring results"""
    question: str
    user_answer: str
    correct_answer: str
    score: float
    feedback: str
    max_score: float = 10.0

@dataclass
class AssessmentResult:
    """Data class for storing complete assessment results"""
    assessment_id: str
    document_name: str
    total_score: float
    max_total_score: float
    percentage: float
    question_results: List[ScoringResult]
    timestamp: str
    time_taken: Optional[str] = None

class PDFChatbotConfig:
    """Configuration class for the chatbot"""
    def __init__(self):
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAkHRbkUvKvnfzWpoX1pks8hNUc78PXqXs")
        self.PDF_DIR = Path("pdfFiles")
        self.VECTOR_DB_DIR = Path("vectorDB")
        self.METADATA_FILE = Path("pdf_metadata.json")
        self.ASSESSMENTS_DIR = Path("assessments")
        self.SCORING_RESULTS_FILE = Path("scoring_results.json")
        self.CHUNK_SIZE = 1000
        self.CHUNK_OVERLAP = 200
        self.MAX_MEMORY_MESSAGES = 10
        self.SUPPORTED_FILE_TYPES = ["pdf"]
        self.MAX_FILE_SIZE_MB = 50

class ScoringEngine:
    """Handles automated scoring functionality"""
    
    def __init__(self, config: PDFChatbotConfig, llm):
        self.config = config
        self.llm = llm
        self.config.ASSESSMENTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Scoring prompt template
        self.scoring_prompt = PromptTemplate(
            input_variables=["question", "correct_answer", "user_answer", "context"],
            template="""You are an expert educational assessor. Your task is to score a student's answer based on the provided context from a document.

SCORING CRITERIA:
- Maximum score: 10 points
- Award points based on accuracy, completeness, and understanding
- Consider partial credit for partially correct answers
- Be fair but maintain academic standards

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

CORRECT/EXPECTED ANSWER: {correct_answer}

STUDENT'S ANSWER: {user_answer}

Please provide your assessment in the following JSON format:
{{
    "score": [numeric score out of 10],
    "feedback": "[detailed feedback explaining the score, what was correct, what was missing, and suggestions for improvement]"
}}

Be constructive in your feedback and explain your reasoning clearly."""
        )
    
    def load_scoring_results(self) -> List[AssessmentResult]:
        """Load previous scoring results"""
        if self.config.SCORING_RESULTS_FILE.exists():
            try:
                with open(self.config.SCORING_RESULTS_FILE, 'r') as f:
                    data = json.load(f)
                    return [AssessmentResult(**result) for result in data]
            except Exception as e:
                logger.error(f"Error loading scoring results: {e}")
                return []
        return []
    
    def save_scoring_results(self, results: List[AssessmentResult]):
        """Save scoring results"""
        try:
            with open(self.config.SCORING_RESULTS_FILE, 'w') as f:
                json.dump([asdict(result) for result in results], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving scoring results: {e}")
    
    def generate_questions_for_assessment(self, vectorstore, num_questions: int = 5, difficulty: str = "medium") -> List[Dict]:
        """Generate questions for assessment based on document content"""
        try:
            difficulty_prompts = {
                "easy": "Create basic recall and understanding questions that test fundamental concepts.",
                "medium": "Create analytical questions that require understanding and application of concepts.",
                "hard": "Create complex questions that require critical thinking, analysis, and synthesis."
            }
            
            prompt = f"""Based on the provided document context, generate {num_questions} examination questions for assessment.

REQUIREMENTS:
- {difficulty_prompts.get(difficulty, difficulty_prompts["medium"])}
- Include the correct answer for each question
- Questions should be answerable based on the document content
- Vary question types (short answer, explanation, analysis)
- Each question should test different aspects of the material

Please format your response as a JSON array with this structure:
[
    {{
        "question": "Your question here",
        "correct_answer": "The correct/expected answer",
        "question_type": "short_answer|explanation|analysis",
        "difficulty": "{difficulty}"
    }}
]

Focus on the most important concepts and information from the document."""
            
            # Get relevant context from vectorstore
            retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
            docs = retriever.get_relevant_documents("key concepts main topics important information")
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate questions
            response = self.llm.invoke(f"Context: {context}\n\n{prompt}")
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
                if json_match:
                    questions_data = json.loads(json_match.group())
                    return questions_data
                else:
                    # Fallback: create questions manually if JSON parsing fails
                    return self._create_fallback_questions(context)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from LLM response, using fallback")
                return self._create_fallback_questions(context)
                
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return self._create_fallback_questions("Document content")
    
    def _create_fallback_questions(self, context: str) -> List[Dict]:
        """Create fallback questions if automatic generation fails"""
        return [
            {
                "question": "What are the main topics discussed in this document?",
                "correct_answer": "Summarize the key topics and themes presented in the document based on your reading.",
                "question_type": "explanation",
                "difficulty": "medium"
            },
            {
                "question": "Explain the most important concept presented in the document.",
                "correct_answer": "Identify and explain the central concept or principle discussed in the document.",
                "question_type": "explanation", 
                "difficulty": "medium"
            },
            {
                "question": "What conclusions or recommendations are made in the document?",
                "correct_answer": "Summarize any conclusions, recommendations, or key takeaways from the document.",
                "question_type": "analysis",
                "difficulty": "medium"
            }
        ]
    
    def score_answer(self, question: str, correct_answer: str, user_answer: str, vectorstore) -> ScoringResult:
        """Score a single answer using LLM"""
        try:
            # Get relevant context for the question
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate scoring prompt
            scoring_input = self.scoring_prompt.format(
                question=question,
                correct_answer=correct_answer,
                user_answer=user_answer,
                context=context
            )
            
            # Get scoring from LLM
            response = self.llm.invoke(scoring_input)
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    scoring_data = json.loads(json_match.group())
                    score = float(scoring_data.get("score", 5.0))
                    feedback = scoring_data.get("feedback", "Score provided")
                else:
                    # Fallback scoring
                    score, feedback = self._fallback_scoring(user_answer, correct_answer)
            except (json.JSONDecodeError, ValueError):
                logger.warning("Failed to parse scoring JSON, using fallback")
                score, feedback = self._fallback_scoring(user_answer, correct_answer)
            
            return ScoringResult(
                question=question,
                user_answer=user_answer,
                correct_answer=correct_answer,
                score=max(0, min(10, score)),  # Ensure score is between 0-10
                feedback=feedback
            )
            
        except Exception as e:
            logger.error(f"Error scoring answer: {e}")
            return ScoringResult(
                question=question,
                user_answer=user_answer,
                correct_answer=correct_answer,
                score=5.0,
                feedback=f"Unable to score automatically. Error: {str(e)}"
            )
    
    def _fallback_scoring(self, user_answer: str, correct_answer: str) -> Tuple[float, str]:
        """Provide fallback scoring when LLM scoring fails"""
        if not user_answer.strip():
            return 0.0, "No answer provided."
        
        # Simple keyword matching for basic scoring
        user_words = set(user_answer.lower().split())
        correct_words = set(correct_answer.lower().split())
        common_words = user_words.intersection(correct_words)
        
        if len(correct_words) > 0:
            similarity = len(common_words) / len(correct_words)
            score = min(10.0, similarity * 10)
        else:
            score = 5.0
        
        return score, f"Basic similarity score based on keyword matching. Consider reviewing the expected answer for a complete response."

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
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            logger.error(f"File not found when generating hash: {file_path}")
            raise
    
    def get_document_specific_db_path(self, file_hash: str) -> Path:
        """Get document-specific vector database path"""
        return self.config.VECTOR_DB_DIR / f"doc_{file_hash}"
    
    def load_metadata(self) -> Dict:
        """Load metadata about processed files"""
        if self.config.METADATA_FILE.exists():
            try:
                with open(self.config.METADATA_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Error loading metadata: {e}. Creating new metadata file.")
                return {}
        return {}
    
    def save_metadata(self, metadata: Dict):
        """Save metadata about processed files"""
        try:
            with open(self.config.METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def is_file_processed(self, file_path: Path) -> tuple[bool, str]:
        """Check if file has already been processed and return file hash"""
        try:
            if not file_path.exists():
                return False, ""
            
            file_hash = self.get_file_hash(file_path)
            metadata = self.load_metadata()
            doc_db_path = self.get_document_specific_db_path(file_hash)
            
            # Check if metadata exists and vector db directory exists
            is_processed = (file_hash in metadata and 
                          doc_db_path.exists() and 
                          any(doc_db_path.iterdir()))
            
            return is_processed, file_hash
        except Exception as e:
            logger.error(f"Error checking if file is processed: {e}")
            return False, ""
    
    def process_pdf(self, file_path: Path) -> List:
        """Process PDF and return document chunks"""
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            # Convert Path to string for PyPDFLoader
            loader = PyPDFLoader(str(file_path.resolve()))
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
    
    def create_vectorstore(self, documents: List, file_hash: str) -> Chroma:
        """Create document-specific vector store"""
        try:
            # Get document-specific database path
            doc_db_path = self.get_document_specific_db_path(file_hash)
            doc_db_path.mkdir(parents=True, exist_ok=True)
            
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(doc_db_path)
            )
            vectorstore.persist()
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def load_vectorstore(self, file_hash: str) -> Optional[Chroma]:
        """Load document-specific vector store"""
        try:
            doc_db_path = self.get_document_specific_db_path(file_hash)
            
            if doc_db_path.exists() and any(doc_db_path.iterdir()):
                vectorstore = Chroma(
                    persist_directory=str(doc_db_path),
                    embedding_function=self.embeddings
                )
                logger.info(f"Vectorstore loaded for document hash: {file_hash}")
                return vectorstore
        except Exception as e:
            logger.warning(f"Could not load vectorstore for {file_hash}: {e}")
        return None
    
    def cleanup_old_vectorstores(self, keep_hash: str = None):
        """Clean up old vector stores (optional - for storage management)"""
        try:
            if not self.config.VECTOR_DB_DIR.exists():
                return
                
            for item in self.config.VECTOR_DB_DIR.iterdir():
                if item.is_dir() and item.name.startswith("doc_"):
                    if keep_hash is None or not item.name.endswith(keep_hash):
                        shutil.rmtree(item)
                        logger.info(f"Cleaned up old vectorstore: {item}")
        except Exception as e:
            logger.warning(f"Error cleaning up old vectorstores: {e}")

class ChatbotUI:
    """Handles the Streamlit UI"""
    
    def __init__(self):
        self.config = PDFChatbotConfig()
        self.processor = PDFProcessor(self.config)
        self.setup_directories()
        self.initialize_session_state()
    
    def setup_directories(self):
        """Create necessary directories"""
        for directory in [self.config.PDF_DIR, self.config.VECTOR_DB_DIR, self.config.ASSESSMENTS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created/verified: {directory}")
    
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
            - IMPORTANT: Only use information from the currently active document context

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
                streaming=True
            )
        
        if 'scoring_engine' not in st.session_state:
            st.session_state.scoring_engine = ScoringEngine(self.config, st.session_state.llm)
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None
        
        if 'current_document' not in st.session_state:
            st.session_state.current_document = None
        
        if 'current_document_hash' not in st.session_state:
            st.session_state.current_document_hash = None
        
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = None
        
        # Assessment related state
        if 'current_assessment' not in st.session_state:
            st.session_state.current_assessment = None
        
        if 'assessment_mode' not in st.session_state:
            st.session_state.assessment_mode = False
        
        if 'assessment_questions' not in st.session_state:
            st.session_state.assessment_questions = []
        
        if 'current_question_index' not in st.session_state:
            st.session_state.current_question_index = 0
        
        if 'user_answers' not in st.session_state:
            st.session_state.user_answers = []
        
        if 'assessment_start_time' not in st.session_state:
            st.session_state.assessment_start_time = None
    
    def reset_document_session(self):
        """Reset session for new document"""
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.session_state.vectorstore = None
        st.session_state.qa_chain = None
        st.session_state.current_document = None
        st.session_state.current_document_hash = None
        
        # Reset assessment state
        st.session_state.current_assessment = None
        st.session_state.assessment_mode = False
        st.session_state.assessment_questions = []
        st.session_state.current_question_index = 0
        st.session_state.user_answers = []
        st.session_state.assessment_start_time = None
        
        logger.info("Document session reset")
    
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
        self.config.PDF_DIR.mkdir(parents=True, exist_ok=True)
        file_path = self.config.PDF_DIR / uploaded_file.name
        
        try:
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())
            logger.info(f"File saved successfully: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise
    
    def process_uploaded_file(self, uploaded_file):
        """Process the uploaded PDF file"""
        try:
            # Create a placeholder for status messages
            status_placeholder = st.empty()
            
            # Save the uploaded file
            file_path = self.save_uploaded_file(uploaded_file)
            logger.info(f"Processing file: {file_path}")
            
            # Check if this is a different document than currently loaded
            is_processed, file_hash = self.processor.is_file_processed(file_path)
            
            # If switching to a different document, reset session
            if (st.session_state.current_document_hash and 
                st.session_state.current_document_hash != file_hash):
                status_placeholder.info("Switching to a different document. Resetting chat session...")
                time.sleep(5)
                status_placeholder.empty()
                self.reset_document_session()
            
            # Set current document info
            st.session_state.current_document = uploaded_file.name
            st.session_state.current_document_hash = file_hash
            
            if is_processed:
                status_placeholder.info("Document already processed. Loading existing analysis...")
                time.sleep(5)
                status_placeholder.empty()
                st.session_state.vectorstore = self.processor.load_vectorstore(file_hash)
                if st.session_state.vectorstore is None:
                    status_placeholder.warning("Could not load existing analysis. Reprocessing document...")
                    time.sleep(5)
                    status_placeholder.empty()
                    self._process_new_file(file_path, uploaded_file, file_hash)
            else:
                self._process_new_file(file_path, uploaded_file, file_hash)
            
            # Create QA chain
            self.create_qa_chain()
            
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            status_placeholder.error(error_msg)
            time.sleep(5)
            status_placeholder.empty()
            logger.error(error_msg)
            
            # Clean up if file was partially saved
            file_path = self.config.PDF_DIR / uploaded_file.name
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up partial file: {file_path}")
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up file: {cleanup_error}")
    
    def _process_new_file(self, file_path: Path, uploaded_file, file_hash: str):
        """Process a new PDF file"""
        status_placeholder = st.empty()
        
        with st.status("Processing PDF...", expanded=True) as status:
            status_placeholder.write("📄 Reading PDF content...")
            time.sleep(5)
            status_placeholder.empty()
            
            # Process PDF
            status_placeholder.write("🔄 Splitting document into chunks...")
            time.sleep(5)
            status_placeholder.empty()
            documents = self.processor.process_pdf(file_path)
            
            status_placeholder.write("🧮 Creating embeddings and vector store...")
            time.sleep(5)
            status_placeholder.empty()
            st.session_state.vectorstore = self.processor.create_vectorstore(documents, file_hash)
            
            # Update metadata
            metadata = self.processor.load_metadata()
            metadata[file_hash] = {
                'filename': uploaded_file.name,
                'processed_at': datetime.now().isoformat(),
                'chunks_count': len(documents)
            }
            self.processor.save_metadata(metadata)
            
            # Show success message and clear after 5 seconds
            status.update(label="✅ PDF processed successfully!", state="complete")
            time.sleep(5)
            status.update(label="", state="complete")
            status_placeholder.empty()
    
    def create_qa_chain(self):
        """Create the QA chain"""
        if st.session_state.vectorstore is not None:
            try:
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
                logger.info("QA chain created successfully")
            except Exception as e:
                logger.error(f"Error creating QA chain: {e}")
                st.error("Error setting up question-answering system. Please try again.")
    
    def generate_automated_response(self, feature_type: str):
        """Generate automated responses for quick features"""
        prompts = {
            "summarize": f"""Please provide a comprehensive summary of the current document: "{st.session_state.current_document}". Include:
            - Main topic and purpose
            - Key points and findings
            - Important conclusions or recommendations
            - Any significant data or statistics mentioned
            Keep the summary concise but thorough. Focus only on this document.""",
            
            "study_guide": f"""Create a detailed study guide from the current document: "{st.session_state.current_document}" that includes:
            - Main concepts and definitions
            - Important facts and figures
            - Key processes or procedures explained
            - Critical points students should remember
            - Potential areas of focus for deeper study
            Format it as a structured study guide. Use only information from this document.""",
            
            "exam_questions": f"""Generate a set of potential examination questions based on the current document: "{st.session_state.current_document}". Include:
            - 3-5 multiple choice questions with options
            - 3-5 short answer questions
            - 2-3 essay/long answer questions
            - Include the correct answers or key points for each question
            Focus on testing understanding of key concepts from this specific document."""
        }
        
        return prompts.get(feature_type, "")
    
    def start_assessment(self, difficulty: str, num_questions: int):
        """Start a new assessment"""
        try:
            with st.spinner("🎯 Generating assessment questions..."):
                questions = st.session_state.scoring_engine.generate_questions_for_assessment(
                    st.session_state.vectorstore, 
                    num_questions, 
                    difficulty
                )
                
                if questions:
                    st.session_state.assessment_questions = questions
                    st.session_state.current_question_index = 0
                    # Initialize user_answers with empty strings for all questions
                    st.session_state.user_answers = [""] * len(questions)
                    st.session_state.assessment_mode = True
                    st.session_state.assessment_start_time = datetime.now()
                    
                    # Generate assessment ID
                    assessment_id = f"assessment_{st.session_state.current_document_hash[:8]}_{int(time.time())}"
                    st.session_state.current_assessment = assessment_id
                    
                    st.success("✅ Assessment generated successfully! You can now start answering questions.")
                    st.rerun()
                else:
                    st.error("Failed to generate assessment questions. Please try again.")
        except Exception as e:
            st.error(f"Error starting assessment: {str(e)}")
            logger.error(f"Assessment generation error: {e}")
    
    def display_assessment_interface(self):
        """Display the enhanced assessment interface for taking a test"""
        st.header("📝 Assessment")
        
        # Display current question
        current_question = st.session_state.assessment_questions[st.session_state.current_question_index]
        total_questions = len(st.session_state.assessment_questions)
        current_index = st.session_state.current_question_index
        
        # Show progress
        progress = (current_index + 1) / total_questions
        st.progress(progress)
        st.write(f"Question {current_index + 1} of {total_questions}")
        
        # Display question
        st.markdown(f"{current_question['question']}")
        
        # Get current answer if it exists
        current_answer = st.session_state.user_answers[current_index] if current_index < len(st.session_state.user_answers) else ""
        
        # Create a container for the answer input and error message
        with st.container():
            # Display answer input
            user_answer = st.text_area(
                "Your Answer:", 
                value=current_answer,
                key=f"answer_input_{current_index}_{st.session_state.current_assessment}",
                height=150,
                help="Please provide a detailed answer to this question."
            )
            
            # Add a placeholder for the error message right after the text area
            error_placeholder = st.empty()
        
        # Update the answer in session state immediately
        st.session_state.user_answers[current_index] = user_answer
        
        # Create navigation buttons with three columns for better alignment
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Previous button (only show if not on first question)
            if current_index > 0:
                if st.button("⬅️ Previous", key=f"prev_btn_{current_index}_{st.session_state.current_assessment}"):
                    st.session_state.current_question_index -= 1
                    st.rerun()
        
        with col3:
            # Right-aligned column for main action
            if current_index < total_questions - 1:
                # Not the last question - show Next button
                if st.button(
                    "➡️ Next Question", 
                    type="primary", 
                    key=f"next_btn_{current_index}_{st.session_state.current_assessment}",
                    help="Click to proceed to the next question"
                ):
                    # Validate answer before proceeding
                    if not user_answer.strip():
                        error_placeholder.error("⚠️ Please provide an answer before proceeding to the next question.")
                    else:
                        st.session_state.current_question_index += 1
                        st.rerun()
            else:
                # Last question - show Finish button
                if st.button(
                    "🏁 Finish Assessment", 
                    type="primary",
                    key=f"finish_btn_{current_index}_{st.session_state.current_assessment}",
                    help="Click to finish the assessment"
                ):
                    # Validate answer before finishing
                    if not user_answer.strip():
                        error_placeholder.error("⚠️ Please provide an answer before finishing the assessment.")
                    else:
                        self.calculate_assessment_score()
    
    def calculate_assessment_score(self):
        """Calculate and display the final assessment score with validation"""
        try:
            # Validate all questions are answered
            unanswered_questions = []
            for i, answer in enumerate(st.session_state.user_answers):
                if not answer.strip():
                    unanswered_questions.append(i + 1)
            
            if unanswered_questions:
                st.error(f"Please answer all questions before finishing. Unanswered questions: {', '.join(map(str, unanswered_questions))}")
                return
            
            # Calculate time taken
            if st.session_state.assessment_start_time:
                time_taken = datetime.now() - st.session_state.assessment_start_time
                time_str = str(time_taken).split('.')[0]  # Remove microseconds
            else:
                time_str = None
            
            # Score each answer
            scoring_results = []
            total_score = 0
            max_total_score = 0
            
            with st.spinner("🔄 Calculating your scores..."):
                for i, (question, user_answer) in enumerate(zip(st.session_state.assessment_questions, st.session_state.user_answers)):
                    result = st.session_state.scoring_engine.score_answer(
                        question=question['question'],
                        correct_answer=question['correct_answer'],
                        user_answer=user_answer,
                        vectorstore=st.session_state.vectorstore
                    )
                    
                    scoring_results.append(result)
                    total_score += result.score
                    max_total_score += result.max_score
            
            # Calculate percentage
            percentage = (total_score / max_total_score * 100) if max_total_score > 0 else 0
            
            # Create assessment result
            assessment_result = AssessmentResult(
                assessment_id=st.session_state.current_assessment,
                document_name=st.session_state.current_document,
                total_score=total_score,
                max_total_score=max_total_score,
                percentage=percentage,
                question_results=scoring_results,
                timestamp=datetime.now().isoformat(),
                time_taken=time_str
            )
            
            # Save results
            results = st.session_state.scoring_engine.load_scoring_results()
            results.append(assessment_result)
            st.session_state.scoring_engine.save_scoring_results(results)
            
            # Reset assessment state
            st.session_state.assessment_mode = False
            st.session_state.assessment_questions = []
            st.session_state.user_answers = []
            st.session_state.current_question_index = 0
            st.session_state.assessment_start_time = None
            
            # Show results with celebration
            st.success("🎉 Assessment completed successfully!")
            st.balloons()
            
            # Display comprehensive score summary
            st.header("📊 Assessment Results")
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final Score", f"{total_score:.1f}/{max_total_score:.1f}")
            
            with col2:
                st.metric("Percentage", f"{percentage:.1f}%")
            
            with col3:
                st.metric("Questions Answered", len(scoring_results))
            
            with col4:
                if time_str:
                    st.metric("Time Taken", time_str)
            
            # Performance grade
            if percentage >= 90:
                st.success("🏆 Outstanding Performance! Grade: A")
            elif percentage >= 80:
                st.success("🌟 Excellent Work! Grade: B")
            elif percentage >= 70:
                st.info("👍 Good Job! Grade: C")
            elif percentage >= 60:
                st.warning("📚 Satisfactory. Grade: D")
            else:
                st.error("🔄 Needs Improvement. Grade: F")
            
            # Show detailed results immediately
            st.markdown("---")
            st.header("📝 Detailed Question Analysis")
            
            # Quick performance overview
            correct_count = sum(1 for r in scoring_results if r.score == r.max_score)
            partial_count = sum(1 for r in scoring_results if 0 < r.score < r.max_score)
            incorrect_count = sum(1 for r in scoring_results if r.score == 0)
            
            overview_col1, overview_col2, overview_col3 = st.columns(3)
            with overview_col1:
                st.success(f"✅ Fully Correct: {correct_count}")
            with overview_col2:
                st.info(f"🔶 Partially Correct: {partial_count}")
            with overview_col3:
                st.error(f"❌ Incorrect: {incorrect_count}")
            
            # Display each question result in detail
            for i, result in enumerate(scoring_results):
                st.markdown("---")
                st.subheader(f"Question {i+1}")
                
                # Question and scoring
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Question:** {result.question}")
                with col2:
                    score_color = "success" if result.score == result.max_score else "info" if result.score > 0 else "error"
                    st.markdown(f"**Score:** {result.score:.1f}/{result.max_score:.1f}", help=f"Question {i+1} Score")
                
                # Answers comparison
                st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Your Answer:</span>", unsafe_allow_html=True)
                if result.score == result.max_score:
                    st.success(result.user_answer)
                elif result.score > 0:
                    st.info(result.user_answer)
                else:
                    st.error(result.user_answer)
                
                st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Expected Answer:</span>", unsafe_allow_html=True)
                st.success(result.correct_answer)
                
                # Feedback
                if hasattr(result, 'feedback') and result.feedback:
                    st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Feedback:</span>", unsafe_allow_html=True)
                    feedback_container = st.container()
                    with feedback_container:
                        if result.score == result.max_score:
                            st.success(result.feedback)
                        elif result.score > 0:
                            st.info(result.feedback)
                        else:
                            st.error(result.feedback)
            
            # Action buttons
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            # Define callback functions
            def on_view_history():
                st.session_state.show_history = True
            
            def on_new_assessment():
                if 'show_history' in st.session_state:
                    st.session_state.show_history = False
                st.session_state.assessment_mode = False
                st.session_state.assessment_questions = []
                st.session_state.user_answers = []
                st.session_state.current_question_index = 0
                st.session_state.assessment_start_time = None
            
            # Get a unique key base for the buttons
            button_key_base = assessment_result.assessment_id
            
            with col1:
                st.button(
                    "📊 View Assessment History",
                    key=f"view_history_{button_key_base}",
                    on_click=on_view_history
                )
            
            with col2:
                st.button(
                    "🔄 Take Another Assessment",
                    key=f"new_assessment_{button_key_base}",
                    on_click=on_new_assessment
                )
                    
        except Exception as e:
            st.error(f"Error calculating assessment score: {str(e)}")
            logger.error(f"Assessment scoring error: {e}")

    def display_assessment_history(self):
        """Display comprehensive assessment history with analytics"""
        try:
            results = st.session_state.scoring_engine.load_scoring_results()
            
            if not results:
                st.info("📋 No assessment history found. Take your first assessment to see analytics here!")
                return
            
            st.header("📊 Assessment Analytics Dashboard")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                # Document filter
                all_documents = list(set([r.document_name for r in results if r.document_name]))
                selected_doc = st.selectbox(
                    "Filter by Document:",
                    ["All Documents"] + all_documents,
                    key="doc_filter"
                )
            
            with col2:
                # Time range filter
                time_range = st.selectbox(
                    "Time Range:",
                    ["All Time", "Last 7 Days", "Last 30 Days", "Last 3 Months"],
                    key="time_filter"
                )
            
            # Apply filters
            filtered_results = self.apply_history_filters(results, selected_doc, time_range)
            
            if not filtered_results:
                st.warning("No assessments found for the selected filters.")
                return
            
            # Display analytics sections
            self.display_performance_overview(filtered_results)
            self.display_progress_trends(filtered_results)
            self.display_detailed_assessment_list(filtered_results)
            self.display_question_analytics(filtered_results)
            
        except Exception as e:
            st.error(f"Error loading assessment history: {str(e)}")
            st.exception(e)

    def apply_history_filters(self, results, selected_doc, time_range):
        """Apply filters to assessment results"""
        filtered_results = results.copy()
        
        # Document filter
        if selected_doc != "All Documents":
            filtered_results = [r for r in filtered_results if self._get_value(r, 'document_name') == selected_doc]
        
        # Time range filter
        if time_range != "All Time":
            cutoff_date = datetime.now()
            if time_range == "Last 7 Days":
                cutoff_date -= timedelta(days=7)
            elif time_range == "Last 30 Days":
                cutoff_date -= timedelta(days=30)
            elif time_range == "Last 3 Months":
                cutoff_date -= timedelta(days=90)
            
            filtered_results = [
                r for r in filtered_results 
                if datetime.fromisoformat(self._get_value(r, 'timestamp').replace('Z', '+00:00').replace('+00:00', '')) >= cutoff_date
            ]
        
        return filtered_results

    def display_performance_overview(self, results):
        """Display overall performance metrics"""
        st.subheader("🎯 Performance Overview")
        
        # Calculate metrics
        total_assessments = len(results)
        percentages = [self._get_value(r, 'percentage') for r in results]
        avg_score = statistics.mean(percentages)
        highest_score = max(percentages)
        latest_score = percentages[0] if results else 0
        
        # Performance improvement calculation
        if len(results) >= 2:
            recent_scores = [self._get_value(r, 'percentage') for r in sorted(results, key=lambda x: self._get_value(x, 'timestamp'), reverse=True)[:3]]
            older_scores = [self._get_value(r, 'percentage') for r in sorted(results, key=lambda x: self._get_value(x, 'timestamp'))[:3]]
            improvement = statistics.mean(recent_scores) - statistics.mean(older_scores)
        else:
            improvement = 0
        
        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Assessments", total_assessments)
        
        with col2:
            st.metric("Average Score", f"{avg_score:.1f}%")
        
        with col3:
            st.metric("Highest Score", f"{highest_score:.1f}%")
        
        with col4:
            st.metric("Latest Score", f"{latest_score:.1f}%")
        
        with col5:
            st.metric(
                "Improvement", 
                f"{improvement:+.1f}%",
                delta=f"{improvement:+.1f}%" if improvement != 0 else None
            )
        
        # Performance grade distribution
        self.display_grade_distribution(results)

    def display_grade_distribution(self, results):
        """Display grade distribution chart"""
        st.subheader("📈 Grade Distribution")
        
        # Categorize scores
        grades = {"A (90-100%)": 0, "B (80-89%)": 0, "C (70-79%)": 0, "D (60-69%)": 0, "F (<60%)": 0}
        
        for result in results:
            score = self._get_value(result, 'percentage')
            if score >= 90:
                grades["A (90-100%)"] += 1
            elif score >= 80:
                grades["B (80-89%)"] += 1
            elif score >= 70:
                grades["C (70-79%)"] += 1
            elif score >= 60:
                grades["D (60-69%)"] += 1
            else:
                grades["F (<60%)"] += 1
        
        # Create pie chart
        fig = px.pie(
            values=list(grades.values()),
            names=list(grades.keys()),
            title="Grade Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    def display_progress_trends(self, results):
        """Display progress trends over time"""
        st.subheader("📊 Progress Trends")
        
        if len(results) < 2:
            st.info("Take more assessments to see progress trends!")
            return
        
        # Sort results by timestamp
        sorted_results = sorted(results, key=lambda x: self._get_value(x, 'timestamp'))
        
        # Prepare data for plotting
        dates = [datetime.fromisoformat(self._get_value(r, 'timestamp').replace('Z', '+00:00').replace('+00:00', '')) for r in sorted_results]
        scores = [self._get_value(r, 'percentage') for r in sorted_results]
        documents = [self._get_value(r, 'document_name') for r in sorted_results]
        
        # Create line chart
        fig = go.Figure()
        
        # Group by document for different lines
        doc_data = defaultdict(lambda: {'dates': [], 'scores': []})
        for date, score, doc in zip(dates, scores, documents):
            doc_data[doc]['dates'].append(date)
            doc_data[doc]['scores'].append(score)
        
        colors = px.colors.qualitative.Set1
        for i, (doc, data) in enumerate(doc_data.items()):
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=data['scores'],
                mode='lines+markers',
                name=doc,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Score Progress Over Time",
            xaxis_title="Date",
            yaxis_title="Score (%)",
            yaxis=dict(range=[0, 100]),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display trend insights
        self.display_trend_insights(sorted_results)

    def display_trend_insights(self, sorted_results):
        """Display insights about performance trends"""
        if len(sorted_results) < 3:
            return
        
        st.subheader("🧠 Performance Insights")
        
        # Calculate trends
        recent_scores = [self._get_value(r, 'percentage') for r in sorted_results[-3:]]
        older_scores = [self._get_value(r, 'percentage') for r in sorted_results[:3]]
        
        avg_recent = statistics.mean(recent_scores)
        avg_older = statistics.mean(older_scores)
        improvement = avg_recent - avg_older
        
        # Consistency analysis
        score_variance = statistics.variance([self._get_value(r, 'percentage') for r in sorted_results]) if len(sorted_results) > 1 else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            if improvement > 5:
                st.success(f"🚀 **Excellent Progress!** You've improved by {improvement:.1f}% on average")
            elif improvement > 0:
                st.info(f"📈 **Steady Improvement:** {improvement:.1f}% average increase")
            elif improvement > -5:
                st.warning(f"📊 **Stable Performance:** {abs(improvement):.1f}% average change")
            else:
                st.error(f"📉 **Focus Needed:** {abs(improvement):.1f}% average decrease")
        
        with col2:
            if score_variance < 25:
                st.success("🎯 **Consistent Performance:** Your scores are very stable")
            elif score_variance < 100:
                st.info("📊 **Moderate Consistency:** Some variation in performance")
            else:
                st.warning("🎢 **Variable Performance:** Consider reviewing study methods")

    def display_question_analytics(self, results):
        """Display analytics about question performance"""
        st.subheader("🎯 Question Performance Analytics")
        
        if not results:
            return
        
        # Aggregate question performance data
        question_stats = defaultdict(lambda: {'total_score': 0, 'max_score': 0, 'count': 0, 'questions': []})
        
        for result in results:
            question_results = self._get_value(result, 'question_results', [])
            for i, qr in enumerate(question_results):
                key = f"Q{i+1}"
                
                # Handle both dict and object access patterns
                score = self._get_value(qr, 'score', 0)
                max_score = self._get_value(qr, 'max_score', 1)
                question = self._get_value(qr, 'question', '')
                
                question_stats[key]['total_score'] += score
                question_stats[key]['max_score'] += max_score
                question_stats[key]['count'] += 1
                question_stats[key]['questions'].append(question)
        
        # Calculate averages and create visualization
        question_performance = []
        for q_num, stats in question_stats.items():
            if stats['count'] > 0 and stats['max_score'] > 0:
                avg_percentage = (stats['total_score'] / stats['max_score']) * 100
                question_performance.append({
                    'Question': q_num,
                    'Average Score (%)': avg_percentage,
                    'Attempts': stats['count']
                })
        
        # Display the performance data
        if question_performance:
            df = pd.DataFrame(question_performance)
            st.dataframe(df, use_container_width=True)
            
            # Create a bar chart for question performance
            fig = px.bar(
                df, 
                x='Question', 
                y='Average Score (%)',
                title="Average Performance by Question",
                color='Average Score (%)',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No question performance data available.")

    def display_detailed_assessment_list(self, results):
        """Display a detailed list of all assessments with their results"""
        st.subheader("📋 Detailed Assessment History")
        
        # Sort results by timestamp (most recent first)
        sorted_results = sorted(results, key=lambda x: self._get_value(x, 'timestamp'), reverse=True)
        
        for result in sorted_results:
            # Create a unique key for each assessment expander
            expander_key = f"assessment_{self._get_value(result, 'assessment_id')}"
            
            # Format the timestamp
            try:
                timestamp_str = self._get_value(result, 'timestamp')
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00').replace('+00:00', ''))
                formatted_date = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_date = self._get_value(result, 'timestamp', 'Unknown')
            
            # Create expander title with key metrics
            document_name = self._get_value(result, 'document_name', 'Unknown Document')
            percentage = self._get_value(result, 'percentage', 0)
            expander_title = f"📝 {document_name} - {formatted_date} - Score: {percentage:.1f}%"
            
            with st.expander(expander_title):
                # Assessment details in columns
                col1, col2, col3, col4 = st.columns(4)
                
                total_score = self._get_value(result, 'total_score', 0)
                max_total_score = self._get_value(result, 'max_total_score', 1)
                question_results = self._get_value(result, 'question_results', [])
                time_taken = self._get_value(result, 'time_taken')
                
                with col1:
                    st.metric("Total Score", f"{total_score:.1f}/{max_total_score:.1f}")
                
                with col2:
                    st.metric("Percentage", f"{percentage:.1f}%")
                
                with col3:
                    st.metric("Questions", len(question_results))
                
                with col4:
                    if time_taken:
                        st.metric("Time Taken", time_taken)
                
                # Performance grade
                if percentage >= 90:
                    grade = "A"
                    grade_color = "success"
                elif percentage >= 80:
                    grade = "B"
                    grade_color = "success"
                elif percentage >= 70:
                    grade = "C"
                    grade_color = "info"
                elif percentage >= 60:
                    grade = "D"
                    grade_color = "warning"
                else:
                    grade = "F"
                    grade_color = "error"
                
                st.markdown(f"**Grade: {grade}**", help=f"Based on {percentage:.1f}% score")
                
                # Question breakdown
                st.markdown("---")
                st.markdown("### 📝 Question Breakdown")
                
                for i, qr in enumerate(question_results):
                    # Create a container for each question with a border
                    with st.container():
                        # Get question data using the helper method
                        score = self._get_value(qr, 'score', 0)
                        max_score = self._get_value(qr, 'max_score', 1)
                        question = self._get_value(qr, 'question', 'No question available')
                        user_answer = self._get_value(qr, 'user_answer', 'No answer provided')
                        correct_answer = self._get_value(qr, 'correct_answer', 'No correct answer available')
                        feedback = self._get_value(qr, 'feedback', '')
                        
                        # Determine question status and color
                        if score == max_score:
                            status_icon = "✅"
                            status_color = "success"
                        elif score > 0:
                            status_icon = "🔶"
                            status_color = "info"
                        else:
                            status_icon = "❌"
                            status_color = "error"
                        
                        # Question header with score
                        st.markdown(f"""
                        <div style='padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; margin: 10px 0;'>
                            <h6 style='margin: 0;'>{status_icon} Question {i+1} - <span style='font-size: 0.9em; font-style: italic;'>Score: {score:.1f}/{max_score:.1f} points</span></h6>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Question content in a wide container
                        with st.container():
                            # Question
                            st.markdown("#### Question:")
                            st.markdown(f"<div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin: 5px 0;'>{question}</div>", unsafe_allow_html=True)
                            
                            # Answers in a single column
                            st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Your Answer:</span>", unsafe_allow_html=True)
                            if status_color == "success":
                                st.success(user_answer)
                            elif status_color == "info":
                                st.info(user_answer)
                            else:
                                st.error(user_answer)
                            
                            st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Expected Answer:</span>", unsafe_allow_html=True)
                            st.success(correct_answer)
                            
                            # Feedback in full width
                            if feedback:
                                st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Feedback:</span>", unsafe_allow_html=True)
                                feedback_container = st.container()
                                with feedback_container:
                                    if status_color == "success":
                                        st.success(feedback)
                                    elif status_color == "info":
                                        st.info(feedback)
                                    else:
                                        st.error(feedback)
                        
                        # Add spacing between questions
                        st.markdown("<br>", unsafe_allow_html=True)

    def _get_value(self, obj, key: str, default=None):
        """Helper method to get a value from either a dict or object safely.
        
        Args:
            obj: The object or dict to get the value from
            key: The key or attribute name to access
            default: Default value to return if key/attribute doesn't exist
            
        Returns:
            The value if found, otherwise the default value
        """
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def run(self):
        """Main application runner"""
        # Page configuration
        st.set_page_config(
            page_title="📚 PDF Assistant with Automated Scoring",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("📚 PDF Assistant with Automated Scoring")
        st.markdown("Upload a PDF document to chat with it and take automated assessments!")
        
        # Sidebar
        with st.sidebar:
            st.header("📁 Document Management")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload PDF Document",
                type=self.config.SUPPORTED_FILE_TYPES,
                help=f"Maximum file size: {self.config.MAX_FILE_SIZE_MB}MB"
            )
            
            # Create a placeholder for status messages
            status_placeholder = st.empty()
            
            # Process uploaded file
            if uploaded_file is not None:
                if self.validate_uploaded_file(uploaded_file):
                    # Check if this is a new file or the same file
                    if (st.session_state.current_document != uploaded_file.name or 
                        st.session_state.vectorstore is None):
                        with st.spinner("Processing document..."):
                            self.process_uploaded_file(uploaded_file)
                    
                    status_placeholder.success(f"✅ Document loaded: {uploaded_file.name}")
                    
                    # Document info
                    if st.session_state.current_document_hash:
                        metadata = self.processor.load_metadata()
                        if st.session_state.current_document_hash in metadata:
                            doc_info = metadata[st.session_state.current_document_hash]
                            status_placeholder.info(f"Chunks: {doc_info.get('chunks_count', 'Unknown')}")
                    
                    # Clear the status message after 5 seconds
                    time.sleep(5)
                    status_placeholder.empty()
            
            # Quick Actions
            if st.session_state.vectorstore is not None:
                st.markdown("---")
                st.header("🚀 Quick Actions")
                
                # First row with two buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📝 Summarize", use_container_width=True):
                        with st.spinner("Generating summary..."):
                            try:
                                response = st.session_state.qa_chain.run(
                                    "Please provide a comprehensive summary of the current document. Include main topic, key points, findings, and important conclusions."
                                )
                                st.session_state.chat_history.append(("user", "Summarize the document"))
                                st.session_state.chat_history.append(("assistant", response))
                            except Exception as e:
                                st.error(f"Error generating summary: {str(e)}")
                
                with col2:
                    if st.button("📚 Study Guide", use_container_width=True):
                        with st.spinner("Creating study guide..."):
                            try:
                                response = st.session_state.qa_chain.run(
                                    "Create a detailed study guide that includes main concepts, definitions, important facts, key processes, and critical points to remember."
                                )
                                st.session_state.chat_history.append(("user", "Create a study guide"))
                                st.session_state.chat_history.append(("assistant", response))
                            except Exception as e:
                                st.error(f"Error creating study guide: {str(e)}")
                
                # Second row with centered button
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                if st.button("❓ Sample Questions", use_container_width=True):
                    with st.spinner("Generating questions..."):
                        try:
                            response = st.session_state.qa_chain.run(
                                "Generate a set of potential examination questions including multiple choice, short answer, and essay questions with their answers."
                            )
                            st.session_state.chat_history.append(("user", "Generate sample questions"))
                            st.session_state.chat_history.append(("assistant", response))
                        except Exception as e:
                            st.error(f"Error generating questions: {str(e)}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Assessment Section
                st.markdown("---")
                st.header("🎯 Automated Assessment")
                
                if not st.session_state.assessment_mode:
                    difficulty = st.selectbox(
                        "Select Difficulty:",
                        ["easy", "medium", "hard"],
                        index=1
                    )
                    
                    num_questions = st.slider(
                        "Number of Questions:",
                        min_value=3,
                        max_value=10,
                        value=5
                    )
                    
                    if st.button("🎯 Start Assessment", type="primary"):
                        self.start_assessment(difficulty, num_questions)
                else:
                    st.info("📝 Assessment in progress...")
                    if st.button("❌ Cancel Assessment"):
                        st.session_state.assessment_mode = False
                        st.session_state.assessment_questions = []
                        st.session_state.user_answers = []
                        st.rerun()
                
                # Assessment History
                if st.button("📊 View Assessment History"):
                    st.session_state.show_history = True
        
        # Main content area
        if st.session_state.vectorstore is None:
            # Welcome message
            st.markdown("""
            ## Welcome to PDF Assistant with Automated Scoring! 🎉
            
            This application allows you to:
            - 📄 Upload and chat with PDF documents
            - 🤖 Get AI-powered answers to your questions
            - 🎯 Take automated assessments based on the document content
            - 📊 Track your learning progress with detailed scoring
            
            **To get started:**
            1. Upload a PDF document using the sidebar
            2. Wait for the document to be processed
            3. Start chatting or take an assessment!
            """)
        else:
            # Check if we should show assessment interface
            if st.session_state.assessment_mode:
                self.display_assessment_interface()
            elif getattr(st.session_state, 'show_history', False):
                self.display_assessment_history()
                if st.button("🔙 Back to Chat"):
                    st.session_state.show_history = False
                    st.rerun()
            else:
                # Regular chat interface
                # Display chat history
                for role, message in st.session_state.chat_history:
                    if role == "user":
                        with st.chat_message("user"):
                            st.write(message)
                    else:
                        with st.chat_message("assistant"):
                            st.write(message)
                
                # Chat input
                if user_question := st.chat_input("Ask a question about the document..."):
                    # Add user message to chat history
                    st.session_state.chat_history.append(("user", user_question))
                    
                    # Display user message
                    with st.chat_message("user"):
                        st.write(user_question)
                    
                    # Generate and display assistant response
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        
                        try:
                            # Create callback handler for streaming
                            callback_handler = StreamlitCallbackHandler(message_placeholder)
                            
                            # Get response
                            with st.spinner("Thinking..."):
                                response = st.session_state.qa_chain.run(
                                    user_question,
                                    callbacks=[callback_handler]
                                )
                            
                            # Final response
                            message_placeholder.markdown(response)
                            
                            # Add assistant response to chat history
                            st.session_state.chat_history.append(("assistant", response))
                            
                        except Exception as e:
                            error_msg = f"Sorry, I encountered an error: {str(e)}"
                            message_placeholder.error(error_msg)
                            st.session_state.chat_history.append(("assistant", error_msg))
                            logger.error(f"Error in chat: {e}")
        
        # Footer
        st.markdown("---")
        st.markdown("<div style='text-align: center;'>🤖 Powered by Google Gemini AI | CLINTON AGEBOBA</div>", unsafe_allow_html=True)

def main():
    """Main entry point"""
    try:
        app = ChatbotUI()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application startup error: {e}")

if __name__ == "__main__":
    main()
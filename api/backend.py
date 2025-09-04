import os
import time
import logging
import asyncio
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from core.llm_service import LLMService, GenerationRequest
from core.rag_service import RAGService
from core.preference_db import PreferenceDatabase, PreferenceRecord
from utils.prompts import ResponseStyle, get_next_style_in_sequence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Request/Response Models
class QuestionRequest(BaseModel):
    """Question request model"""
    question: str = Field(..., min_length=1, max_length=1000)
    user_id: str = Field(default="default")
    style: Optional[str] = Field(default=None)


class RegenerateRequest(BaseModel):
    """Regenerate request model"""
    question: str = Field(...)
    user_id: str = Field(...)
    session_id: str = Field(...)
    current_styles: List[str] = Field(...)


class PreferenceRequest(BaseModel):
    """Preference recording request"""
    user_id: str = Field(...)
    session_id: str = Field(...)
    question: str = Field(...)
    chosen_style: str = Field(...)
    generation_sequence: List[str] = Field(...)
    interaction_time: float = Field(default=0.0)


class QuestionResponse(BaseModel):
    """Question response model"""
    session_id: str
    question: str
    answer: str
    style: str
    generation_time: float
    context_used: bool
    sources_count: int
    metadata: Dict[str, Any]


class RegenerateResponse(BaseModel):
    """Regenerate response model"""
    session_id: str
    answer: str
    style: str
    generation_time: float
    metadata: Dict[str, Any]


class SystemStatusResponse(BaseModel):
    """System status response"""
    status: str
    model_status: str
    total_chunks: int
    performance_stats: Dict[str, Any]
    uptime: float


# Session Management
@dataclass
class ConversationSession:
    """Conversation session data"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    question: str
    generated_styles: List[str]  # Styles already generated
    current_answer: str
    current_style: str
    rag_context: str
    sources_count: int
    metadata: Dict[str, Any]


class SessionManager:
    """Simple in-memory session manager"""

    def __init__(self, max_sessions: int = 100, session_timeout: int = 3600):
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.logger = logging.getLogger("session_manager")

    def create_session(self, user_id: str, question: str, answer: str,
                       style: str, rag_context: str, sources_count: int) -> str:
        """Create new conversation session"""
        session_id = str(uuid.uuid4())
        now = datetime.now()

        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            question=question,
            generated_styles=[style],
            current_answer=answer,
            current_style=style,
            rag_context=rag_context,
            sources_count=sources_count,
            metadata={}
        )

        # Cleanup if needed
        self._cleanup_expired_sessions()

        if len(self.sessions) >= self.max_sessions:
            oldest_session_id = min(
                self.sessions.keys(),
                key=lambda sid: self.sessions[sid].last_activity
            )
            del self.sessions[oldest_session_id]

        self.sessions[session_id] = session
        self.logger.info(f"Created session {session_id} for user {user_id}")

        return session_id

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get session by ID"""
        session = self.sessions.get(session_id)
        if session:
            session.last_activity = datetime.now()
        return session

    def update_session(self, session_id: str, answer: str, style: str):
        """Update session with new answer"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.current_answer = answer
            session.current_style = style
            if style not in session.generated_styles:
                session.generated_styles.append(style)
            session.last_activity = datetime.now()

    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if (now - session.last_activity).seconds > self.session_timeout
        ]

        for sid in expired_sessions:
            del self.sessions[sid]

        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            "active_sessions": len(self.sessions),
            "max_sessions": self.max_sessions,
            "session_timeout": self.session_timeout
        }


# Main API Service
class APIService:
    """Main API service"""

    def __init__(self, config_path: str = "config.yaml"):
        self.logger = logging.getLogger("api_service")
        self.config = self._load_config(config_path)

        # Initialize services
        self.llm_service: Optional[LLMService] = None
        self.rag_service: Optional[RAGService] = None
        self.preference_db: Optional[PreferenceDatabase] = None
        self.session_manager = SessionManager(
            max_sessions=100,
            session_timeout=self.config.get("api", {}).get("session_timeout", 3600)
        )

        # System state
        self.start_time = time.time()
        self.request_count = 0
        self.initialization_complete = False

        self.logger.info("API service initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}

    async def initialize_services(self):
        """Initialize core services with proper model loading"""
        try:
            self.logger.info("Starting service initialization...")

            # Initialize LLM service
            self.logger.info("Initializing LLM service...")
            from core.llm_service import create_llm_service
            self.llm_service = create_llm_service(self.config)

            # Critical: Load the model immediately
            self.logger.info("Loading LLM model... This may take a minute...")
            model_loaded = self.llm_service.load_model()

            if not model_loaded:
                self.logger.error("Failed to load LLM model! Service will run but won't be able to generate responses.")
            else:
                self.logger.info("✓ LLM model loaded successfully!")
                # Test the model with a simple query
                try:
                    test_request = GenerationRequest(
                        question="Hello",
                        context="",
                        style=ResponseStyle.BALANCED,
                        user_id="system_test",
                        request_id="test_001"
                    )
                    test_response = self.llm_service.generate_response(test_request)
                    self.logger.info(f"✓ Model test successful. Response length: {len(test_response.content)}")
                except Exception as e:
                    self.logger.error(f"Model test failed: {e}")

            # Initialize RAG service
            self.logger.info("Initializing RAG service...")
            from core.rag_service import create_rag_service
            self.rag_service = create_rag_service(self.config)
            self.logger.info("✓ RAG service initialized")

            # Initialize preference database
            self.logger.info("Initializing preference database...")
            from core.preference_db import create_preference_database
            db_path = self.config.get("database", {}).get("path", "./data/database/preferences.db")
            self.preference_db = create_preference_database(db_path, self.config)
            self.logger.info("✓ Preference database initialized")

            # Load documents if directory exists
            doc_dir = self.config.get("paths", {}).get("documents_dir", "./data/documents")
            if os.path.exists(doc_dir):
                self.logger.info(f"Loading documents from {doc_dir}...")
                try:
                    chunk_count = self.rag_service.add_documents_from_directory(doc_dir)
                    self.logger.info(f"✓ Loaded {chunk_count} document chunks")
                except Exception as e:
                    self.logger.warning(f"Failed to load documents: {e}")
            else:
                self.logger.warning(f"Document directory not found: {doc_dir}")

            self.initialization_complete = True
            self.logger.info("=" * 50)
            self.logger.info("✓ All services initialized successfully!")
            self.logger.info("=" * 50)

        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Don't raise - let the service run even if initialization partially fails

    def _ensure_services_ready(self):
        """Ensure all services are ready with detailed error messages"""
        if not self.initialization_complete:
            raise HTTPException(
                status_code=503,
                detail="Services are still initializing. Please wait a moment and try again."
            )

        if self.llm_service is None:
            raise HTTPException(
                status_code=503,
                detail="LLM service not initialized. Check server logs for initialization errors."
            )

        if self.rag_service is None:
            raise HTTPException(
                status_code=503,
                detail="RAG service not initialized. Check server logs for initialization errors."
            )

        if not self.llm_service.is_model_ready():
            # Try to load model one more time
            self.logger.info("Model not ready, attempting to reload...")
            if not self.llm_service.load_model():
                raise HTTPException(
                    status_code=503,
                    detail="LLM model not loaded. Please check: 1) Model file exists at configured path, 2) Sufficient memory available, 3) llama-cpp-python is properly installed"
                )

    async def ask_question(self, request: QuestionRequest) -> QuestionResponse:
        """Handle question request"""
        self._ensure_services_ready()
        self.request_count += 1

        start_time = time.time()

        try:
            # Get RAG context
            rag_context = self.rag_service.get_context_for_query(request.question)
            retrieval_result = self.rag_service.search(request.question)
            sources_count = len(retrieval_result.chunks)

            # Determine style
            if request.style:
                style = ResponseStyle(request.style)
            else:
                # Use user's preferred style or default to balanced
                preferred_style = self.preference_db.get_preferred_style(request.user_id)
                style = ResponseStyle(preferred_style)

            # Generate response
            llm_request = GenerationRequest(
                question=request.question,
                context=rag_context,
                style=style,
                user_id=request.user_id,
                request_id=f"{request.user_id}_{int(time.time())}"
            )

            response = self.llm_service.generate_response(llm_request)

            # Create session
            session_id = self.session_manager.create_session(
                user_id=request.user_id,
                question=request.question,
                answer=response.content,
                style=style.value,
                rag_context=rag_context,
                sources_count=sources_count
            )

            generation_time = time.time() - start_time

            return QuestionResponse(
                session_id=session_id,
                question=request.question,
                answer=response.content,
                style=style.value,
                generation_time=generation_time,
                context_used=bool(rag_context.strip()),
                sources_count=sources_count,
                metadata={
                    "llm_generation_time": response.generation_time,
                    "token_count": response.token_count
                }
            )

        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

    async def regenerate_answer(self, request: RegenerateRequest) -> RegenerateResponse:
        """Handle regenerate request"""
        self._ensure_services_ready()

        session = self.session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        start_time = time.time()

        try:
            # Get next style in sequence
            next_style_name = get_next_style_in_sequence(request.current_styles)
            next_style = ResponseStyle(next_style_name)

            # Generate new response
            llm_request = GenerationRequest(
                question=session.question,
                context=session.rag_context,
                style=next_style,
                user_id=request.user_id,
                request_id=f"{request.user_id}_regen_{int(time.time())}"
            )

            response = self.llm_service.generate_response(llm_request)

            # Update session
            self.session_manager.update_session(
                request.session_id,
                response.content,
                next_style.value
            )

            generation_time = time.time() - start_time

            return RegenerateResponse(
                session_id=request.session_id,
                answer=response.content,
                style=next_style.value,
                generation_time=generation_time,
                metadata={
                    "llm_generation_time": response.generation_time,
                    "token_count": response.token_count
                }
            )

        except Exception as e:
            self.logger.error(f"Error regenerating answer: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to regenerate answer: {str(e)}")

    async def record_preference(self, request: PreferenceRequest):
        """Record user preference"""
        try:
            session = self.session_manager.get_session(request.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            # Create preference record
            record = PreferenceRecord(
                user_id=request.user_id,
                session_id=request.session_id,
                question=request.question,
                chosen_style=request.chosen_style,
                generation_sequence=request.generation_sequence,
                interaction_time=request.interaction_time,
                context_length=len(session.rag_context),
                timestamp=datetime.now()
            )

            # Store in database
            self.preference_db.record_preference(record)

            self.logger.info(f"Recorded preference for user {request.user_id}: {request.chosen_style}")

            return {"status": "success", "message": "Preference recorded"}

        except Exception as e:
            self.logger.error(f"Error recording preference: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to record preference: {str(e)}")

    async def get_system_status(self) -> SystemStatusResponse:
        """Get system status"""
        try:
            # LLM status
            if self.llm_service:
                model_status = self.llm_service.model_status.value
                llm_stats = self.llm_service.get_performance_stats()
            else:
                model_status = "not_initialized"
                llm_stats = {}

            # RAG status
            if self.rag_service:
                rag_stats = self.rag_service.get_statistics()
                total_chunks = rag_stats.get("total_chunks", 0)
            else:
                total_chunks = 0

            # System uptime
            uptime = time.time() - self.start_time

            return SystemStatusResponse(
                status="running" if self.initialization_complete else "initializing",
                model_status=model_status,
                total_chunks=total_chunks,
                performance_stats={
                    "llm_stats": llm_stats,
                    "session_stats": self.session_manager.get_session_stats(),
                    "total_requests": self.request_count,
                    "initialization_complete": self.initialization_complete
                },
                uptime=uptime
            )

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


# Create API service instance
api_service = APIService()

# Create FastAPI app
app = FastAPI(
    title="TutorAI API",
    description="Intelligent Academic Advisor with Preference Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],  # Added * for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logging.getLogger("api").info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s"
    )

    return response


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logging.info("=" * 50)
    logging.info("Starting TutorAI API Service...")
    logging.info("=" * 50)

    # Run initialization in background to not block startup
    asyncio.create_task(api_service.initialize_services())

    # Give it a moment to start initialization
    await asyncio.sleep(0.1)

    logging.info("API server is ready to accept requests.")
    logging.info("Service initialization continues in background...")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "TutorAI API",
        "status": "running",
        "initialization_complete": api_service.initialization_complete,
        "docs_url": "/docs"
    }


@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask question endpoint"""
    return await api_service.ask_question(request)


@app.post("/api/regenerate", response_model=RegenerateResponse)
async def regenerate_answer(request: RegenerateRequest):
    """Regenerate answer endpoint"""
    return await api_service.regenerate_answer(request)


@app.post("/api/preference")
async def record_preference(request: PreferenceRequest):
    """Record preference endpoint"""
    return await api_service.record_preference(request)


@app.get("/api/status", response_model=SystemStatusResponse)
async def get_system_status():
    """System status endpoint"""
    return await api_service.get_system_status()


@app.get("/api/admin/statistics")
async def get_admin_statistics():
    """Admin statistics endpoint"""
    if not api_service.preference_db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        stats = api_service.preference_db.get_system_statistics()
        return {"statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/user/{user_id}")
async def get_user_data(user_id: str):
    """Get user data endpoint"""
    if not api_service.preference_db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        user_stats = api_service.preference_db.get_user_statistics(user_id)
        return {"user_data": user_stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/reload-documents")
async def reload_documents():
    """Reload documents endpoint"""
    if not api_service.rag_service:
        raise HTTPException(status_code=503, detail="RAG service not initialized")

    try:
        doc_dir = api_service.config.get("paths", {}).get("documents_dir", "./data/documents")
        api_service.rag_service.rebuild_index(doc_dir)
        stats = api_service.rag_service.get_statistics()

        return {
            "status": "success",
            "message": "Documents reloaded successfully",
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/reset-user/{user_id}")
async def reset_user_preferences(user_id: str):
    """Reset user preferences endpoint"""
    if not api_service.preference_db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    try:
        success = api_service.preference_db.reset_user_preferences(user_id)
        if success:
            return {"status": "success", "message": f"Reset preferences for user {user_id}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reset preferences")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "initialization_complete": api_service.initialization_complete
    }

    # Check LLM service
    if api_service.llm_service and api_service.llm_service.is_model_ready():
        health_status["services"]["llm"] = "ready"
    else:
        health_status["services"]["llm"] = "not_ready"
        health_status["status"] = "degraded"

    # Check RAG service
    if api_service.rag_service:
        rag_stats = api_service.rag_service.get_statistics()
        if rag_stats.get("total_chunks", 0) > 0:
            health_status["services"]["rag"] = "ready"
        else:
            health_status["services"]["rag"] = "no_documents"
    else:
        health_status["services"]["rag"] = "not_initialized"
        health_status["status"] = "degraded"

    # Check database
    if api_service.preference_db:
        health_status["services"]["database"] = "ready"
    else:
        health_status["services"]["database"] = "not_initialized"

    return health_status


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "status_code": exc.status_code,
            "detail": exc.detail,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logging.getLogger("api").error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "status_code": 500,
            "detail": "An unexpected error occurred",
            "path": str(request.url.path)
        }
    )


def main():
    """Run the API server"""
    import argparse

    parser = argparse.ArgumentParser(description="TutorAI API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"Starting TutorAI API server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "api.backend:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
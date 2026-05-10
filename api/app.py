"""
CloudDash Support — REST API
Endpoints:
  POST /conversations                         — start a new conversation
  POST /conversations/{id}/messages          — send a message
  GET  /conversations/{id}                   — get conversation history
  GET  /conversations                         — list all conversations
  GET  /conversations/{id}/handovers         — audit log for a conversation
  GET  /health                                — system health check
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agents.models import (
    ConversationHistoryResponse,
    HealthResponse,
    SendMessageRequest,
    SendMessageResponse,
    StartConversationRequest,
    StartConversationResponse,
)
from agents.orchestrator import get_orchestrator
from config.logging_config import get_logger, setup_logging
from config.settings import get_settings
from handover.protocol import get_handover_logs
from retrieval.retriever import get_retriever

settings = get_settings()
setup_logging(level=settings.log_level, fmt=settings.log_format)
logger = get_logger(__name__)


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CloudDash Support API", env=settings.app_env)

    # Lazy loading for Render deployment
    # Heavy components like ChromaDB + SentenceTransformer
    # will initialize only on first request instead of startup.
    
    logger.info("System ready")
    
    yield
    
    logger.info("Shutting down CloudDash Support API")

# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CloudDash Multi-Agent Support API",
    description="AI-powered customer support system for CloudDash SaaS platform.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request logging ────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "HTTP request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


# ── Error handler ──────────────────────────────────────────────────────────

@app.exception_handler(KeyError)
async def key_error_handler(request: Request, exc: KeyError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Please try again."})


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """Lightweight system health check."""
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "CloudDash Support API"
    }


@app.post("/conversations", response_model=StartConversationResponse, tags=["Conversations"])
async def start_conversation(body: StartConversationRequest = StartConversationRequest()):
    """
    Start a new support conversation.
    Optionally provide an initial message to kick off intent classification immediately.
    """
    orchestrator = get_orchestrator()
    return orchestrator.start_conversation(initial_message=body.initial_message)


@app.post(
    "/conversations/{conversation_id}/messages",
    response_model=SendMessageResponse,
    tags=["Conversations"],
)
async def send_message(conversation_id: str, body: SendMessageRequest):
    """
    Send a message in an existing conversation.
    The system routes to the appropriate agent and returns the response.
    """
    orchestrator = get_orchestrator()
    try:
        return orchestrator.send_message(conversation_id, body.message)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id!r} not found.")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"LLM service error: {exc}")


@app.get(
    "/conversations/{conversation_id}",
    response_model=ConversationHistoryResponse,
    tags=["Conversations"],
)
async def get_conversation(conversation_id: str):
    """Retrieve full conversation history and metadata."""
    orchestrator = get_orchestrator()
    try:
        state = orchestrator.get_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id!r} not found.")

    return ConversationHistoryResponse(
        conversation_id=state.conversation_id,
        trace_id=state.trace_id,
        created_at=state.created_at,
        updated_at=state.updated_at,
        current_agent=state.current_agent,
        messages=state.messages,
        entities=state.entities,
        is_escalated=state.is_escalated,
        ticket_id=state.ticket_id,
    )


@app.get("/conversations", tags=["Conversations"])
async def list_conversations():
    """List all active conversations (useful for admin/debug)."""
    orchestrator = get_orchestrator()
    return orchestrator.list_conversations()


@app.get("/conversations/{conversation_id}/handovers", tags=["Audit"])
async def get_handovers(conversation_id: str):
    """Retrieve the handover audit log for a specific conversation."""
    logs = get_handover_logs(conversation_id=conversation_id)
    return {"conversation_id": conversation_id, "handovers": logs}


@app.get("/handovers", tags=["Audit"])
async def get_all_handovers():
    """Retrieve all handover events across all conversations."""
    return {"handovers": get_handover_logs()}

"""
Typed data models for conversation state, agent responses, and handover payloads.
All inter-module data transfer uses these Pydantic models.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class AgentType(str, Enum):
    TRIAGE = "triage"
    TECHNICAL_SUPPORT = "technical_support"
    BILLING = "billing"
    ESCALATION = "escalation"


class IntentCategory(str, Enum):
    TECHNICAL = "TECHNICAL"
    BILLING = "BILLING"
    ACCOUNT = "ACCOUNT"
    GENERAL = "GENERAL"
    ESCALATION = "ESCALATION"
    UNKNOWN = "UNKNOWN"


class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ─────────────────────────────────────────────────────────────────────────────
# Message & Conversation
# ─────────────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: Optional[AgentType] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KBCitation(BaseModel):
    article_id: str
    title: str
    category: str
    relevance_score: float


class AgentResponse(BaseModel):
    agent: AgentType
    content: str
    citations: list[KBCitation] = Field(default_factory=list)
    intent_detected: Optional[IntentCategory] = None
    entities: dict[str, Any] = Field(default_factory=dict)
    requires_handover: bool = False
    handover_to: Optional[AgentType] = None
    handover_reason: Optional[str] = None
    requires_escalation: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationState(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    current_agent: AgentType = AgentType.TRIAGE
    messages: list[Message] = Field(default_factory=list)
    entities: dict[str, Any] = Field(default_factory=dict)   # customer_id, plan, issue_type, etc.
    intent_history: list[IntentCategory] = Field(default_factory=list)
    handover_count: int = 0
    is_escalated: bool = False
    ticket_id: Optional[str] = None
    summary: Optional[str] = None   # Updated after each handover

    def add_message(self, role: MessageRole, content: str, agent: Optional[AgentType] = None, metadata: dict | None = None) -> None:
        self.messages.append(Message(
            role=role,
            content=content,
            agent=agent,
            metadata=metadata or {},
        ))
        self.updated_at = datetime.now(timezone.utc)

    def get_history_for_llm(self, max_messages: int = 20) -> list[dict[str, str]]:
        """Return recent messages formatted for the Gemini API."""
        relevant = [m for m in self.messages if m.role != MessageRole.SYSTEM]
        recent = relevant[-max_messages:]
        return [{"role": m.role.value, "parts": [{"text": m.content}]} for m in recent]


# ─────────────────────────────────────────────────────────────────────────────
# Handover Protocol
# ─────────────────────────────────────────────────────────────────────────────

class HandoverPayload(BaseModel):
    handover_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_agent: AgentType
    target_agent: AgentType
    reason: str
    conversation_id: str
    trace_id: str
    context_snapshot: dict[str, Any]   # entities, intent, summary
    priority: Priority = Priority.NORMAL
    success: bool = True
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# API Schemas
# ─────────────────────────────────────────────────────────────────────────────

class StartConversationRequest(BaseModel):
    initial_message: Optional[str] = None


class StartConversationResponse(BaseModel):
    conversation_id: str
    trace_id: str
    message: str
    agent: AgentType
    citations: list[KBCitation] = Field(default_factory=list)


class SendMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class SendMessageResponse(BaseModel):
    conversation_id: str
    trace_id: str
    message: str
    agent: AgentType
    citations: list[KBCitation] = Field(default_factory=list)
    intent: Optional[str] = None
    ticket_id: Optional[str] = None
    handover_occurred: bool = False


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    trace_id: str
    created_at: datetime
    updated_at: datetime
    current_agent: AgentType
    messages: list[Message]
    entities: dict[str, Any]
    is_escalated: bool
    ticket_id: Optional[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    kb_articles: int
    chroma_ready: bool

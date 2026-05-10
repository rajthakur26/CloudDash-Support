"""
Orchestrator — routes messages between agents, manages conversation state,
and coordinates handovers.

Design: stateless orchestrator + stateful ConversationState.
Adding a new agent = add it to AGENT_REGISTRY; no other changes needed.
"""
from __future__ import annotations

import uuid
from typing import Any

from agents.base_agent import BaseAgent
from agents.billing_agent import BillingAgent
from agents.escalation_agent import EscalationAgent
from agents.guardrails import check_input, redact_pii
from agents.models import (
    AgentResponse,
    AgentType,
    ConversationState,
    MessageRole,
    SendMessageResponse,
    StartConversationResponse,
)
from agents.technical_agent import TechnicalSupportAgent
from agents.triage_agent import TriageAgent
from config.logging_config import get_logger
from handover.protocol import execute_handover
from retrieval.retriever import KBRetriever, get_retriever

logger = get_logger(__name__)


def _build_registry(retriever: KBRetriever) -> dict[AgentType, BaseAgent]:
    """
    Central agent registry. To add a new agent:
    1. Create the agent class in agents/
    2. Add it here. The orchestrator requires NO other changes.
    """
    return {
        AgentType.TRIAGE: TriageAgent(retriever),
        AgentType.TECHNICAL_SUPPORT: TechnicalSupportAgent(retriever),
        AgentType.BILLING: BillingAgent(retriever),
        AgentType.ESCALATION: EscalationAgent(retriever),
    }


class Orchestrator:
    """
    Central controller for the multi-agent support system.
    Manages conversation state, routes messages, and handles agent handovers.
    """

    MAX_HANDOVERS_PER_TURN = 3  # prevent infinite loops

    def __init__(self) -> None:
        self._retriever: KBRetriever = get_retriever()
        self._agents: dict[AgentType, BaseAgent] = _build_registry(self._retriever)
        self._conversations: dict[str, ConversationState] = {}

        logger.info(
            "Orchestrator initialized",
            agents=list(a.value for a in self._agents.keys()),
            kb_chunks=self._retriever.article_count,
        )

    # ── Conversation lifecycle ─────────────────────────────────────────────

    def start_conversation(self, initial_message: str | None = None) -> StartConversationResponse:
        """Create a new conversation and optionally send the first message."""
        state = ConversationState()
        self._conversations[state.conversation_id] = state

        logger.info(
            "Conversation started",
            conversation_id=state.conversation_id,
            trace_id=state.trace_id,
        )

        if initial_message:
            response = self.send_message(state.conversation_id, initial_message)
            return StartConversationResponse(
                conversation_id=state.conversation_id,
                trace_id=state.trace_id,
                message=response.message,
                agent=response.agent,
                citations=response.citations,
            )

        # Default greeting from triage
        greeting = (
            "Hello! Welcome to CloudDash Support. I'm here to help you with any questions "
            "about your CloudDash account, technical issues, or billing inquiries. "
            "How can I assist you today?"
        )
        state.add_message(MessageRole.ASSISTANT, greeting, AgentType.TRIAGE)

        return StartConversationResponse(
            conversation_id=state.conversation_id,
            trace_id=state.trace_id,
            message=greeting,
            agent=AgentType.TRIAGE,
            citations=[],
        )

    def send_message(self, conversation_id: str, user_message: str) -> SendMessageResponse:
        """Process a user message and return the agent response."""
        state = self._get_state(conversation_id)

        logger.info(
            "Message received",
            conversation_id=conversation_id,
            trace_id=state.trace_id,
            current_agent=state.current_agent.value,
            message_preview=user_message[:80],
        )

        # ── Input guardrails ──────────────────────────────────────────
        check = check_input(user_message)
        if not check.allowed:
            state.add_message(MessageRole.USER, user_message)
            state.add_message(MessageRole.ASSISTANT, check.reason or "I cannot process that request.")
            return SendMessageResponse(
                conversation_id=conversation_id,
                trace_id=state.trace_id,
                message=check.reason or "I cannot process that request.",
                agent=state.current_agent,
            )

        clean_message = check.sanitized_text or user_message
        state.add_message(MessageRole.USER, clean_message)

        # ── Agent processing with handover loop ───────────────────────
        handover_count = 0
        final_response: AgentResponse | None = None
        handover_occurred = False

        while handover_count <= self.MAX_HANDOVERS_PER_TURN:
            current_agent = self._agents.get(state.current_agent)
            if current_agent is None:
                logger.error("Unknown agent type", agent=state.current_agent.value)
                break

            response = current_agent.process(clean_message, state)
            final_response = response

            # ── Handover decision ──────────────────────────────────────
            if response.requires_escalation and state.current_agent != AgentType.ESCALATION:
                payload = execute_handover(
                    state=state,
                    target_agent=AgentType.ESCALATION,
                    reason=response.handover_reason or "Agent requested escalation",
                )
                self._update_summary(state, response)
                handover_count += 1
                handover_occurred = True
                logger.info(
                    "Escalation handover",
                    handover_id=payload.handover_id,
                    conversation_id=conversation_id,
                )
                continue

            if response.requires_handover and response.handover_to:
                payload = execute_handover(
                    state=state,
                    target_agent=response.handover_to,
                    reason=response.handover_reason or "Agent requested handover",
                )
                self._update_summary(state, response)
                handover_count += 1
                handover_occurred = True
                logger.info(
                    "Agent handover",
                    from_agent=response.agent.value,
                    to_agent=response.handover_to.value,
                    conversation_id=conversation_id,
                )
                continue

            # No handover needed — we're done
            break

        if final_response is None:
            final_response = AgentResponse(
                agent=state.current_agent,
                content="I'm sorry, I encountered an issue processing your request. Please try again.",
            )

        # ── Output guardrails ─────────────────────────────────────────
        output_check = redact_pii(final_response.content)
        if output_check.was_modified:
            logger.info("PII redacted from output", conversation_id=conversation_id)
        clean_content = output_check.text

        state.add_message(
            MessageRole.ASSISTANT,
            clean_content,
            final_response.agent,
        )

        ticket_id = state.ticket_id or final_response.metadata.get("ticket_id")

        return SendMessageResponse(
            conversation_id=conversation_id,
            trace_id=state.trace_id,
            message=clean_content,
            agent=final_response.agent,
            citations=final_response.citations,
            intent=final_response.intent_detected.value if final_response.intent_detected else None,
            ticket_id=ticket_id,
            handover_occurred=handover_occurred,
        )

    def get_conversation(self, conversation_id: str) -> ConversationState:
        return self._get_state(conversation_id)

    def list_conversations(self) -> list[dict]:
        return [
            {
                "conversation_id": s.conversation_id,
                "created_at": s.created_at.isoformat(),
                "current_agent": s.current_agent.value,
                "message_count": len(s.messages),
                "is_escalated": s.is_escalated,
            }
            for s in self._conversations.values()
        ]

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_state(self, conversation_id: str) -> ConversationState:
        state = self._conversations.get(conversation_id)
        if state is None:
            raise KeyError(f"Conversation {conversation_id!r} not found.")
        return state

    def _update_summary(self, state: ConversationState, response: AgentResponse) -> None:
        """Generate a brief summary of the conversation for the receiving agent."""
        entities_str = ", ".join(f"{k}={v}" for k, v in state.entities.items()) if state.entities else "none"
        intents = ", ".join(i.value for i in state.intent_history)
        state.summary = (
            f"Conversation context: Customer contacted support with intent(s): {intents}. "
            f"Known entities: {entities_str}. "
            f"Last agent response: {response.content[:300]}..."
            if len(response.content) > 300
            else f"Last agent response: {response.content}"
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator

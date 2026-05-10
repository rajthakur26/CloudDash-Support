"""
Escalation Agent — packages context and simulates handover to human operator.
"""
from __future__ import annotations

import random
import string

from agents.base_agent import BaseAgent
from agents.models import (
    AgentResponse,
    AgentType,
    ConversationState,
    KBCitation,
    Priority,
)
from config.logging_config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Escalation Agent for CloudDash customer support.
Your role is to handle situations where the AI agents cannot fully resolve the customer's issue.

Your responsibilities:
1. Acknowledge the customer's frustration or unresolved concern with genuine empathy.
2. Assure them that a real human specialist is taking over.
3. Summarize the issue clearly so the customer knows you understand.
4. Provide a support ticket number and realistic expected response time:
   - CRITICAL (system outage, data loss): within 1 hour
   - HIGH (billing disputes, urgent access issues): within 4 business hours
   - NORMAL: within 24 business hours (Starter), 4 business hours (Pro), 1 hour (Enterprise)
5. Ask the customer to check their email for ticket confirmation.

IMPORTANT:
- Never minimize or dismiss the customer's concern.
- Be warm, professional, and reassuring.
- The human agent will have full context — tell the customer this explicitly.
- Generate the ticket ID in format TKT-XXXXXX (6 alphanumeric chars).
- You do NOT need to resolve the issue — your job is a smooth, compassionate handover.
"""


def _generate_ticket_id() -> str:
    chars = string.ascii_uppercase + string.digits
    return "TKT-" + "".join(random.choices(chars, k=6))


def _determine_priority(state: ConversationState) -> Priority:
    entities = state.entities
    intent_history = [i.value for i in state.intent_history]

    urgency = entities.get("urgency", "normal")
    plan = entities.get("plan", "").lower()

    if urgency == "high" or any(w in str(state.messages).lower() for w in ["outage", "down", "data loss"]):
        return Priority.CRITICAL

    if "BILLING" in intent_history or plan == "enterprise":
        return Priority.HIGH

    return Priority.NORMAL


class EscalationAgent(BaseAgent):
    agent_type = AgentType.ESCALATION
    name = "Escalation Agent"
    system_prompt = SYSTEM_PROMPT
    temperature = 0.2
    max_tokens = 512

    def _should_retrieve_kb(self, user_message: str) -> bool:
        return False  # Escalation doesn't need KB lookup

    def _build_user_prompt(self, user_message: str, state: ConversationState) -> str:
        priority = _determine_priority(state)
        plan = state.entities.get("plan", "Unknown")
        issue_summary = state.summary or user_message

        return (
            f"Customer Issue Summary: {issue_summary}\n"
            f"Customer Plan: {plan}\n"
            f"Priority Level: {priority.value}\n"
            f"Number of agents involved: {state.handover_count}\n\n"
            f"Customer's latest message: {user_message}\n\n"
            f"Please handle this escalation professionally and provide a ticket ID."
        )

    def _parse_response(
        self,
        raw_response: str,
        citations: list[KBCitation],
        state: ConversationState,
    ) -> AgentResponse:
        ticket_id = _generate_ticket_id()
        priority = _determine_priority(state)

        # Inject ticket ID if LLM didn't generate one naturally
        if "TKT-" not in raw_response:
            raw_response = raw_response.rstrip() + f"\n\n**Your Support Ticket:** {ticket_id}"

        # Update state
        state.is_escalated = True
        state.ticket_id = ticket_id

        logger.info(
            "Escalation created",
            ticket_id=ticket_id,
            priority=priority.value,
            conversation_id=state.conversation_id,
            trace_id=state.trace_id,
            plan=state.entities.get("plan"),
            handover_count=state.handover_count,
        )

        return AgentResponse(
            agent=AgentType.ESCALATION,
            content=raw_response,
            citations=[],
            requires_escalation=True,
            metadata={
                "ticket_id": ticket_id,
                "priority": priority.value,
            },
        )

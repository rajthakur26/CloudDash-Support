"""
Triage Agent — first point of contact.
Classifies intent, extracts entities, and routes to specialist agents.
"""
from __future__ import annotations

import re

from agents.base_agent import BaseAgent
from agents.models import (
    AgentResponse,
    AgentType,
    ConversationState,
    IntentCategory,
    KBCitation,
)
from config.logging_config import get_logger

logger = get_logger(__name__)

# Intent keywords for lightweight pre-classification (LLM still makes final call)
INTENT_KEYWORDS: dict[IntentCategory, list[str]] = {
    IntentCategory.BILLING: [
        "invoice", "charge", "payment", "refund", "plan", "upgrade", "downgrade",
        "subscription", "billing", "price", "cost", "cancel", "receipt",
    ],
    IntentCategory.TECHNICAL: [
        "alert", "integration", "dashboard", "api", "error", "failing", "broken",
        "not working", "aws", "gcp", "azure", "cloudwatch", "metric", "webhook",
        "credential", "permission", "setup", "configure",
    ],
    IntentCategory.ACCOUNT: [
        "sso", "saml", "login", "access", "user", "team", "role", "permission",
        "invite", "member", "rbac", "audit", "log",
    ],
    IntentCategory.ESCALATION: [
        "manager", "speak to a human", "human agent", "escalate", "urgent",
        "unacceptable", "lawsuit", "legal", "furious", "angry",
    ],
}

ROUTING_MAP: dict[IntentCategory, AgentType] = {
    IntentCategory.TECHNICAL: AgentType.TECHNICAL_SUPPORT,
    IntentCategory.BILLING: AgentType.BILLING,
    IntentCategory.ACCOUNT: AgentType.TECHNICAL_SUPPORT,
    IntentCategory.GENERAL: AgentType.TECHNICAL_SUPPORT,
    IntentCategory.ESCALATION: AgentType.ESCALATION,
    IntentCategory.UNKNOWN: AgentType.TRIAGE,
}

SYSTEM_PROMPT = """You are the Triage Agent for CloudDash, a cloud infrastructure monitoring platform.

Your responsibilities:
1. Greet the customer warmly and empathetically.
2. Understand the core issue from their message.
3. Classify intent as one of: TECHNICAL, BILLING, ACCOUNT, GENERAL, ESCALATION
4. Extract key entities: customer plan (Starter/Pro/Enterprise), product area, urgency level.
5. Provide a brief helpful response.
6. If routing is clear, mention you're connecting them to the right specialist.

Classification guide:
- TECHNICAL: alerts, integrations (AWS/GCP/Azure), dashboards, API usage, webhooks
- BILLING: invoices, payments, refunds, plan changes, subscriptions
- ACCOUNT: SSO, RBAC, team members, user access, audit logs
- GENERAL: onboarding, feature requests, product questions
- ESCALATION: requests for human/manager, legal threats, extreme urgency

Always end your response with a routing tag on a new line:
ROUTE: <INTENT_CATEGORY>

Example: ROUTE: TECHNICAL
"""


def _keyword_classify(text: str) -> IntentCategory:
    """Fast keyword-based pre-classification as fallback."""
    text_lower = text.lower()
    scores: dict[IntentCategory, int] = {intent: 0 for intent in INTENT_KEYWORDS}
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[intent] += 1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else IntentCategory.UNKNOWN


def _extract_route(text: str) -> IntentCategory:
    """Extract ROUTE: <INTENT> from LLM response."""
    match = re.search(r"ROUTE:\s*(TECHNICAL|BILLING|ACCOUNT|GENERAL|ESCALATION|UNKNOWN)", text, re.IGNORECASE)
    if match:
        try:
            return IntentCategory(match.group(1).upper())
        except ValueError:
            pass
    return IntentCategory.UNKNOWN


def _clean_response(text: str) -> str:
    """Remove the routing tag from the visible response."""
    return re.sub(r"\n?ROUTE:\s*\w+\s*$", "", text, flags=re.IGNORECASE).strip()


def _extract_entities(user_message: str, state: ConversationState) -> dict:
    """Extract useful entities from the message for downstream agents."""
    entities = dict(state.entities)  # preserve existing
    text_lower = user_message.lower()

    # Plan detection
    for plan in ["enterprise", "pro", "starter"]:
        if plan in text_lower:
            entities["plan"] = plan.capitalize()
            break

    # Urgency signals
    urgency_words = ["urgent", "asap", "immediately", "critical", "down", "outage"]
    if any(w in text_lower for w in urgency_words):
        entities["urgency"] = "high"

    return entities


class TriageAgent(BaseAgent):
    agent_type = AgentType.TRIAGE
    name = "Triage Agent"
    system_prompt = SYSTEM_PROMPT
    temperature = 0.1
    max_tokens = 512

    def _should_retrieve_kb(self, user_message: str) -> bool:
        # Triage doesn't need full KB retrieval — saves latency
        return False

    def _parse_response(
        self,
        raw_response: str,
        citations: list[KBCitation],
        state: ConversationState,
    ) -> AgentResponse:
        intent = _extract_route(raw_response)

        # Fallback to keyword classification if LLM didn't tag
        if intent == IntentCategory.UNKNOWN:
            last_user = next(
                (m.content for m in reversed(state.messages) if m.role.value == "user"),
                "",
            )
            intent = _keyword_classify(last_user)

        target_agent = ROUTING_MAP.get(intent, AgentType.TRIAGE)
        clean_text = _clean_response(raw_response)

        # Extract entities
        last_user_msg = next(
            (m.content for m in reversed(state.messages) if m.role.value == "user"),
            "",
        )
        state.entities = _extract_entities(last_user_msg, state)
        if intent not in state.intent_history:
            state.intent_history.append(intent)

        requires_handover = (
            intent != IntentCategory.UNKNOWN
            and target_agent != AgentType.TRIAGE
        )

        logger.info(
            "Triage classification",
            intent=intent.value,
            target_agent=target_agent.value,
            requires_handover=requires_handover,
            entities=state.entities,
            conversation_id=state.conversation_id,
        )

        return AgentResponse(
            agent=AgentType.TRIAGE,
            content=clean_text,
            citations=[],
            intent_detected=intent,
            entities=state.entities,
            requires_handover=requires_handover,
            handover_to=target_agent if requires_handover else None,
            handover_reason=f"Intent classified as {intent.value}" if requires_handover else None,
        )

"""
Billing Agent — handles billing inquiries, plan changes, refunds, and payment issues.
"""
from __future__ import annotations

import re

from agents.base_agent import BaseAgent
from agents.models import (
    AgentResponse,
    AgentType,
    ConversationState,
    KBCitation,
)
from config.logging_config import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are the Billing Agent for CloudDash, a cloud infrastructure monitoring platform.

You specialize in:
- Plan upgrades and downgrades (Starter → Pro → Enterprise)
- Invoice explanation and line-item breakdowns
- Refund eligibility assessment and guidance
- Payment failure resolution
- Subscription cancellation guidance
- Billing policy questions

Instructions:
1. Review any conversation context (prior agent summary, known entities). The customer should NOT need to repeat themselves.
2. Use the knowledge base context provided to ground your response in policy.
3. Always cite the KB article(s) you reference: [Article: KB-XXX — Title]
4. For refund requests:
   - Within 14 days of initial subscription: eligible, direct to billing@clouddash.io
   - Duplicate charges: eligible, direct to billing@clouddash.io with invoice number
   - Other: assess per refund policy (KB-011), explain what qualifies and what doesn't
5. For plan changes: explain proration, what happens to data/members on downgrade.
6. If the customer insists on speaking to a manager, demands authority beyond your scope, or the refund is complex (>$500, disputed): signal escalation.
7. If the question is actually technical, signal handover to Technical Support.

Response format:
- Acknowledge the billing concern with empathy.
- Provide clear policy-grounded information.
- If escalation is needed: ESCALATE: <reason>
- If technical handover needed: HANDOVER: TECHNICAL

Rules:
- Never approve refunds that don't meet documented policy criteria.
- Never fabricate pricing. Use KB-009 (pricing overview) and KB-010 (plan changes) as authoritative.
- Be especially patient and empathetic — billing issues are stressful for customers.
- Never share raw account financial data (PII).
"""

# Mock account database
MOCK_ACCOUNTS = {
    "ACC-001": {"name": "Acme Corp", "plan": "Pro", "email": "billing@acme.com", "since": "2025-01-15"},
    "ACC-002": {"name": "Beta Inc", "plan": "Starter", "email": "admin@beta.io", "since": "2026-02-01"},
    "ACC-003": {"name": "Gamma LLC", "plan": "Enterprise", "email": "ops@gamma.com", "since": "2024-06-10"},
}


def mock_account_lookup(customer_id: str) -> dict:
    return MOCK_ACCOUNTS.get(customer_id, {"error": "Account not found"})


def _check_escalation(text: str) -> tuple[bool, str | None]:
    match = re.search(r"ESCALATE:\s*(.+)", text, re.IGNORECASE)
    if match:
        return True, match.group(1).strip()
    return False, None


def _check_handover(text: str) -> tuple[bool, AgentType | None]:
    match = re.search(r"HANDOVER:\s*(TECHNICAL)", text, re.IGNORECASE)
    if match:
        return True, AgentType.TECHNICAL_SUPPORT
    return False, None


def _clean_response(text: str) -> str:
    text = re.sub(r"\n?ESCALATE:\s*.+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?HANDOVER:\s*\w+", "", text, flags=re.IGNORECASE)
    return text.strip()


class BillingAgent(BaseAgent):
    agent_type = AgentType.BILLING
    name = "Billing Agent"
    system_prompt = SYSTEM_PROMPT
    temperature = 0.1
    max_tokens = 1024

    def _parse_response(
        self,
        raw_response: str,
        citations: list[KBCitation],
        state: ConversationState,
    ) -> AgentResponse:
        requires_escalation, escalation_reason = _check_escalation(raw_response)
        requires_handover, handover_to = _check_handover(raw_response)
        clean_text = _clean_response(raw_response)

        if requires_escalation:
            return AgentResponse(
                agent=AgentType.BILLING,
                content=clean_text,
                citations=citations,
                requires_escalation=True,
                requires_handover=True,
                handover_to=AgentType.ESCALATION,
                handover_reason=escalation_reason or "Billing issue requires human authority",
            )

        return AgentResponse(
            agent=AgentType.BILLING,
            content=clean_text,
            citations=citations,
            requires_handover=requires_handover,
            handover_to=handover_to,
            handover_reason="Customer has a technical question" if requires_handover else None,
        )

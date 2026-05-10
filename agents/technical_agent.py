"""
Technical Support Agent — resolves technical issues using KB articles.
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

SYSTEM_PROMPT = """You are the Technical Support Agent for CloudDash, a cloud infrastructure monitoring platform.

You specialize in:
- Alert configuration and troubleshooting (alert policies, thresholds, notification channels)
- Cloud integrations: AWS CloudWatch, GCP Cloud Monitoring, Azure Monitor
- Dashboard performance and configuration
- API usage, webhooks, and SDK issues
- SSO (SAML), RBAC, and access management

Instructions:
1. Review any conversation context provided (prior agent summary, known entities).
2. The knowledge base context will be provided to you — use it to ground your response.
3. Provide clear, numbered step-by-step troubleshooting steps where applicable.
4. Always cite the KB article(s) you used using the format: [Article: KB-XXX — Title]
5. If the KB contains no relevant information, say so explicitly and offer to escalate.
6. Generate code snippets where they help (e.g., IAM policies, API calls).
7. If the issue crosses into billing territory, signal a handover.
8. If you genuinely cannot resolve the issue, signal escalation.

Response format:
- Start with a brief acknowledgment of the issue.
- Provide your resolution steps or information.
- End citations naturally in the text.
- If you need to hand over, add on a new line: HANDOVER: BILLING or HANDOVER: ESCALATION

Rules:
- Never invent steps, features, or permissions not in the KB.
- If KB context says "No relevant knowledge base articles found," acknowledge the gap honestly.
- Be concise but complete. Bullet points and numbered lists are encouraged.
"""


def _check_handover(text: str) -> tuple[bool, AgentType | None, str | None]:
    match = re.search(r"HANDOVER:\s*(BILLING|ESCALATION)", text, re.IGNORECASE)
    if match:
        target_str = match.group(1).upper()
        target = AgentType.BILLING if target_str == "BILLING" else AgentType.ESCALATION
        reason = f"Technical agent requested handover to {target_str}"
        return True, target, reason
    return False, None, None


def _clean_response(text: str) -> str:
    return re.sub(r"\n?HANDOVER:\s*\w+\s*$", "", text, flags=re.IGNORECASE).strip()


class TechnicalSupportAgent(BaseAgent):
    agent_type = AgentType.TECHNICAL_SUPPORT
    name = "Technical Support Agent"
    system_prompt = SYSTEM_PROMPT
    temperature = 0.2
    max_tokens = 1024

    def _parse_response(
        self,
        raw_response: str,
        citations: list[KBCitation],
        state: ConversationState,
    ) -> AgentResponse:
        requires_handover, handover_to, handover_reason = _check_handover(raw_response)
        clean_text = _clean_response(raw_response)

        # Check if KB was empty — signal escalation
        kb_empty = "no relevant knowledge base articles found" in clean_text.lower()
        if kb_empty and not requires_handover:
            requires_escalation = True
        else:
            requires_escalation = False

        return AgentResponse(
            agent=AgentType.TECHNICAL_SUPPORT,
            content=clean_text,
            citations=citations,
            requires_handover=requires_handover,
            handover_to=handover_to,
            handover_reason=handover_reason,
            requires_escalation=requires_escalation,
        )

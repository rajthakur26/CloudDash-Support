"""
Agent Handover Protocol
- Packages conversation context for receiving agent
- Validates handover eligibility
- Logs every handover event with full audit trail
- Handles failed handovers with fallback to triage
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agents.models import (
    AgentType,
    ConversationState,
    HandoverPayload,
    MessageRole,
    Priority,
)
from config.logging_config import get_logger

logger = get_logger(__name__)

HANDOVER_LOG_PATH = Path("logs/handovers.jsonl")


def _ensure_log_dir() -> None:
    HANDOVER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _classify_priority(state: ConversationState, reason: str) -> Priority:
    """Classify handover priority based on conversation signals."""
    reason_lower = reason.lower()
    entities = state.entities

    if any(kw in reason_lower for kw in ["urgent", "down", "outage", "data loss", "critical"]):
        return Priority.CRITICAL
    if any(kw in reason_lower for kw in ["refund", "charge", "billing", "manager", "frustrated", "angry"]):
        return Priority.HIGH
    if entities.get("urgency") == "high":
        return Priority.HIGH

    return Priority.NORMAL


def _build_context_snapshot(state: ConversationState) -> dict:
    """Build a serializable snapshot of conversation context."""
    return {
        "conversation_id": state.conversation_id,
        "entities": state.entities,
        "intent_history": [i.value for i in state.intent_history],
        "handover_count": state.handover_count,
        "summary": state.summary or "",
        "message_count": len(state.messages),
        "last_user_message": next(
            (m.content for m in reversed(state.messages) if m.role == MessageRole.USER),
            "",
        ),
    }


def _log_handover(payload: HandoverPayload) -> None:
    """Append handover record to JSONL audit log."""
    _ensure_log_dir()
    record = {
        "handover_id": payload.handover_id,
        "timestamp": payload.timestamp.isoformat(),
        "source_agent": payload.source_agent.value,
        "target_agent": payload.target_agent.value,
        "reason": payload.reason,
        "conversation_id": payload.conversation_id,
        "trace_id": payload.trace_id,
        "priority": payload.priority.value,
        "success": payload.success,
        "error": payload.error,
        "context_snapshot": payload.context_snapshot,
    }
    with open(HANDOVER_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        "Handover logged",
        handover_id=payload.handover_id,
        source=payload.source_agent.value,
        target=payload.target_agent.value,
        reason=payload.reason,
        priority=payload.priority.value,
        success=payload.success,
        conversation_id=payload.conversation_id,
        trace_id=payload.trace_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

VALID_TRANSITIONS: dict[AgentType, set[AgentType]] = {
    AgentType.TRIAGE: {AgentType.TECHNICAL_SUPPORT, AgentType.BILLING, AgentType.ESCALATION},
    AgentType.TECHNICAL_SUPPORT: {AgentType.BILLING, AgentType.ESCALATION, AgentType.TRIAGE},
    AgentType.BILLING: {AgentType.TECHNICAL_SUPPORT, AgentType.ESCALATION, AgentType.TRIAGE},
    AgentType.ESCALATION: {AgentType.TRIAGE},
}


def execute_handover(
    state: ConversationState,
    target_agent: AgentType,
    reason: str,
) -> HandoverPayload:
    """
    Execute a handover from the current agent to target_agent.
    Updates conversation state. Logs the handover event.
    Falls back to TRIAGE if transition is invalid.
    """
    source_agent = state.current_agent

    # Validate transition
    allowed = VALID_TRANSITIONS.get(source_agent, set())
    if target_agent not in allowed:
        logger.warning(
            "Invalid handover transition — falling back to triage",
            source=source_agent.value,
            attempted_target=target_agent.value,
        )
        fallback_payload = HandoverPayload(
            source_agent=source_agent,
            target_agent=AgentType.TRIAGE,
            reason=f"Invalid transition from {source_agent} to {target_agent}. Falling back to Triage.",
            conversation_id=state.conversation_id,
            trace_id=state.trace_id,
            context_snapshot=_build_context_snapshot(state),
            priority=Priority.NORMAL,
            success=False,
            error=f"Transition {source_agent} → {target_agent} not permitted.",
        )
        _log_handover(fallback_payload)
        state.current_agent = AgentType.TRIAGE
        state.handover_count += 1
        return fallback_payload

    priority = _classify_priority(state, reason)
    context = _build_context_snapshot(state)

    payload = HandoverPayload(
        source_agent=source_agent,
        target_agent=target_agent,
        reason=reason,
        conversation_id=state.conversation_id,
        trace_id=state.trace_id,
        context_snapshot=context,
        priority=priority,
        success=True,
    )

    # Update state
    state.current_agent = target_agent
    state.handover_count += 1

    _log_handover(payload)
    return payload


def get_handover_logs(conversation_id: str | None = None) -> list[dict]:
    """Read and optionally filter handover logs."""
    if not HANDOVER_LOG_PATH.exists():
        return []
    records = []
    with open(HANDOVER_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    if conversation_id is None or record.get("conversation_id") == conversation_id:
                        records.append(record)
                except json.JSONDecodeError:
                    continue
    return records

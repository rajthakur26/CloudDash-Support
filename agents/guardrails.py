"""
Input and output guardrails:
  - Input:  prompt-injection detection, off-topic filtering, length check
  - Output: PII redaction using simple regex patterns (no spacy/presidio needed for deployment simplicity)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from config.logging_config import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────────────────────────────────────

INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore\s+(previous|all|your)\s+instructions?",
        r"disregard\s+(your\s+)?(system\s+prompt|instructions?)",
        r"you\s+are\s+now\s+",
        r"act\s+as\s+(if\s+)?",
        r"jailbreak",
        r"DAN\s+mode",
        r"pretend\s+you\s+(are|have\s+no)",
        r"forget\s+(everything|all)\s+(you|your)",
        r"your\s+new\s+(instructions?|role|task)\s+(is|are)",
        r"<\s*system\s*>",
        r"\[\s*system\s*\]",
    ]
]

# Very loose off-topic check — only block clear non-support requests
OFF_TOPIC_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(write\s+me\s+a\s+(poem|song|story|essay))\b",
        r"\b(generate\s+(image|picture|art))\b",
        r"\b(give\s+me\s+(lottery|stock)\s+tips?)\b",
    ]
]

# PII redaction patterns
PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Credit card numbers
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CREDIT_CARD_REDACTED]"),
    # SSN
    (re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "[SSN_REDACTED]"),
    # Generic long numeric sequences that look like account numbers
    (re.compile(r"\b\d{10,}\b"), "[ACCOUNT_NUMBER_REDACTED]"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InputCheckResult:
    allowed: bool
    reason: str | None = None
    sanitized_text: str | None = None


@dataclass
class OutputCheckResult:
    text: str
    was_modified: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail functions
# ─────────────────────────────────────────────────────────────────────────────

def check_input(text: str, max_length: int = 2000) -> InputCheckResult:
    """
    Run all input guardrails. Returns InputCheckResult.
    allowed=False means the message must be rejected and reason returned to user.
    """
    # Length check
    if len(text) > max_length:
        logger.warning("Input rejected: exceeds max length" + f" (length={len(text)})")
        return InputCheckResult(
            allowed=False,
            reason=f"Your message is too long (max {max_length} characters). Please shorten it.",
        )

    # Prompt injection detection
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning(
                "Input rejected: prompt injection detected",
                pattern=pattern.pattern,
                input_preview=text[:100],
            )
            return InputCheckResult(
                allowed=False,
                reason="I'm not able to process that request. Please describe your CloudDash support issue.",
            )

    # Off-topic filter
    for pattern in OFF_TOPIC_PATTERNS:
        if pattern.search(text):
            logger.warning("Input rejected: off-topic", pattern=pattern.pattern)
            return InputCheckResult(
                allowed=False,
                reason="I'm CloudDash's support assistant and can only help with CloudDash-related questions. How can I assist you with CloudDash today?",
            )

    return InputCheckResult(allowed=True, sanitized_text=text.strip())


def redact_pii(text: str) -> OutputCheckResult:
    """
    Scan output for PII patterns and redact them.
    """
    modified = False
    result = text
    for pattern, replacement in PII_PATTERNS:
        new_result, count = pattern.subn(replacement, result)
        if count > 0:
            modified = True
            result = new_result
            logger.info("PII redacted from output", pattern=pattern.pattern, count=count)

    return OutputCheckResult(text=result, was_modified=modified)

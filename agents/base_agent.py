"""
Base Agent — all specialist agents inherit from this.
Provides KB retrieval, LLM chat, and response parsing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agents.llm_client import GeminiClient
from agents.models import (
    AgentResponse,
    AgentType,
    ConversationState,
    KBCitation,
    MessageRole,
)
from config.logging_config import get_logger
from retrieval.retriever import KBRetriever

logger = get_logger(__name__)


class BaseAgent(ABC):
    agent_type: AgentType
    name: str
    system_prompt: str
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.2
    max_tokens: int = 1024

    def __init__(self, retriever: KBRetriever) -> None:
        self.retriever = retriever
        self._llm = GeminiClient(model_name=self.model_name, temperature=self.temperature)

    # ── Overridable hooks ─────────────────────────────────────────────────

    def _should_retrieve_kb(self, user_message: str) -> bool:
        """Override to skip KB retrieval for simple messages."""
        return True

    def _get_system_prompt(self, kb_context: str, state: ConversationState) -> str:
        """Build the full system prompt with KB context injected."""
        summary_block = ""
        if state.summary:
            summary_block = f"\n\n[CONVERSATION SUMMARY FROM PREVIOUS AGENT]\n{state.summary}\n"

        entities_block = ""
        if state.entities:
            entities_str = "\n".join(f"  - {k}: {v}" for k, v in state.entities.items())
            entities_block = f"\n\n[KNOWN CUSTOMER CONTEXT]\n{entities_str}\n"

        kb_block = f"\n\n[KNOWLEDGE BASE CONTEXT — use this to answer, always cite article IDs]\n{kb_context}" if kb_context else ""

        return self.system_prompt + summary_block + entities_block + kb_block

    def _build_user_prompt(self, user_message: str, state: ConversationState) -> str:
        return user_message

    # ── Core process method ───────────────────────────────────────────────

    def process(self, user_message: str, state: ConversationState) -> AgentResponse:
        """
        Main entry point. Called by the orchestrator for every user message.
        """
        logger.info(
            "Agent processing message",
            agent=self.agent_type.value,
            conversation_id=state.conversation_id,
            trace_id=state.trace_id,
            message_preview=user_message[:80],
        )

        # KB retrieval
        citations: list[KBCitation] = []
        kb_context = ""
        if self._should_retrieve_kb(user_message):
            query = self.retriever.rewrite_query(user_message, state.summary or "")
            results = self.retriever.retrieve(query, top_k=4)
            kb_context = self.retriever.format_context_for_prompt(results)
            citations = self.retriever.format_citations(results)

            logger.info(
                "KB retrieval done",
                agent=self.agent_type.value,
                results=len(results),
                trace_id=state.trace_id,
            )

        # Build prompts
        system_prompt = self._get_system_prompt(kb_context, state)
        user_prompt = self._build_user_prompt(user_message, state)

        # LLM call
        history = state.get_history_for_llm(max_messages=16)
        raw_response = self._llm.chat(
            system_prompt=system_prompt,
            history=history,
            user_message=user_prompt,
            max_tokens=self.max_tokens,
        )

        # Parse and return
        response = self._parse_response(raw_response, citations, state)

        logger.info(
            "Agent response generated",
            agent=self.agent_type.value,
            conversation_id=state.conversation_id,
            trace_id=state.trace_id,
            requires_handover=response.requires_handover,
            requires_escalation=response.requires_escalation,
            citations_count=len(response.citations),
        )
        return response

    @abstractmethod
    def _parse_response(
        self,
        raw_response: str,
        citations: list[KBCitation],
        state: ConversationState,
    ) -> AgentResponse:
        """Parse the raw LLM response into a structured AgentResponse."""
        ...

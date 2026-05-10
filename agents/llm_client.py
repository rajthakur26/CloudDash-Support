"""
Gemini API client wrapper using the new google-genai SDK.
Handles authentication, retry logic, and response parsing.
"""
from __future__ import annotations

import time
from typing import Any

from google import genai
from google.genai import types

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


class GeminiClient:
    """Thin wrapper around the google-genai SDK with retry logic."""

    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.2) -> None:
        self.model_name = model_name
        self.temperature = temperature

    def chat(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_message: str,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> str:
        """
        Send a chat message and return the text response.
        history: list of {"role": "user"|"model", "parts": [{"text": "..."}]}
        """
        client = _get_client()

        # Build contents list from history
        contents: list[types.Content] = []
        for msg in history:
            role = msg["role"]
            if role == "assistant":
                role = "model"
            text = msg["parts"][0]["text"] if msg.get("parts") else msg.get("content", "")
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))

        # Add current user message
        contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            max_output_tokens=max_tokens,
            temperature=self.temperature,
        )

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                text = response.text.strip()
                logger.debug(
                    "Gemini response received",
                    model=self.model_name,
                    attempt=attempt,
                    response_length=len(text),
                )
                return text

            except Exception as exc:
                last_error = exc
                wait = 2 ** attempt
                logger.warning(
                    "Gemini API error — retrying",
                    attempt=attempt,
                    wait_seconds=wait,
                    error=str(exc),
                )
                if attempt < retries:
                    time.sleep(wait)

        raise RuntimeError(
            f"Gemini API failed after {retries} attempts: {last_error}"
        ) from last_error

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """Simple single-turn completion (no history)."""
        return self.chat(system_prompt="", history=[], user_message=prompt, max_tokens=max_tokens)

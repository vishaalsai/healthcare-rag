"""
Phase 1 – Generation
Anthropic Claude API client wrapper.
Supports standard completions and streaming, with retry logic.
"""

from __future__ import annotations

import os
import time
from typing import Iterator

from loguru import logger


class AnthropicClient:
    """
    Thin wrapper around the Anthropic SDK.

    Reads ANTHROPIC_API_KEY from environment (via python-dotenv).
    Implements simple exponential-backoff retry for rate-limit errors.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_tokens: int = 1500,
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self._client = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Send a single-turn message and return the text response.

        Args:
            system_prompt: Claude's system instruction.
            user_prompt:   The user message (contains context + question).

        Returns:
            Stripped text content of the first TextBlock.
        """
        client = self._get_client()

        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text = response.content[0].text
                logger.debug(
                    f"Claude response: {response.usage.input_tokens} in / "
                    f"{response.usage.output_tokens} out tokens"
                )
                return text.strip()

            except Exception as exc:
                is_rate_limit = "rate_limit" in str(exc).lower() or "529" in str(exc)
                if is_rate_limit and attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"Rate limit hit (attempt {attempt}/{self.max_retries}). "
                        f"Retrying in {delay:.1f}s …"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Anthropic API error on attempt {attempt}: {exc}")
                    raise

        raise RuntimeError("Max retries exceeded for Anthropic API call")

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Iterator[str]:
        """
        Streaming variant — yields text delta strings as they arrive.
        Useful for interactive CLI/UI usage.
        """
        client = self._get_client()

        with client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    # ------------------------------------------------------------------ #
    #  Private                                                             #
    # ------------------------------------------------------------------ #

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "anthropic SDK not installed. Run: pip install anthropic"
                ) from exc

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY not set. Add it to your .env file."
                )

            self._client = anthropic.Anthropic(api_key=api_key)
            logger.debug(f"Anthropic client initialised (model={self.model})")
        return self._client
